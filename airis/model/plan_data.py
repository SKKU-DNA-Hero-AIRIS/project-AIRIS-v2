"""계획 데이터셋 → 계획 모델의 출력 공간·한도·도장. 소유자: F.

계획 데이터셋의 스키마는 docs/experiments_model.md 4.2절 (C 와 합의 2026-10-06). 단계 수 N 과 계획 한도는
C 의 설계 결정값이라 코드에 고정하지 않고 데이터셋에서 읽는다.

    plan_limits_from_dataset(df)    데이터셋을 만든 한도 (C 의 PlanLimits)
    plan_space_from_dataset(df, …)  그 한도로 만든 출력 공간 (flow.PlanSpace)
    check_plan_dataset(df)          도장 확인: 설정 도장이 한 값인지, 지금 설정과 같은지
    plan_meta_from_dataset(df)      산출물 meta 에 옮길 도장·한도
"""
from __future__ import annotations

from collections.abc import Sequence

from airis.sim import Scenario

from .flow import PlanSpace, plan_phase_count
from .knn import PLAN_LIMIT_COLS, PLAN_STAMP_COLS, PROVENANCE_COLS, stamp_value

#: 한 값이어야 하는 열: 설정 도장과 계획 한도. 여러 값이면 서로 비교할 수 없는 행이 섞인 것이다
#: (계획 점수는 에너지 가중·단계 수·한도에 따라 달라진다).
SINGLE_VALUED_COLS = tuple(c for c in PLAN_STAMP_COLS if c not in PROVENANCE_COLS) + PLAN_LIMIT_COLS


def _single(df, col: str):
    """열의 값 하나 (없으면 None). 여러 값이면 ValueError."""
    if col not in df.columns or not len(df):
        return None
    values = sorted(set(df[col].astype(str)))
    if len(values) != 1:
        raise ValueError(f"계획 데이터셋의 {col} 가 여러 값이다 {values}. 한 조건(물리·단계 수·에너지 가중·한도)의 "
                         "행만 쓴다.")
    return df[col].iloc[0]


def plan_limits_from_dataset(df):
    """데이터셋을 만든 계획 한도 (C 의 PlanLimits).

    한도 열(duration_lo_s, duration_hi_s, min_phase_s, transition_s, s_max, cap_ratio)이 있으면 그 값이다.
    없는 값은 설정 파일에서 읽되, 총 시간 하한은 N × min_phase_s 이상으로 올린다 (C 의 run_e7 과 같은 규칙:
    N 단계가 각각 단계 최소 시간을 가지려면 총 시간이 그만큼은 되어야 한다).
    단계 수는 `n_phases` 열이 있으면 열 수(`plan_p<k>_…`)와 같아야 한다.
    """
    from dataclasses import replace

    from airis.optimize.plan_encoding import PlanLimits
    from airis.sim.scenario import load_physics

    n = plan_phase_count(df.columns)
    stamped = _single(df, "n_phases")
    if stamped is not None and int(stamped) != n:
        raise ValueError(f"n_phases 열({int(stamped)})과 계획 열의 단계 수({n})가 다르다")
    lim = replace(PlanLimits.from_config(load_physics()), n_phases=n)
    values = {c: _single(df, c) for c in PLAN_LIMIT_COLS}
    for key in ("min_phase_s", "transition_s", "s_max", "cap_ratio"):
        if values[key] is not None:
            lim = replace(lim, **{key: float(values[key])})
    lo, hi = values["duration_lo_s"], values["duration_hi_s"]
    lo = float(lo) if lo is not None else max(lim.duration_bounds_s[0], n * lim.min_phase_s)
    hi = float(hi) if hi is not None else max(lim.duration_bounds_s[1], lo)
    if lo < n * lim.min_phase_s - 1e-9:
        raise ValueError(f"총 시간 하한 {lo:g} s 가 단계 {n}개 × 최소 {lim.min_phase_s:g} s 보다 짧다")
    return replace(lim, duration_bounds_s=(lo, hi))


def plan_space_from_dataset(df, scenarios: Sequence[Scenario]) -> PlanSpace:
    """데이터셋의 단계 수와 한도로 만든 출력 공간. scenarios 는 학습에 쓰는 시나리오들 (자세 범위의 합집합)."""
    lim = plan_limits_from_dataset(df)
    return PlanSpace.from_scenarios(list(scenarios), n_phases=lim.n_phases, duration_bounds_s=lim.duration_bounds_s,
                                    s_max=lim.s_max, min_phase_s=lim.min_phase_s)


def plan_meta_from_dataset(df) -> dict:
    """산출물 meta 에 옮길 도장과 한도 (기본 타입). 출처 열(commit)은 여러 값이면 전부 적는다."""
    meta: dict = {}
    for col in PLAN_STAMP_COLS + PLAN_LIMIT_COLS:
        if col not in df.columns or not len(df):
            continue
        v = stamp_value(df[col]) if col in PROVENANCE_COLS else _single(df, col)
        meta["dataset_commit" if col == "commit" else col] = v.item() if hasattr(v, "item") else v
    lim = plan_limits_from_dataset(df)
    meta.update(n_phases=lim.n_phases, min_phase_s=lim.min_phase_s, duration_lo_s=lim.duration_bounds_s[0],
                duration_hi_s=lim.duration_bounds_s[1], transition_s=lim.transition_s, s_max=lim.s_max,
                cap_ratio=lim.cap_ratio)
    return meta


def check_plan_dataset(df) -> tuple[dict, list[str]]:
    """(데이터셋 도장, 경고 목록). 설정 도장·한도가 여러 값이면 ValueError.

    지금 설정과 비교하는 것: 노즐·물리 해시, 그리고 계획 채점 설정(kinetics_enabled, time_constant_s).
    기준은 predict.stamp_mismatch (산출물을 로드할 때와 같다). 출처 열(commit)은 여러 값이어도 경고만 한다.
    """
    from . import predict

    stamp: dict = {}
    warns: list[str] = []
    for col in PLAN_STAMP_COLS + PLAN_LIMIT_COLS:
        if col not in df.columns:
            continue
        if col in PROVENANCE_COLS:
            counts = df[col].astype(str).value_counts().to_dict()
            if len(counts) > 1:
                warns.append(f"{col}: 데이터셋에 여러 값이 있다 {counts}. 출처 기록이라 그대로 진행한다.")
            stamp[col] = stamp_value(df[col])
        else:
            v = _single(df, col)
            stamp[col] = v.item() if hasattr(v, "item") else v
    plan_limits_from_dataset(df)                                # 단계 수·한도가 서로 맞는지
    now = predict.current_stamp()
    for k in predict.stamp_mismatch(stamp, now):
        warns.append(f"{k}: 데이터셋 {stamp[k]} ≠ 지금 설정 {now[k]}. 물리·노즐 설정이나 계획 채점 설정이 "
                     "바뀐 것이다. 데이터셋부터 다시 만든다(C).")
    missing = [c for c in SINGLE_VALUED_COLS if c not in df.columns]
    if missing:
        warns.append(f"도장·한도 열이 없다: {missing}. 없는 한도는 설정 파일에서 읽고, 없는 도장은 비교하지 못한다.")
    return stamp, warns
