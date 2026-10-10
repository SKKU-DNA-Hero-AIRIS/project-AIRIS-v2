"""계획 데이터셋 → 계획 모델의 출력 공간·한도·도장. 소유자: F.

계획 데이터셋의 스키마는 docs/experiments_model.md 4.2절 (C 와 합의 2026-10-06). 단계 수 N 과 계획 한도는
C 의 설계 결정값이라 코드에 고정하지 않고 데이터셋에서 읽는다.

    plan_limits_from_dataset(df)    데이터셋을 만든 한도 (C 의 PlanLimits)
    plan_space_from_dataset(df, …)  그 한도로 만든 출력 공간 (flow.PlanSpace)
    check_plan_dataset(df)          도장 확인: 설정 도장이 한 값인지, 지금 설정과 같은지
    read_plan_dataset(path)         파일을 읽고 후보를 다시 거른다 (학습·평가는 이 함수로 읽는다)
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
    """열의 값 하나 (열이 없으면 None). 여러 값이거나 빈 값(NaN)이 있으면 ValueError.

    C 의 계획 데이터셋은 행마다 도장 전체를 넣으므로 도장·한도 열이 비는 일이 없다. 비어 있다면 그 열이 없던
    옛 코드의 파일을 이어 만든 것이라 행끼리 조건이 같은지 알 수 없다 (C 확인 2026-10-07).
    """
    if col not in df.columns or not len(df):
        return None
    filled = df[col].dropna()
    if len(filled) != len(df):
        raise ValueError(f"계획 데이터셋의 {col} 에 빈 값이 {len(df) - len(filled)}행 있다. 도장·한도 열은 모든 행에 "
                         "있어야 한다 (옛 코드로 만든 파일을 이어 만든 경우). 데이터셋을 다시 만든다(C).")
    values = sorted(set(filled.astype(str)))
    if len(values) != 1:
        raise ValueError(f"계획 데이터셋의 {col} 가 여러 값이다 {values}. 한 조건(물리·단계 수·에너지 가중·한도)의 "
                         "행만 쓴다.")
    return filled.iloc[0]


#: 후보를 다시 거르는 데 필요한 열 (C 의 plan_dataset.refilter_candidate_frame 이 행에서 읽는다).
REFILTER_COLS = ("cand_plan_raw", "cand_score", "candidate_min_dist", "n_phases") + PLAN_LIMIT_COLS


def refilter_plan_candidates(df, min_dist: float | None = None):
    """(후보를 다시 거른 DataFrame, 기록). 거를 수 없으면 df 를 그대로 돌려주고 기록에 이유를 적는다.

    2026-10-08 이전에 만든 계획 데이터셋은 후보 거리 필터가 탐색 공간 값으로 돌아, 저장된 후보에 거의 같은 계획이
    섞여 있다 (첫 60행 기준 후보의 약 1/3. C 의 PR #176). C 의 `refilter_candidate_frame` 으로 읽은 직후 한 번
    거르면 중복이 없어진다. 다만 후보가 `candidate_k` 에 닿은 행은 새 코드로 다시 만든 결과와 후보 구성이 다를 수
    있다 (interfaces.md). 새 코드로 만든 파일에는 아무 일도 하지 않는다 (같은 임계로 다시 거르는 것이라).

    임계는 주지 않으면 데이터셋의 `candidate_min_dist` 도장이다 (값을 코드에 고정하지 않는다). min_dist 를 주면 그
    값으로 거른다: 옛 파일의 저장 후보는 생성 때 잘못된 거리로 걸러진 것이라, 도장보다 낮은 임계를 주면 다시 만들지
    않고도 후보가 더 남는다 (총괄 2026-10-10: 0.15 와 0.12 를 비교). 새 코드로 만든 파일은 이미 도장 임계로 걸러져
    있어 더 낮은 임계를 줘도 달라지지 않는다.
    kNN 표는 후보 열을 쓰지 않아 영향이 없고, flow 학습만 달라진다 (행 안에서 중복된 계획이 두 번 세어지지 않는다).
    기록: {"applied", "reason", "min_dist"(쓴 임계), "stamp_min_dist"(파일 도장), "before", "after", "rows_changed"}
    (before·after 는 전체 후보 수).
    """
    info = {"applied": False, "reason": "", "min_dist": None, "stamp_min_dist": None, "before": None, "after": None,
            "rows_changed": 0}
    if min_dist is not None and not float(min_dist) > 0:
        raise ValueError(f"재필터 임계는 0 보다 커야 한다: {min_dist}")
    missing = [c for c in REFILTER_COLS if c not in df.columns]
    if missing:
        info["reason"] = f"열 없음: {missing}"
        return df, info
    if not len(df):
        info["reason"] = "행 없음"
        return df, info
    from airis.optimize.plan_dataset import refilter_candidate_frame

    stamp = float(_single(df, "candidate_min_dist"))
    min_dist = stamp if min_dist is None else float(min_dist)
    before = df["cand_score"].map(len)
    out = refilter_candidate_frame(df, min_dist=min_dist)
    after = out["cand_score"].map(len)
    info.update(applied=True, min_dist=min_dist, stamp_min_dist=stamp, before=int(before.sum()),
                after=int(after.sum()),
                rows_changed=int((before.to_numpy() != after.to_numpy()).sum()))
    return out, info


def read_plan_dataset(path, *, refilter: bool = True, min_dist: float | None = None):
    """(계획 데이터셋 DataFrame, 후보 재필터 기록). 학습·평가 스크립트는 이 함수로 읽는다.

    min_dist: 재필터 임계 (기본 None = 파일의 candidate_min_dist 도장). refilter_plan_candidates 참고.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    if not refilter:
        if min_dist is not None:
            raise ValueError("재필터를 끄면서 임계(min_dist)를 줄 수는 없다")
        return df, {"applied": False, "reason": "끔 (--no-refilter)", "min_dist": None, "stamp_min_dist": None,
                    "before": None, "after": None, "rows_changed": 0}
    return refilter_plan_candidates(df, min_dist)


def refilter_message(info: dict) -> str:
    """재필터 기록 한 줄."""
    if not info.get("applied"):
        return f"후보 재필터 안 함 ({info.get('reason')})"
    stamp = info.get("stamp_min_dist")
    note = "" if stamp is None or stamp == info["min_dist"] else f", 파일 도장 {stamp:g}"
    return (f"후보 재필터: {info['before']} → {info['after']}개 (임계 {info['min_dist']:g}{note}, "
            f"바뀐 행 {info['rows_changed']})")


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
        if v is None:
            continue
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
