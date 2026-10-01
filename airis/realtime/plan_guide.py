"""운전 계획 추천과 장비 제어 출력. 소유자: E. (`docs/plan_extension.md` 5절·7절 5번 "계획 시각화, 장비 제어 출력")

단일 자세 안내(`recommend.py`)를 **계획**(자세 순서 + 구역 세기 + 시간, `Plan`)으로 넓힌다.

`recommend_plan(body, scenario) -> PlanRecommendation`

1. 후보 풀 = C의 계획 모델 후보(`airis.model.predict.predict_plan_candidates`, 산출물 `data/models/plan_flow.pt`)
   + **계획 후보표**(`PLAN_STUB_TABLE`, E7 P5)를 `extra_candidates`로 함께 넘긴 것
   + **제품 안내 P1**(기본 자세로 12방향 회전, C의 `plan_baseline("P1")`).
   P1 은 12단계라 2단계 인코더(`clip_plan`)를 지나지 못해 모델 풀에 넣지 않고 따로 채점해 같은 자로 비교한다.
2. 전부 이 체형·몸 모델의 패치판으로 `evaluate_plan` 재채점해 최고를 고른다. 재채점 설정은 계획 경로 공통의
   `plan_physics_cfg()`(시간 의존 제거 켬)다. 산출물이 없거나 torch 가 없으면 표 + P1 만 쓴다.

P1 을 풀에 넣는 이유: E7(PR #118)에서 2단계 계획(P5)이 P1 보다 낮은 경우가 있었다. pregnant 는 에너지 가중과
무관하게 P1 이 높았고(w = 0.1: 0.4975 > 0.4532), wheelchair 는 w = 0 에서 P1 이 높았다(0.6386 > 0.6162).
제품 안내보다 낮은 계획을 화면이 "추천"으로 내보내지 않게, P1 이 이기면 P1 을 그대로 안내한다.

`device_control(plan, scenario) -> dict` 는 장비에 넘길 JSON 이다 (`docs/interfaces.md` "계획 모델" 장비 제어
출력): 몸 기준 구역 세기와 단계별 실제 노즐 12개 속도 비율(`apply_zone_strengths(…).strengths`), 풍량, 시각표.
E는 물리 계산을 하지 않는다. 점수·에너지는 D의 `PatchEvaluator.evaluate_plan` 값을 그대로 옮긴다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache, partial

import numpy as np

from ..optimize.baselines import plan_baseline
from ..optimize.plan_encoding import PlanEncoder, PlanLimits, plan_physics_cfg
from ..sim.body import build_body
from ..sim.patch_baseline import PatchEvaluator
from ..sim.scenario import (apply_zone_strengths, chest_wall_sign, load_physics, nozzle_zone_index,
                            zone_nozzle_counts, zone_strength_caps)
from ..sim.types import PART_NAMES, ZONE_NAMES, BodyParams, EvalResult, NozzleConfig, Phase, Plan, PoseParams, Scenario
from .recommend import (_nozzles, default_body, pose_instructions, resolve_model,
                        scoring_patches_per_m2)

#: 장비 제어 JSON 형식 이름. 필드를 바꾸면 숫자를 올린다.
DEVICE_SCHEMA = "airis.device_control.v1"
#: 단계 사이 전환 동안 팬 처리. D의 `evaluate_plan` 은 전환 동안 제거 0·에너지 0 으로 본다(00_common 4.6).
#: 그 가정과 같게 기본은 끈다. 장비 쪽 결정이 나면 "hold"(앞 단계 값 유지)로 바꿀 수 있다.
TRANSITION_FANS = ("off", "hold")
#: 계획 모델에서 받는 샘플 수. 2단계 계획 재채점이 1회 약 0.1~0.2 s 라 자세 추천(flow 8개)과 같은 수로 둔다.
PLAN_MODEL_SAMPLES = 8
ZONE_LABELS = {"chest_low": "가슴 쪽 벽 하단", "chest_high": "가슴 쪽 벽 상단",
               "back_low": "등 쪽 벽 하단", "back_high": "등 쪽 벽 상단", "top": "천장"}


@dataclass(frozen=True)
class PlanStubEntry:
    label: str
    plan: Plan
    source: str


def _phase(duration_s: float, *pose: float) -> Phase:
    return Phase(PoseParams(*pose), duration_s)


# E7 본 실행 P5 (자세 순서 + 세기 + 시간 전부 최적, 21차원), `scoring.energy_weight` 0.1 (설정 파일 값), 시드 0.
#   묶음 e7k14_20261001_012825_948aed (main 4edeb82, 메시 1,500/m², kinetics T_r 2 s), PR #118 docs/e7_reference.json.
#   자세는 0.1°, 시간은 0.01 s, 세기는 0.0001 로 반올림했다. yaw 는 접지 않은 원래 값이다 (2단계 yaw 는 1단계와의
#   상대 방향이 의미라 따로 접으면 안 된다). pregnant 가슴 구역 0.6 은 최적화 결과가 아니라 쾌적 상한이다.
#   PoseParams 순서: 어깨 벌림, 어깨 굽힘, 팔꿈치, 상체 숙임, 상체 회전(yaw), 고관절, 무릎.
E7_SOURCE = "E7 P5 e7k14_20261001_012825_948aed w=0.1 시드 0"

PLAN_STUB_TABLE: dict[str, list[PlanStubEntry]] = {
    "default": [PlanStubEntry(
        "만세로 한쪽 → 반대쪽 회전 (2단계)",
        Plan([_phase(7.20, 169.1, 13.9, 2.5, 3.1, 47.7, 0.2, 3.8),
              _phase(6.99, 169.7, 17.0, 20.2, 0.9, -55.6, 0.3, 9.4)],
             np.array([0.9998, 0.9992, 0.9957, 0.9980, 0.7996])),
        E7_SOURCE)],                                                      # E7 점수 0.8533, e 0.591, 14.2 s
    "pregnant": [PlanStubEntry(
        "팔 앞으로 들기 → 만세로 옆 회전 (2단계)",
        Plan([_phase(4.39, 154.6, 50.9, 106.7, 3.3, 0.4, 29.5, 8.5),
              _phase(4.96, 177.4, -0.9, 0.1, 0.8, -70.5, 1.5, 4.3)],
             np.array([0.6000, 0.6000, 0.9954, 0.9954, 0.5546])),
        E7_SOURCE)],                                                      # E7 점수 0.4532, e 0.214, 9.4 s
    "wheelchair": [PlanStubEntry(
        "만세로 한쪽 → 반대쪽 회전 (2단계)",
        Plan([_phase(5.74, 178.8, 3.1, 22.5, -3.3, 42.4, 90.0, 90.0),
              _phase(5.94, 179.4, 11.1, 10.3, -5.1, -36.7, 90.0, 90.0)],
             np.array([0.9887, 1.0000, 0.9914, 0.9969, 0.8625])),
        E7_SOURCE)],                                                      # E7 점수 0.5500, e 0.507, 11.7 s
}

P1_LABEL = "제품 안내: 기본 자세로 12방향 회전"
P0_LABEL = "현행 운전: 기본 자세, 전 구역 최대, 20초"


@dataclass(frozen=True)
class PlanCandidate:
    label: str
    plan: Plan
    source: str                      # "model" | "stub" | "P1"
    score: float
    infeasible: bool


@dataclass
class PlanRecommendation:
    plan: Plan
    label: str
    source: str                      # "model: plan_flow" | "stub: <exp>" | "baseline: P1"
    result: EvalResult
    candidates: list[PlanCandidate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0


@lru_cache(maxsize=1)
def plan_limits() -> PlanLimits:
    return PlanLimits.from_config(load_physics())


def plan_encoder(scenario: Scenario) -> PlanEncoder:
    """C의 계획 데이터셋·`predict_plan` 과 같은 인코더 (구역별 노즐 수 = 실제 배치)."""
    return PlanEncoder(scenario, plan_limits(), zone_nozzle_counts(_nozzles()))


def plan_evaluator(model: str | None = None) -> PatchEvaluator:
    """계획 채점기. 자세 채점기와 같은 몸 모델·밀도이고, 설정만 `plan_physics_cfg()`(시간 항 켬)다."""
    return _plan_evaluator(resolve_model(model))


@lru_cache(maxsize=4)
def _plan_evaluator(model: str) -> PatchEvaluator:
    return PatchEvaluator(plan_physics_cfg(), partial(build_body, model=model),
                          patches_per_m2=scoring_patches_per_m2(model))


def evaluate_plan_safely(ev: PatchEvaluator, plan: Plan, nozzle: NozzleConfig, body: BodyParams,
                         scenario: Scenario) -> tuple[EvalResult, str | None]:
    """`evaluate_plan` 결과와 오류 메시지. 평가기가 이 계획을 못 재면(ValueError) 불가 결과로 바꾼다.

    알려진 경우: 캡슐 몸 휠체어에서 yaw 에 따라 단계마다 패치 수가 달라져 D가 `ValueError` 를 낸다
    (B의 "패치 = 자세와 무관한 물질점" 보장 위반, 메시 몸에서는 재현되지 않음). 화면 전체를 멈추지 않고
    그 후보만 빼며, 사유는 호출자가 `notes` 로 보여 준다.
    """
    try:
        return ev.evaluate_plan(plan, nozzle, body, scenario), None
    except ValueError as exc:
        return EvalResult(score=float("-inf"), removal_by_part=np.zeros(len(PART_NAMES)), total_removal=0.0,
                          discomfort=0.0, extra={"infeasible": True, "error": str(exc),
                                                 "energy": float("nan"), "duration_s": plan.duration_s}), str(exc)


def product_rotation_plan(scenario: Scenario) -> Plan:
    """P1 (C의 `plan_baseline`). 휠체어는 yaw 범위(±45°)로 투영되고 임산부 가슴 구역은 쾌적 상한으로 줄어든다."""
    return plan_baseline("P1", scenario, plan_limits())


def current_operation_plan(scenario: Scenario) -> Plan:
    """P0 현행 운전 (C의 `plan_baseline`)."""
    return plan_baseline("P0", scenario, plan_limits())


def _model_candidates(body: BodyParams, scenario: Scenario, notes: list[str]) -> list[Plan] | None:
    """C의 계획 모델 샘플 (`clip_plan` 투영 후). 쓸 수 없으면 None 과 사유(`notes`).

    재채점은 E가 한다 (12단계 P1 과 같은 자로 재야 하므로). 그래서 `rescore=False` 로 샘플만 받고, 표 후보는
    `recommend_plan` 이 같은 풀에 더한다 (`rescore=False` 면 C는 `extra_candidates` 를 붙이지 않는다).
    """
    try:
        from ..model.predict import predict_plan_candidates
    except ImportError as exc:
        notes.append(f"계획 모델을 쓸 수 없어 계획 후보표(E7)로 안내합니다 ({exc})")
        return None
    try:
        pred = predict_plan_candidates(body, scenario, n_samples=PLAN_MODEL_SAMPLES, rescore=False,
                                       nozzle=_nozzles())
    except FileNotFoundError:
        notes.append("계획 모델 산출물(data/models/plan_flow.pt)이 없어 계획 후보표(E7)로 안내합니다")
        return None
    except ImportError as exc:
        notes.append(f"계획 모델 산출물은 있지만 torch 가 없어 계획 후보표(E7)로 안내합니다 ({exc})")
        return None
    except KeyError as exc:
        notes.append(f"계획 모델이 이 입력을 처리하지 못해 계획 후보표(E7)로 안내합니다 (KeyError: {exc})")
        return None
    return list(pred.candidates)


def recommend_plan(body: BodyParams | None, scenario: Scenario, *, use_model: bool = True,
                   include_product_rotation: bool = True, model: str | None = None) -> PlanRecommendation:
    """추천 계획 + 출처·후보 점수. 항상 계획 범위 안이고, 가능하면 부스 안이다.

    모든 후보를 같은 채점기(`plan_evaluator`)로 재서 고르므로, 결과는 표(E7 P5)나 제품 안내(P1)보다
    나빠지지 않는다. 쓸 후보가 없으면(전부 부스 밖이거나 평가 불가) P0(기본 자세)으로 안내한다.
    """
    t0 = time.perf_counter()
    model = resolve_model(model)
    body = body if body is not None else default_body(model)
    entries = PLAN_STUB_TABLE.get(scenario.name)
    if entries is None:
        raise KeyError(f"계획 후보표에 없는 시나리오: {scenario.name}")
    enc = plan_encoder(scenario)
    notes: list[str] = []

    pool: list[tuple[str, Plan, str]] = []
    if use_model:
        sampled = _model_candidates(body, scenario, notes)
        pool += [("모델 추천", p, "model") for p in sampled or []]
    pool += [(e.label, enc.clip_plan(e.plan), "stub") for e in entries]
    if include_product_rotation:
        pool.append((P1_LABEL, product_rotation_plan(scenario), "P1"))

    ev, nz = plan_evaluator(model), _nozzles()
    results = []
    for label, plan, _ in pool:
        r, err = evaluate_plan_safely(ev, plan, nz, body, scenario)
        if err is not None:
            notes.append(f"'{label}' 계획은 이 몸 모델에서 평가할 수 없어 제외 ({err})")
        results.append(r)
    cands = [PlanCandidate(label, plan, src, float(r.score), bool(r.extra.get("infeasible")))
             for (label, plan, src), r in zip(pool, results)]
    feasible = [i for i, c in enumerate(cands) if not c.infeasible]
    if not feasible:
        notes.append("쓸 수 있는 계획 후보가 없어(부스 밖 또는 평가 불가) 현행 운전(P0, 기본 자세)으로 안내")
        plan = current_operation_plan(scenario)
        return PlanRecommendation(plan, P0_LABEL, "fallback: P0",
                                  evaluate_plan_safely(ev, plan, nz, body, scenario)[0],
                                  cands, notes, time.perf_counter() - t0)

    best = max(feasible, key=lambda i: cands[i].score)
    pick = cands[best]
    if pick.source == "P1":
        two_phase = [cands[i].score for i in feasible if cands[i].source != "P1"]
        if two_phase:
            notes.append(f"이 체형·시나리오에서는 2단계 계획(최고 {max(two_phase):.3f})보다 제품 안내 12방향 회전"
                         f"({pick.score:.3f})이 높아 회전 안내를 그대로 씁니다")
        source = "baseline: P1"
    elif pick.source == "stub":
        source = f"stub: {next(e.source for e in entries if e.label == pick.label)}"
    else:
        source = "model: plan_flow"
    return PlanRecommendation(pick.plan, pick.label, source, results[best], cands, notes,
                              time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# 기준과 비교
# ---------------------------------------------------------------------------
@dataclass
class PlanScoreRow:
    name: str
    plan: Plan
    score: float
    total_removal: float
    energy: float
    duration_s: float
    discomfort: float
    removal_by_part: np.ndarray
    infeasible: bool


def _row(name: str, plan: Plan, r: EvalResult) -> PlanScoreRow:
    return PlanScoreRow(name, plan, float(r.score), float(r.total_removal),
                        float(r.extra.get("energy", float("nan"))), float(plan.duration_s),
                        float(r.discomfort), np.asarray(r.removal_by_part, dtype=np.float64),
                        bool(r.extra.get("infeasible")))


def compare_plan_with_baselines(body: BodyParams | None, scenario: Scenario, rec: PlanRecommendation,
                                model: str | None = None) -> list[PlanScoreRow]:
    """[추천, P0 현행 운전, P1 제품 안내]. 같은 채점기라 점수를 세로로 비교할 수 있다.

    에너지 e 는 현행 운전(전 팬 s = 1, 20 s) = 1 기준의 무차원 값이다 (00_common 4.7).
    """
    model = resolve_model(model)
    body = body if body is not None else default_body(model)
    ev, nz = plan_evaluator(model), _nozzles()
    rows = [_row("추천", rec.plan, rec.result)]
    for name, plan in (("P0 현행 운전", current_operation_plan(scenario)),
                       ("P1 제품 안내 (12방향 회전)", product_rotation_plan(scenario))):
        rows.append(_row(name, plan, evaluate_plan_safely(ev, plan, nz, body, scenario)[0]))
    return rows


# ---------------------------------------------------------------------------
# 안내 문장과 장비 제어 출력
# ---------------------------------------------------------------------------
def plan_instructions(plan: Plan, scenario: Scenario) -> list[tuple[str, list[str]]]:
    """단계별 (제목, 안내 문장). 단계가 많으면(P1) 한 덩어리로 줄인다."""
    if len(plan.phases) > 3:
        step = plan.phases[0].duration_s
        return [(f"{len(plan.phases)}단계 · 총 {plan.duration_s:.0f}초",
                 [f"기본 자세로 서서 약 {step:.1f}초마다 몸을 같은 방향으로 조금씩 돌려 한 바퀴 도세요."
                  if scenario.name != "wheelchair" else
                  f"앉은 채로 약 {step:.1f}초마다 몸을 좌우로 조금씩 돌리세요 (최대 약 45°)."])]
    out = []
    for k, ph in enumerate(plan.phases, start=1):
        lines = pose_instructions(ph.pose, scenario)
        if k > 1 and chest_wall_sign(ph.pose.torso_yaw) != chest_wall_sign(plan.phases[k - 2].pose.torso_yaw):
            lines = ["반대쪽 벽을 보도록 몸을 돌리세요."] + lines
        out.append((f"{k}단계 · {ph.duration_s:.1f}초", lines))
    return out


def _wall_label(zone: str, chest_sign: int) -> str:
    if zone == "top":
        return "top"
    sign = chest_sign if zone.startswith("chest") else -chest_sign
    return "+y" if sign > 0 else "-y"


def _r(x: float, nd: int = 4) -> float:
    return float(round(float(x), nd))


def transition_for(plan: Plan) -> float:
    """단계 사이 전환 시간. 인코더 계획(단계 ≤ `plan.n_phases`)은 `plan.transition_s`.

    단계가 그보다 많은 계획은 P1 처럼 **이어서 도는** 회전이라 멈춰 자세를 바꾸는 전환이 없다 (0).
    """
    limits = plan_limits()
    return limits.transition_s if len(plan.phases) <= limits.n_phases else 0.0


def device_control(plan: Plan, scenario: Scenario, *, nozzle: NozzleConfig | None = None,
                   physics_cfg: dict | None = None, result: EvalResult | None = None,
                   transition_fans: str = "off") -> dict:
    """장비 제어 JSON (`DEVICE_SCHEMA`). 표준 JSON 타입만 쓴다 (json.dumps 가능).

    - `segments`: 단계(`phase`)와 단계 사이 전환(`transition`)을 시간 순으로. 단계마다 노즐별 구역, 속도 비율
      (`apply_zone_strengths(nozzle, zone_strengths, 그 단계 yaw).strengths`), 풍량(비율 × 정격 풍량).
    - `zone_strengths`: 몸 기준 구역 세기 (계획 전체에서 하나).
    - `energy`: `result`(D의 `evaluate_plan`)가 있을 때만 옮긴다. E는 계산하지 않는다.
    """
    if transition_fans not in TRANSITION_FANS:
        raise ValueError(f"transition_fans 는 {TRANSITION_FANS} 중 하나: {transition_fans!r}")
    nozzle = nozzle if nozzle is not None else _nozzles()
    cfg = physics_cfg if physics_cfg is not None else load_physics()
    rated = float(cfg["fan"]["rated_flow_m3_min"])
    transition = transition_for(plan)
    zones = np.asarray(plan.zone_strengths, dtype=np.float64)

    segments: list[dict] = []
    t = 0.0
    prev: dict | None = None
    instructions = plan_instructions(plan, scenario) if len(plan.phases) <= 3 else None
    for k, ph in enumerate(plan.phases, start=1):
        if prev is not None and transition > 0:
            speed = prev["speed_ratio"] if transition_fans == "hold" else [0.0] * nozzle.count
            segments.append({"kind": "transition", "start_s": _r(t, 2), "end_s": _r(t + transition, 2),
                             "speed_ratio": speed,
                             "flow_m3_min": [_r(s * rated, 3) for s in speed],
                             "note": "자세 전환. 시뮬레이터는 이 동안 제거 0으로 본다"})
            t += transition
        yaw = float(ph.pose.torso_yaw)
        sign = chest_wall_sign(yaw)
        zone_idx = nozzle_zone_index(nozzle, yaw)
        speed = [_r(s) for s in apply_zone_strengths(nozzle, zones, yaw).strengths]
        seg = {"kind": "phase", "phase": k, "start_s": _r(t, 2), "end_s": _r(t + ph.duration_s, 2),
               "torso_yaw_deg": _r(yaw, 1), "chest_wall": "+y" if sign > 0 else "-y",
               "pose": {key: _r(v, 1) for key, v in vars(ph.pose).items()},
               "nozzle_zone": [ZONE_NAMES[i] for i in zone_idx],
               "nozzle_wall": [_wall_label(ZONE_NAMES[i], sign) for i in zone_idx],
               "speed_ratio": speed,
               "flow_m3_min": [_r(s * rated, 3) for s in speed],
               "total_flow_m3_min": _r(sum(speed) * rated, 2)}
        if instructions is not None:
            seg["instructions"] = instructions[k - 1][1]
        segments.append(seg)
        prev = seg
        t += ph.duration_s

    out = {
        "schema": DEVICE_SCHEMA,
        "scenario": scenario.name,
        "n_nozzles": int(nozzle.count),
        "zone_names": list(ZONE_NAMES),
        "zone_strengths": {z: _r(s) for z, s in zip(ZONE_NAMES, zones)},
        "zone_strength_caps": {z: (None if not np.isfinite(c) else _r(c))
                               for z, c in zip(ZONE_NAMES, zone_strength_caps(scenario))},
        "nozzle_positions_m": [[_r(v, 3) for v in p] for p in np.asarray(nozzle.positions)],
        "rated_flow_m3_min": rated,
        "spray_duration_s": _r(plan.duration_s, 2),
        "transition_s": _r(transition, 2),
        "transition_fans": transition_fans,
        "timeline_s": _r(t, 2),
        "segments": segments,
        "energy": None,
        "energy_reference": "현행 운전(전 팬 s = 1, 20 s) = 1",
    }
    if result is not None:
        out["energy"] = _r(result.extra.get("energy", float("nan")))
        out["score"] = _r(result.score)
        out["total_removal"] = _r(result.total_removal)
        out["infeasible"] = bool(result.extra.get("infeasible"))
        out["removal_by_part"] = {p: _r(v) for p, v in zip(PART_NAMES, result.removal_by_part)}
    return out
