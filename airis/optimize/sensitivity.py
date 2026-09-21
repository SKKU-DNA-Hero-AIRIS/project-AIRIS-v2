"""E3 민감도: 설정 덮어쓰기, 봉우리 판정, 순위 상관. 소유자: C. docs/tracks/C_optimize.md 단계 8.

설정은 configs/ 파일을 건드리지 않고 load_physics() 결과 dict(또는 Scenario)를 복사해 바꾼다.

키 형식
    physics 키            "jet.impingement.wall_jet_gain", "adhesion.fabric_roughness_factor" ...
    시나리오 키           "scenario.discomfort_weights.shoulder_abduction"

패치판에서 서로 같은 손잡이인 키 (00_common.md 4.2, 4.3):
    τ = 0.5·ρ·Cf·|u|², R = Φ((ln τ − ln τ_med)/σ), τ_med = critical_shear_pa_median × fabric_roughness_factor,
    u 는 출구 속도 U0 에 선형이다. 그래서 제거율은 Cf·U0²/τ_med 에만 의존하고
        air.friction_coeff × k      ≡ fabric_roughness_factor × 1/k
        jet.slot.exit_velocity_mps × k ≡ fabric_roughness_factor × 1/k²
    이다 (슬롯 노즐만 있는 배치에서, 수치로 1e−8 안 일치 확인). 입자판은 속도가 입자 수송에도
    들어가므로 이 등가가 성립하지 않는다.
"""
from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np

from airis.sim import PoseParams, Scenario

from .e4 import fold_pose

SCENARIO_PREFIX = "scenario."

#: 총괄·통합 관리가 정한 E3 기본 스윕 (2026-09-18). (키, 방식, 값들). 방식 "factor" 는 기준값에 곱한다.
#: friction_coeff 와 slot.exit_velocity_mps 는 패치판에서 fabric_roughness_factor 와 같은 손잡이라 뺐다
#: (roughness ×0.5 ≡ friction ×2 ≡ 출구 속도 ×√2, roughness ×2 ≡ friction ×0.5 ≡ 출구 속도 ×1/√2).
CORE_PRESET: list[tuple[str, str, list]] = [
    ("jet.impingement.wall_jet_gain", "value", [0.5, 2.0]),
    ("jet.impingement.enabled", "value", [False]),
    ("adhesion.fabric_roughness_factor", "factor", [0.5, 2.0]),
    ("jet.slot.spread_rate", "factor", [0.75, 1.25]),
    ("scoring.discomfort_weight", "factor", [0.5, 2.0]),
    ("scenario.discomfort_weights.shoulder_abduction", "factor", [0.5, 2.0]),
]


def get_setting(cfg: dict, scenario: Scenario, key: str):
    """현재 값. 없는 키면 KeyError (오타를 조용히 넘기지 않는다)."""
    if key.startswith(SCENARIO_PREFIX):
        field_name, _, sub = key[len(SCENARIO_PREFIX):].partition(".")
        value = getattr(scenario, field_name)
        if sub:
            if sub not in value:
                raise KeyError(f"{key}: 시나리오 {scenario.name} 의 {field_name} 에 {sub} 가 없다")
            value = value[sub]
        return value
    node = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"physics 설정에 {key} 가 없다")
        node = node[part]
    return node


def apply_setting(cfg: dict, scenario: Scenario, key: str, value) -> tuple[dict, Scenario]:
    """key 를 value 로 바꾼 (physics dict 복사본, Scenario 복사본). 원본은 건드리지 않는다."""
    get_setting(cfg, scenario, key)                       # 키 확인
    if key.startswith(SCENARIO_PREFIX):
        field_name, _, sub = key[len(SCENARIO_PREFIX):].partition(".")
        if sub:
            new_field = dict(getattr(scenario, field_name))
            new_field[sub] = value
        else:
            new_field = value
        return cfg, replace(scenario, **{field_name: new_field})
    new_cfg = copy.deepcopy(cfg)
    node = new_cfg
    parts = key.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    return new_cfg, scenario


def resolve_value(cfg: dict, scenario: Scenario, key: str, mode: str, v):
    """mode "factor" 면 기준값 × v, "value" 면 v 그대로."""
    if mode == "factor":
        return float(get_setting(cfg, scenario, key)) * float(v)
    if mode == "value":
        return v
    raise ValueError(f"알 수 없는 방식: {mode}")


def arm_class(pose: PoseParams | dict) -> str:
    """팔 봉우리: 벌림 90° 이상이면 hands_up, 아니면 arms_down."""
    abd = pose.shoulder_abduction if isinstance(pose, PoseParams) else pose["shoulder_abduction"]
    return "hands_up" if abd >= 90.0 else "arms_down"


def same_peak(pose, ref_pose, yaw_tol_deg: float = 30.0) -> bool:
    """같은 봉우리인가: 팔 봉우리가 같고 0~90° 로 접은 yaw(e4.fold_yaw) 차이가 yaw_tol_deg 이하."""
    return (arm_class(pose) == arm_class(ref_pose)
            and abs(fold_pose(pose)["torso_yaw"] - fold_pose(ref_pose)["torso_yaw"]) <= yaw_tol_deg)


def diverse_top(candidates: list[dict], k: int, min_dist: float) -> list[dict]:
    """가능한 후보를 점수순으로 훑어, 이미 고른 후보와 정규화 거리 min_dist 이상인 것만 k 개 고른다.

    CMA-ES 후보는 한 봉우리 근처에 몰려 있어 그냥 상위 k 개를 고르면 거의 같은 자세가 된다.
    """
    feasible = sorted((c for c in candidates if not c["infeasible"]), key=lambda c: -c["score"])
    chosen: list[dict] = []
    for c in feasible:
        if all(np.linalg.norm(c["x"] - o["x"]) >= min_dist for o in chosen):
            chosen.append(c)
            if len(chosen) == k:
                break
    return chosen


def spearman(a, b) -> float:
    """Spearman 순위 상관 (동점은 평균 순위). 원소가 2개 미만이거나 분산이 0 이면 NaN."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.size != b.size:
        return float("nan")

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, dtype=np.float64)
        r[order] = np.arange(v.size, dtype=np.float64)
        for val in np.unique(v):                      # 동점 평균 순위
            idx = np.where(v == val)[0]
            if idx.size > 1:
                r[idx] = r[idx].mean()
        return r

    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def regret(ref_pose_score: float, new_best: float) -> float:
    """설정이 바뀐 뒤 기준 설정의 최적 자세를 그대로 쓰면 잃는 비율: 1 − score(기준 자세)/새 best."""
    if not math.isfinite(new_best) or abs(new_best) < 1e-12:
        return float("nan")
    return 1.0 - ref_pose_score / new_best
