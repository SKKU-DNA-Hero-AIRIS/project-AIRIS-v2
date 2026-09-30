"""E4 기준선 B0, B1, B2 정의와 평가. 소유자: C. docs/tracks/C_optimize.md 단계 6.

    B0  기본 자세 (PoseParams() 기본값)                   — 안내 없이 통과할 때
    B1  기본 자세로 서서 몸 회전 (torso_yaw 0, 30, …, 330) — 퓨리움 공식 안내
        "머문 상태에서 몸을 회전"의 정적 등가물. 가능한 yaw 의 평균.
    B2  만세 (shoulder_abduction=180, elbow_flexion=0)     — 체형에 따라 천장에 걸리면 불가

총괄 결정 ⑧′. 팔 수평 90° 조건은 새 부스에서 불가라 삭제했다.

B1 집계:
    score, removal_by_part, total_removal = 가능한 yaw 결과의 평균
    discomfort                            = 기본 자세(yaw 0) 값
    가능한 yaw 가 없으면 불가 (score 는 전체 yaw 벌점 평균)
    시나리오 pose_bounds 밖의 yaw 는 뺀다 (휠체어는 torso_yaw [-45, 45] 라 −30, 0, 30 만 남는다).

scripts/run_baselines.py 와 scripts/run_e4.py 가 함께 쓴다.
"""
from __future__ import annotations

import numpy as np

from airis.sim import PART_NAMES, ZONE_NAMES, BodyParams, Evaluator, NozzleConfig, PoseParams, Scenario

from .cmaes_runner import is_infeasible
from .encoding import PoseEncoder

#: B1 의 몸 회전 각도 (deg). 180 초과는 pose_bounds [-180, 180] 에 맞춰 음수로 감싼다.
B1_YAWS_DEG: list[float] = [float(y) for y in range(0, 360, 30)]

#: 조건별 자세 목록. 자세가 여러 개면 가능한 자세의 평균을 낸다 (B1).
BASELINES: dict[str, list[PoseParams]] = {
    "B0": [PoseParams()],
    "B1": [PoseParams(torso_yaw=y if y <= 180 else y - 360) for y in B1_YAWS_DEG],
    "B2": [PoseParams(shoulder_abduction=180.0, elbow_flexion=0.0)],
}


def in_bounds(pose: PoseParams, scenario: Scenario) -> bool:
    """자유 변수가 전부 시나리오 pose_bounds 안인가 (fixed_pose 는 투영 때 덮어쓰므로 보지 않는다)."""
    for key, (lo, hi) in scenario.pose_bounds.items():
        if key in scenario.fixed_pose:
            continue
        if not lo <= getattr(pose, key) <= hi:
            return False
    return True


def evaluate_condition(
    evaluator: Evaluator,
    poses: list[PoseParams],
    encoder: PoseEncoder,
    nozzle: NozzleConfig,
    body: BodyParams,
    scenario: Scenario,
) -> dict:
    """조건 하나를 평가해 집계한다. 반환 dict 는 CSV 행의 재료가 된다."""
    usable = [p for p in poses if in_bounds(p, scenario)]
    if not usable:
        raise ValueError(
            f"시나리오 {scenario.name} 의 pose_bounds 안에 드는 기준 자세가 없다 "
            f"(자세 {len(poses)}개). 범위를 좁혀 돌리는 중이라면 기준선은 원래 범위로 평가한다."
        )
    # 시나리오 제약(pose_bounds + fixed_pose)으로 투영한다. 휠체어는 여기서 고관절·무릎이 90도가 된다.
    projected = [encoder.clip_pose(p) for p in usable]
    results = [evaluator.evaluate(p, nozzle, body, scenario) for p in projected]
    feasible = [r for r in results if not is_infeasible(r)]

    pool = feasible or results                 # 전부 불가면 벌점 평균을 남긴다
    return {
        "infeasible": not feasible,
        "n_poses": len(poses),
        "n_in_bounds": len(usable),
        "n_feasible": len(feasible),
        "yaws": ";".join(f"{p.torso_yaw:g}" for p in projected) if len(poses) > 1 else "",
        "score": float(np.mean([r.score for r in pool])),
        "total_removal": float(np.mean([r.total_removal for r in feasible])) if feasible else 0.0,
        # 첫 자세 = 단일 자세 또는 회전 없는 기본 자세(yaw 0).
        "discomfort": float(results[0].discomfort),
        "removal_by_part": (np.mean([r.removal_by_part for r in feasible], axis=0)
                            if feasible else np.zeros(len(PART_NAMES))),
        "pose": projected[0],
    }


def evaluate_all(
    evaluator: Evaluator,
    nozzle: NozzleConfig,
    body: BodyParams,
    scenario: Scenario,
) -> dict[str, dict]:
    """시나리오 하나에서 B0, B1, B2 를 모두 평가한다. {조건: evaluate_condition 결과}."""
    encoder = PoseEncoder(scenario)
    return {
        cond: evaluate_condition(evaluator, poses, encoder, nozzle, body, scenario)
        for cond, poses in BASELINES.items()
    }


# ---------- 계획 기준선 (E7, docs/plan_extension.md 6절) ----------

#: P1 의 몸 회전 단계 수 (제품 안내 "머문 상태에서 몸을 회전"을 순서로 평가한다).
P1_PHASES = 12


def plan_baseline(name: str, scenario: Scenario, limits, best_pose: PoseParams | None = None) -> "Plan":
    """계획 기준선.

        P0  기본 자세 1단계, 전 구역 최대 세기, 총 시간 상한 (현행 운전)
        P1  기본 자세로 몸을 12단계 회전 (제품 안내), 세기·시간은 P0 과 같다
        P2  단일 자세 최적(best_pose)을 K 단계 모두에, 세기·시간은 P0 과 같다

    시나리오 yaw 범위를 넘는 회전 단계는 범위 안으로 투영된다(휠체어 ±45°).
    """
    from airis.sim import Phase, Plan

    from .plan_encoding import PlanEncoder

    enc = PlanEncoder(scenario, limits)
    total = enc.limits.duration_bounds_s[1]
    zones = np.full(len(ZONE_NAMES), enc.limits.s_max, dtype=np.float64)
    if name == "P0":
        plan = Plan([Phase(enc.pose_encoder.clip_pose(PoseParams()), total)], zones)
    elif name == "P1":
        step = total / P1_PHASES
        plan = Plan([Phase(enc.pose_encoder.clip_pose(PoseParams(torso_yaw=y if y <= 180 else y - 360)), step)
                     for y in (360 * i / P1_PHASES for i in range(P1_PHASES))], zones)
    elif name == "P2":
        if best_pose is None:
            raise ValueError("P2 는 단일 자세 최적(best_pose)이 필요하다")
        plan = enc.plan_from_pose(best_pose, duration_s=total)
    else:
        raise ValueError(f"계획 기준선 이름은 P0 | P1 | P2: {name!r}")
    return Plan(plan.phases, enc.clip_zone_strengths(plan.zone_strengths))


def evaluate_plan_row(evaluator, plan, nozzle, body, scenario: Scenario) -> dict:
    """계획 하나를 평가해 CSV 행 재료로 만든다 (기준선·최적 결과 공통)."""
    from .cmaes_runner import is_infeasible

    result = evaluator.evaluate_plan(plan, nozzle, body, scenario)
    extra = result.extra or {}
    return {
        "infeasible": is_infeasible(result),
        "score": float(result.score),
        "total_removal": float(result.total_removal),
        "discomfort": float(result.discomfort),
        "energy": float(extra.get("energy", float("nan"))),
        "duration_s": float(extra.get("duration_s", plan.duration_s)),
        "n_phases": len(plan.phases),
        "phase_durations_s": ";".join(f"{p.duration_s:.2f}" for p in plan.phases),
        "zone_strengths": ";".join(f"{v:.3f}" for v in np.asarray(plan.zone_strengths, dtype=float)),
        "removal_by_part": np.asarray(result.removal_by_part, dtype=np.float64),
        "plan": plan,
    }
