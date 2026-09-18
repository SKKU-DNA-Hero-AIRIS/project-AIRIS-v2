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

from airis.sim import PART_NAMES, BodyParams, Evaluator, NozzleConfig, PoseParams, Scenario

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
