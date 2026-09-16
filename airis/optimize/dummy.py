"""평가기가 없는 동안 최적화 루프를 검증하기 위한 더미. 소유자: C.

D 의 PatchEvaluator 가 병합되면 이 파일은 단위 테스트 전용으로 남는다.
docs/tracks/C_optimize.md 단계 2.
"""
from __future__ import annotations

import numpy as np

from airis.sim import (
    PART_NAMES, Evaluator, EvalResult, NozzleConfig, PoseParams, Scenario,
)

from .encoding import PoseEncoder


# TODO(C): tests/fakes.py (D 소유) 가 병합되면 이 함수를 지우고 fakes.fake_nozzles 를 import 한다.
#          D 병합 전이라 docs/tracks/00_common.md 3절의 fake_nozzles 명세
#          ("좌우 벽 각 2개, 높이 1.0/1.4 m, 안쪽 향함") 를 그대로 옮겨 임시로 둔다.
#          configs/nozzles.yaml 의 실제 16개 배치와는 다르다. B 의 load_nozzles 가 병합되면 그쪽이 우선.
def temp_nozzles() -> NozzleConfig:
    """노즐 4개짜리 임시 배치. 부스 중앙 x=1.0, 벽면 y=±0.6, 높이 1.0/1.4 m."""
    positions, directions = [], []
    for wall_y, inward in ((-0.6, 1.0), (0.6, -1.0)):
        for z in (1.0, 1.4):
            positions.append((1.0, wall_y, z))
            directions.append((0.0, inward, 0.0))
    return NozzleConfig(
        positions=np.array(positions, dtype=np.float32),
        directions=np.array(directions, dtype=np.float32),
        strengths=np.ones(len(positions), dtype=np.float32),
    )


def discomfort(pose: PoseParams, scenario: Scenario, encoder: PoseEncoder | None = None) -> float:
    """docs/tracks/00_common.md 4.4 의 불편도.

    TODO(D): scoring.py 가 병합되면 그쪽 구현을 import 하고 이 함수를 지운다.
    """
    enc = encoder or PoseEncoder(scenario)
    base = PoseParams()
    total = 0.0
    for key, weight in scenario.discomfort_weights.items():
        if not weight or not hasattr(base, key):
            continue
        lo, hi = enc.bounds_for(key)
        total += float(weight) * abs(getattr(pose, key) - getattr(base, key)) / (hi - lo)
    return float(total)


class DummyEvaluator(Evaluator):
    """정규화 공간에서 목표 자세와의 거리에 음수를 취한 점수.

    물리가 없으므로 removal_by_part 는 0 이다. 루프·로그·재현성 검증 전용.
    """

    def __init__(self, target: PoseParams, scenario: Scenario) -> None:
        self.scenario = scenario
        self.encoder = PoseEncoder(scenario)
        # 목표도 시나리오 제약 안으로 투영해 둔다 (fixed_pose 가 있으면 도달 가능한 목표가 된다).
        self.target = self.encoder.clip_pose(target)
        self._target_x = self.encoder.encode(self.target)

    def evaluate(
        self,
        pose: PoseParams,
        nozzle: NozzleConfig,
        body,
        scenario: Scenario,
    ) -> EvalResult:
        x = self.encoder.encode(pose)
        dist2 = float(np.sum((x - self._target_x) ** 2))
        return EvalResult(
            score=-dist2,
            removal_by_part=np.zeros(len(PART_NAMES), dtype=np.float32),
            total_removal=0.0,
            discomfort=discomfort(pose, self.scenario, self.encoder),
            extra={"distance": float(np.sqrt(dist2)), "evaluator": "dummy"},
        )
