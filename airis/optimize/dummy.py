"""평가기가 없는 동안 최적화 루프를 검증하기 위한 더미. 소유자: C.

D 의 PatchEvaluator 가 병합되면 이 파일은 단위 테스트 전용으로 남는다.
노즐은 평가에 쓰지 않는다. 실행 스크립트는 configs/nozzles.yaml 의 load_nozzles() 를 넘긴다.
docs/tracks/C_optimize.md 단계 2.
"""
from __future__ import annotations

import numpy as np

from airis.sim import (
    PART_NAMES, Evaluator, EvalResult, NozzleConfig, PoseParams, Scenario,
)

from .encoding import PoseEncoder


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
