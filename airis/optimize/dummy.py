"""평가기가 없는 동안 최적화 루프를 검증하기 위한 더미. 소유자: C.

실제 실행은 --evaluator patch (D 의 PatchEvaluator) 를 쓰고, 이 파일은 루프 단위 테스트 전용이다.
노즐은 평가에 쓰지 않는다. 실행 스크립트는 configs/nozzles.yaml 의 load_nozzles() 를 넘긴다.
docs/tracks/C_optimize.md 단계 2.
"""
from __future__ import annotations

import numpy as np

from airis.sim import (
    PART_NAMES, ZONE_NAMES, Evaluator, EvalResult, NozzleConfig, Plan, PoseParams, Scenario,
)
from airis.sim.scoring import discomfort

from .encoding import PoseEncoder


class DummyEvaluator(Evaluator):
    """정규화 공간에서 목표 자세와의 거리에 음수를 취한 점수.

    물리가 없으므로 removal_by_part 는 0 이다. 루프·로그·재현성 검증 전용.
    """

    def __init__(self, target: PoseParams, scenario: Scenario, energy_weight: float = 0.1) -> None:
        self.scenario = scenario
        self.energy_weight = float(energy_weight)      # evaluate_plan 의 에너지 가중 (E7 스윕 확인용)
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
            discomfort=discomfort(pose, self.scenario),
            extra={"distance": float(np.sqrt(dist2)), "evaluator": "dummy"},
        )

    def evaluate_plan(self, plan: Plan, nozzle: NozzleConfig, body, scenario: Scenario) -> EvalResult:
        """계획용 더미 (D 의 evaluate_plan 병합 전 개발·테스트용).

        단계마다 자세 품질 exp(점수) ∈ (0, 1] 에 시간 포화를 곱해 시간으로 가중 평균하고,
        세기·시간이 클수록 커지는 에너지를 뺀다. 품질을 양수로 두어야 "시간이 길수록 이득"이 된다.
        물리는 없지만 계획 탐색이 (a) 시간을 늘리면 이득, (b) 에너지가 크면 손해라는
        구조를 흉내 내 루프·인코더·로그를 검증할 수 있다 (docs/plan_extension.md 4.4·4.6·4.7 형태).
        """
        t_ref = 20.0
        total = plan.duration_s
        zones = np.asarray(plan.zone_strengths, dtype=np.float64)
        parts = []
        for phase in plan.phases:
            base = self.evaluate(phase.pose, nozzle, body, scenario)
            saturation = 1.0 - np.exp(-phase.duration_s / 2.0)          # 4.6 시간 포화 흉내
            quality = float(np.exp(base.score))                          # 자세 점수(≤ 0) → (0, 1]
            parts.append((phase.duration_s, quality * saturation, base.discomfort))
        weight = sum(t for t, _, _ in parts) or 1.0
        score = sum(t * v for t, v, _ in parts) / weight
        disc = sum(t * d for t, _, d in parts) / weight
        energy = float((zones ** 3).sum() / max(1, len(ZONE_NAMES)) * total / t_ref)   # 4.7 에너지
        return EvalResult(
            score=float(score - self.energy_weight * energy),
            removal_by_part=np.zeros(len(PART_NAMES), dtype=np.float32),
            total_removal=0.0,
            discomfort=float(disc),
            extra={"evaluator": "dummy", "energy": energy, "duration_s": total,
                   "removal_by_part_per_phase": [np.zeros(len(PART_NAMES), dtype=np.float32)
                                                 for _ in plan.phases]},
        )
