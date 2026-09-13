"""입자 기반 평가기 (Taichi GPU). 소유자: A.

설계 원칙
- 처음부터 배치 구조: 후보 B개 × 입자 N개를 한 커널에서 처리
- 입자 상태: 위치, 속도, 지름, 임계 전단, 부착 여부, 소속 후보 ID, 소속 부위
- 스텝: 속도장 조회 → 항력 → 적분 → 캡슐 충돌 → 부착/이탈 판정 → 재부착
- 결정론: seed 고정, 원자 연산 순서 의존 최소화

TODO(A):
- 1주차: 단일 후보, 항력 + 캡슐 충돌
- 2주차: 부착력 분포, 크기 분포, 배치 실행, batch_evaluate 오버라이드
- 3주차: 성능 (입자 2만 × 후보 100 → 수십 ms 목표)
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .interface import Evaluator
from .types import BodyParams, PoseParams, NozzleConfig, Scenario, EvalResult


class ParticleEvaluator(Evaluator):
    def __init__(self, physics_cfg: dict, arch: str = "gpu"):
        self.cfg = physics_cfg
        self.arch = arch

    def evaluate(self, pose: PoseParams, nozzle: NozzleConfig,
                 body: BodyParams, scenario: Scenario) -> EvalResult:
        raise NotImplementedError("A: 1주차 구현 대상")

    def batch_evaluate(self, candidates: Sequence[tuple[PoseParams, NozzleConfig]],
                       body: BodyParams, scenario: Scenario) -> np.ndarray:
        raise NotImplementedError("A: 2주차 구현 대상")
