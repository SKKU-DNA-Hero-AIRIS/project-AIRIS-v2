"""패치 기반 빠른 평가기. 소유자: D.

역할
- 입자판 완성 전까지 최적화 파이프라인을 끝까지 돌리는 기준선
- 입자판 디버깅 시 비교 대상
- 밀리초 단위여야 함 (numpy 벡터화)

TODO(D, 1주차):
- build_body → 패치별 유효 풍속 (velocity_field + 법선 각도 + 캡슐 가림)
- 풍속 → 제거율 (임계값 + 시그모이드)
- 부위별 집계, 불편도 페널티, EvalResult
"""
from __future__ import annotations

from .interface import Evaluator
from .types import BodyParams, PoseParams, NozzleConfig, Scenario, EvalResult


class PatchEvaluator(Evaluator):
    def __init__(self, physics_cfg: dict):
        self.cfg = physics_cfg

    def evaluate(self, pose: PoseParams, nozzle: NozzleConfig,
                 body: BodyParams, scenario: Scenario) -> EvalResult:
        raise NotImplementedError("D: 1주차 구현 대상")
