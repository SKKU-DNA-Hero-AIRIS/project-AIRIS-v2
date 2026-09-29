"""평가기 공통 인터페이스. 입자판(A)과 패치판(D)이 동일 시그니처로 구현한다.

C(최적화)는 Evaluator만 알고, 어느 구현인지는 모른다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from .types import BodyParams, PoseParams, NozzleConfig, Scenario, EvalResult, Plan


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        pose: PoseParams,
        nozzle: NozzleConfig,
        body: BodyParams,
        scenario: Scenario,
    ) -> EvalResult:
        """후보 하나 평가. 결정론적이어야 한다 (같은 입력 → 같은 점수)."""

    def evaluate_plan(
        self,
        plan: Plan,
        nozzle: NozzleConfig,
        body: BodyParams,
        scenario: Scenario,
    ) -> EvalResult:
        """계획(자세 순서 + 구역 세기 + 시간) 평가. docs/plan_extension.md, 00_common 4.4·4.6·4.7.

        nozzle 은 기준 배치(구역 세기 적용 전)다. 평가기가 단계마다 B 의 apply_zone_strengths 로
        구역 세기를 입힌다. extra 에 energy, duration_s, removal_by_part_per_phase 를 넣는다.
        구현하지 않은 평가기는 NotImplementedError.
        """
        raise NotImplementedError(f"{type(self).__name__} 은 evaluate_plan 을 구현하지 않았다")

    def batch_evaluate(
        self,
        candidates: Sequence[tuple[PoseParams, NozzleConfig]],
        body: BodyParams,
        scenario: Scenario,
    ) -> np.ndarray:
        """후보 여러 개 평가 → (len(candidates),) 점수.

        기본 구현은 순차 호출. 입자판은 GPU 배치로 오버라이드할 것.
        """
        return np.array(
            [self.evaluate(p, n, body, scenario).score for p, n in candidates],
            dtype=np.float32,
        )
