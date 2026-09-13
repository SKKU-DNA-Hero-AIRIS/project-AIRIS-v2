"""CMA-ES 실행기. 소유자: C.

TODO(C, 1주차):
- 시나리오의 pose_bounds + 노즐 범위 → 정규화된 벡터 [-1, 1]^D 로 인코딩/디코딩
- fixed_pose 변수는 벡터에서 제외
- cma.CMAEvolutionStrategy 로 루프, 각 세대는 Evaluator.batch_evaluate 호출
- 로그: 실험 ID, 커밋 해시, 설정 파일 해시, 세대별 best → outputs/<exp_id>/
"""
from __future__ import annotations

from airis.sim import Evaluator, BodyParams, Scenario


def run_cmaes(evaluator: Evaluator, body: BodyParams, scenario: Scenario,
              n_nozzles: int, max_evals: int = 10_000, seed: int = 0,
              popsize: int | None = None):
    raise NotImplementedError("C: 1주차 구현 대상")
