"""CMA-ES 실행기. 소유자: C. docs/tracks/C_optimize.md 단계 3.

자세 벡터를 [-1, 1]^dim 으로 정규화(encoding.PoseEncoder)하고 CMA-ES 로 탐색한다.
cma 는 최소화기이므로 점수에 음수를 취해 tell 한다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from airis.sim import BodyParams, EvalResult, Evaluator, NozzleConfig, PoseParams, Scenario

from . import explog
from .encoding import PoseEncoder

#: 정체 판정 기준. 이보다 작은 best 개선은 개선으로 치지 않는다.
STAGNATION_TOL = 1e-4


@dataclass
class OptResult:
    best_pose: PoseParams
    best_score: float
    best_result: EvalResult
    history: list[dict]           # per generation: gen, evals, best, mean, sigma
    n_evals: int
    exp_id: str
    elapsed_s: float = 0.0
    stop_reason: str = ""
    free_keys: list[str] = field(default_factory=list)


def cma_seed(seed: int) -> int:
    """사용자 시드를 cma 옵션 시드로 매핑한다.

    cma 는 seed 가 0 이나 None 이면 시각 기반으로 재시드해 재현성이 깨진다
    (CMAOptions: "`None` and `0` equate to `time`"). 항상 1 이상이 되도록 옮긴다.
    """
    return int(seed) % (2 ** 31 - 2) + 1


def run_cmaes(
    evaluator: Evaluator,
    body: BodyParams,
    scenario: Scenario,
    nozzle: NozzleConfig,
    *,
    max_evals: int = 3000,
    seed: int = 0,
    popsize: int = 100,
    sigma0: float = 0.5,
    tol_stagnation_gens: int = 30,
    log_dir: Path | str | None = None,
    exp_id: str | None = None,
    tag: str = "opt",
) -> OptResult:
    """시나리오 제약 안에서 자세를 탐색한다.

    같은 seed 와 결정론적 평가기면 history 가 완전히 같아야 한다.

    log_dir 이 주어지면 세대마다 <log_dir>/<exp_id>/history.csv 에 한 줄씩 덧붙인다.
    meta.json / best.json 은 호출자가 explog.write_run 으로 쓴다.
    """
    import cma  # 무거운 import 라 함수 안에서 한다.

    enc = PoseEncoder(scenario)
    if enc.dim < 2:
        raise ValueError(
            f"자유 자세 변수가 {enc.dim}개라 CMA-ES 를 돌릴 수 없다 (시나리오 {scenario.name})"
        )

    exp_id = exp_id or explog.new_exp_id(tag)
    es = cma.CMAEvolutionStrategy(
        enc.default_x().tolist(),
        sigma0,
        {
            "bounds": [-1, 1],
            "popsize": popsize,
            "seed": cma_seed(seed),
            "maxfevals": max_evals,
            "verbose": -9,
        },
    )

    history: list[dict] = []
    best_score = -np.inf
    best_x: np.ndarray | None = None
    n_evals = 0
    gen = 0
    stagnant = 0
    stop_reason = ""
    t0 = time.perf_counter()

    while not es.stop():
        X = es.ask()
        poses = [enc.decode(x) for x in X]
        scores = np.asarray(
            evaluator.batch_evaluate([(p, nozzle) for p in poses], body, scenario),
            dtype=np.float64,
        ).reshape(-1)
        if scores.size != len(X):
            raise ValueError(
                f"batch_evaluate 가 후보 {len(X)}개에 점수 {scores.size}개를 돌려줬다"
            )

        es.tell(X, (-scores).tolist())   # cma 는 최소화

        n_evals += len(X)
        gen += 1

        top = int(np.argmax(scores))
        gain = float(scores[top]) - best_score
        if gain > 0:
            best_score = float(scores[top])
            best_x = np.asarray(X[top], dtype=np.float64).copy()
        stagnant = 0 if gain > STAGNATION_TOL else stagnant + 1

        row = {
            "gen": gen,
            "evals": n_evals,
            "best": best_score,            # 누적 최고점
            "mean": float(np.mean(scores)),  # 해당 세대 평균
            "sigma": float(es.sigma),
        }
        history.append(row)
        if log_dir is not None:
            explog.append_history_row(log_dir, exp_id, row)

        if stagnant >= tol_stagnation_gens:
            stop_reason = f"stagnation({tol_stagnation_gens} gens)"
            break
        if n_evals >= max_evals:
            # cma 의 maxfevals 는 초과한 뒤에 멈추므로 예산을 정확히 지키도록 여기서 끊는다.
            stop_reason = f"maxfevals({max_evals})"
            break

    if not stop_reason:
        stop_reason = ", ".join(es.stop()) or "maxfevals"

    if best_x is None:                     # 한 세대도 못 돌린 경우
        best_x = enc.default_x()
        best_score = float("nan")

    best_pose = enc.decode(best_x)
    # 부위별 값을 채우기 위해 best 자세를 한 번 더 평가한다.
    best_result = evaluator.evaluate(best_pose, nozzle, body, scenario)

    return OptResult(
        best_pose=best_pose,
        best_score=float(best_result.score),
        best_result=best_result,
        history=history,
        n_evals=n_evals,
        exp_id=exp_id,
        elapsed_s=time.perf_counter() - t0,
        stop_reason=stop_reason,
        free_keys=list(enc.free_keys),
    )
