"""CMA-ES 실행기. 소유자: C. docs/tracks/C_optimize.md 단계 3.

자세 벡터를 [-1, 1]^dim 으로 정규화(encoding.PoseEncoder)하고 CMA-ES 로 탐색한다.
cma 는 최소화기이므로 점수에 음수를 취해 tell 한다.

평가기는 부스 밖으로 나가는 자세를 score = −1 − 10·d_out (≤ −1.0), extra["infeasible"]=True 로
돌려준다 (docs/tracks/00_common.md 5절, d_out 은 벽 초과 거리 m).
CMA-ES 에는 그 점수를 그대로 벌점으로 넘겨 가능 구간 쪽 기울기를 주고, best 갱신에서는 제외한다.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from airis.sim import BodyParams, EvalResult, Evaluator, NozzleConfig, PoseParams, Scenario

from . import explog
from .encoding import PoseEncoder

#: 정체 판정 기준. 이보다 작은 best 개선은 개선으로 치지 않는다.
STAGNATION_TOL = 1e-4

def is_infeasible(result: EvalResult) -> bool:
    """평가 결과 하나가 불가(부스 밖)인가. extra["infeasible"] 만 본다.

    점수로 판정하지 않는 이유: DummyEvaluator 는 −(정규화 거리)² 라 가능한 자세도 −1 아래로 내려간다.
    """
    return bool(result.extra.get("infeasible", False))


@dataclass
class OptResult:
    best_pose: PoseParams
    best_score: float
    best_result: EvalResult
    history: list[dict]           # per generation: gen, evals, best, mean, sigma, infeasible_frac, start
    n_evals: int
    exp_id: str
    elapsed_s: float = 0.0
    stop_reason: str = ""
    free_keys: list[str] = field(default_factory=list)
    n_infeasible: int = 0         # 불가 판정을 받은 후보 수 (전 세대 합)
    per_start: list[dict] = field(default_factory=list)  # 시작점별 요약 (start, best_score, ...)
    # record_candidates=True 일 때만 채운다: 평가한 후보 전부 {"x": 정규화 벡터, "score", "infeasible", "start"}
    candidates: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class Start:
    """CMA-ES 시작점. sigma0 가 None 이면 run_cmaes 의 sigma0 를 쓴다."""
    name: str
    pose: PoseParams
    sigma0: float | None = None


def cma_seed(seed: int) -> int:
    """사용자 시드를 cma 옵션 시드로 매핑한다.

    cma 는 seed 가 0 이나 None 이면 시각 기반으로 재시드해 재현성이 깨진다
    (CMAOptions: "`None` and `0` equate to `time`"). 항상 1 이상이 되도록 옮긴다.
    """
    return int(seed) % (2 ** 31 - 2) + 1


def score_batch(
    evaluator: Evaluator,
    poses: list[PoseParams],
    nozzle: NozzleConfig,
    body: BodyParams,
    scenario: Scenario,
) -> tuple[np.ndarray, np.ndarray]:
    """후보들의 (점수, 불가 여부)를 돌려준다.

    batch_evaluate 는 점수만 돌려주므로 extra["infeasible"] 을 볼 수 없다.
    - batch_evaluate 를 오버라이드하지 않은 평가기(패치판, 더미)는 기본 구현과 같은
      순차 evaluate 를 직접 불러 extra["infeasible"] 을 읽는다.
    - 오버라이드한 평가기(입자판)는 batch_evaluate 를 쓰고 score <= -1.0 으로 판정한다
      (00_common.md 5절. 가능한 자세의 점수는 −discomfort_weight·불편도 > −1 이상이다).
    """
    if type(evaluator).batch_evaluate is Evaluator.batch_evaluate:
        results = [evaluator.evaluate(p, nozzle, body, scenario) for p in poses]
        # 기본 batch_evaluate 와 같은 float32 를 거쳐 기존 실행과 궤적을 맞춘다.
        scores = np.array([r.score for r in results], dtype=np.float32).astype(np.float64)
        infeasible = np.array([is_infeasible(r) for r in results], dtype=bool)
    else:
        scores = np.asarray(
            evaluator.batch_evaluate([(p, nozzle) for p in poses], body, scenario),
            dtype=np.float64,
        ).reshape(-1)
        infeasible = scores <= -1.0
    if scores.size != len(poses):
        raise ValueError(
            f"batch_evaluate 가 후보 {len(poses)}개에 점수 {scores.size}개를 돌려줬다"
        )
    return scores, infeasible


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
    starts: Sequence[Start] | None = None,
    record_candidates: bool = False,
    log_dir: Path | str | None = None,
    exp_id: str | None = None,
    tag: str = "opt",
) -> OptResult:
    """시나리오 제약 안에서 자세를 탐색한다.

    같은 seed 와 결정론적 평가기면 history 가 완전히 같아야 한다.

    starts 를 주면 시작점마다 CMA-ES 를 따로 돌린다 (점수 지형의 봉우리가 여럿일 때).
    - 예산 max_evals 를 시작점 수로 나눈다 (나머지는 마지막 시작점).
    - 시작점 i 의 시드는 seed + i, 초기 스텝은 start.sigma0 (None 이면 sigma0).
    - 정체 판정은 시작점마다 따로, best 는 모든 시작점의 가능한 후보 중 최고.
    - history 의 gen / evals 는 시작점을 넘어 누적하고 start 열에 시작점 이름을 남긴다.
    starts=None 이면 기본 자세 하나 (예전 동작과 같다).

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
    starts = list(starts) if starts is not None else [Start("default", PoseParams())]
    if not starts:
        raise ValueError("starts 가 비어 있다")
    names = [st.name for st in starts]
    if len(set(names)) != len(names):
        raise ValueError(f"시작점 이름이 겹친다: {names}")
    # 예산은 시작점 수로 나누고 나머지는 마지막 시작점에 준다.
    budgets = [max_evals // len(starts)] * len(starts)
    budgets[-1] += max_evals - sum(budgets)

    history: list[dict] = []
    per_start: list[dict] = []
    candidates: list[dict] = []
    best_score = -np.inf
    best_x: np.ndarray | None = None
    n_evals = 0
    n_infeasible = 0
    gen = 0
    t0 = time.perf_counter()

    for i, (start, budget) in enumerate(zip(starts, budgets)):
        s0 = sigma0 if start.sigma0 is None else start.sigma0
        es = cma.CMAEvolutionStrategy(
            enc.encode(start.pose).tolist(),
            s0,
            {
                "bounds": [-1, 1],
                "popsize": popsize,
                "seed": cma_seed(seed + i),
                "maxfevals": budget,
                "verbose": -9,
            },
        )
        s_best = -np.inf
        s_best_x: np.ndarray | None = None
        s_evals = 0
        s_infeasible = 0
        stagnant = 0
        stop_reason = ""

        while budget > 0 and not es.stop():
            X = es.ask()
            poses = [enc.decode(x) for x in X]
            scores, infeasible = score_batch(evaluator, poses, nozzle, body, scenario)
            if record_candidates:
                candidates.extend(
                    {"x": np.asarray(x, dtype=np.float64).copy(), "score": float(sc),
                     "infeasible": bool(bad), "start": start.name}
                    for x, sc, bad in zip(X, scores, infeasible)
                )

            es.tell(X, (-scores).tolist())   # cma 는 최소화. 불가 후보도 벌점 점수 그대로 넘긴다.

            s_evals += len(X)
            s_infeasible += int(infeasible.sum())
            gen += 1

            # best 는 가능한 후보 중에서만 고른다. 전부 불가면 이번 세대는 개선 없음.
            feasible_scores = np.where(infeasible, -np.inf, scores)
            top = int(np.argmax(feasible_scores))
            gain = float(feasible_scores[top]) - s_best if not infeasible.all() else 0.0
            if gain > 0:
                s_best = float(scores[top])
                s_best_x = np.asarray(X[top], dtype=np.float64).copy()
            stagnant = 0 if gain > STAGNATION_TOL else stagnant + 1
            if s_best > best_score:
                best_score, best_x = s_best, s_best_x

            row = {
                "gen": gen,                    # 시작점을 넘어 누적
                "evals": n_evals + s_evals,    # 시작점을 넘어 누적
                "best": best_score,            # 전체 누적 최고점
                "mean": float(np.mean(scores)),  # 해당 세대 평균
                "sigma": float(es.sigma),
                "infeasible_frac": float(infeasible.mean()),  # 해당 세대 불가 후보 비율
                "start": start.name,
            }
            history.append(row)
            if log_dir is not None:
                explog.append_history_row(log_dir, exp_id, row)

            if stagnant >= tol_stagnation_gens:
                stop_reason = f"stagnation({tol_stagnation_gens} gens)"
                break
            if s_evals >= budget:
                # cma 의 maxfevals 는 초과한 뒤에 멈추므로 예산을 정확히 지키도록 여기서 끊는다.
                stop_reason = f"maxfevals({budget})"
                break

        if not stop_reason:
            stop_reason = ", ".join(es.stop()) or "maxfevals"
        n_evals += s_evals
        n_infeasible += s_infeasible
        per_start.append({
            "start": start.name,
            "start_pose": asdict(enc.clip_pose(start.pose)),
            "sigma0": float(s0),
            "seed": int(seed + i),
            "n_evals": s_evals,
            "n_infeasible": s_infeasible,
            "best_score": float(s_best) if s_best_x is not None else float("nan"),
            "best_pose": asdict(enc.decode(s_best_x)) if s_best_x is not None else None,
            "stop_reason": stop_reason,
        })

    if len(per_start) == 1:
        stop_reason = per_start[0]["stop_reason"]
    else:
        stop_reason = "; ".join(f"{ps['start']}: {ps['stop_reason']}" for ps in per_start)

    if best_x is None:                     # 한 세대도 못 돌렸거나 가능한 후보가 없던 경우
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
        n_infeasible=n_infeasible,
        per_start=per_start,
        candidates=candidates,
    )
