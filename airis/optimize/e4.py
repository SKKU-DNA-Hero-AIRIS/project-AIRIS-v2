"""E4 (최적 자세 vs 기준 자세) 요약 계산. 소유자: C. docs/tracks/C_optimize.md 단계 7.

scripts/run_e4.py 가 시나리오 × 시드 실행 결과를 모아 여기서 요약한다.

- 개선율 = (best − 기준선) / |기준선|. 기준선 점수가 양수면 best / 기준선 − 1 과 같다.
  기준선이 불가거나 0 이면 NaN.
- 시드 간 자세 편차는 torso_yaw 를 |yaw| 로 접어서 계산한다. 노즐·부스가 좌우 대칭이라
  yaw +θ 와 −θ 가 같은 점수이므로 (docs/interfaces.md 회귀 모델 절, 거울 정규화)
  접지 않으면 같은 해가 ±90 으로 갈려 편차가 부풀려진다.
"""
from __future__ import annotations

import math
from dataclasses import asdict, fields

import numpy as np

from airis.sim import PoseParams

POSE_KEYS: list[str] = [f.name for f in fields(PoseParams)]
BASELINE_NAMES: tuple[str, ...] = ("B0", "B1", "B2")


def improvement(best: float, base: float, base_infeasible: bool = False) -> float:
    """(best − base) / |base|. 기준선이 불가거나 0 이면 NaN."""
    if base_infeasible or not math.isfinite(base) or abs(base) < 1e-12:
        return float("nan")
    return (best - base) / abs(base)


def fold_pose(pose: PoseParams | dict) -> dict[str, float]:
    """좌우 거울 정규화: torso_yaw 를 |yaw| 로 접는다."""
    d = asdict(pose) if isinstance(pose, PoseParams) else dict(pose)
    d["torso_yaw"] = abs(float(d["torso_yaw"]))
    return {k: float(d[k]) for k in POSE_KEYS}


def winning_start(per_start: list[dict]) -> str:
    """best 를 낸 시작점 이름. 가능한 후보가 없던 시작점(NaN)은 건너뛴다."""
    scored = [ps for ps in per_start if math.isfinite(ps["best_score"])]
    if not scored:
        return ""
    return max(scored, key=lambda ps: ps["best_score"])["start"]


def summarize_scenario(runs: list[dict], baselines: dict[str, dict]) -> dict:
    """시나리오 하나의 시드별 실행을 요약한다.

    runs 의 각 원소: {"exp_id", "seed", "best_score", "best_pose"(PoseParams), "per_start",
                     "elapsed_s", "n_evals", "n_infeasible"}
    baselines: baselines.evaluate_all 결과 ({조건: {"score", "infeasible", ...}})
    """
    scores = np.array([r["best_score"] for r in runs], dtype=np.float64)
    folded = np.array([[fold_pose(r["best_pose"])[k] for k in POSE_KEYS] for r in runs])
    ddof = 1 if len(runs) > 1 else 0
    best_mean = float(scores.mean())

    row: dict = {
        "n_seeds": len(runs),
        "seeds": ";".join(str(r["seed"]) for r in runs),
        "best_mean": best_mean,
        "best_std": float(scores.std(ddof=ddof)),
        "best_min": float(scores.min()),
        "best_max": float(scores.max()),
    }
    for name in BASELINE_NAMES:
        base = baselines[name]
        row[f"{name}_score"] = float(base["score"])
        row[f"{name}_infeasible"] = bool(base["infeasible"])
        row[f"imp_vs_{name}"] = improvement(best_mean, float(base["score"]), bool(base["infeasible"]))

    wins = [winning_start(r["per_start"]) for r in runs]
    row["best_start_counts"] = ";".join(f"{s}:{wins.count(s)}" for s in sorted(set(wins)))
    n_evals = sum(r["n_evals"] for r in runs)
    row["infeasible_frac"] = sum(r["n_infeasible"] for r in runs) / n_evals if n_evals else 0.0
    row["elapsed_mean_s"] = float(np.mean([r["elapsed_s"] for r in runs]))
    for j, key in enumerate(POSE_KEYS):
        row[f"pose_mean_{key}"] = float(folded[:, j].mean())
    for j, key in enumerate(POSE_KEYS):
        row[f"pose_std_{key}"] = float(folded[:, j].std(ddof=ddof))
    return row
