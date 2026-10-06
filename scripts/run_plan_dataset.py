"""계획 데이터셋 만들기 (체형 × 시나리오 → 최적 계획). 소유자: C.

자세 데이터셋(`run_dataset.py`)의 계획판이다. F 의 계획 추천 모델이 학습·평가에 쓴다.
열 규약은 `docs/interfaces.md` "계획 데이터셋 스키마" 절(C·F 합의 2026-10-06).

사용 예:
    python scripts/run_plan_dataset.py --n-bodies 50 --processes 8          # 기본: 단계 9, 가중 0.1
    python scripts/run_plan_dataset.py --n-bodies 2 --evaluator dummy       # 파이프라인 확인

기본 설정은 총괄이 채택한 **따뜻한 시작**(10-06)이다 — 단일 자세 최적에서 출발, 초기 스텝 0.2,
예산 12,000, 정체 20세대. 1건 실측에서 예산 24,000 차가운 시작과 점수가 같고(1.1169 vs 1.1168)
평가 횟수는 2.2배 적었다.

체형은 `--body-seed 0` 이라 **자세 데이터셋과 같은 body_idx = 같은 체형**이다(F 의 5-fold 가
자세 모델 추천을 고정 후보로 쓰려면 이 성질이 필요하다).

중단되면 같은 명령을 다시 돌린다 — 끝난 행은 건너뛴다. 긴 실행은 셸과 분리해 띄운다
(PowerShell `Start-Process -WindowStyle Hidden -RedirectStandardOutput`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airis.optimize import cli                                      # noqa: E402
from airis.optimize.plan_dataset import (                           # noqa: E402
    PlanDatasetConfig, build_plan_dataset,
)
from airis.sim.scenario import load_scenarios                       # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="계획 데이터셋 만들기")
    ap.add_argument("--n-bodies", type=int, required=True, help="체형 수 (0..n-1 번). 늘리면 뒷부분만 추가")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--evaluator", choices=cli.EVALUATOR_CHOICES, default="patch")
    ap.add_argument("--n-phases", type=int, default=9, help="계획 단계 수 (N × min_phase_s < 총 시간 상한)")
    ap.add_argument("--energy-weight", type=float, default=0.1)
    ap.add_argument("--max-evals", type=int, default=12000, help="계획 최적화 예산 (따뜻한 시작 기준)")
    ap.add_argument("--pose-max-evals", type=int, default=3000, help="단일 자세 최적 예산")
    ap.add_argument("--popsize", type=int, default=100)
    ap.add_argument("--sigma0", type=float, default=0.5)
    ap.add_argument("--warm-start-sigma0", type=float, default=0.2)
    ap.add_argument("--tol-stagnation-gens", type=int, default=20)
    ap.add_argument("--patches-per-m2", type=float, default=1500.0)
    ap.add_argument("--seed", type=int, default=0, help="CMA-ES 시드 기준값")
    ap.add_argument("--body-seed", type=int, default=0, help="체형 샘플러 시드 (자세 데이터셋과 맞춘다)")
    ap.add_argument("--candidate-k", type=int, default=16, help="0 이면 후보를 저장하지 않는다")
    ap.add_argument("--candidate-tol", type=float, default=0.02)
    ap.add_argument("--candidate-min-dist", type=float, default=0.05,
                    help="encode 공간 거리 ÷√dim 기준. 자세(7차원 0.1)와 다르다")
    ap.add_argument("--no-baselines", action="store_true",
                    help="score_p1_10 · score_p1opt_10 · pose_* 를 넣지 않는다 (행당 계획 평가 2회 절약)")
    ap.add_argument("--processes", type=int, default=8,
                    help="8 이 가장 빨랐다 (12 는 2.5배 느림, 메모리 대역폭 경합)")
    ap.add_argument("--flush-every", type=int, default=20, help="몇 행마다 저장할지 (중단 대비)")
    ap.add_argument("--out", default=str(ROOT / "data" / "datasets" / "plan_dataset.parquet"))
    return ap


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)

    all_scenarios = load_scenarios()
    names = args.scenarios or list(all_scenarios)
    missing = [n for n in names if n not in all_scenarios]
    if missing:
        print(f"없는 시나리오: {missing} (가능: {', '.join(all_scenarios)})", file=sys.stderr)
        return 2

    cfg = PlanDatasetConfig(
        n_phases=args.n_phases, energy_weight=args.energy_weight, evaluator=args.evaluator,
        patches_per_m2=args.patches_per_m2, max_evals=args.max_evals,
        pose_max_evals=args.pose_max_evals, popsize=args.popsize, sigma0=args.sigma0,
        warm_start_sigma0=args.warm_start_sigma0, tol_stagnation_gens=args.tol_stagnation_gens,
        seed=args.seed, body_seed=args.body_seed, candidate_k=args.candidate_k,
        candidate_tol=args.candidate_tol, candidate_min_dist=args.candidate_min_dist,
        baselines=not args.no_baselines,
    )
    try:
        out = build_plan_dataset(args.out, n_bodies=args.n_bodies, scenarios=names, cfg=cfg,
                                 processes=args.processes, flush_every=args.flush_every)
    except (ValueError, cli.TrackNotMerged) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"\n행 {out['n_rows']} (새로 {out['n_new']}, 건너뜀 {out['n_skipped']})")
    print(f"저장 위치    {out['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
