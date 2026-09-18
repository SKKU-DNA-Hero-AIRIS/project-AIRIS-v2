"""회귀용 데이터셋을 만든다(이어 만든다). 소유자: C. docs/tracks/C_optimize.md 단계 9.

사용 예:
    python scripts/run_dataset.py --n-bodies 100 --processes 8               # 패치판, 시나리오 3개
    python scripts/run_dataset.py --n-bodies 200 --processes 8               # 같은 파일에 101~200 번 체형만 추가
    python scripts/run_dataset.py --evaluator dummy --n-bodies 4 --max-evals 40 --out data/datasets/try.parquet

기본 출력은 data/datasets/pose_dataset.parquet (git 밖). 끝나면 키 구간별 봉우리 요약을 출력하고
같은 폴더에 <이름>_height_bins.csv 로 남긴다. 스키마와 재개 규칙은 airis/optimize/dataset.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airis.optimize import cli, dataset, explog              # noqa: E402
from airis.sim.scenario import load_scenarios                # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="체형 × 시나리오 최적 자세 데이터셋")
    ap.add_argument("--n-bodies", type=int, required=True, help="체형 수 (0..n-1 번). 늘리면 뒷부분만 추가")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--evaluator", choices=cli.EVALUATOR_CHOICES, default="patch")
    ap.add_argument("--max-evals", type=int, default=2000)
    ap.add_argument("--popsize", type=int, default=50)
    ap.add_argument("--starts", default=cli.DEFAULT_STARTS)
    ap.add_argument("--patches-per-m2", type=float, default=cli.DEFAULT_PATCHES_PER_M2)
    ap.add_argument("--seed", type=int, default=0, help="CMA-ES 시드 기준값")
    ap.add_argument("--body-seed", type=int, default=0, help="체형 샘플러 시드")
    ap.add_argument("--processes", type=int, default=1)
    ap.add_argument("--flush-every", type=int, default=100)
    ap.add_argument("--out", default=str(ROOT / "data" / "datasets" / "pose_dataset.parquet"))
    return ap


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)
    all_scenarios = load_scenarios()
    names = args.scenarios or list(all_scenarios)
    missing = [n for n in names if n not in all_scenarios]
    if missing:
        print(f"없는 시나리오: {missing}", file=sys.stderr)
        return 2

    out = Path(args.out)
    cfg = dataset.DatasetConfig(
        evaluator=args.evaluator, max_evals=args.max_evals, popsize=args.popsize, starts=args.starts,
        patches_per_m2=args.patches_per_m2, seed=args.seed, body_seed=args.body_seed,
        dataset_id=out.stem,
    )
    try:
        info = dataset.build_dataset(out, n_bodies=args.n_bodies, scenarios=names, cfg=cfg,
                                     processes=args.processes, flush_every=args.flush_every)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except cli.TrackNotMerged as exc:
        print(str(exc), file=sys.stderr)
        return 3

    import pandas as pd

    df = pd.read_parquet(out)
    bins = dataset.height_bin_summary(df)
    bins_path = out.with_name(out.stem + "_height_bins.csv")
    bins.to_csv(bins_path, index=False, encoding="utf-8")
    print()
    print(f"행 {info['n_rows']} (새로 {info['n_new']}, 건너뜀 {info['n_skipped']})  commit {explog.git_commit()}")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(bins.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(f"저장 위치    {out}")
    print(f"키 구간 요약  {bins_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
