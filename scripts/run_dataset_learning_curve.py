"""Generate nested AIRIS pose datasets for learning-curve experiments.

Schema stays:
    one row = one body x one scenario

With candidate_k=16, each row additionally contains near-optimal CMA-ES
candidate lists (cand_pose_*, cand_score, n_candidates).

Datasets are nested:
LC100 ⊂ LC300 ⊂ LC600 ⊂ LC1000 ⊂ LC1500 ⊂ LC3000

The master parquet is incrementally extended, so earlier bodies are reused
instead of recomputed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from airis.optimize import dataset  # noqa: E402
from airis.sim.scenario import load_scenarios  # noqa: E402

DEFAULT_SIZES = (100, 300, 600, 1000, 1500, 3000)


def parse_sizes(value: str) -> list[int]:
    sizes = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not sizes or any(n <= 0 for n in sizes):
        raise argparse.ArgumentTypeError("sizes는 양의 정수 목록이어야 합니다.")
    return sizes


def stable_fold(body_idx: int, folds: int = 5, seed: int = 0) -> int:
    """Stable body-level fold across all nested LC datasets."""
    return (int(body_idx) + int(seed)) % int(folds)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="AIRIS nested candidate-k learning-curve datasets")
    ap.add_argument("--sizes", type=parse_sizes, default=list(DEFAULT_SIZES))
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--flush-every", type=int, default=100)
    ap.add_argument("--max-evals", type=int, default=2000)
    ap.add_argument("--popsize", type=int, default=50)
    ap.add_argument("--patches-per-m2", type=float, default=1500.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--body-seed", type=int, default=0)
    ap.add_argument("--candidate-k", type=int, default=16)
    ap.add_argument("--candidate-tol", type=float, default=0.02)
    ap.add_argument("--candidate-min-dist", type=float, default=0.1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-seed", type=int, default=0)
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "data" / "datasets" / "learning_curve"),
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_scenarios = load_scenarios()
    names = args.scenarios or list(all_scenarios)
    missing = [name for name in names if name not in all_scenarios]
    if missing:
        raise SystemExit(f"없는 시나리오: {missing}")
    if args.candidate_k <= 0:
        raise SystemExit("Flow 검증용 learning curve는 --candidate-k > 0 이어야 합니다.")
    if args.folds < 2:
        raise SystemExit("--folds는 2 이상이어야 합니다.")

    master = out_dir / "pose_dataset_mesh_cand_master.parquet"
    cfg = dataset.DatasetConfig(
        evaluator="patch",
        max_evals=args.max_evals,
        popsize=args.popsize,
        starts="default,hands_up",
        patches_per_m2=args.patches_per_m2,
        seed=args.seed,
        body_seed=args.body_seed,
        dataset_id="lc_master",
        candidate_k=args.candidate_k,
        candidate_tol=args.candidate_tol,
        candidate_min_dist=args.candidate_min_dist,
    )

    manifest_rows: list[dict] = []
    for n_bodies in args.sizes:
        print(f"\n=== LC{n_bodies}: master를 {n_bodies} bodies까지 확장 ===")
        dataset.build_dataset(
            master,
            n_bodies=n_bodies,
            scenarios=names,
            cfg=cfg,
            processes=args.processes,
            flush_every=args.flush_every,
        )

        full = pd.read_parquet(master)
        snap = full[full["body_idx"].astype(int) < n_bodies].copy()
        expected = n_bodies * len(names)
        if len(snap) != expected:
            raise RuntimeError(f"LC{n_bodies}: {len(snap)}행, 기대 {expected}행")

        # Same body -> same fold for all scenarios and all LC sizes.
        snap["body_fold"] = snap["body_idx"].map(
            lambda x: stable_fold(int(x), args.folds, args.fold_seed)
        ).astype("int8")

        if int(snap.groupby("body_idx")["body_fold"].nunique().max()) != 1:
            raise RuntimeError("같은 body_idx가 둘 이상의 fold에 들어갔습니다.")

        path = out_dir / f"pose_dataset_mesh_cand_LC{n_bodies:04d}.parquet"
        snap.to_parquet(path, index=False)

        body_fold_counts = (
            snap[["body_idx", "body_fold"]]
            .drop_duplicates()["body_fold"]
            .value_counts()
            .sort_index()
        )
        manifest_rows.append({
            "dataset": path.name,
            "bodies": n_bodies,
            "scenario_rows": len(snap),
            "candidate_k": args.candidate_k,
            "candidate_mean": float(snap["n_candidates"].mean()),
            "folds": args.folds,
            **{
                f"fold_{fold}_bodies": int(body_fold_counts.get(fold, 0))
                for fold in range(args.folds)
            },
        })
        print(
            f"{path.name}: rows={len(snap)}, "
            f"candidate_mean={float(snap['n_candidates'].mean()):.2f}"
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(
        out_dir / "learning_curve_manifest.csv",
        index=False,
        encoding="utf-8",
    )

    config = {
        "sizes": args.sizes,
        "scenarios": names,
        "max_evals": args.max_evals,
        "popsize": args.popsize,
        "patches_per_m2": args.patches_per_m2,
        "seed": args.seed,
        "body_seed": args.body_seed,
        "candidate_k": args.candidate_k,
        "candidate_tol": args.candidate_tol,
        "candidate_min_dist": args.candidate_min_dist,
        "folds": args.folds,
        "fold_seed": args.fold_seed,
        "fold_rule": "(body_idx + fold_seed) % folds",
    }
    (out_dir / "learning_curve_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n완료:", out_dir / "learning_curve_manifest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
