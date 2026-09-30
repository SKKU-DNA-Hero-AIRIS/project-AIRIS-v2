"""Run stable Body Group 5-fold learning-curve validation.

Uses the current repository's scripts/run_e5_flow.py unchanged, but overrides
its fold_indices() at runtime with a stable body-level assignment:
    fold = (body_idx + fold_seed) % K

Methods:
- knn+stub
- flow+stub
- hybrid = flow 8 + kNN 8 + fixed candidates

Outputs per-size row CSVs plus one combined learning_curve_summary.csv.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_e5_module():
    path = ROOT / "scripts" / "run_e5_flow.py"
    spec = importlib.util.spec_from_file_location("airis_e5_flow_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stable_fold_indices(body_idx: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    ids = np.unique(body_idx).astype(int)
    return [
        np.sort(ids[(ids + int(seed)) % int(folds) == fold])
        for fold in range(int(folds))
    ]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="AIRIS stable Body 5-fold learning curve")
    ap.add_argument(
        "--dataset-dir",
        default=str(ROOT / "data" / "datasets" / "learning_curve"),
    )
    ap.add_argument("--sizes", default="100,300,600,1000,1500,3000")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-flow", type=int, default=8)
    ap.add_argument("--n-knn", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs" / "learning_curve_cv"),
    )
    return ap


def run_train(dataset: Path, model: Path, args) -> None:
    cmd = [
        sys.executable,
        "scripts/train_pose_flow.py",
        "--dataset", str(dataset),
        "--out", str(model),
        "--holdout-frac", "0",
        "--steps", str(args.steps),
        "--device", args.device,
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def aggregate(rows_csv: Path, n_bodies: int) -> list[dict]:
    df = pd.read_csv(rows_csv)
    out = []
    for method, g in df.groupby("method"):
        ratio = g["ratio"].to_numpy(dtype=float)
        ms = g["ms"].to_numpy(dtype=float)
        out.append({
            "bodies": n_bodies,
            "scenario_rows": int(
                g[["body_idx", "scenario"]].drop_duplicates().shape[0]
            ),
            "method": method,
            "ratio_median": float(np.nanmedian(ratio)),
            "ratio_p05": float(np.nanpercentile(ratio, 5)),
            "below_095": float(np.nanmean(ratio < 0.95)),
            "infeasible_frac": float(g["infeasible"].mean()),
            "ms_mean": float(np.nanmean(ms)),
            "ms_p95": float(np.nanpercentile(ms, 95)),
            "n_evals_mean": float(g["n_evals"].mean()),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    e5 = load_e5_module()
    e5.fold_indices = stable_fold_indices

    all_rows: list[dict] = []
    for n_bodies in sizes:
        dataset = dataset_dir / f"pose_dataset_mesh_cand_LC{n_bodies:04d}.parquet"
        if not dataset.exists():
            raise SystemExit(f"데이터셋이 없습니다: {dataset}")

        model = out_dir / f"LC{n_bodies:04d}_base_flow.pt"
        rows_csv = out_dir / f"LC{n_bodies:04d}_e5cv.csv"

        run_train(dataset, model, args)

        e5_args = [
            "--model", str(model),
            "--dataset", str(dataset),
            "--methods", "knn+stub,flow+stub,hybrid",
            "--n-samples", str(args.n_samples),
            "--n-flow", str(args.n_flow),
            "--n-knn", str(args.n_knn),
            "--folds", str(args.folds),
            "--fold-seed", str(args.fold_seed),
            "--out", str(rows_csv),
        ]
        print(
            f"\n=== LC{n_bodies}: stable Body Group {args.folds}-fold ===",
            flush=True,
        )
        rc = e5.main(e5_args)
        if rc not in (0, None):
            raise SystemExit(int(rc))

        all_rows.extend(aggregate(rows_csv, n_bodies))

    summary = (
        pd.DataFrame(all_rows)
        .sort_values(["bodies", "method"])
        .reset_index(drop=True)
    )
    summary_path = out_dir / "learning_curve_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    print("\n=== Learning curve summary ===")
    with pd.option_context("display.width", 180, "display.max_columns", 20):
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n저장:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
