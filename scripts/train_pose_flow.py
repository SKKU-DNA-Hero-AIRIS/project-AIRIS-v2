"""조건부 flow matching 자세 모델을 학습한다. 소유자: C. docs/proposals/flow_matching.md.

사용 예:
    python scripts/train_pose_flow.py --dataset data/datasets/pose_dataset_mesh.parquet
    python scripts/train_pose_flow.py --dataset data/datasets/pose_dataset_mesh_cand.parquet --steps 8000 --device auto

데이터셋에 cand_pose_* 열(run_dataset.py --candidate-k)이 있으면 체형마다 후보 여러 개로, 없으면 최적 자세 하나로
학습한다. 체형(body_idx) 단위로 holdout 을 떼어 산출물 meta 에 적어 두고, scripts/run_e5_flow.py 가 그 체형으로
평가한다. 산출물 기본 위치는 data/models/pose_flow.pt (git 밖, airis/model/predict.py 가 읽는다).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                           # noqa: E402

from airis.optimize import cli, explog                       # noqa: E402
from airis.sim.scenario import load_scenarios                # noqa: E402

#: 데이터셋에서 한 값이어야 하는 열 → 산출물 meta 로 옮긴다 (predict.py 가 현재 설정과 비교).
STAMP_COLS = ("nozzle_layout_hash", "physics_hash", "body_model", "patches_per_m2", "evaluator")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="조건부 flow matching 자세 모델 학습")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=str(ROOT / "data" / "models" / "pose_flow.pt"))
    ap.add_argument("--holdout-frac", type=float, default=0.2, help="평가용으로 뗄 체형 비율 (0 이면 전부 학습)")
    ap.add_argument("--holdout-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--source", choices=("gaussian", "default_pose"), default="gaussian")
    ap.add_argument("--source-sigma", type=float, default=1.0)
    ap.add_argument("--ode-steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="cpu | cuda | auto")
    return ap


def split_bodies(body_idx: np.ndarray, frac: float, seed: int) -> list[int]:
    """body_idx 단위 holdout. 같은 (체형 목록, frac, seed) 이면 같은 결과."""
    ids = np.unique(body_idx)
    n = int(round(len(ids) * frac))
    return sorted(int(i) for i in np.random.default_rng(seed).permutation(ids)[:n])


def arm_mode_agreement(model, df, scenarios, n: int = 64) -> dict[str, float]:
    """holdout 행마다 샘플 n 개의 다수 팔 봉우리가 데이터의 arm_class 와 같은 비율 (시뮬레이터 없이 보는 점검)."""
    from airis.model.flow import body_from_row
    from airis.optimize.sensitivity import arm_class

    out: dict[str, list[bool]] = {}
    for _, row in df.iterrows():
        body = body_from_row(row)
        poses = model.sample_poses(body, scenarios[row["scenario"]], n, seed=int(row["body_idx"]))
        up = np.mean([arm_class(p) == "hands_up" for p in poses])
        out.setdefault(row["scenario"], []).append(("hands_up" if up >= 0.5 else "arms_down") == row["arm_class"])
    return {s: float(np.mean(v)) for s, v in out.items()}


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)

    import pandas as pd

    from airis.model.flow import FlowConfig, flow_loss, train_pose_flow

    df = pd.read_parquet(args.dataset)
    meta: dict = {"dataset": str(Path(args.dataset).resolve()), "commit": explog.git_commit(),
                  "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    for col in STAMP_COLS:
        if col in df.columns:
            vals = sorted(set(df[col].astype(str)))
            if len(vals) != 1:
                print(f"데이터셋의 {col} 가 여러 값이다 {vals}. 한 물리 기준의 행만 쓴다.", file=sys.stderr)
                return 2
            meta[col] = df[col].iloc[0].item() if hasattr(df[col].iloc[0], "item") else df[col].iloc[0]
    meta["candidate_k"] = int(df["candidate_k"].iloc[0]) if "candidate_k" in df.columns else 0

    holdout = split_bodies(df["body_idx"].to_numpy(), args.holdout_frac, args.holdout_seed)
    meta["holdout_body_idx"] = holdout
    train = df[~df["body_idx"].isin(holdout)].reset_index(drop=True)
    test = df[df["body_idx"].isin(holdout)].reset_index(drop=True)

    cfg = FlowConfig(hidden=args.hidden, layers=args.layers, steps=args.steps, batch_size=args.batch_size,
                     lr=args.lr, source=args.source, source_sigma=args.source_sigma,
                     ode_steps=args.ode_steps, seed=args.seed, device=args.device)
    scenarios = load_scenarios()
    t0 = time.perf_counter()
    model = train_pose_flow(train, scenarios, cfg, meta=meta, log=print)
    elapsed = time.perf_counter() - t0
    model.meta["train_rows"] = int(len(train))
    model.meta["train_s"] = round(elapsed, 1)
    if len(test):
        model.meta["holdout_flow_loss"] = flow_loss(model, test, scenarios)
    path = model.save(args.out)

    print()
    print(f"학습 {len(train)}행 (학습 점 {model.meta['n_train_points']}), holdout 체형 {len(holdout)}개 "
          f"({len(test)}행), {elapsed:.0f} s")
    print(f"학습 손실(마지막 100스텝) {model.history[-1]['loss']:.4f}"
          + (f"   holdout 손실 {model.meta['holdout_flow_loss']:.4f}" if len(test) else ""))
    if len(test):
        agree = arm_mode_agreement(model, test, scenarios)
        print("holdout 팔 봉우리 일치율 (샘플 다수결 vs 데이터 arm_class): "
              + ", ".join(f"{s} {v:.2f}" for s, v in sorted(agree.items())))
    print(f"저장 위치 {path}")
    print("점수 비율 평가: python scripts/run_e5_flow.py --model", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
