"""E5 일반화: flow matching 추천 vs 비교군, holdout 체형에서 점수 비율. 소유자: C.
README 6절 E5, docs/proposals/flow_matching.md 5절.

    점수 비율 = score(예측 자세) / score(그 체형·시나리오를 직접 CMA-ES 로 최적화한 자세 = 데이터셋 score)

두 점수는 같은 평가기(학습 데이터의 몸 모델·패치 밀도, predict.default_rescorer)로 잰다.

방법
    flow    flow matching 샘플 n 개 → 패치판 재채점 → 최고 (제안안, predict_pose 기본 동작)
    flow1   flow matching 샘플 1개, 재채점 없음
    mlp     MLPRegressor (최적 자세 평균 회귀, README 5.6 기본안). 학습 체형의 최적 자세로 이 스크립트에서 학습
    stub    recommend.py 스텁: 시나리오별 고정 후보표(만세·팔 내림) → 이 체형으로 재채점 → 최고

사용 예:
    python scripts/run_e5_flow.py --model data/models/pose_flow.pt
    python scripts/run_e5_flow.py --model data/models/pose_flow.pt --methods flow,stub --n-samples 32
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

METHODS = ("flow", "flow1", "mlp", "stub")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E5: flow matching vs 비교군 점수 비율")
    ap.add_argument("--model", default=str(ROOT / "data" / "models" / "pose_flow.pt"))
    ap.add_argument("--dataset", default=None, help="기본값: 산출물 meta 의 dataset")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="holdout 행 수 상한 (빠른 점검용)")
    ap.add_argument("--out", default=None, help="기본값: outputs/e5flow_<시각>.csv")
    return ap


def fit_mlp(train, model):
    """최적 자세(pose_*)를 평균 회귀하는 MLP. 입력은 flow 모델과 같은 조건 벡터."""
    from sklearn.neural_network import MLPRegressor

    from airis.model.flow import BODY_KEYS, POSE_KEYS

    body = train[[f"body_{k}" for k in BODY_KEYS]].to_numpy(dtype=np.float32)
    X = model._cond(body, list(train["scenario"].astype(str))).numpy()
    Y = model.space.encode(train[[f"pose_{k}" for k in POSE_KEYS]].to_numpy())
    return MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=2000, random_state=0).fit(X, Y)


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        print(f"알 수 없는 방법 {bad} (가능: {', '.join(METHODS)})", file=sys.stderr)
        return 2

    import pandas as pd

    from airis.model import predict as pred
    from airis.model.flow import BODY_KEYS, body_from_row
    from airis.optimize.cmaes_runner import score_batch
    from airis.optimize.encoding import PoseEncoder

    model = pred.load_model(args.model)
    df = pd.read_parquet(args.dataset or model.meta["dataset"])
    holdout = set(model.meta.get("holdout_body_idx", []))
    if not holdout:
        print("산출물에 holdout 체형이 없다 (train_pose_flow.py --holdout-frac 0 으로 학습).", file=sys.stderr)
        return 2
    test = df[df["body_idx"].isin(holdout)].reset_index(drop=True)
    if args.limit:
        test = test.head(args.limit)
    train = df[~df["body_idx"].isin(holdout)].reset_index(drop=True)
    scenarios = load_scenarios()
    evaluator, nozzle = pred.default_rescorer(model)
    mlp = fit_mlp(train, model) if "mlp" in methods else None
    stub_table = None
    if "stub" in methods:
        from airis.realtime.recommend import STUB_TABLE as stub_table

    rows = []
    t_all = time.perf_counter()
    for i, row in test.iterrows():
        body, scenario = body_from_row(row), scenarios[row["scenario"]]
        ref = float(row["score"])
        for m in methods:
            t0 = time.perf_counter()
            if m in ("flow", "flow1"):
                pose = pred.predict(body, scenario, n_samples=args.n_samples if m == "flow" else 1,
                                    rescore=(m == "flow"), seed=args.seed + int(row["body_idx"]),
                                    path=args.model, evaluator=evaluator, nozzle=nozzle).pose
            elif m == "mlp":
                bvec = np.array([[getattr(body, k) for k in BODY_KEYS]], dtype=np.float32)
                x = mlp.predict(model._cond(bvec, [scenario.name]).numpy())
                pose = model.space.to_pose(x[0], scenario)
            else:
                enc = PoseEncoder(scenario)
                cands = [enc.clip_pose(e.pose) for e in stub_table[scenario.name]]
                sc, bad_ = score_batch(evaluator, cands, nozzle, body, scenario)
                pose = cands[int(np.argmax(np.where(bad_, -np.inf, sc) if not bad_.all() else sc))]
            ms = (time.perf_counter() - t0) * 1000.0
            score, infeasible = score_batch(evaluator, [pose], nozzle, body, scenario)
            rows.append({
                "body_idx": int(row["body_idx"]), "scenario": scenario.name, "method": m,
                "height_m": body.height_m, "ref_score": ref, "score": float(score[0]),
                "ratio": float(score[0]) / ref if ref > 0 else float("nan"),
                "infeasible": bool(infeasible[0]), "ms": ms,
                "shoulder_abduction": pose.shoulder_abduction, "torso_yaw": pose.torso_yaw,
                "ref_arm_class": row.get("arm_class"),
            })
        if (i + 1) % 10 == 0 or i + 1 == len(test):
            print(f"[e5flow] {i + 1}/{len(test)}  {time.perf_counter() - t_all:.0f} s")

    res = pd.DataFrame(rows)
    summary = (res.groupby(["method", "scenario"])
                  .agg(n=("ratio", "size"),
                       ratio_median=("ratio", "median"),
                       ratio_p05=("ratio", lambda s: float(np.nanpercentile(s, 5))),
                       ratio_mean=("ratio", "mean"),
                       infeasible_frac=("infeasible", "mean"),
                       ms_mean=("ms", "mean"))
                  .reset_index())
    out = Path(args.out) if args.out else ROOT / "outputs" / f"{explog.new_exp_id('e5flow')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False, encoding="utf-8")
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False, encoding="utf-8")

    print()
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print("채택 기준 (proposal 5절): flow 중앙값 ≥ 0.95, 하위 5%·불가 비율이 비교군보다 나을 것")
    print(f"행별 결과 {out}")
    print(f"요약      {out.with_name(out.stem + '_summary.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
