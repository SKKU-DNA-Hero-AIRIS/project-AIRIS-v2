"""E5 일반화: flow matching 추천 vs 비교군, holdout 체형에서 점수 비율. 소유자: F.
README 6절 E5, docs/proposals/flow_matching.md 5절, docs/interfaces.md "계획 모델" 채택 기준.

    점수 비율 = score(예측 자세) / score(그 체형·시나리오를 직접 CMA-ES 로 최적화한 자세 = 데이터셋 score)

두 점수는 같은 평가기(학습 데이터의 몸 모델·패치 밀도, predict.default_rescorer)로 잰다.

방법 (괄호 = 체형 하나에 쓰는 재채점 횟수)
    flow      flow matching 샘플 n 개 → 패치판 재채점 → 최고 (n, 제안안, predict_pose 기본 동작)
    flow+stub flow 샘플 n − 2 개 + 고정 후보표 2개를 함께 재채점 (n)
    flow1     flow matching 샘플 1개, 재채점 없음 (0)
    knn       가까운 학습 체형 n 개(같은 시나리오)의 최적 자세를 후보로 재채점 (n). 학습 없는 기준선
    clsreg    팔 봉우리 분류 + 봉우리별 회귀(HGB). 봉우리마다 후보 1개씩 재채점 (2, interfaces.md 우회책)
    hgb       HistGradientBoostingRegressor 평균 회귀 (0, README 5.6 기본안)
    mlp       MLPRegressor 평균 회귀 (0, README 5.6 기본안)
    stub      recommend.py 스텁: 시나리오별 고정 후보표(만세·팔 내림) → 이 체형으로 재채점 → 최고 (2)

채택 기준: 중앙값이 아니라 **하위 5% 점수 비율과 0.95 미만 비율**이 stub·knn 보다 나을 것.
중앙값 0.95(README H3)는 stub 이 이미 넘는다 (메시판 300행: 중앙값 0.996~0.999, 하위 5% 0.90~0.96).
경계 구간(wheelchair 키 1.45~1.60 m, 선 자세 키 1.83 m 이상)은 따로 집계한다.

사용 예:
    python scripts/run_e5_flow.py --model data/models/pose_flow.pt
    python scripts/run_e5_flow.py --model data/models/pose_flow.pt --methods flow,knn,stub --n-samples 32
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

METHODS = ("hybrid", "flow", "flow+stub", "flow1", "knn", "knn+stub", "clsreg", "hgb", "mlp", "stub")
RATIO_FLOOR = 0.95


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E5: flow matching vs 비교군 점수 비율")
    ap.add_argument("--model", default=str(ROOT / "data" / "models" / "pose_flow.pt"))
    ap.add_argument("--dataset", default=None, help="기본값: 산출물 meta 의 dataset")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--n-samples", type=int, default=16, help="flow·knn 의 재채점 후보 수 (같은 값으로 맞춘다)")
    ap.add_argument("--n-flow", type=int, default=8, help="hybrid 의 flow 후보 수")
    ap.add_argument("--n-knn", type=int, default=8, help="hybrid 의 kNN 후보 수")
    ap.add_argument("--folds", type=int, default=0,
                    help="체형 K-fold 교차검증 (0 이면 산출물 meta 의 holdout 체형만). fold 마다 flow 를 다시 학습한다")
    ap.add_argument("--fold-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="holdout 행 수 상한 (빠른 점검용)")
    ap.add_argument("--out", default=None, help="기본값: outputs/e5flow_<시각>.csv")
    return ap


def in_boundary(scenario: str, height_m: float) -> bool:
    """봉우리 전환·천장 경계 구간 (docs/proposals/flow_matching.md 5절)."""
    return (1.45 <= height_m <= 1.60) if scenario == "wheelchair" else height_m >= 1.83


def pose_matrix(df, keys) -> np.ndarray:
    return df[[f"pose_{k}" for k in keys]].to_numpy(dtype=np.float64)


def fit_mean_regressors(train, model):
    """최적 자세(pose_*)를 평균 회귀하는 MLP·HGB. 입력은 flow 모델과 같은 조건 벡터."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neural_network import MLPRegressor

    from airis.model.flow import BODY_KEYS, POSE_KEYS

    body = train[[f"body_{k}" for k in BODY_KEYS]].to_numpy(dtype=np.float32)
    X = model._cond(body, list(train["scenario"].astype(str))).numpy()
    Y = model.space.encode(pose_matrix(train, POSE_KEYS))
    mlp = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=2000, random_state=0).fit(X, Y)
    hgb = MultiOutputRegressor(HistGradientBoostingRegressor(max_depth=3, max_iter=200, random_state=0)).fit(X, Y)
    return mlp, hgb


class PeakRegressor:
    """시나리오별로 봉우리(arm_class)마다 자세 회귀. 행이 min_rows 보다 적은 봉우리는 평균 자세를 쓴다.

    후보는 학습에 있던 봉우리마다 하나씩이고(최대 2개), 재채점으로 고른다. 봉우리 분류기는 후보 순서만 정한다.
    """

    def __init__(self, train, space, min_rows: int = 12) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor

        from airis.model.flow import BODY_KEYS, POSE_KEYS

        self.space, self.body_keys = space, BODY_KEYS
        self.models: dict[tuple[str, str], object] = {}
        for (scenario, peak), g in train.groupby(["scenario", "arm_class"]):
            X = g[[f"body_{k}" for k in BODY_KEYS]].to_numpy(dtype=np.float64)
            Y = space.encode(pose_matrix(g, POSE_KEYS))
            if len(g) >= min_rows:
                reg = MultiOutputRegressor(
                    HistGradientBoostingRegressor(max_depth=3, max_iter=200, random_state=0)).fit(X, Y)
                self.models[(scenario, peak)] = reg
            else:
                self.models[(scenario, peak)] = Y.mean(axis=0)

    def candidates(self, body, scenario) -> list:
        x = np.array([[getattr(body, k) for k in self.body_keys]], dtype=np.float64)
        out = []
        for (name, _peak), m in self.models.items():
            if name != scenario.name:
                continue
            y = m if isinstance(m, np.ndarray) else m.predict(x)[0]
            out.append(self.space.to_pose(y, scenario))
        return out


class NearestBodies:
    """가까운 학습 체형의 최적 자세를 후보로 돌려준다 (표준화한 체형 5개의 유클리드 거리)."""

    def __init__(self, train, space) -> None:
        from airis.model.flow import BODY_KEYS, POSE_KEYS

        self.space, self.body_keys = space, BODY_KEYS
        cols = [f"body_{k}" for k in BODY_KEYS]
        self.mean = train[cols].to_numpy(dtype=np.float64).mean(axis=0)
        self.std = np.maximum(train[cols].to_numpy(dtype=np.float64).std(axis=0), 1e-9)
        self.by_scenario = {s: ((g[cols].to_numpy(dtype=np.float64) - self.mean) / self.std,
                                space.encode(pose_matrix(g, POSE_KEYS)))
                            for s, g in train.groupby("scenario")}

    def candidates(self, body, scenario, k: int) -> list:
        B, Y = self.by_scenario[scenario.name]
        b = (np.array([getattr(body, key) for key in self.body_keys], dtype=np.float64) - self.mean) / self.std
        idx = np.argsort(((B - b) ** 2).sum(axis=1))[:k]
        return [self.space.to_pose(Y[i], scenario) for i in idx]


def best_of(evaluator, cands, nozzle, body, scenario):
    """후보를 재채점해 (최고 자세, 쓴 평가 횟수). 가능한 후보가 없으면 벌점이 가장 작은 것."""
    from airis.optimize.cmaes_runner import score_batch

    sc, bad = score_batch(evaluator, cands, nozzle, body, scenario)
    pick = np.where(bad, -np.inf, sc) if not bad.all() else sc
    return cands[int(np.argmax(pick))], len(cands)


def summarize(res):
    """방법 × 시나리오 요약. 채택 기준 열: ratio_p05, below_095."""
    return (res.groupby(["method", "scenario"])
               .agg(n=("ratio", "size"),
                    ratio_median=("ratio", "median"),
                    ratio_p05=("ratio", lambda s: float(np.nanpercentile(s, 5))),
                    ratio_min=("ratio", "min"),
                    below_095=("ratio", lambda s: float(np.mean(s < RATIO_FLOOR))),
                    infeasible_frac=("infeasible", "mean"),
                    n_evals=("n_evals", "mean"),
                    ms_mean=("ms", "mean"))
               .reset_index())


def fold_indices(body_idx: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    """체형 단위 K-fold. 같은 (체형 목록, folds, seed) 면 같은 분할."""
    ids = np.unique(body_idx)
    shuffled = np.random.default_rng(seed).permutation(ids)
    return [np.sort(part) for part in np.array_split(shuffled, folds)]


def train_fold_model(train, scenarios, base_model, meta: dict):
    """fold 학습 체형만으로 flow 를 다시 학습한다 (설정은 산출물과 같게)."""
    from airis.model.flow import train_pose_flow

    return train_pose_flow(train, scenarios, base_model.cfg, meta=dict(meta), log=None)


def run_split(test, train, model, methods, args, evaluator, nozzle, scenarios, *, fold=None,
              knn_path=None, model_path=None) -> list[dict]:
    """holdout(test) 행마다 방법별로 자세를 예측하고 점수 비율을 잰다."""
    import time as _time

    from airis.model import predict as pred
    from airis.model.flow import BODY_KEYS, body_from_row
    from airis.optimize.cmaes_runner import score_batch
    from airis.optimize.encoding import PoseEncoder

    mlp = hgb = None
    if "mlp" in methods or "hgb" in methods:
        mlp, hgb = fit_mean_regressors(train, model)
    peaks = PeakRegressor(train, model.space) if "clsreg" in methods else None
    near = (NearestBodies(train, model.space)
            if {"knn", "knn+stub"} & set(methods) else None)
    stub_table = None
    if {"stub", "flow+stub", "knn+stub", "hybrid"} & set(methods):
        from airis.realtime.recommend import STUB_TABLE as stub_table

    n = max(1, int(args.n_samples))
    rows: list[dict] = []
    t_all = _time.perf_counter()
    for i, row in test.iterrows():
        body, scenario = body_from_row(row), scenarios[row["scenario"]]
        ref = float(row["score"])
        enc = PoseEncoder(scenario)
        stub_cands = [enc.clip_pose(e.pose) for e in stub_table[scenario.name]] if stub_table else []
        seed = args.seed + int(row["body_idx"])
        for m in methods:
            t0 = _time.perf_counter()
            if m == "hybrid":
                p = pred.predict(body, scenario, backend="hybrid", n_flow=args.n_flow, n_knn=args.n_knn,
                                 seed=seed, path=model_path or args.model, knn_path=knn_path,
                                 evaluator=evaluator, nozzle=nozzle, extra_candidates=stub_cands)
                pose, n_evals = p.pose, len(p.candidates)
            elif m == "flow":
                p = pred.predict(body, scenario, backend="flow", n_samples=n, seed=seed, path=model_path or args.model,
                                 evaluator=evaluator, nozzle=nozzle)
                pose, n_evals = p.pose, len(p.candidates)
            elif m == "flow+stub":
                p = pred.predict(body, scenario, backend="flow", n_samples=max(1, n - len(stub_cands)),
                                 seed=seed, path=model_path or args.model, evaluator=evaluator, nozzle=nozzle,
                                 extra_candidates=stub_cands)
                pose, n_evals = p.pose, len(p.candidates)
            elif m == "flow1":
                pose = pred.predict(body, scenario, backend="flow", n_samples=1, rescore=False,
                                    seed=seed, path=model_path or args.model).pose
                n_evals = 0
            elif m == "knn":
                pose, n_evals = best_of(evaluator, near.candidates(body, scenario, n), nozzle, body, scenario)
            elif m == "knn+stub":
                cands = near.candidates(body, scenario, max(1, n - len(stub_cands))) + stub_cands
                pose, n_evals = best_of(evaluator, cands, nozzle, body, scenario)
            elif m == "clsreg":
                pose, n_evals = best_of(evaluator, peaks.candidates(body, scenario), nozzle, body, scenario)
            elif m in ("mlp", "hgb"):
                bvec = np.array([[getattr(body, k) for k in BODY_KEYS]], dtype=np.float32)
                x = (mlp if m == "mlp" else hgb).predict(model._cond(bvec, [scenario.name]).numpy())
                pose, n_evals = model.space.to_pose(x[0], scenario), 0
            else:
                pose, n_evals = best_of(evaluator, stub_cands, nozzle, body, scenario)
            ms = (_time.perf_counter() - t0) * 1000.0
            score, infeasible = score_batch(evaluator, [pose], nozzle, body, scenario)
            rows.append({
                "body_idx": int(row["body_idx"]), "scenario": scenario.name, "method": m,
                "height_m": body.height_m, "boundary": in_boundary(scenario.name, body.height_m),
                "ref_score": ref, "score": float(score[0]),
                "ratio": float(score[0]) / ref if ref > 0 else float("nan"),
                "infeasible": bool(infeasible[0]), "n_evals": n_evals, "ms": ms,
                "shoulder_abduction": pose.shoulder_abduction, "torso_yaw": pose.torso_yaw,
                "ref_arm_class": row.get("arm_class"), "fold": fold,
            })
        if (i + 1) % 10 == 0 or i + 1 == len(test):
            tag = "" if fold is None else f"fold {fold} "
            print(f"[e5flow] {tag}{i + 1}/{len(test)}  {_time.perf_counter() - t_all:.0f} s")

    return rows


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

    model = pred.load_model(args.model, kind="pose")
    df = pd.read_parquet(args.dataset or model.meta["dataset"])
    scenarios = load_scenarios()
    evaluator, nozzle = pred.default_rescorer(model)

    if args.folds and args.folds > 1:
        rows = []
        folds = fold_indices(df["body_idx"].to_numpy(), args.folds, args.fold_seed)
        tmp = Path(args.out).parent if args.out else ROOT / "outputs"
        tmp.mkdir(parents=True, exist_ok=True)
        for f, ids in enumerate(folds):
            test = df[df["body_idx"].isin(ids)].reset_index(drop=True)
            train = df[~df["body_idx"].isin(ids)].reset_index(drop=True)
            if args.limit:
                test = test.head(args.limit)
            print(f"[e5flow] fold {f}: 학습 {len(train)}행 / 평가 {len(test)}행 — flow 재학습")
            fold_model = train_fold_model(train, scenarios, model, model.meta)
            fold_model_path = tmp / f"_fold{f}_flow.pt"
            fold_model.save(fold_model_path)
            knn_path = None
            if {"hybrid"} & set(methods):
                from airis.model.knn import PoseKNN

                knn_path = tmp / f"_fold{f}_knn.parquet"
                PoseKNN.from_dataset(train).save(knn_path)
            rows += run_split(test, train, fold_model, methods, args, evaluator, nozzle, scenarios,
                              fold=f, knn_path=knn_path, model_path=fold_model_path)
    else:
        holdout = set(model.meta.get("holdout_body_idx", []))
        if not holdout:
            print("산출물에 holdout 체형이 없다 (train_pose_flow.py --holdout-frac 0 으로 학습, 또는 --folds 5).",
                  file=sys.stderr)
            return 2
        test = df[df["body_idx"].isin(holdout)].reset_index(drop=True)
        if args.limit:
            test = test.head(args.limit)
        train = df[~df["body_idx"].isin(holdout)].reset_index(drop=True)
        rows = run_split(test, train, model, methods, args, evaluator, nozzle, scenarios, fold=None)

    res = pd.DataFrame(rows)
    summary = summarize(res)
    edge = res[res["boundary"]]
    out = Path(args.out) if args.out else ROOT / "outputs" / f"{explog.new_exp_id('e5flow')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False, encoding="utf-8")
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False, encoding="utf-8")

    fmt = lambda v: f"{v:.3f}"                                                   # noqa: E731
    print()
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string(index=False, float_format=fmt))
        if len(edge):
            edge_summary = summarize(edge)
            edge_summary.to_csv(out.with_name(out.stem + "_boundary.csv"), index=False, encoding="utf-8")
            print("\n경계 구간 (wheelchair 키 1.45~1.60 m, 선 자세 키 1.83 m 이상)")
            print(edge_summary.to_string(index=False, float_format=fmt))
    print()
    print(f"채택 기준: ratio_p05 와 below_095(점수 비율 {RATIO_FLOOR} 미만 비율)가 stub·knn 보다 나을 것. "
          "n_evals 가 같은 방법끼리 비교한다.")
    print(f"행별 결과 {out}")
    print(f"요약      {out.with_name(out.stem + '_summary.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
