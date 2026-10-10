"""계획 추천의 체형 K-fold 평가: 추천 계획 점수 / 그 체형을 직접 최적화한 계획 점수. 소유자: F.

docs/experiments_model.md 4절, 완성도 F2(하위 5% ≥ 0.97, 불가 0)·F3(응답 ≤ 1.5 s).

    점수 비율 = score(추천 계획) / score(그 체형·시나리오를 직접 최적화한 계획 = 데이터셋 score)

두 점수는 같은 평가기(계획 채점 설정 plan_physics_cfg, 학습 데이터의 몸 모델·패치 밀도)로 잰다.
fold 마다 학습 체형만으로 flow 를 다시 학습하고 kNN 표를 다시 만든다 (평가 체형은 모델이 본 적이 없다).

방법 (괄호 = 체형 하나에 쓰는 계획 채점 횟수. F = 고정 회전 계획 수: 제안 자세가 있으면 2, 없으면 1)
    hybrid           flow n_flow + kNN n_knn + 고정 회전 계획 (n_flow + n_knn + F, predict_plan 기본 동작)
    hybrid-nofixed   flow n_flow + kNN n_knn (n_flow + n_knn). 고정 회전 계획이 얼마나 받쳐 주는지 본다
    flow+fixed       flow (n_flow + n_knn) + 고정 회전 계획
    knn+fixed        kNN (n_flow + n_knn) + 고정 회전 계획
    fixed            고정 회전 계획만 (F). "그냥 회전" 기준선

고정 회전 계획의 "제안 자세" (--rotation-pose)
    none      기본 자세 회전만 넣는다 (기본값. 새지 않는다)
    dataset   데이터셋 행의 단일 자세 최적(pose_<값>)으로 돈다. **그 체형을 직접 최적화한 자세**라 실제 추천에서는
              쓸 수 없는 값이다. 회전 후보의 상한을 보는 용도이고 결과에 그렇게 적는다
    model     자세 추천 모델(--pose-model, --pose-knn)의 추천으로 돈다. 그 산출물이 평가 체형을 학습에 썼으면
              새는 평가다 (행에 pose_artifacts 로 경로를 남긴다)

사용 예:
    python scripts/run_e5_plan.py --model outputs/plan_refresh/plan_flow.pt --dataset data/datasets/<계획>.parquet
    python scripts/run_e5_plan.py --model … --dataset … --folds 2 --limit 2 --methods hybrid,fixed   # 빠른 점검

응답 시간 열(ms)은 참고용이다 (다른 실행과 CPU 를 나눠 쓴다).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np                                           # noqa: E402

from airis.optimize import cli, explog                       # noqa: E402

METHODS = ("hybrid", "hybrid-nofixed", "flow+fixed", "knn+fixed", "fixed")
ROTATION_POSE_MODES = ("none", "dataset", "model")


def build_parser() -> argparse.ArgumentParser:
    import run_e5_flow

    e5 = run_e5_flow.build_parser()
    ap = argparse.ArgumentParser(description="계획 추천 체형 K-fold 평가")
    ap.add_argument("--model", required=True, help="학습 설정을 읽을 계획 flow 산출물 (fold 마다 같은 설정으로 다시 학습)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--methods", default="hybrid,hybrid-nofixed,knn+fixed,fixed")
    ap.add_argument("--n-flow", type=int, default=e5.get_default("n_flow"))
    ap.add_argument("--n-knn", type=int, default=e5.get_default("n_knn"))
    ap.add_argument("--rotation-steps", type=int, default=None, help="고정 회전 계획의 단계 수 (기본: predict 의 값)")
    ap.add_argument("--rotation-pose", choices=ROTATION_POSE_MODES, default="none")
    ap.add_argument("--pose-model", default=None, help="--rotation-pose model 의 자세 flow 산출물")
    ap.add_argument("--pose-knn", default=None, help="--rotation-pose model 의 자세 kNN 표")
    ap.add_argument("--rescore-top", type=int, default=None,
                    help="재채점할 후보 수 (기본: 전부). 후보는 그대로 만들고 predict.rescore_subset 이 고른 것만 채점한다")
    ap.add_argument("--n-threads", type=int, default=1, help="재채점 스레드 수 (결과는 순차와 같다)")
    ap.add_argument("--feasibility-margin", type=float, default=0.0)
    ap.add_argument("--feasibility-margin-mode", choices=("all", "reach"), default="all")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-seed", type=int, default=e5.get_default("fold_seed"))
    ap.add_argument("--fold-rule", choices=run_e5_flow.FOLD_RULES, default="permutation")
    ap.add_argument("--seed", type=int, default=e5.get_default("seed"))
    ap.add_argument("--limit", type=int, default=None, help="fold 마다 평가 행 상한 (빠른 점검용)")
    ap.add_argument("--refilter-min-dist", type=float, default=None,
                    help="후보 재필터 임계 (기본: 파일의 candidate_min_dist 도장). 옛 파일은 낮추면 후보가 더 남는다")
    ap.add_argument("--no-refilter", action="store_true",
                    help="읽은 직후의 후보 재필터(plan_data.read_plan_dataset)를 끈다")
    ap.add_argument("--out", default=None, help="기본값: outputs/e5plan_<시각>.csv")
    return ap


def rotation_pose_for(row, scenario, body, args, seed: int):
    """고정 회전 계획에 쓸 제안 자세 (없으면 None). 방식은 --rotation-pose."""
    from airis.model.flow import POSE_KEYS
    from airis.optimize.encoding import PoseEncoder
    from airis.sim import PoseParams

    if args.rotation_pose == "none":
        return None
    if args.rotation_pose == "dataset":
        missing = [k for k in POSE_KEYS if f"pose_{k}" not in row.index]
        if missing:
            raise KeyError(f"--rotation-pose dataset 은 데이터셋에 단일 자세 최적 열(pose_<값>)이 필요하다: {missing}")
        return PoseEncoder(scenario).clip_pose(PoseParams(**{k: float(row[f"pose_{k}"]) for k in POSE_KEYS}))
    from airis.model import predict as pred

    return pred.predict(body, scenario, seed=seed, path=args.pose_model, knn_path=args.pose_knn, dedup_deg=3.0).pose


def method_kwargs(method: str, args) -> dict:
    """방법 이름 → predict_plan_candidates 인자. 재채점 횟수를 맞추려고 한쪽만 쓸 때는 두 쪽의 합만큼 쓴다."""
    total = int(args.n_flow) + int(args.n_knn)
    if method == "hybrid":
        return {"backend": "hybrid", "n_flow": args.n_flow, "n_knn": args.n_knn, "fixed": True}
    if method == "hybrid-nofixed":
        return {"backend": "hybrid", "n_flow": args.n_flow, "n_knn": args.n_knn, "fixed": False}
    if method == "flow+fixed":
        return {"backend": "flow", "n_flow": total, "fixed": True}
    if method == "knn+fixed":
        return {"backend": "knn", "n_knn": total, "fixed": True}
    raise ValueError(f"혼합 계열이 아닌 방법: {method!r}")


def run_split(test, methods, args, evaluator, nozzle, scenarios, limits, *, fold=None, model_path=None,
              knn_path=None) -> list[dict]:
    """평가 행마다 방법별로 계획을 추천하고 점수 비율을 잰다."""
    from airis.model import predict as pred
    from airis.model.flow import body_from_row

    steps = args.rotation_steps or pred.ROTATION_STEPS
    rows: list[dict] = []
    t_all = time.perf_counter()
    for i, row in test.iterrows():
        body, scenario = body_from_row(row), scenarios[row["scenario"]]
        ref = float(row["score"])
        seed = args.seed + int(row["body_idx"])
        rot = rotation_pose_for(row, scenario, body, args, seed)
        for m in methods:
            t0 = time.perf_counter()
            if m == "fixed":
                cands = pred.fixed_plans(scenario, limits, rotation_pose=rot, n_steps=steps)
                scores, bad = pred.score_plans(evaluator, cands, nozzle, body, scenario)
                j = pred._pick(scores, bad)
                plan, score, infeasible, source, n_evals = cands[j], scores[j], bad[j], "fixed", len(cands)
            else:
                p = pred.predict_plan_candidates(
                    body, scenario, seed=seed, path=model_path, knn_path=knn_path, evaluator=evaluator, nozzle=nozzle,
                    rotation_pose=rot, rotation_steps=steps, limits=limits,
                    feasibility_margin=args.feasibility_margin, feasibility_margin_mode=args.feasibility_margin_mode,
                    rescore_top=args.rescore_top, n_threads=args.n_threads, **method_kwargs(m, args))
                j = next(k for k, c in enumerate(p.candidates) if c is p.plan)
                plan, score, infeasible, source = p.plan, p.scores[j], p.infeasible[j], p.source
                n_evals = p.n_rescored + p.n_margin_checks
            ms = (time.perf_counter() - t0) * 1000.0
            out = {
                "body_idx": int(row["body_idx"]), "scenario": scenario.name, "method": m,
                "height_m": body.height_m, "ref_score": ref, "score": float(score),
                "ratio": float(score) / ref if ref > 0 else float("nan"),
                "infeasible": bool(infeasible), "n_evals": int(n_evals), "ms": ms, "source": source,
                "plan_duration_s": plan.duration_s, "plan_phases": len(plan.phases),
                "rotation_pose": args.rotation_pose, "fold": fold,
                "rescore_top": args.rescore_top, "n_threads": args.n_threads,
            }
            for col in ("score_p1_10", "score_p1opt_10"):       # 데이터셋의 기준선과 나란히 (있으면)
                if col in row.index and float(row[col]) > 0:
                    out[f"ratio_vs_{col[6:]}"] = float(score) / float(row[col])
            rows.append(out)
        if (i + 1) % 10 == 0 or i + 1 == len(test):
            tag = "" if fold is None else f"fold {fold} "
            print(f"[e5plan] {tag}{i + 1}/{len(test)}  {time.perf_counter() - t_all:.0f} s")
    return rows


def train_fold(train, scenarios, base, space, meta: dict):
    """fold 학습 체형만으로 계획 flow 를 다시 학습한다 (설정은 산출물과 같게)."""
    from airis.model.flow import train_flow

    return train_flow(train, scenarios, base.cfg, space=space, meta=dict(meta), log=None)


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        print(f"알 수 없는 방법 {bad} (가능: {', '.join(METHODS)})", file=sys.stderr)
        return 2
    if args.rotation_pose == "model" and not (args.pose_model or args.pose_knn):
        print("--rotation-pose model 은 --pose-model 또는 --pose-knn 이 필요하다", file=sys.stderr)
        return 2
    if args.folds < 2:
        print("--folds 는 2 이상이어야 한다", file=sys.stderr)
        return 2

    import pandas as pd

    import run_e5_flow
    import train_pose_models
    from airis.model import plan_data
    from airis.model import predict as pred
    from airis.model.knn import PlanKNN
    from airis.sim.scenario import load_scenarios

    try:
        df, refilter = plan_data.read_plan_dataset(args.dataset, refilter=not args.no_refilter,
                                                   min_dist=args.refilter_min_dist)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"[e5plan] {plan_data.refilter_message(refilter)}")
    try:
        limits = plan_data.plan_limits_from_dataset(df)
        meta = plan_data.plan_meta_from_dataset(df)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    base = pred.load_model(args.model, kind="plan")
    scenarios = load_scenarios()
    names = sorted(set(df["scenario"].astype(str)))
    space = plan_data.plan_space_from_dataset(df, [scenarios[n] for n in names])
    evaluator, nozzle = pred.default_rescorer({**dict(base.meta), **meta}, plan=True)

    out = Path(args.out) if args.out else ROOT / "outputs" / f"{explog.new_exp_id('e5plan')}.csv"
    work = out.parent
    work.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    folds = run_e5_flow.fold_indices(df["body_idx"].to_numpy(), args.folds, args.fold_seed, args.fold_rule)
    for f, ids in enumerate(folds):
        test = df[df["body_idx"].isin(ids)].reset_index(drop=True)
        train = df[~df["body_idx"].isin(ids)].reset_index(drop=True)
        if args.limit:
            test = test.head(args.limit)
        print(f"[e5plan] fold {f}: 학습 {len(train)}행 / 평가 {len(test)}행 — 계획 flow 재학습 (단계 {limits.n_phases}개)")
        model_path = knn_path = None
        if any(m in ("hybrid", "hybrid-nofixed", "flow+fixed") for m in methods):
            model_path = train_fold(train, scenarios, base, space, {**dict(base.meta), **meta}).save(
                work / f"_fold{f}_plan_flow.pt")
        if any(m in ("hybrid", "hybrid-nofixed", "knn+fixed") for m in methods):
            knn_path = PlanKNN.from_dataset(train, min_phase_s=limits.min_phase_s).save(
                work / f"_fold{f}_plan_knn.parquet")
        # 로더는 경로별로 캐시한다. 같은 프로세스에서 같은 --out 으로 다시 돌리면 앞 실행의 fold 산출물이 남으므로
        # 비우고, 새 산출물을 미리 읽어 둔다 (응답 시간 열에 로드 시간이 들어가지 않게).
        pred._load_cached.cache_clear()
        pred._load_plan_knn_cached.cache_clear()
        if model_path is not None:
            pred.load_model(model_path, kind="plan")
        if knn_path is not None:
            pred.load_plan_knn(knn_path)
        rows += run_split(test, methods, args, evaluator, nozzle, scenarios, limits, fold=f, model_path=model_path,
                          knn_path=knn_path)

    res = pd.DataFrame(rows)
    if args.rotation_pose == "model":
        res["pose_artifacts"] = f"{args.pose_model};{args.pose_knn}"
    res.to_csv(out, index=False, encoding="utf-8")
    stats = train_pose_models.method_stats(res)
    stats["plan_duration_s"] = stats["method"].map(res.groupby("method")["plan_duration_s"].mean())
    stats["fixed_picked"] = stats["method"].map(res.groupby("method")["source"].apply(lambda s: float((s == "fixed").mean())))
    stats.to_csv(out.with_name(out.stem + "_overall.csv"), index=False, encoding="utf-8")
    cols = ["method", "n", "n_evals", "median", "p05", "min", "below_095", "infeasible", "mean_s", "p95_s",
            "plan_duration_s", "fixed_picked"]
    print()
    print(stats[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.rotation_pose == "dataset":
        print("주의: 제안 자세로 데이터셋의 단일 자세 최적을 썼다 (그 체형을 직접 최적화한 값, 회전 후보의 상한).")
    print(f"행별 결과 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
