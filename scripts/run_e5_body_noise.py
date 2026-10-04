"""체형 입력 오차 내성: 체형 값에 ±p% 오차가 섞였을 때 추천 품질이 얼마나 떨어지는가. 소유자: F.

목적 (총괄 2026-10-04): 카메라 포즈 추정 오차(외투·조명)가 자세 추천에 미치는 영향을 수치로 낸다.
입력 센서를 바꿀지(mmWave 등) 판단하는 근거다.

방법: 체형 5-fold 와 같은 틀이다 (docs/experiments_model.md 7절).
    1. fold 마다 학습 체형으로 flow 를 학습하고 kNN 표를 만든다 (오차 수준과 무관하게 한 번).
    2. 평가 체형마다 체형 값 5개에 독립으로 (1 + U(−p, +p)) 를 곱한 **추정 체형**을 만든다.
    3. 추천(후보 생성과 재채점)은 추정 체형으로 한다. 실제 장비는 진짜 체형을 모른다.
    4. 고른 자세를 **진짜 체형**으로 채점해 점수 비율(= 점수 / 그 체형의 최적 점수)을 낸다.
    5. 오차 수준 p 마다 하위 5%, 0.95 미만 비율, 불가 비율을 낸다. p = 0 이 기준이다.

불가 비율은 "추정 체형에서는 부스 안이던 자세가 진짜 체형에서는 천장·벽에 걸리는" 경우를 잡는다.

결과 (--out-dir):
    rows.csv            행별 결과 (run_e5_flow 의 열 + body_noise, est_height_m)
    summary.csv         오차 수준 × 방법 지표와 p = 0 대비 변화
    manifest.json       커밋, 데이터셋·현재 도장, 오차 수준, 시드, 학습 장치, 소요 시간, 인자

사용 예:
    python scripts/run_e5_body_noise.py --dataset data/datasets/pose_dataset_mesh_k14.parquet \\
        --model outputs/refresh_k14/pose_flow.pt
    python scripts/run_e5_body_noise.py --dataset ... --model ... --levels 0,0.05 --folds 2 --limit 3   # 빠른 점검

품질 실행이라 다른 실험과 동시에 돌려도 된다 (응답 시간 열은 참고용).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np                                           # noqa: E402

from airis.optimize import cli, explog                       # noqa: E402

LEVELS = "0,0.02,0.05,0.10"
METHODS = "hybrid,knn+stub,stub"


def build_parser() -> argparse.ArgumentParser:
    import run_e5_flow

    e5 = run_e5_flow.build_parser()
    ap = argparse.ArgumentParser(description="체형 입력 오차 내성 (추천은 추정 체형, 채점은 진짜 체형)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, help="학습 설정을 읽을 flow 산출물 (fold 마다 같은 설정으로 다시 학습한다)")
    ap.add_argument("--levels", default=LEVELS, help="오차 비율 (0.05 = ±5%%). 0 은 기준으로 항상 넣는다")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--margins", default="0",
                    help="부스 안 판정 여유 (0.05 = 체형을 5%% 키운 몸에서도 부스 안인 후보만). 쉼표로 여러 개. "
                         "혼합 계열에만 적용된다 (docs/experiments_model.md 7.2절)")
    ap.add_argument("--methods", default=METHODS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-seed", type=int, default=e5.get_default("fold_seed"))
    ap.add_argument("--n-flow", type=int, default=e5.get_default("n_flow"))
    ap.add_argument("--n-knn", type=int, default=e5.get_default("n_knn"))
    ap.add_argument("--seed", type=int, default=e5.get_default("seed"))
    ap.add_argument("--limit", type=int, default=None, help="fold 마다 평가 행 상한 (빠른 점검용)")
    ap.add_argument("--out-dir", default=None, help="기본값: outputs/<body_noise_시각>")
    return ap


def parse_levels(text: str) -> list[float]:
    """쉼표로 나눈 오차 비율 → 오름차순, 중복 없음, 0 포함. 음수나 1 이상은 ValueError."""
    levels = sorted({float(x) for x in text.split(",") if x.strip()} | {0.0})
    bad = [v for v in levels if not 0.0 <= v < 1.0]
    if bad:
        raise ValueError(f"오차 비율은 0 이상 1 미만이어야 한다: {bad}")
    return levels


def summarize(rows) -> "object":
    """(여유 ×) 오차 수준 × 방법 지표와 기준 대비 변화. 지표 정의는 train_pose_models.method_stats 와 같다.

    기준은 같은 방법의 "여유 0, 오차 0" 행이다. 여유 열(feasibility_margin)이 없으면 여유 0 으로 본다.
    """
    import pandas as pd

    import train_pose_models

    rows = rows.copy()
    if "feasibility_margin" not in rows.columns:
        rows["feasibility_margin"] = 0.0
    rows["feasibility_margin"] = rows["feasibility_margin"].fillna(0.0).astype(float)
    parts = []
    for (margin, level), g in rows.groupby(["feasibility_margin", "body_noise"], sort=True):
        stats = train_pose_models.method_stats(g)
        stats.insert(0, "body_noise", float(level))
        stats.insert(0, "feasibility_margin", float(margin))
        for col in ("margin_rejected", "margin_fallback"):      # 여유 때문에 추천이 바뀐 행 / 폴백한 행의 비율
            if col in g.columns:
                share = g.groupby("method", sort=False)[col].apply(
                    lambda s: float((s.fillna(0).astype(float) > 0).mean()))
                stats[f"{col}_rows"] = stats["method"].map(share)
        parts.append(stats)
    out = pd.concat(parts, ignore_index=True)
    is_base = (out["body_noise"] == 0.0) & (out["feasibility_margin"] == 0.0)
    base = out[is_base].set_index("method")
    for col in ("p05", "below_095", "infeasible", "median"):
        out[f"d_{col}"] = [float(r[col] - base.at[r["method"], col]) if r["method"] in base.index else float("nan")
                           for _, r in out.iterrows()]
    # 같은 행(체형·시나리오·방법)끼리 짝지은 점수 비율 변화: 평균과 가장 크게 떨어진 값
    key = ["body_idx", "scenario", "method"]
    zero = rows[(rows["body_noise"] == 0.0) & (rows["feasibility_margin"] == 0.0)].set_index(key)["ratio"]
    paired_mean, paired_min, changed = [], [], []
    for _, r in out.iterrows():
        g = rows[(rows["body_noise"] == r["body_noise"]) & (rows["method"] == r["method"])
                 & (rows["feasibility_margin"] == r["feasibility_margin"])].set_index(key)
        d = g["ratio"] - zero.reindex(g.index)
        paired_mean.append(float(d.mean()))
        paired_min.append(float(d.min()))
        changed.append(float((d.abs() > 1e-9).mean()))
    out["paired_ratio_mean"], out["paired_ratio_min"], out["rows_changed"] = paired_mean, paired_min, changed
    return out


def markdown_table(summary, info: dict) -> str:
    """docs/experiments_model.md 7절에 붙일 표."""
    lines = [f"데이터 `{info['dataset']}` ({info['rows']}행), {info['folds']}-fold, 커밋 `{info['commit']}`, "
             f"오차 시드 {info['noise_seed']}, 학습 장치 {info['device']}",
             "",
             "| 방법 | 판정 여유 | 체형 오차 | 하위 5% | 0.95 미만 | 불가(진짜 체형) | 중앙값 | 짝지은 비율 변화(평균) | 가장 큰 하락 | 추천이 바뀐 행 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.sort_values(["method", "feasibility_margin", "body_noise"]).iterrows():
        lines.append(f"| {r['method']} | {100 * r['feasibility_margin']:.0f}% | ±{100 * r['body_noise']:.0f}% | "
                     f"{r['p05']:.4f} | {100 * r['below_095']:.2f}% | "
                     f"{100 * r['infeasible']:.2f}% | {r['median']:.4f} | {r['paired_ratio_mean']:+.4f} | "
                     f"{r['paired_ratio_min']:+.4f} | {100 * r['rows_changed']:.1f}% |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)
    start_commit = explog.git_commit()      # 실행 코드의 커밋은 시작할 때 잡는다 (도중에 HEAD 가 바뀔 수 있다)

    import pandas as pd

    import run_e5_flow
    import train_pose_models
    from airis.model import predict as pred
    from airis.model.knn import PoseKNN
    from airis.sim.scenario import load_scenarios

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in run_e5_flow.METHODS]
    if bad:
        print(f"알 수 없는 방법 {bad} (가능: {', '.join(run_e5_flow.METHODS)})", file=sys.stderr)
        return 2
    try:
        levels = parse_levels(args.levels)
        margins = parse_levels(args.margins)
        dataset = Path(args.dataset).resolve()
        df = pd.read_parquet(dataset)
        stamp, warns = train_pose_models.check_dataset(df)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    for w in warns:
        print(f"[경고] {w}")

    base = pred.load_model(args.model, kind="pose")
    evaluator, nozzle = pred.default_rescorer(base)
    scenarios = load_scenarios()
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / explog.new_exp_id("body_noise")
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    t_all = time.perf_counter()
    rows: list[dict] = []
    devices = set()
    folds = run_e5_flow.fold_indices(df["body_idx"].to_numpy(), args.folds, args.fold_seed)
    for f, ids in enumerate(folds):
        test = df[df["body_idx"].isin(ids)].reset_index(drop=True)
        train = df[~df["body_idx"].isin(ids)].reset_index(drop=True)
        if args.limit:
            test = test.head(args.limit)
        print(f"[body-noise] fold {f}: 학습 {len(train)}행 / 평가 {len(test)}행 — flow 재학습")
        fold_model = run_e5_flow.train_fold_model(train, scenarios, base, base.meta)
        devices.add(str(fold_model.meta.get("device")))
        model_path = fold_model.save(work / f"fold{f}_flow.pt")
        knn_path = PoseKNN.from_dataset(train).save(work / f"fold{f}_knn.parquet")
        for margin in margins:
            for level in levels:
                e5_args = run_e5_flow.build_parser().parse_args(
                    ["--model", str(model_path), "--n-flow", str(args.n_flow), "--n-knn", str(args.n_knn),
                     "--seed", str(args.seed), "--body-noise", str(level), "--body-noise-seed", str(args.noise_seed),
                     "--feasibility-margin", str(margin)])
                print(f"[body-noise] fold {f} 판정 여유 {100 * margin:g}% 체형 오차 ±{100 * level:g}%")
                rows += run_e5_flow.run_split(test, train, fold_model, methods, e5_args, evaluator, nozzle,
                                              scenarios, fold=f, knn_path=knn_path, model_path=model_path)
        pd.DataFrame(rows).to_csv(out_dir / "rows.csv", index=False, encoding="utf-8")   # 끊겨도 남게

    res = pd.DataFrame(rows)
    summary = summarize(res)
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8")
    info = {"dataset": dataset.name, "rows": int(len(df)), "folds": args.folds, "commit": start_commit,
            "noise_seed": args.noise_seed, "device": "·".join(sorted(devices))}
    (out_dir / "summary.md").write_text(markdown_table(summary, info), encoding="utf-8")
    manifest = {"commit": info["commit"], "dataset": str(dataset), "dataset_stamp": stamp,
                "current_stamp": pred.current_stamp(), "warnings": warns, "levels": levels, "margins": margins,
                "margin": "체형 값 전부 × (1 + 여유) 인 몸에서도 부스 안인 후보만 고른다 (혼합 계열)",
                "noise": "체형 값마다 독립, (1 + U(−p, +p)) 를 곱한다", "noise_seed": args.noise_seed,
                "methods": methods, "folds": args.folds, "fold_rule": "permutation", "devices": sorted(devices),
                "limit": args.limit, "total_s": round(time.perf_counter() - t_all, 1), "args": vars(args)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
                                           encoding="utf-8")
    print()
    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    if args.limit:
        print(f"(--limit {args.limit} 빠른 점검: 수치를 결론에 쓰지 않는다)")
    print(f"결과 {out_dir}  (총 {manifest['total_s'] / 60:.1f}분)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
