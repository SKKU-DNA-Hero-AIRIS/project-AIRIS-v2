"""계획 추천의 응답 시간 측정: 재채점 수(rescore_top) × 스레드 수. 소유자: F. 총괄 2026-10-11.

품질은 run_e5_plan.py 5-fold 로 잰다. 여기는 시간만이다. CPU 가 비어 있을 때 돌린다 (다른 계산과 겹치면 모든 값이
같은 배율로 느려진다). 자세 추천의 run_e5_timing.py 와 같은 방식이다: 데이터셋에서 행 --n-rows 개를 뽑고, 행마다
구성 순서를 돌려 가며 부르고, 첫 호출(산출물 로드·캐시)은 뺀다.

후보는 장비와 같게 만든다: 가까운 체형의 최적 계획 n_knn 개 + 고정 회전 계획 2개(제안 자세 회전, 기본 자세 회전).
제안 자세는 데이터셋 행의 단일 자세 최적(pose_<값>)을 쓴다 (시간만 재므로 새는 것은 문제가 아니다).

사용 예:
    python scripts/run_e5_plan_timing.py --dataset data/datasets/<계획>.parquet --knn outputs/<실행>/plan_knn.parquet
    python scripts/run_e5_plan_timing.py --dataset … --knn … --density 1500 --tops 5,8,all --threads 1,3
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="계획 추천 응답 시간 (재채점 수 × 스레드 수)")
    ap.add_argument("--dataset", required=True, help="체형·유형 표본을 뽑을 계획 데이터셋")
    ap.add_argument("--knn", required=True, help="계획 kNN 표")
    ap.add_argument("--model", default=None, help="계획 flow 산출물 (--backend hybrid·flow 일 때)")
    ap.add_argument("--backend", default="knn", choices=("knn", "hybrid", "flow"))
    ap.add_argument("--n-knn", type=int, default=16)
    ap.add_argument("--n-flow", type=int, default=8)
    ap.add_argument("--tops", default="5,8,all", help="재채점 수 (쉼표로, all = 전부)")
    ap.add_argument("--threads", default="1,3", help="스레드 수 (쉼표로)")
    ap.add_argument("--density", type=float, default=2000.0, help="재채점 패치 밀도 (/m²)")
    ap.add_argument("--n-rows", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-rotation-pose", action="store_true", help="제안 자세 회전을 빼고 기본 자세 회전만 넣는다")
    ap.add_argument("--out", default=None, help="기본값: outputs/e5plantiming_<시각>.csv")
    return ap


def parse_tops(text: str) -> list[int | None]:
    return [None if t.strip() in ("all", "전부") else int(t) for t in text.split(",") if t.strip()]


def timing(args):
    import pandas as pd

    import run_e5_plan
    from airis.model import plan_data
    from airis.model import predict as pred
    from airis.model.flow import body_from_row
    from airis.sim.scenario import load_scenarios

    df, _ = plan_data.read_plan_dataset(args.dataset)
    limits = plan_data.plan_limits_from_dataset(df)
    meta = plan_data.plan_meta_from_dataset(df)
    idx = np.random.default_rng(args.seed).choice(len(df), size=min(args.n_rows, len(df)), replace=False)
    sample = df.iloc[np.sort(idx)].reset_index(drop=True)
    scenarios = load_scenarios()
    evaluator = pred._rescore_evaluator(str(meta.get("body_model") or "mesh"), float(args.density), True)
    nozzle = cli.resolve_nozzles()[0]
    configs = [(k, t) for k in parse_tops(args.tops) for t in [int(x) for x in args.threads.split(",") if x.strip()]]
    rot_args = argparse.Namespace(rotation_pose="none" if args.no_rotation_pose else "dataset")

    def call(row, k, threads):
        body, scenario = body_from_row(row), scenarios[row["scenario"]]
        rot = run_e5_plan.rotation_pose_for(row, scenario, body, rot_args, 0)
        return pred.predict_plan_candidates(
            body, scenario, backend=args.backend, n_knn=args.n_knn, n_flow=args.n_flow, fixed=True,
            seed=args.seed + int(row["body_idx"]), path=args.model, knn_path=args.knn, evaluator=evaluator,
            nozzle=nozzle, rotation_pose=rot, limits=limits, rescore_top=k, n_threads=threads)

    for k, t in configs:                                        # 산출물 로드·캐시 데우기 (측정 제외)
        call(sample.iloc[0], k, t)

    out = []
    for r, row in sample.iterrows():
        order = configs[r % len(configs):] + configs[:r % len(configs)]     # 순서 효과를 줄이려고 돌린다
        for k, t in order:
            t0 = time.perf_counter()
            p = call(row, k, t)
            ms = (time.perf_counter() - t0) * 1000.0
            out.append({"body_idx": int(row["body_idx"]), "scenario": str(row["scenario"]),
                        "rescore_top": "all" if k is None else k, "n_threads": t, "ms": ms,
                        "n_cands": len(p.candidates), "n_rescored": p.n_rescored, "picked": p.source,
                        "plan_phases": len(p.plan.phases), "density": float(args.density)})
        print(f"[plan-timing] {r + 1}/{len(sample)}", flush=True)
    res = pd.DataFrame(out)
    summary = (res.groupby(["rescore_top", "n_threads"], sort=False)
                  .agg(n=("ms", "size"), mean_s=("ms", lambda s: s.mean() / 1000),
                       median_s=("ms", lambda s: s.median() / 1000),
                       p95_s=("ms", lambda s: np.percentile(s, 95) / 1000), max_s=("ms", lambda s: s.max() / 1000),
                       n_cands=("n_cands", "mean"), n_rescored=("n_rescored", "mean"))
                  .reset_index())
    summary["ms_per_plan"] = 1000.0 * summary["mean_s"] / summary["n_rescored"]
    summary["density"] = float(args.density)
    return res, summary


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)
    out = Path(args.out) if args.out else ROOT / "outputs" / f"{explog.new_exp_id('e5plantiming')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res, summary = timing(args)
    res.to_csv(out, index=False, encoding="utf-8")
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False, encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"밀도 {args.density:g}/m², {len(res) // max(len(summary), 1)}행, 후보 = {args.backend} "
          f"(kNN {args.n_knn}) + 고정 회전 계획. 결과 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
