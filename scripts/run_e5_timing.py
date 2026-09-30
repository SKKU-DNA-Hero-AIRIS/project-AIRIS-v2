"""재채점 단축안의 응답 시간 측정. 소유자: F. 총괄 2026-09-30 (품질은 run_e5_flow.py 5-fold, 여기는 시간만).

두 가지를 잰다. 둘 다 CPU 가 비어 있을 때 돌린다 (다른 계산과 겹치면 모든 값이 같은 배율로 느려진다).

1. --bench: 패치 밀도별 1회 채점 비용 (자세 몇 개 × 반복, 중앙값). 저밀도 선별(hybrid-b·c)이 이득이 있는지 가른다.
   1회 비용의 대부분이 밀도와 무관한 고정비(자세별 메시 변형·가림 준비)면 선별의 이득은 작다.
2. 응답 시간: 데이터셋에서 체형 --n-bodies 개(행)를 뽑아 R0(hybrid)·hybrid-a~d 를 실제 화면 밀도(--density,
   기본 메시 2,000/m²)로 부른다. 체형·시나리오마다 방법 순서를 돌려 가며 부르고, 첫 호출(산출물 로드)은 뺀다.

사용 예:
    python scripts/run_e5_timing.py --bench
    python scripts/run_e5_timing.py --dataset <본 폴더>/data/datasets/pose_dataset_mesh_cand.parquet \\
        --model <본 폴더>/data/models/pose_flow.pt --knn <본 폴더>/data/models/pose_knn.parquet
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np                                           # noqa: E402

from airis.optimize import cli, explog                       # noqa: E402

VARIANTS = ("hybrid", "hybrid-a", "hybrid-b", "hybrid-c", "hybrid-d", "hybrid-e")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="재채점 단축안 응답 시간")
    ap.add_argument("--bench", action="store_true", help="밀도별 1회 채점 비용만 잰다")
    ap.add_argument("--bench-densities", default="400,800,1500,2000")
    ap.add_argument("--bench-poses", type=int, default=6)
    ap.add_argument("--bench-repeat", type=int, default=5)
    ap.add_argument("--body-model", default="mesh")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--model", default=None, help="기본값: predict 의 산출물 경로")
    ap.add_argument("--knn", default=None)
    ap.add_argument("--n-bodies", type=int, default=30)
    ap.add_argument("--density", type=float, default=2000.0, help="최종 재채점 밀도 (E 화면)")
    ap.add_argument("--methods", default=",".join(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    # run_e5_flow 와 같은 변형 설정
    ap.add_argument("--n-flow", type=int, default=8)
    ap.add_argument("--n-knn", type=int, default=8)
    ap.add_argument("--dedup-deg", type=float, default=3.0)
    ap.add_argument("--screen-b", default="400:4")
    ap.add_argument("--screen-c", default="800:3")
    ap.add_argument("--threads", type=int, default=3)
    return ap


def bench(args) -> "object":
    """밀도별 1회 채점 비용 (ms 중앙값). 자세는 고정 후보표와 기본 자세."""
    import pandas as pd

    from airis.model.predict import _rescore_evaluator
    from airis.optimize.encoding import PoseEncoder
    from airis.realtime.recommend import STUB_TABLE
    from airis.sim import PoseParams
    from airis.sim.human_mesh import MESH_DEFAULT_BODY
    from airis.sim.scenario import load_scenarios

    sc = load_scenarios()["default"]
    enc = PoseEncoder(sc)
    poses = [enc.clip_pose(e.pose) for e in STUB_TABLE["default"]] + [enc.clip_pose(PoseParams())]
    rng = np.random.default_rng(args.seed)
    while len(poses) < args.bench_poses:
        base = poses[len(poses) % 3]
        poses.append(enc.clip_pose(PoseParams(**{k: float(v) + rng.normal(0, 5)
                                                  for k, v in asdict(base).items()})))
    nozzle = cli.resolve_nozzles()[0]
    rows = []
    for d in [float(x) for x in args.bench_densities.split(",")]:
        ev = _rescore_evaluator(args.body_model, d)
        ev.evaluate(poses[0], nozzle, MESH_DEFAULT_BODY, sc)                 # 첫 호출(캐시) 제외
        times = []
        for _ in range(args.bench_repeat):
            for pose in poses[:args.bench_poses]:
                t0 = time.perf_counter()
                ev.evaluate(pose, nozzle, MESH_DEFAULT_BODY, sc)
                times.append((time.perf_counter() - t0) * 1000.0)
        rows.append({"density": d, "ms_median": float(np.median(times)), "ms_p95": float(np.percentile(times, 95)),
                     "n": len(times)})
        print(f"[bench] {d:>6.0f}/m²  중앙값 {rows[-1]['ms_median']:.1f} ms  95% {rows[-1]['ms_p95']:.1f} ms")
    df = pd.DataFrame(rows)
    hi = df["ms_median"].iloc[-1]
    df["ratio_to_last"] = df["ms_median"] / hi
    return df


def timing(args) -> "object":
    import pandas as pd

    import run_e5_flow
    from airis.model import predict as pred
    from airis.model.flow import body_from_row
    from airis.optimize.encoding import PoseEncoder
    from airis.realtime.recommend import STUB_TABLE
    from airis.sim.scenario import load_scenarios

    model = pred.load_model(args.model, kind="pose")
    df = pd.read_parquet(args.dataset or model.meta["dataset"])
    rows_idx = np.random.default_rng(args.seed).choice(len(df), size=min(args.n_bodies, len(df)), replace=False)
    sample = df.iloc[np.sort(rows_idx)].reset_index(drop=True)
    scenarios = load_scenarios()
    body_model = str(model.meta.get("body_model") or "mesh")
    evaluator = pred._rescore_evaluator(body_model, float(args.density))
    nozzle = cli.resolve_nozzles()[0]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    def call(m, body, scenario, stubs, seed):
        return pred.predict(body, scenario, backend="hybrid", n_flow=args.n_flow, n_knn=args.n_knn, seed=seed,
                            path=args.model, knn_path=args.knn, evaluator=evaluator, nozzle=nozzle,
                            extra_candidates=stubs, **run_e5_flow.variant_kwargs(m, args))

    first = sample.iloc[0]                                                  # 산출물 로드·캐시 데우기 (측정 제외)
    for m in methods:
        sc0 = scenarios[first["scenario"]]
        call(m, body_from_row(first), sc0, [PoseEncoder(sc0).clip_pose(e.pose) for e in STUB_TABLE[sc0.name]], 0)

    out = []
    for r, row in sample.iterrows():
        body, scenario = body_from_row(row), scenarios[row["scenario"]]
        stubs = [PoseEncoder(scenario).clip_pose(e.pose) for e in STUB_TABLE[scenario.name]]
        order = methods[r % len(methods):] + methods[:r % len(methods)]    # 순서 효과를 줄이려고 돌린다
        for m in order:
            t0 = time.perf_counter()
            p = call(m, body, scenario, stubs, args.seed + int(row["body_idx"]))
            ms = (time.perf_counter() - t0) * 1000.0
            out.append({"body_idx": int(row["body_idx"]), "scenario": scenario.name, "method": m, "ms": ms,
                        "n_cands": len(p.candidates), "n_dropped": p.n_dropped, "n_rescored": p.n_rescored,
                        "picked": p.source})
        print(f"[timing] {r + 1}/{len(sample)}")
    res = pd.DataFrame(out)
    summary = (res.groupby("method", sort=False)
                  .agg(n=("ms", "size"), mean_s=("ms", lambda s: s.mean() / 1000),
                       p95_s=("ms", lambda s: np.percentile(s, 95) / 1000),
                       n_cands=("n_cands", "mean"), n_dropped=("n_dropped", "mean"),
                       n_rescored=("n_rescored", "mean"))
                  .reset_index())
    return res, summary


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)
    out = Path(args.out) if args.out else ROOT / "outputs" / f"{explog.new_exp_id('e5timing')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.bench:
        df = bench(args)
        df.to_csv(out.with_name(out.stem + "_bench.csv"), index=False, encoding="utf-8")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"결과 {out.with_name(out.stem + '_bench.csv')}")
        return 0
    res, summary = timing(args)
    res.to_csv(out, index=False, encoding="utf-8")
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False, encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"최종 밀도 {args.density:g}/m², 체형 {len(res['body_idx'].unique())}개. 결과 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
