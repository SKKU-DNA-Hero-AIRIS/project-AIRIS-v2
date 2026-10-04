"""학습 곡선: 데이터셋 크기(체형 수)에 따른 자세 추천 품질. 소유자: F.

출처: 팀원 제안 `scripts/run_learning_curve_cv.py` (브랜치 feat/privacy-learning-curve, 커밋 3ab035e).
그 스크립트는 실행 중에 run_e5_flow.fold_indices 를 바꿔치기했다. 여기서는 run_e5_flow 의 정식 옵션
(`--fold-rule modulo`)과 공개 함수(`run_split`)를 그대로 부른다.

체형이 가장 많은 데이터셋 하나에서 체형 부분 집합을 뽑아 크기별로 flow 를 학습하고 kNN 표를 만든 뒤 평가한다.
평가 집합을 정하는 설계가 두 가지다 (docs/experiments_model.md 6절).

    holdout (권장)  체형 번호가 가장 큰 --holdout 개는 어느 크기의 학습에도 넣지 않고 공통 평가 집합으로 쓴다.
                    크기 N 의 학습 체형은 나머지에서 앞에서부터 N 개 (작은 크기는 큰 크기의 부분 집합).
                    --repeats R 이면 --repeat-max-size 이하 크기에서 학습 체형을 다르게 뽑아 R 번 반복한다
                    (반복 0 은 앞에서부터 N 개, 나머지는 시드로 뽑는다. 학습 체형 뽑기에 따른 흔들림을 본다).
    cv              크기 N 마다 앞에서부터 N 체형 전부를 K-fold 로 평가한다 (팀원 방식). fold 는 modulo 규칙이라
                    체형을 늘려도 같은 체형이 같은 fold 에 남는다. 크기마다 평가 집합이 달라진다.

결과 (--out-dir):
    rows_N<크기>_r<반복>.csv        행별 결과 (run_e5_flow 와 같은 열 + size, repeat, design)
    learning_curve_summary.csv     크기 × 반복 × 방법 지표 (중앙값, 하위 5%, 0.95 미만, 불가, 응답, 재채점 수)
    manifest.json                  커밋, 데이터셋·현재 도장, 설계, 체형 분할, 학습 장치, 소요 시간, 인자

사용 예:
    python scripts/run_e5_learning_curve.py --dataset data/datasets/<1000체형>.parquet \\
        --design holdout --holdout 200 --sizes 100,300,600,800 --repeats 3 --repeat-max-size 300
    python scripts/run_e5_learning_curve.py --dataset ... --design cv --sizes 100,300,600,1000
    python scripts/run_e5_learning_curve.py --dataset ... --sizes 20,40 --holdout 10 --limit 4   # 빠른 점검

응답 시간 열은 참고용이다 (다른 실행과 CPU 를 나눠 쓴다). 시간 비교는 run_e5_timing.py 로 따로 잰다.
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

DESIGNS = ("holdout", "cv")
METHODS = "knn+stub,flow+stub,hybrid"
SOURCE = "팀원 제안 scripts/run_learning_curve_cv.py (feat/privacy-learning-curve, 3ab035e)"


def build_parser() -> argparse.ArgumentParser:
    import run_e5_flow
    import train_pose_models

    e5 = run_e5_flow.build_parser()
    tpm = train_pose_models.build_parser()
    ap = argparse.ArgumentParser(description="학습 곡선: 체형 수에 따른 자세 추천 품질")
    ap.add_argument("--dataset", required=True, help="체형이 가장 많은 데이터셋 (부분 집합을 여기서 뽑는다)")
    ap.add_argument("--design", choices=DESIGNS, default="holdout")
    ap.add_argument("--sizes", default="100,300,600,800", help="학습 체형 수 (쉼표로)")
    ap.add_argument("--holdout", type=int, default=200, help="holdout 설계의 공통 평가 체형 수")
    ap.add_argument("--repeats", type=int, default=1, help="holdout 설계에서 학습 체형을 다르게 뽑는 반복 수")
    ap.add_argument("--repeat-max-size", type=int, default=300, help="이 크기 이하에서만 반복한다")
    ap.add_argument("--seed", type=int, default=0, help="반복의 체형 뽑기 시드와 예측 시드")
    ap.add_argument("--methods", default=METHODS)
    # 기본값은 기존 스크립트의 것을 그대로 쓴다 (따로 정하지 않는다)
    ap.add_argument("--folds", type=int, default=tpm.get_default("folds"), help="cv 설계의 fold 수")
    ap.add_argument("--fold-seed", type=int, default=e5.get_default("fold_seed"))
    ap.add_argument("--steps", type=int, default=tpm.get_default("steps"))
    ap.add_argument("--device", default=tpm.get_default("device"))
    ap.add_argument("--n-flow", type=int, default=e5.get_default("n_flow"))
    ap.add_argument("--n-knn", type=int, default=e5.get_default("n_knn"))
    ap.add_argument("--limit", type=int, default=None, help="평가 행 수 상한 (빠른 점검용)")
    ap.add_argument("--resume", action="store_true", help="행별 결과가 이미 있는 (크기, 반복)은 건너뛴다")
    ap.add_argument("--out-dir", default=None, help="기본값: outputs/<learning_curve_시각>")
    return ap


def plan_runs(body_ids, sizes, *, design: str = "holdout", holdout: int = 200, repeats: int = 1,
              repeat_max_size: int = 300, seed: int = 0) -> list[dict]:
    """(크기, 반복)마다 학습·평가 체형을 정한다. 같은 인자면 같은 분할.

    반환 항목: {"size", "repeat", "train_ids" (정렬된 배열), "test_ids" (holdout 설계만, cv 는 None)}
    holdout: 평가 = 번호가 가장 큰 holdout 개. 학습 후보(pool) = 나머지. 반복 0 은 pool 의 앞 N 개라
             작은 크기가 큰 크기의 부분 집합이다. 반복 r > 0 은 pool 에서 시드로 N 개를 뽑는다.
    cv:      학습·평가 = 앞 N 체형 전부 (fold 로 나눈다). 반복은 쓰지 않는다.
    """
    ids = np.unique(np.asarray(body_ids))
    sizes = sorted({int(s) for s in sizes})
    if not sizes or sizes[0] <= 0:
        raise ValueError(f"크기는 양수여야 한다: {sizes}")
    if design == "cv":
        if sizes[-1] > len(ids):
            raise ValueError(f"크기 {sizes[-1]} 가 데이터셋 체형 수 {len(ids)} 보다 크다")
        return [{"size": n, "repeat": 0, "train_ids": ids[:n], "test_ids": None} for n in sizes]
    if design != "holdout":
        raise ValueError(f"설계는 {DESIGNS} 중 하나: {design!r}")
    if not 0 < holdout < len(ids):
        raise ValueError(f"holdout 체형 수 {holdout} 는 1 이상, 데이터셋 체형 수 {len(ids)} 미만이어야 한다")
    pool, test = ids[:-holdout], ids[-holdout:]
    if sizes[-1] > len(pool):
        raise ValueError(f"크기 {sizes[-1]} 가 학습에 쓸 수 있는 체형 수 {len(pool)} "
                         f"(전체 {len(ids)} − holdout {holdout}) 보다 크다")
    runs = []
    for n in sizes:
        runs.append({"size": n, "repeat": 0, "train_ids": pool[:n], "test_ids": test})
        if n <= repeat_max_size and n < len(pool):
            for r in range(1, max(1, int(repeats))):
                rng = np.random.default_rng([int(seed), n, r])
                runs.append({"size": n, "repeat": r, "test_ids": test,
                             "train_ids": np.sort(rng.choice(pool, size=n, replace=False))})
    return runs


def train_artifacts(train_df, tag: str, work: Path, args) -> tuple[Path, Path, float]:
    """학습 체형으로 flow 와 kNN 표를 만든다. (flow 경로, kNN 경로, 학습 시간 s)."""
    import train_pose_flow
    from airis.model.knn import PoseKNN

    data = work / f"{tag}_train.parquet"
    flow_path, knn_path = work / f"{tag}_flow.pt", work / f"{tag}_knn.parquet"
    train_df.to_parquet(data)
    t0 = time.perf_counter()
    rc = train_pose_flow.main(["--dataset", str(data), "--holdout-frac", "0", "--steps", str(args.steps),
                               "--device", args.device, "--out", str(flow_path)])
    if rc:
        raise RuntimeError(f"flow 학습 실패 (코드 {rc}): {tag}")
    PoseKNN.from_dataset(train_df).save(knn_path)
    return flow_path, knn_path, time.perf_counter() - t0


def evaluate_holdout(train_df, test_df, flow_path: Path, knn_path: Path, methods: list[str], args):
    """공통 평가 체형에서 방법별 점수 비율. run_e5_flow.run_split 을 그대로 쓴다."""
    import pandas as pd

    import run_e5_flow
    from airis.model import predict as pred
    from airis.sim.scenario import load_scenarios

    model = pred.load_model(flow_path, kind="pose")
    evaluator, nozzle = pred.default_rescorer(model)
    e5_args = run_e5_flow.build_parser().parse_args(
        ["--model", str(flow_path), "--n-flow", str(args.n_flow), "--n-knn", str(args.n_knn),
         "--seed", str(args.seed)])
    rows = run_e5_flow.run_split(test_df, train_df, model, methods, e5_args, evaluator, nozzle, load_scenarios(),
                                 knn_path=knn_path, model_path=flow_path)
    return pd.DataFrame(rows)


def evaluate_cv(tag: str, flow_path: Path, work: Path, methods: list[str], args):
    """크기 N 의 전 체형을 K-fold 로 평가 (modulo 규칙). run_e5_flow.main 을 그대로 부른다."""
    import pandas as pd

    import run_e5_flow

    out = work / f"{tag}_cv.csv"
    argv = ["--model", str(flow_path), "--dataset", str(work / f"{tag}_train.parquet"),
            "--folds", str(args.folds), "--fold-rule", "modulo", "--fold-seed", str(args.fold_seed),
            "--methods", ",".join(methods), "--n-flow", str(args.n_flow), "--n-knn", str(args.n_knn),
            "--seed", str(args.seed), "--out", str(out)]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    rc = run_e5_flow.main(argv)
    if rc:
        raise RuntimeError(f"5-fold 실패 (코드 {rc}): {tag}")
    return pd.read_csv(out)


def summarize(rows, run: dict, design: str, extra: dict) -> "object":
    """한 (크기, 반복)의 방법별 지표. 지표 정의는 train_pose_models.method_stats 와 같다."""
    import train_pose_models

    stats = train_pose_models.method_stats(rows)
    stats.insert(0, "repeat", run["repeat"])
    stats.insert(0, "size", run["size"])
    stats.insert(0, "design", design)
    for k, v in extra.items():
        stats[k] = v
    return stats


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)
    start_commit = explog.git_commit()      # 실행 코드의 커밋은 시작할 때 잡는다 (도중에 HEAD 가 바뀔 수 있다)

    import pandas as pd

    import run_e5_flow
    import train_pose_models
    from airis.model import predict as pred

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in run_e5_flow.METHODS]
    if bad:
        print(f"알 수 없는 방법 {bad} (가능: {', '.join(run_e5_flow.METHODS)})", file=sys.stderr)
        return 2
    dataset = Path(args.dataset).resolve()
    df = pd.read_parquet(dataset)
    try:
        stamp, warns = train_pose_models.check_dataset(df)
        runs = plan_runs(df["body_idx"].to_numpy(), [s for s in args.sizes.split(",") if s.strip()],
                         design=args.design, holdout=args.holdout, repeats=args.repeats,
                         repeat_max_size=args.repeat_max_size, seed=args.seed)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    for w in warns:
        print(f"[경고] {w}")

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / explog.new_exp_id("learning_curve")
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    summaries, devices, done = [], set(), []
    for i, run in enumerate(runs, start=1):
        tag = f"N{run['size']:04d}_r{run['repeat']}"
        rows_csv = out_dir / f"rows_{tag}.csv"
        train_df = df[df["body_idx"].isin(run["train_ids"])].reset_index(drop=True)
        print(f"[learning-curve] {i}/{len(runs)} {args.design} 크기 {run['size']} 반복 {run['repeat']} "
              f"(학습 {len(train_df)}행)")
        extra = {"n_train_bodies": int(len(run["train_ids"])), "n_train_rows": int(len(train_df))}
        if args.resume and rows_csv.exists():
            rows = pd.read_csv(rows_csv)
            print(f"[learning-curve] {rows_csv.name} 가 있어 건너뛴다 (--resume)")
        else:
            flow_path, knn_path, train_s = train_artifacts(train_df, tag, work, args)
            devices.add(str(train_pose_models.trained_device(flow_path)))
            t0 = time.perf_counter()
            if args.design == "holdout":
                test_df = df[df["body_idx"].isin(run["test_ids"])].reset_index(drop=True)
                if args.limit:
                    test_df = test_df.head(args.limit)
                rows = evaluate_holdout(train_df, test_df, flow_path, knn_path, methods, args)
            else:
                rows = evaluate_cv(tag, flow_path, work, methods, args)
            rows.insert(0, "repeat", run["repeat"])
            rows.insert(0, "size", run["size"])
            rows.insert(0, "design", args.design)
            rows.to_csv(rows_csv, index=False, encoding="utf-8")
            extra.update(train_s=round(train_s, 1), eval_s=round(time.perf_counter() - t0, 1))
        summaries.append(summarize(rows, run, args.design, extra))
        done.append({"size": run["size"], "repeat": run["repeat"], "rows": rows_csv.name,
                     "train_body_idx_min": int(np.min(run["train_ids"])),
                     "train_body_idx_max": int(np.max(run["train_ids"]))})
        # 도중에 끊겨도 그때까지의 요약이 남게 매번 쓴다
        pd.concat(summaries, ignore_index=True).to_csv(out_dir / "learning_curve_summary.csv", index=False,
                                                        encoding="utf-8")

    summary = pd.concat(summaries, ignore_index=True)
    test_ids = runs[0]["test_ids"]
    manifest = {
        "source": SOURCE, "commit": start_commit, "dataset": str(dataset), "dataset_rows": int(len(df)),
        "dataset_bodies": int(df["body_idx"].nunique()), "dataset_stamp": stamp,
        "current_stamp": pred.current_stamp(), "warnings": warns,
        "design": args.design, "fold_rule": "modulo" if args.design == "cv" else None,
        "sizes": sorted({r["size"] for r in runs}), "methods": methods,
        "holdout_bodies": (None if test_ids is None else
                           {"n": int(len(test_ids)), "body_idx_min": int(test_ids.min()),
                            "body_idx_max": int(test_ids.max())}),
        "runs": done, "devices": sorted(devices), "limit": args.limit,
        "total_s": round(time.perf_counter() - t_all, 1), "args": vars(args),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
                                           encoding="utf-8")

    cols = ["size", "repeat", "method", "n", "p05", "below_095", "infeasible", "median", "mean_s"]
    print()
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.limit:
        print(f"(--limit {args.limit} 빠른 점검: 수치를 결론에 쓰지 않는다)")
    print(f"결과 {out_dir}  (총 {manifest['total_s'] / 60:.1f}분)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
