"""자세 추천 산출물 갱신: 데이터셋 → flow 학습 → kNN 표 → 체형 5-fold → 합격 판정 → 설치. 소유자: F.

물리 기준(configs/physics.yaml 의 계수 k 등)이 바뀌면 산출물은 무효다. 트랙 C 가 데이터셋을 다시 만든 뒤
이 명령 하나로 두 산출물을 다시 만들고 확인한다. 절차는 docs/experiments_model.md "산출물 갱신 절차".

    1. 데이터셋 도장 확인  설정 도장(STAMP_COLS)이 한 값인지, 지금 설정 해시와 같은지 (다르면 경고만)
    2. flow 학습          scripts/train_pose_flow.py --holdout-frac 0 (전 체형)
    3. kNN 표             scripts/build_pose_knn.py
    4. 체형 5-fold         scripts/run_e5_flow.py --folds 5 (fold 마다 flow 재학습·kNN 표 재생성)
    5. 합격 판정 (혼합)    하위 5% ≥ 0.99, 0.95 미만 ≤ 2%, 불가 0, 평균 응답 ≤ 1.5 s
    6. 설치               합격이면 산출물 폴더(data/models 또는 AIRIS_MODEL_DIR)에 복사. 기존 파일은 *.prev 로 둔다

산출물은 먼저 --out-dir 에 만들고 합격했을 때만 설치하므로, 불합격 모델이 쓰던 모델을 덮지 않는다.
결과: <out-dir>/e5cv.csv(행별), e5cv_summary.csv, gate.json, gate.md(docs/experiments_model.md 에 붙일 표).

사용 예:
    python scripts/train_pose_models.py --dataset data/datasets/pose_dataset_mesh_cand.parquet
    python scripts/train_pose_models.py --dataset ... --no-install --tag k1-repro        # 재현만
    python scripts/train_pose_models.py --dataset ... --skip-cv --install                 # 급할 때 (판정 없이 설치)
    python scripts/train_pose_models.py --dataset ... --folds 5 --limit 4 --no-install    # 빠른 점검

5-fold 는 CPU 로 약 25분이다 (메시 300행, 1,500/m²). 다른 세션과 CPU 를 나눠 쓰면 먼저 알린다.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np                                           # noqa: E402

from airis.optimize import cli, explog                       # noqa: E402

METHODS = "hybrid,knn+stub,flow+stub,stub"
GATE_METHOD = "hybrid"
FLOW_FILE, KNN_FILE = "pose_flow.pt", "pose_knn.parquet"


@dataclass(frozen=True)
class Gate:
    """합격 기준 (docs/tracks/F_model.md 단계 1). 전체 행(시나리오 합산) 기준."""
    min_p05: float = 0.99
    max_below_095: float = 0.02
    max_infeasible: float = 0.0
    max_mean_s: float = 1.5


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="자세 추천 산출물 갱신 (flow + kNN + 5-fold 판정)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model-dir", default=None, help="설치 폴더. 기본값: AIRIS_MODEL_DIR 또는 data/models")
    ap.add_argument("--out-dir", default=None, help="기본값: outputs/<refresh_시각[_tag]>")
    ap.add_argument("--tag", default="")
    ap.add_argument("--steps", type=int, default=4000, help="flow 학습 스텝")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--methods", default=METHODS, help=f"5-fold 비교 방법 (판정은 {GATE_METHOD})")
    ap.add_argument("--n-flow", type=int, default=8)
    ap.add_argument("--n-knn", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="fold 마다 평가 행 상한 (빠른 점검용, 판정 무효)")
    ap.add_argument("--skip-cv", action="store_true", help="5-fold 와 판정을 건너뛴다 (--install 필요)")
    inst = ap.add_mutually_exclusive_group()
    inst.add_argument("--install", action="store_true", help="판정과 무관하게 설치")
    inst.add_argument("--no-install", action="store_true", help="판정과 무관하게 설치하지 않는다")
    return ap


# ---------- 1. 데이터셋 도장 ----------

def current_stamp() -> dict[str, str]:
    """predict 가 산출물을 로드할 때 쓰는 것과 같은 판정 기준."""
    from airis.model import predict

    return predict.current_stamp()


def check_dataset(df) -> tuple[dict, list[str]]:
    """(데이터셋 도장, 경고 목록). 도장 열이 여러 값이면 ValueError."""
    from airis.model.knn import STAMP_COLS

    stamp: dict = {}
    for col in STAMP_COLS:
        if col in df.columns:
            vals = sorted(set(df[col].astype(str)))
            if len(vals) != 1:
                raise ValueError(f"데이터셋의 {col} 가 여러 값이다 {vals}. 한 물리 기준의 행만 쓴다.")
            stamp[col] = vals[0]
    warns = []
    now = current_stamp()
    for k, v in now.items():
        if k in stamp and stamp[k] != v:
            warns.append(f"{k}: 데이터셋 {stamp[k]} ≠ 지금 설정 {v}. 해시는 파일 바이트 전체라 물리 값이 같아도 "
                         "키 추가·줄바꿈으로 바뀐다. 물리 값이 바뀐 것이면 데이터셋부터 다시 만든다(C).")
    return stamp, warns


# ---------- 5. 합격 판정 ----------

def method_stats(res) -> "object":
    """방법별 전체 지표 (시나리오 합산). 응답은 초."""
    import pandas as pd

    rows = []
    for m, g in res.groupby("method", sort=False):
        r = g["ratio"].to_numpy(dtype=np.float64)
        s = g["ms"].to_numpy(dtype=np.float64) / 1000.0
        edge = g[g["boundary"]]["ratio"].to_numpy(dtype=np.float64) if "boundary" in g else np.array([])
        rows.append({
            "method": m, "n": len(g), "n_evals": float(g["n_evals"].mean()),
            "median": float(np.nanmedian(r)), "p05": float(np.nanpercentile(r, 5)), "min": float(np.nanmin(r)),
            "below_095": float(np.mean(r < 0.95)), "infeasible": float(g["infeasible"].mean()),
            "mean_s": float(s.mean()), "p95_s": float(np.percentile(s, 95)),
            "boundary_n": int(len(edge)),
            "boundary_p05": float(np.nanpercentile(edge, 5)) if len(edge) else float("nan"),
        })
    return pd.DataFrame(rows)


def judge(stats, gate: Gate = Gate(), method: str = GATE_METHOD) -> dict:
    """판정 결과 {passed, checks: {이름: (값, 기준, 통과)}}. 해당 방법 행이 없으면 passed=False."""
    row = stats[stats["method"] == method]
    if row.empty:
        return {"method": method, "passed": False, "checks": {}, "reason": f"{method} 결과 없음"}
    r = row.iloc[0]
    checks = {
        "p05": (float(r["p05"]), f">= {gate.min_p05}", bool(r["p05"] >= gate.min_p05)),
        "below_095": (float(r["below_095"]), f"<= {gate.max_below_095}", bool(r["below_095"] <= gate.max_below_095)),
        "infeasible": (float(r["infeasible"]), f"<= {gate.max_infeasible}",
                       bool(r["infeasible"] <= gate.max_infeasible)),
        "mean_s": (float(r["mean_s"]), f"<= {gate.max_mean_s}", bool(r["mean_s"] <= gate.max_mean_s)),
    }
    return {"method": method, "passed": all(c[2] for c in checks.values()), "checks": checks}


NAMES = {"hybrid": "혼합 (flow 8 + kNN 8 + 고정 2)", "knn+stub": "kNN + 고정 후보", "flow+stub": "flow + 고정 후보",
         "stub": "고정 후보표(E 스텁)", "knn": "kNN", "flow": "flow"}


def markdown_table(stats, verdict: dict, info: dict) -> str:
    """docs/experiments_model.md 에 붙일 표."""
    lines = [f"데이터 `{info['dataset']}` ({info['rows']}행, 도장 {info['stamp_text']}), 커밋 `{info['commit']}`, "
             f"{info['folds']}-fold, 결과 `{info['out_dir']}`",
             "",
             "| 방법 | 재채점 | 중앙값 | 하위 5% | 최솟값 | 0.95 미만 | 불가 | 평균 응답 | 95% 응답 | 경계 하위 5% |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in stats.iterrows():
        name = NAMES.get(r["method"], r["method"])
        if r["method"] == verdict.get("method"):
            name = f"**{name}**"
        lines.append(f"| {name} | {r['n_evals']:.0f} | {r['median']:.4f} | {r['p05']:.4f} | {r['min']:.4f} | "
                     f"{100 * r['below_095']:.2f}% | {100 * r['infeasible']:.1f}% | {r['mean_s']:.2f} s | "
                     f"{r['p95_s']:.2f} s | {r['boundary_p05']:.4f} |")
    mark = "합격" if verdict["passed"] else "불합격"
    detail = ", ".join(f"{k} {v[0]:.4g} ({v[1]}) {'통과' if v[2] else '미달'}"
                       for k, v in verdict.get("checks", {}).items())
    note = f" ({verdict['reason']})" if detail and verdict.get("reason") else ""
    lines += ["", f"판정({verdict['method']}): **{mark}**{note} — {detail or verdict.get('reason', '')}"]
    return "\n".join(lines) + "\n"


# ---------- 6. 설치 ----------

def install(src_dir: Path, model_dir: Path, names=(FLOW_FILE, KNN_FILE)) -> list[Path]:
    """src_dir 의 산출물을 model_dir 로 설치. 기존 산출물은 <이름>.prev 로 남긴다 (1세대, 쌍으로).

    flow 와 kNN 표가 섞인 상태(새 flow + 옛 kNN)가 남지 않게 세 단계로 한다.

    1. 준비: 새 파일을 <이름>.new, 기존 파일을 <이름>.prev.new 로 복사한다. 여기서 실패하면 아무것도 바뀌지 않는다.
    2. 교체: os.replace 로 하나씩 바꾼다. 도중에 실패하면(Windows 에서 대시보드가 파일을 열고 있을 때 등)
       이미 바꾼 파일을 준비해 둔 옛 파일로 되돌리고 예외를 올린다.
    3. 확정: <이름>.prev.new → <이름>.prev. 모델 교체는 끝났으므로 여기서 실패하면 세대가 어긋나지 않게
       .prev 를 모두 지우고 경고만 한다.

    어느 경우든 끝나면 임시 파일(.new, .prev.new)은 남지 않는다. 설치 중에는 산출물을 여는 프로세스(E 대시보드)를 내린다.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    dsts = [model_dir / name for name in names]
    news = [d.with_name(d.name + ".new") for d in dsts]
    olds = [d.with_name(d.name + ".prev.new") if d.exists() else None for d in dsts]
    replaced: list[int] = []
    try:
        for name, new in zip(names, news):                           # 1. 준비
            shutil.copy2(src_dir / name, new)
        for dst, old in zip(dsts, olds):
            if old is not None:
                shutil.copy2(dst, old)
        try:                                                         # 2. 교체
            for i, (new, dst) in enumerate(zip(news, dsts)):
                os.replace(new, dst)
                replaced.append(i)
        except BaseException:
            for i in reversed(replaced):                             # 되돌리기
                if olds[i] is not None:
                    shutil.copy2(olds[i], dsts[i])
                else:
                    dsts[i].unlink(missing_ok=True)
            raise
        prevs = [dst.with_name(dst.name + ".prev") for dst in dsts]  # 3. 확정
        try:
            for old, prev in zip(olds, prevs):
                if old is not None:
                    os.replace(old, prev)
        except OSError as exc:
            # 세대가 어긋난 .prev 쌍을 남기지 않는다 (없는 편이 섞인 것보다 낫다)
            for prev in prevs:
                prev.unlink(missing_ok=True)
            print(f"[경고] .prev 갱신 실패 ({exc}). 모델은 새것으로 설치됐고 .prev 는 지웠다.", file=sys.stderr)
    finally:
        for tmp in news + [o for o in olds if o is not None]:
            tmp.unlink(missing_ok=True)
    return dsts


def default_model_dir() -> Path:
    from airis.model import predict

    return predict.MODEL_DIR


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)
    if args.skip_cv and not (args.install or args.no_install):
        print("--skip-cv 는 판정이 없으므로 --install 또는 --no-install 을 함께 준다", file=sys.stderr)
        return 2

    import pandas as pd

    import build_pose_knn
    import run_e5_flow
    import train_pose_flow

    dataset = Path(args.dataset).resolve()
    df = pd.read_parquet(dataset)
    try:
        stamp, warns = check_dataset(df)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    for w in warns:
        print(f"[경고] {w}")

    exp = explog.new_exp_id("refresh" + (f"_{args.tag}" if args.tag else ""))
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir) if args.model_dir else default_model_dir()
    t_all = time.perf_counter()
    timings: dict[str, float] = {}

    print(f"[refresh] 1/4 flow 학습 ({args.steps} 스텝)")
    t0 = time.perf_counter()
    rc = train_pose_flow.main(["--dataset", str(dataset), "--holdout-frac", "0", "--steps", str(args.steps),
                               "--out", str(out_dir / FLOW_FILE)])
    timings["train_s"] = time.perf_counter() - t0
    if rc:
        return rc

    print("[refresh] 2/4 kNN 표")
    rc = build_pose_knn.main(["--dataset", str(dataset), "--out", str(out_dir / KNN_FILE)])
    if rc:
        return rc

    verdict: dict = {"method": GATE_METHOD, "passed": False, "checks": {}, "reason": "5-fold 건너뜀"}
    if not args.skip_cv:
        print(f"[refresh] 3/4 체형 {args.folds}-fold ({args.methods})")
        t0 = time.perf_counter()
        cv_args = ["--model", str(out_dir / FLOW_FILE), "--dataset", str(dataset), "--folds", str(args.folds),
                   "--methods", args.methods, "--n-flow", str(args.n_flow), "--n-knn", str(args.n_knn),
                   "--out", str(out_dir / "e5cv.csv")]
        if args.limit:
            cv_args += ["--limit", str(args.limit)]
        rc = run_e5_flow.main(cv_args)
        timings["cv_s"] = time.perf_counter() - t0
        if rc:
            return rc
        stats = method_stats(pd.read_csv(out_dir / "e5cv.csv"))
        verdict = judge(stats)
        if args.limit:
            verdict = {**verdict, "passed": False, "reason": f"--limit {args.limit} 빠른 점검 (판정 무효)"}
        stats.to_csv(out_dir / "e5cv_overall.csv", index=False, encoding="utf-8")
        info = {"dataset": dataset.name, "rows": len(df), "commit": explog.git_commit(), "folds": args.folds,
                "out_dir": out_dir.relative_to(ROOT).as_posix() if out_dir.is_relative_to(ROOT) else str(out_dir),
                "stamp_text": ", ".join(f"{k}={v}" for k, v in stamp.items() if k != "commit")}
        (out_dir / "gate.md").write_text(markdown_table(stats, verdict, info), encoding="utf-8")

    do_install = args.install or (verdict["passed"] and not args.no_install)
    print("[refresh] 4/4 설치" + ("" if do_install else " 안 함"))
    installed = [str(p) for p in install(out_dir, model_dir)] if do_install else []

    timings["total_s"] = time.perf_counter() - t_all
    report = {"dataset": str(dataset), "dataset_stamp": stamp, "current_stamp": current_stamp(),
              "warnings": warns, "gate": asdict(Gate()), "verdict": verdict, "installed": installed,
              "model_dir": str(model_dir), "timings": {k: round(v, 1) for k, v in timings.items()},
              "args": vars(args)}
    (out_dir / "gate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                                       encoding="utf-8")

    print()
    if (out_dir / "gate.md").exists():
        print((out_dir / "gate.md").read_text(encoding="utf-8"))
    print(f"설치: {', '.join(installed) if installed else '안 함'}")
    print(f"결과 {out_dir}  (총 {timings['total_s'] / 60:.1f}분)")
    # 판정을 돌렸는데 불합격이면 1 (자동화에서 알아채게). 건너뛰었거나 빠른 점검이면 0.
    return 0 if args.skip_cv or args.limit or verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
