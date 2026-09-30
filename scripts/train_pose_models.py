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
import filecmp
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
    from airis.model import predict

    now = current_stamp()
    # 자세 산출물이므로 설정 해시(STAMP_KEYS)만 본다. kinetics 는 계획 평가에만 쓰여 자세 데이터셋과 무관하다.
    diff = [k for k in predict.stamp_mismatch(stamp, now) if k in predict.STAMP_KEYS]
    warns = [f"{k}: 데이터셋 {stamp[k]} ≠ 지금 설정 {now[k]}. 해시는 설정 YAML 내용(정렬 직렬화) 기준이라 "
             "물리·노즐 설정 값이 바뀐 것이다. 데이터셋부터 다시 만든다(C). "
             "(#98 이전 옛 규약 도장이면 값이 같아도 다르게 나온다.)" for k in diff]
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
            # 혼합 계열의 중복 제거 뒤 후보 수 (다른 방법은 NaN)
            "n_cands": (float(g["n_cands"].mean()) if "n_cands" in g and g["n_cands"].notna().any()
                        else float("nan")),
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
         "stub": "고정 후보표(E 스텁)", "knn": "kNN", "flow": "flow",
         "hybrid-a": "혼합 A (중복 제거)", "hybrid-b": "혼합 B (중복 제거 + 선별 b)",
         "hybrid-c": "혼합 C (중복 제거 + 선별 c)", "hybrid-d": "혼합 D (중복 제거 + 스레드)"}


def markdown_table(stats, verdict: dict, info: dict) -> str:
    """docs/experiments_model.md 에 붙일 표."""
    lines = [f"데이터 `{info['dataset']}` ({info['rows']}행, 도장 {info['stamp_text']}), 커밋 `{info['commit']}`, "
             f"{info['folds']}-fold, 결과 `{info['out_dir']}`",
             "",
             "| 방법 | 후보 | 재채점 | 중앙값 | 하위 5% | 최솟값 | 0.95 미만 | 불가 | 평균 응답 | 95% 응답 | 경계 하위 5% |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in stats.iterrows():
        name = NAMES.get(r["method"], r["method"])
        if r["method"] == verdict.get("method"):
            name = f"**{name}**"
        cands = "" if np.isnan(r.get("n_cands", float("nan"))) else f"{r['n_cands']:.1f}"
        lines.append(f"| {name} | {cands} | {r['n_evals']:.1f} | {r['median']:.4f} | {r['p05']:.4f} | {r['min']:.4f} | "
                     f"{100 * r['below_095']:.2f}% | {100 * r['infeasible']:.1f}% | {r['mean_s']:.2f} s | "
                     f"{r['p95_s']:.2f} s | {r['boundary_p05']:.4f} |")
    mark = "합격" if verdict["passed"] else "불합격"
    detail = ", ".join(f"{k} {v[0]:.4g} ({v[1]}) {'통과' if v[2] else '미달'}"
                       for k, v in verdict.get("checks", {}).items())
    note = f" ({verdict['reason']})" if detail and verdict.get("reason") else ""
    lines += ["", f"판정({verdict['method']}): **{mark}**{note} — {detail or verdict.get('reason', '')}"]
    return "\n".join(lines) + "\n"


# ---------- 6. 설치 ----------

class InstallRollbackError(RuntimeError):
    """교체가 실패했고 되돌리기도 실패했다 (이중 실패). 옛 산출물 백업(<이름>.prev.new)이 남아 있다."""


def install(src_dir: Path, model_dir: Path, names=(FLOW_FILE, KNN_FILE)) -> list[Path]:
    """src_dir 의 산출물을 model_dir 로 설치. 기존 산출물은 <이름>.prev 로 남긴다 (1세대, 쌍으로).

    flow 와 kNN 표가 섞인 상태(새 flow + 옛 kNN)가 남지 않게 세 단계로 한다.

    0. 확인: <이름>.prev.new 가 이미 있으면 앞선 설치가 이중 실패로 남긴 옛 산출물 백업이므로, 덮어쓰지 않고
       RuntimeError 로 멈춘다 (손으로 복구한 뒤 다시 돌린다).
    1. 준비: 새 파일을 <이름>.new, 기존 파일을 <이름>.prev.new 로 복사한다. 여기서 실패하면 아무것도 바뀌지 않는다.
    2. 교체: os.replace 로 하나씩 바꾼다. 도중에 실패하면(Windows 에서 대시보드가 파일을 열고 있을 때, Ctrl+C 등)
       바꾼 파일을 백업에서 os.replace 로 되돌리고 원래 예외를 올린다. 되돌리기마저 실패하면(되돌리는 중 Ctrl+C 포함)
       백업을 지우지 않고 남긴 채 InstallRollbackError 를 올린다 (Ctrl+C 였으면 KeyboardInterrupt).
    3. 확정: <이름>.prev.new → <이름>.prev. 옛 파일이 없던 쪽의 묵은 .prev 는 지운다 (.prev 는 늘 같은 세대의 쌍).
       모델 교체는 끝났으므로 여기서 실패하면(잠김, Ctrl+C) .prev 를 모두 지우려 하고 경고한다.
       지우지 못한 .prev 는 경고에 적는다. Ctrl+C 는 정리 뒤 다시 올리고, 그 밖의 실패로는 예외를 내지 않는다.

    단계는 한 try 안에서 phase 로 구분하므로 단계 사이에 온 Ctrl+C 도 어느 한 단계의 처리를 받는다.
    끝나면 임시 파일(.new, .prev.new)은 남지 않는다 (2의 이중 실패가 남긴 백업은 예외).
    보장 범위: 한 번의 실패로는 쌍이 섞이지 않고 옛 산출물을 잃지 않는다. 이중 실패에서는 백업을 남기고 알린다.
    설치 중에는 산출물을 여는 프로세스(E 대시보드)를 내린다.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    dsts = [model_dir / name for name in names]
    news = [d.with_name(d.name + ".new") for d in dsts]
    prevs = [d.with_name(d.name + ".prev") for d in dsts]
    leftover = [d.with_name(d.name + ".prev.new") for d in dsts if d.with_name(d.name + ".prev.new").exists()]
    if leftover:                                                     # 0. 확인
        raise RuntimeError(
            f"앞선 설치가 남긴 옛 산출물 백업이 있다: {', '.join(p.name for p in leftover)}. 덮어쓰지 않고 멈춘다. "
            "각 <이름>.prev.new 를 <이름> 으로 되돌려(또는 지금 파일이 맞으면 지워) 쌍을 맞춘 뒤 다시 돌린다.")
    olds = [d.with_name(d.name + ".prev.new") if d.exists() else None for d in dsts]
    keep: set[Path] = set()
    touched: list[int] = []
    phase = "prepare"
    try:
        for name, new in zip(names, news):                           # 1. 준비
            shutil.copy2(src_dir / name, new)
        for dst, old in zip(dsts, olds):
            if old is not None:
                shutil.copy2(dst, old)
        phase = "swap"
        for i, (new, dst) in enumerate(zip(news, dsts)):             # 2. 교체
            touched.append(i)                    # replace 전에 적는다 (직후 Ctrl+C 에도 되돌린다)
            os.replace(new, dst)
        phase = "commit"
        for old, prev in zip(olds, prevs):                           # 3. 확정
            if old is not None:
                os.replace(old, prev)
            else:
                prev.unlink(missing_ok=True)
    except BaseException as exc:
        if phase == "swap":
            failed: list[str] = []
            interrupted = None
            for i in reversed(touched):
                try:
                    if olds[i] is not None:
                        # 교체되지 않은 파일(잠겨서 실패한 그 파일 등)은 이미 옛것이다
                        if not (dsts[i].exists() and filecmp.cmp(olds[i], dsts[i], shallow=False)):
                            os.replace(olds[i], dsts[i])
                    else:
                        dsts[i].unlink(missing_ok=True)
                except BaseException as rb:                  # 되돌리는 중 Ctrl+C 도 백업을 남긴다
                    failed.append(f"{dsts[i].name}: {type(rb).__name__}: {rb}")
                    if not isinstance(rb, Exception):
                        interrupted = rb
            if failed:
                keep.update(o for o in olds if o is not None and o.exists())
                msg = (f"교체 실패({type(exc).__name__}: {exc}) 뒤 되돌리기도 실패했다 ({'; '.join(failed)}). "
                       f"옛 산출물 백업을 남긴다: {', '.join(str(k) for k in sorted(keep))}. "
                       "백업을 <이름> 으로 되돌려 쌍을 맞춘 뒤 다시 설치한다.")
                print(f"[경고] {msg}", file=sys.stderr)
                if interrupted is not None:
                    raise interrupted from exc
                raise InstallRollbackError(msg) from exc
            raise
        if phase == "commit":
            stuck = []
            for prev in prevs:                   # 세대가 어긋난 .prev 쌍을 남기지 않는다
                try:
                    prev.unlink(missing_ok=True)
                except OSError:
                    stuck.append(prev.name)
            note = f" 지우지 못한 묵은 .prev: {', '.join(stuck)} (직접 지울 것)" if stuck else ""
            print(f"[경고] .prev 갱신 실패 ({type(exc).__name__}: {exc}). 모델은 새 쌍으로 설치됐고 .prev 는 지웠다."
                  + note, file=sys.stderr)
            if not isinstance(exc, Exception):
                raise
            return dsts
        raise                                                        # 준비 단계: 바뀐 것 없음
    finally:
        for tmp in news + [o for o in olds if o is not None]:
            if tmp not in keep:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
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
    installed: list[str] = []
    install_error = None
    interrupted: BaseException | None = None
    if do_install:
        try:
            installed = [str(p) for p in install(out_dir, model_dir)]
        except BaseException as exc:        # 설치 실패·Ctrl+C 도 gate.json 에 남긴다
            install_error = f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, Exception):
                interrupted = exc
    # 실패했을 때 설치 폴더의 실제 상태 (파일마다 이번 산출물과 같은가). 섞였는지 바로 보이게.
    install_state = ({name: ("new" if (model_dir / name).exists()
                                       and filecmp.cmp(out_dir / name, model_dir / name, shallow=False)
                             else "old" if (model_dir / name).exists() else "missing")
                      for name in (FLOW_FILE, KNN_FILE)}
                     if install_error else None)

    timings["total_s"] = time.perf_counter() - t_all
    report = {"dataset": str(dataset), "dataset_stamp": stamp, "current_stamp": current_stamp(),
              "warnings": warns, "gate": asdict(Gate()), "verdict": verdict,
              "installed": installed, "install_error": install_error, "install_state": install_state,
              "model_dir": str(model_dir), "timings": {k: round(v, 1) for k, v in timings.items()},
              "args": vars(args)}
    (out_dir / "gate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                                       encoding="utf-8")
    if interrupted is not None:
        raise interrupted

    print()
    if (out_dir / "gate.md").exists():
        print((out_dir / "gate.md").read_text(encoding="utf-8"))
    state = (", ".join(f"{k}={v}" for k, v in install_state.items()) if install_state else "")
    print(f"설치: {', '.join(installed) if installed else '안 함'}"
          + (f"  — 설치 실패 {install_error}. 설치 폴더 상태: {state}" if install_error else ""))
    print(f"결과 {out_dir}  (총 {timings['total_s'] / 60:.1f}분)")
    # 판정을 돌렸는데 불합격이거나 설치가 실패하면 1 (자동화에서 알아채게). 건너뛰었거나 빠른 점검이면 0.
    if install_error:
        return 1
    return 0 if args.skip_cv or args.limit or verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
