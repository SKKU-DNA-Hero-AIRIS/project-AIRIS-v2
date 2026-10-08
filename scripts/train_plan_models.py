"""계획 추천 산출물 만들기: 계획 데이터셋 → flow 학습 → kNN 표 → 체형 5-fold → 합격 판정 → 설치. 소유자: F.

자세 추천의 scripts/train_pose_models.py 와 같은 절차다 (docs/experiments_model.md 3·4절).
단계 수 N 과 계획 한도는 데이터셋에서 읽는다 (C 의 설계 결정값, 코드에 고정하지 않는다).

    1. 데이터셋 도장 확인  설정 도장·한도가 한 값인지, 지금 설정(물리·노즐·계획 채점)과 같은지 (다르면 경고)
    2. flow 학습          전 체형, 출력 공간은 데이터셋의 단계 수·한도로 (plan_data.plan_space_from_dataset)
    3. kNN 표             knn.PlanKNN
    4. 체형 5-fold         scripts/run_e5_plan.py (fold 마다 flow 재학습·kNN 표 재생성)
    5. 합격 판정 (혼합)    하위 5% ≥ 0.97, 불가 0, 평균 응답 ≤ 1.5 s (완성도 F2·F3)
    6. 설치               합격이면 산출물 폴더에 plan_flow.pt · plan_knn.parquet. 기존 파일은 *.prev 로 둔다

산출물은 먼저 --out-dir 에 만들고 합격했을 때만 설치한다. 5-fold 와 설치를 나누려면 --no-install 로 돌린 뒤
`--install-from <결과 폴더>` 로 설치만 한다 (설치 전에 산출물을 여는 프로세스를 내린다).
결과: <out-dir>/e5plan.csv(행별), e5plan_overall.csv, gate.json, gate.md.

사용 예:
    python scripts/train_plan_models.py --dataset data/datasets/<계획 데이터셋>.parquet --no-install
    python scripts/train_plan_models.py --install-from outputs/<결과 폴더> --model-dir <설치 폴더>
    python scripts/train_plan_models.py --dataset … --folds 2 --limit 2 --steps 300 --no-install    # 빠른 점검
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from airis.optimize import cli, explog                       # noqa: E402

FLOW_FILE, KNN_FILE = "plan_flow.pt", "plan_knn.parquet"
NAMES = (FLOW_FILE, KNN_FILE)
GATE_METHOD = "hybrid"
METHODS = "hybrid,hybrid-nofixed,knn+fixed,fixed"


@dataclass(frozen=True)
class PlanGate:
    """합격 기준 (완성도 F2·F3). 전체 행(시나리오 합산) 기준."""
    min_p05: float = 0.97
    max_infeasible: float = 0.0
    max_mean_s: float = 1.5


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="계획 추천 산출물 만들기 (flow + kNN + 5-fold 판정)")
    ap.add_argument("--dataset", default=None, help="계획 데이터셋 (--install-from 이 아니면 필요)")
    ap.add_argument("--install-from", default=None,
                    help="이미 끝난 실행의 결과 폴더. 학습·5-fold 없이 그 산출물을 설치만 한다 (판정이 합격이어야 한다)")
    ap.add_argument("--model-dir", default=None, help="설치 폴더. 기본값: AIRIS_MODEL_DIR 또는 data/models")
    ap.add_argument("--out-dir", default=None, help="기본값: outputs/<plan_refresh_시각[_tag]>")
    ap.add_argument("--tag", default="")
    ap.add_argument("--steps", type=int, default=4000, help="flow 학습 스텝")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="flow 학습 장치 cpu | cuda | auto")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--methods", default=METHODS, help=f"5-fold 비교 방법 (판정은 {GATE_METHOD})")
    ap.add_argument("--n-flow", type=int, default=8)
    ap.add_argument("--n-knn", type=int, default=8)
    ap.add_argument("--rescore-top", type=int, default=None, help="재채점할 후보 수 (기본: 전부). run_e5_plan.py 와 같다")
    ap.add_argument("--n-threads", type=int, default=1, help="재채점 스레드 수")
    ap.add_argument("--rotation-pose", default="none", help="run_e5_plan.py 의 --rotation-pose (none | dataset | model)")
    ap.add_argument("--pose-model", default=None)
    ap.add_argument("--pose-knn", default=None)
    ap.add_argument("--limit", type=int, default=None, help="fold 마다 평가 행 상한 (빠른 점검용, 판정 무효)")
    ap.add_argument("--no-refilter", action="store_true",
                    help="읽은 직후의 후보 재필터(plan_data.read_plan_dataset)를 끈다")
    ap.add_argument("--skip-cv", action="store_true", help="5-fold 와 판정을 건너뛴다 (--install 또는 --no-install 필요)")
    inst = ap.add_mutually_exclusive_group()
    inst.add_argument("--install", action="store_true", help="판정과 무관하게 설치")
    inst.add_argument("--no-install", action="store_true", help="판정과 무관하게 설치하지 않는다")
    return ap


def judge(stats, gate: PlanGate = PlanGate(), method: str = GATE_METHOD) -> dict:
    """판정 결과 {passed, checks: {이름: (값, 기준, 통과)}}. 해당 방법 행이 없으면 passed=False."""
    row = stats[stats["method"] == method]
    if row.empty:
        return {"method": method, "passed": False, "checks": {}, "reason": f"{method} 결과 없음"}
    r = row.iloc[0]
    checks = {
        "p05": (float(r["p05"]), f">= {gate.min_p05}", bool(r["p05"] >= gate.min_p05)),
        "infeasible": (float(r["infeasible"]), f"<= {gate.max_infeasible}",
                       bool(r["infeasible"] <= gate.max_infeasible)),
        "mean_s": (float(r["mean_s"]), f"<= {gate.max_mean_s}", bool(r["mean_s"] <= gate.max_mean_s)),
    }
    return {"method": method, "passed": all(c[2] for c in checks.values()), "checks": checks}


NAMES_KO = {"hybrid": "혼합 (flow + kNN + 고정 회전 계획)", "hybrid-nofixed": "혼합, 고정 회전 계획 없이",
            "flow+fixed": "flow + 고정 회전 계획", "knn+fixed": "kNN + 고정 회전 계획", "fixed": "고정 회전 계획만"}


def markdown_table(stats, verdict: dict, info: dict) -> str:
    """docs/experiments_model.md 에 붙일 표."""
    lines = [f"데이터 `{info['dataset']}` ({info['rows']}행, 단계 {info['n_phases']}개, 에너지 가중 {info['energy_weight']}), "
             f"커밋 `{info['commit']}`, {info['folds']}-fold, 제안 자세 {info['rotation_pose']}, "
             f"학습 장치 {info.get('device') or '기록 없음'}, 결과 `{info['out_dir']}`"
             + (f", {info['refilter']}" if info.get("refilter") else ""),
             "",
             "| 방법 | 채점 수 | 중앙값 | 하위 5% | 최솟값 | 0.95 미만 | 불가 | 평균 응답 | 95% 응답 | 고른 계획의 총 시간 | 회전 계획이 뽑힌 비율 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in stats.iterrows():
        name = NAMES_KO.get(r["method"], r["method"])
        if r["method"] == verdict.get("method"):
            name = f"**{name}**"
        lines.append(f"| {name} | {r['n_evals']:.1f} | {r['median']:.4f} | {r['p05']:.4f} | {r['min']:.4f} | "
                     f"{100 * r['below_095']:.2f}% | {100 * r['infeasible']:.2f}% | {r['mean_s']:.2f} s | "
                     f"{r['p95_s']:.2f} s | {r['plan_duration_s']:.1f} s | {100 * r['fixed_picked']:.0f}% |")
    mark = "합격" if verdict["passed"] else "불합격"
    detail = ", ".join(f"{k} {v[0]:.4g} ({v[1]}) {'통과' if v[2] else '미달'}"
                       for k, v in verdict.get("checks", {}).items())
    note = f" ({verdict['reason']})" if detail and verdict.get("reason") else ""
    lines += ["", f"판정({verdict['method']}): **{mark}**{note} — {detail or verdict.get('reason', '')}"]
    return "\n".join(lines) + "\n"


def default_model_dir() -> Path:
    from airis.model import predict

    return predict.MODEL_DIR


def install_only(run_dir: Path, model_dir: Path, *, force: bool = False) -> int:
    """이미 끝난 실행의 계획 산출물을 설치만 한다. train_pose_models.install_only 와 같은 규칙.

    종료 코드: 0 설치함, 1 불합격이거나 설치 실패, 2 입력이 잘못됨(폴더·파일 없음, 남은 백업).
    """
    import train_pose_models as tpm

    gate_path = run_dir / "gate.json"
    missing = [n for n in NAMES if not (run_dir / n).is_file()] + ([] if gate_path.is_file() else ["gate.json"])
    if missing:
        print(f"{run_dir} 에 설치할 것이 없다 (없는 파일: {', '.join(missing)})", file=sys.stderr)
        return 2
    report = json.loads(gate_path.read_text(encoding="utf-8"))
    verdict = report.get("verdict") or {}
    if not verdict.get("passed") and not force:
        print(f"판정이 합격이 아니라 설치하지 않는다 ({verdict.get('reason') or verdict.get('checks')}). "
              "그래도 설치하려면 --install.", file=sys.stderr)
        return 1
    left = tpm.leftover_backups(model_dir, NAMES)
    if left:
        print(f"설치 폴더에 앞선 설치가 남긴 옛 산출물 백업이 있다: {', '.join(p.name for p in left)}. "
              "각 <이름>.prev.new 를 <이름> 으로 되돌려(또는 지금 파일이 맞으면 지워) 쌍을 맞춘 뒤 다시 돌린다.",
              file=sys.stderr)
        return 2
    try:
        installed, error = [str(p) for p in tpm.install(run_dir, model_dir, NAMES)], None
    except Exception as exc:
        installed, error = [], f"{type(exc).__name__}: {exc}"
    history = list(report.get("installs", []))
    history.append({"model_dir": str(model_dir), "installed": installed, "error": error,
                    "at": time.strftime("%Y-%m-%d %H:%M:%S"), "forced": bool(force and not verdict.get("passed"))})
    report.update(installed=installed, install_error=error, model_dir=str(model_dir), installs=history)
    gate_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if error:
        print(f"설치 실패 {error}", file=sys.stderr)
        return 1
    print(f"설치: {', '.join(installed)}  (판정 {'합격' if verdict.get('passed') else '불합격, 강제 설치'})")
    return 0


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)
    model_dir = Path(args.model_dir) if args.model_dir else default_model_dir()
    if args.install_from:
        return install_only(Path(args.install_from), model_dir, force=args.install)
    if not args.dataset:
        print("--dataset 이 필요하다 (설치만 하려면 --install-from <결과 폴더>)", file=sys.stderr)
        return 2
    if args.skip_cv and not (args.install or args.no_install):
        print("--skip-cv 는 판정이 없으므로 --install 또는 --no-install 을 함께 준다", file=sys.stderr)
        return 2
    start_commit = explog.git_commit()          # 실행 코드의 커밋은 시작할 때 잡는다

    import pandas as pd

    import run_e5_plan
    import train_pose_models as tpm
    from airis.model import flow, plan_data
    from airis.model.knn import PlanKNN
    from airis.sim.scenario import load_scenarios

    dataset = Path(args.dataset).resolve()
    df, refilter = plan_data.read_plan_dataset(dataset, refilter=not args.no_refilter)
    print(f"[plan-refresh] {plan_data.refilter_message(refilter)}")
    try:
        stamp, warns = plan_data.check_plan_dataset(df)
        limits = plan_data.plan_limits_from_dataset(df)
        meta = plan_data.plan_meta_from_dataset(df)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    for w in warns:
        print(f"[경고] {w}")
    if not args.no_install:
        left = tpm.leftover_backups(model_dir, NAMES)
        if left:
            print(f"설치 폴더에 앞선 설치가 남긴 옛 산출물 백업이 있다: {', '.join(p.name for p in left)}. "
                  "쌍을 맞춘 뒤 다시 돌리거나 --no-install 로 돌린다.", file=sys.stderr)
            return 2

    exp = explog.new_exp_id("plan_refresh" + (f"_{args.tag}" if args.tag else ""))
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    timings: dict[str, float] = {}

    scenarios = load_scenarios()
    names = sorted(set(df["scenario"].astype(str)))
    space = plan_data.plan_space_from_dataset(df, [scenarios[n] for n in names])
    print(f"[plan-refresh] 1/4 flow 학습 ({args.steps} 스텝, 단계 {limits.n_phases}개 = {space.dim}차원, "
          f"총 시간 {limits.duration_bounds_s[0]:g}~{limits.duration_bounds_s[1]:g} s)")
    cfg = flow.FlowConfig(hidden=args.hidden, layers=args.layers, steps=args.steps, batch_size=args.batch_size,
                          lr=args.lr, seed=args.train_seed, device=args.device)
    t0 = time.perf_counter()
    model = flow.train_flow(df, scenarios, cfg, space=space, log=print,
                            meta={**meta, "dataset": str(dataset), "commit": start_commit,
                                  "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    timings["train_s"] = time.perf_counter() - t0
    model.meta["train_rows"] = int(len(df))
    model.save(out_dir / FLOW_FILE)
    device = model.meta.get("device")

    print("[plan-refresh] 2/4 kNN 표")
    PlanKNN.from_dataset(df, min_phase_s=limits.min_phase_s).save(out_dir / KNN_FILE)

    verdict: dict = {"method": GATE_METHOD, "passed": False, "checks": {}, "reason": "5-fold 건너뜀"}
    if not args.skip_cv:
        print(f"[plan-refresh] 3/4 체형 {args.folds}-fold ({args.methods})")
        t0 = time.perf_counter()
        cv_args = ["--model", str(out_dir / FLOW_FILE), "--dataset", str(dataset), "--folds", str(args.folds),
                   "--methods", args.methods, "--n-flow", str(args.n_flow), "--n-knn", str(args.n_knn),
                   "--rotation-pose", args.rotation_pose, "--out", str(out_dir / "e5plan.csv")]
        for flag, value in (("--pose-model", args.pose_model), ("--pose-knn", args.pose_knn),
                            ("--limit", args.limit), ("--rescore-top", args.rescore_top),
                            ("--n-threads", args.n_threads if args.n_threads > 1 else None)):
            if value:
                cv_args += [flag, str(value)]
        if args.no_refilter:
            cv_args.append("--no-refilter")
        rc = run_e5_plan.main(cv_args)
        timings["cv_s"] = time.perf_counter() - t0
        if rc:
            return rc
        stats = pd.read_csv(out_dir / "e5plan_overall.csv")
        verdict = judge(stats)
        if args.limit:
            verdict = {**verdict, "passed": False, "reason": f"--limit {args.limit} 빠른 점검 (판정 무효)"}
        info = {"dataset": dataset.name, "rows": len(df), "commit": start_commit, "folds": args.folds,
                "refilter": plan_data.refilter_message(refilter),
                "n_phases": limits.n_phases, "energy_weight": meta.get("energy_weight", "기록 없음"),
                "rotation_pose": args.rotation_pose, "device": device,
                "out_dir": out_dir.relative_to(ROOT).as_posix() if out_dir.is_relative_to(ROOT) else str(out_dir)}
        (out_dir / "gate.md").write_text(markdown_table(stats, verdict, info), encoding="utf-8")

    report = {"commit": start_commit, "dataset": str(dataset), "dataset_stamp": stamp, "warnings": warns,
              "candidate_refilter": refilter,
              "limits": asdict(limits), "space_dim": space.dim, "gate": asdict(PlanGate()), "verdict": verdict,
              "installed": [], "install_error": None, "model_dir": str(model_dir), "device": device,
              "timings": {k: round(v, 1) for k, v in timings.items()}, "args": vars(args)}
    (out_dir / "gate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                                       encoding="utf-8")
    do_install = args.install or (verdict["passed"] and not args.no_install)
    print("[plan-refresh] 4/4 설치" + ("" if do_install else " 안 함"))
    rc_install = install_only(out_dir, model_dir, force=args.install) if do_install else 0

    print()
    if (out_dir / "gate.md").exists():
        print((out_dir / "gate.md").read_text(encoding="utf-8"))
    print(f"결과 {out_dir}  (총 {(time.perf_counter() - t_all) / 60:.1f}분)")
    if rc_install:
        return 1
    # 판정을 돌렸는데 불합격이면 1 (자동화에서 알아채게). 건너뛰었거나 빠른 점검이면 0.
    return 0 if args.skip_cv or args.limit or verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
