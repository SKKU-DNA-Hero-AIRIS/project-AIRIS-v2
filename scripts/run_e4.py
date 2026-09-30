"""E4 러너: 시나리오 × 시드마다 CMA-ES 를 돌리고 기준 자세(B0, B1, B2)와 비교한다. 소유자: C.

사용 예:
    python scripts/run_e4.py                                  # 패치판, 시나리오 3개 × 시드 0~4
    python scripts/run_e4.py --scenarios default --seeds 0 1  # 일부만
    python scripts/run_e4.py --evaluator dummy --max-evals 200 --popsize 20   # 파이프라인 확인용

결과:
    outputs/<exp_id>/                     실행마다 meta.json, history.csv, best.json (tag=e4)
    outputs/<group_id>/e4_summary.csv     시나리오별 best 평균·표준편차, B0·B1·B2 대비 개선율,
                                          시드 간 자세 편차(yaw 는 0~90° 로 접음, e4.fold_yaw)
    outputs/<group_id>/best_poses.json    시드별 best 자세, 시나리오별 평균 자세, 기준선 점수
    outputs/<group_id>/baselines.csv      이번 묶음에서 평가한 B0, B1, B2

재채점 (--rescore-patches-per-m2, 패치판 기본 2000):
    탐색은 400/m² 로 빠르게 하되, 400/m² 패치 격자는 좌우 거울 대칭이 아니라 같은 자세의
    yaw ±θ 점수가 최대 약 3% 다르고 CMA-ES 가 그 격자 잡음을 이용할 수 있다.
    그래서 시드별 best 자세와 B0·B1·B2 를 더 촘촘한 밀도로 다시 평가해 rescore_* 열로 함께 남긴다.
    0 이면 끈다. 패치판이 아니면 무시한다.

요약 파일은 실행 묶음마다 <group_id> 폴더에 따로 남긴다 (다시 돌려도 이전 요약을 덮어쓰지 않는다).
범위를 좁힌 탐색 (--pose-bound):
    configs/scenarios.yaml 을 건드리지 않고 **탐색 상자만** 실행 시점에 좁힌다 (원래 범위와 교집합).
    예: 천장(2.15 m)에 손이 닿지 않는 팔 내림 봉우리를 찾을 때
        --starts default --pose-bound shoulder_abduction=0,90 --pose-bound shoulder_flexion=-30,90
    벌림만 막으면 최적화가 어깨 굽힘(팔을 앞으로 들기)으로 빠져나가니 둘 다 건다.

    **점수는 원래 시나리오로 매긴다.** scoring.discomfort 가 pose_bounds 폭으로 정규화하기 때문에
    (airis/sim/scoring.py) 좁힌 시나리오로 채점하면 그 변수의 불편도가 폭에 반비례해 커져
    점수가 다른 E4 묶음과 비교 불가능해지고 탐색 목적함수까지 달라진다 (통합 2026-09-30 지적).
    그래서 평가기·채점·기준선에는 원래 시나리오를 넘기고, 좁힌 시나리오는 PoseEncoder
    (= 탐색 상자)에만 쓴다. 범위 밖 시작점은 경고와 함께 건너뛴다.
    좁힌 범위는 meta.json·best_poses.json 의 args·pose_bounds_effective, 요약 CSV 의
    pose_bound 열에 남는다.

docs/tracks/C_optimize.md 단계 7.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airis.optimize import baselines, cli, e4, explog         # noqa: E402
from airis.optimize.cmaes_runner import run_cmaes             # noqa: E402
from airis.optimize.encoding import PoseEncoder                # noqa: E402
from airis.sim import PART_NAMES, BodyParams, PoseParams      # noqa: E402
from airis.sim.scenario import load_scenarios                 # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E4: 시나리오 × 시드 CMA-ES 와 기준 자세 비교")
    ap.add_argument("--evaluator", choices=cli.EVALUATOR_CHOICES, default="patch")
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="시나리오. 생략 시 configs/scenarios.yaml 전부")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--max-evals", type=int, default=3000)
    ap.add_argument("--popsize", type=int, default=100)
    ap.add_argument("--sigma0", type=float, default=0.5, help="시작점에 sigma0 가 없을 때의 초기 스텝")
    ap.add_argument("--starts", default=cli.DEFAULT_STARTS,
                    help=f"CMA-ES 시작점, 쉼표 구분 (가능: {', '.join(cli.START_PRESETS)})")
    ap.add_argument("--tol-stagnation-gens", type=int, default=30)
    ap.add_argument("--patches-per-m2", type=float, default=cli.DEFAULT_PATCHES_PER_M2,
                    help="패치판 표면 패치 밀도 (--evaluator patch 에만 적용)")
    ap.add_argument("--rescore-patches-per-m2", type=float, default=2000.0,
                    help="best 자세·기준선 재채점 밀도 (패치판만, 0 이면 끔)")
    ap.add_argument("--pose-bound", action="append", default=[], metavar="변수=lo,hi",
                    help="pose_bounds 를 실행 시점에 좁힌다 (원래 범위와 교집합). 여러 번 쓸 수 있다. "
                         "예: --pose-bound shoulder_abduction=0,90")
    ap.add_argument("--body", default=None, help="BodyParams 덮어쓰기 JSON")
    ap.add_argument("--tag", default="e4", help="exp_id·group_id 접두어")
    ap.add_argument("--log-dir", default=str(ROOT / "outputs"))
    ap.add_argument("--dummy-target", default=None, help="--evaluator dummy 의 목표 자세 JSON")
    return ap


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)

    all_scenarios = load_scenarios()
    names = args.scenarios or list(all_scenarios)
    missing = [n for n in names if n not in all_scenarios]
    if missing:
        print(f"없는 시나리오: {missing} (가능: {', '.join(all_scenarios)})", file=sys.stderr)
        return 2
    if not args.seeds:
        print("--seeds 가 비어 있다", file=sys.stderr)
        return 2

    # 좁힌 범위와 시작점은 루프 전에 전부 검증한다 (뒤쪽 시나리오에서 죽으면 앞선 실행이
    # 요약 없이 버려진다). 시작점이 범위 밖이면 여기서 경고하고 건너뛴다.
    narrowed = {n: cli.narrow_scenario(all_scenarios[n], args.pose_bound) for n in names}

    body = cli.dataclass_from_json(BodyParams, args.body)
    dummy_target = cli.dataclass_from_json(PoseParams, args.dummy_target) if args.dummy_target else None
    starts = cli.parse_starts(args.starts)
    kept_starts = {n: (cli.starts_in_bounds(starts, narrowed[n]) if args.pose_bound else (starts, []))
                   for n in names}
    nozzle, nozzle_source = cli.resolve_nozzles()
    nozzle_hash = cli.nozzle_hash(nozzle)
    physics_hash = explog.file_hash(ROOT / "configs" / "physics.yaml")
    commit = explog.git_commit()

    group_id = explog.new_exp_id(args.tag)
    group_dir = Path(args.log_dir) / group_id
    print(f"[{group_id}] evaluator={args.evaluator} scenarios={names} seeds={args.seeds} "
          f"max_evals={args.max_evals} popsize={args.popsize} starts={args.starts} "
          f"nozzles={nozzle.count}({nozzle_source})"
          + (f" patches_per_m2={args.patches_per_m2:g}" if args.evaluator == "patch" else ""))

    rescore = args.evaluator == "patch" and args.rescore_patches_per_m2 > 0
    summary_rows: list[dict] = []
    baseline_rows: list[dict] = []
    poses_out: dict = {
        "group_id": group_id,
        "commit": commit,
        "physics_hash": physics_hash,
        "nozzle_hash": nozzle_hash,
        "args": vars(args),
        "yaw_note": "좌우 대칭(θ ≡ −θ)·앞뒤 등가(θ ≡ 180° − θ)라 yaw 는 0~90° 로 접어 비교한다. "
                    "mean_pose_folded 는 e4.fold_yaw 로 접은 평균",
        "scenarios": {},
    }

    for name in names:
        # 채점(평가기·기준선·재채점)은 원래 시나리오로, 탐색 상자만 좁힌 시나리오로 한다.
        scenario = all_scenarios[name]           # 점수 척도의 기준 (불편도 정규화 포함)
        search_scenario = narrowed[name]         # 탐색 상자
        encoder = PoseEncoder(search_scenario) if args.pose_bound else None
        scenario_starts, dropped = kept_starts[name]
        if args.pose_bound:
            print(f"  [{name}] 좁힌 탐색 상자: {cli.format_bounds_note(search_scenario, scenario)}"
                  f"  (점수·기준선은 원래 범위)")
            if dropped:
                print(f"  [{name}] 범위 밖 시작점 건너뜀: {', '.join(dropped)}")
        try:
            evaluator = cli.make_evaluator(
                args.evaluator, scenario, body=body, nozzle=nozzle, dummy_target=dummy_target,
                patches_per_m2=args.patches_per_m2,
            )
        except cli.TrackNotMerged as exc:
            print(str(exc), file=sys.stderr)
            return 3

        base = baselines.evaluate_all(evaluator, nozzle, body, scenario)
        for cond, agg in base.items():
            row = {"scenario": name, "condition": cond, "infeasible": agg["infeasible"],
                   "score": agg["score"], "total_removal": agg["total_removal"],
                   "discomfort": agg["discomfort"], "n_feasible": agg["n_feasible"],
                   "n_in_bounds": agg["n_in_bounds"], "yaws": agg["yaws"]}
            row["pose_bound"] = ";".join(args.pose_bound)
            row.update({f"removal_{p}": float(v) for p, v in zip(PART_NAMES, agg["removal_by_part"])})
            baseline_rows.append(row)

        runs: list[dict] = []
        for seed in args.seeds:
            exp_id = explog.new_exp_id(args.tag)
            result = run_cmaes(
                evaluator, body, scenario, nozzle,
                max_evals=args.max_evals, seed=seed, popsize=args.popsize, sigma0=args.sigma0,
                tol_stagnation_gens=args.tol_stagnation_gens, starts=scenario_starts,
                log_dir=args.log_dir, exp_id=exp_id, encoder=encoder,
            )
            explog.write_run(
                args.log_dir, exp_id,
                scenario=scenario, body=body, nozzle_hash=nozzle_hash, physics_hash=physics_hash,
                commit=commit, seed=seed,
                args={**vars(args), "e4_group": group_id, "scenario": name, "seed": seed,
                      "pose_bounds_effective": {k: list(v) for k, v in search_scenario.pose_bounds.items()
                                                if v != scenario.pose_bounds[k]},
                      "nozzle_source": nozzle_source, "free_keys": result.free_keys,
                      "stop_reason": result.stop_reason, "elapsed_s": round(result.elapsed_s, 3)},
                result=result,
            )
            runs.append({
                "exp_id": exp_id, "seed": seed, "best_score": result.best_score,
                "best_pose": result.best_pose, "per_start": result.per_start,
                "elapsed_s": result.elapsed_s, "n_evals": result.n_evals,
                "n_infeasible": result.n_infeasible,
                "total_removal": float(result.best_result.total_removal),
            })
            print(f"  {name:<11} seed {seed}  best {result.best_score:.6f}  "
                  f"start {e4.winning_start(result.per_start) or '-':<9} "
                  f"abd {result.best_pose.shoulder_abduction:6.1f}  yaw {result.best_pose.torso_yaw:7.1f}  "
                  f"{result.elapsed_s:6.1f} s  [{exp_id}]")

        summary = e4.summarize_scenario(runs, base)
        row = {"scenario": name, "pose_bound": ";".join(args.pose_bound), **summary}
        base_fine = None
        if rescore:
            # 같은 자세를 촘촘한 격자로 다시 평가한다 (탐색은 하지 않는다).
            fine = cli.make_evaluator(args.evaluator, scenario, body=body, nozzle=nozzle,
                                      patches_per_m2=args.rescore_patches_per_m2)
            base_fine = baselines.evaluate_all(fine, nozzle, body, scenario)
            for r in runs:
                r["rescore_score"] = float(fine.evaluate(r["best_pose"], nozzle, body, scenario).score)
            fine_summary = e4.summarize_scenario(
                [{**r, "best_score": r["rescore_score"]} for r in runs], base_fine)
            row["rescore_patches_per_m2"] = args.rescore_patches_per_m2
            for key in ("best_mean", "best_std", "best_min", "best_max"):
                row[f"rescore_{key}"] = fine_summary[key]
            for b in e4.BASELINE_NAMES:
                for key in (f"{b}_score", f"{b}_infeasible", f"imp_vs_{b}"):
                    row[f"rescore_{key}"] = fine_summary[key]
        summary_rows.append(row)
        poses_out["scenarios"][name] = {
            "runs": [{
                "exp_id": r["exp_id"], "seed": r["seed"], "best_score": r["best_score"],
                "rescore_score": r.get("rescore_score"), "total_removal": r["total_removal"],
                "start_best": {ps["start"]: ps["best_score"] for ps in r["per_start"]},
                "peak_gap": e4.peak_gap(r["per_start"]), "best_start": e4.winning_start(r["per_start"]),
                "pose": cli.pose_dict(r["best_pose"]), "pose_folded": e4.fold_pose(r["best_pose"]),
            } for r in runs],
            "mean_pose_folded": {k: summary[f"pose_mean_{k}"] for k in e4.POSE_KEYS},
            # 기준선은 밀도별로 따로 남긴다. 시드별 best 의 rescore_score 와 비교할 때는
            # 반드시 baselines_rescored(재채점 밀도) 쪽을 써야 한다 (통합 2026-09-30 지적).
            "baselines": {c: {"score": a["score"], "infeasible": a["infeasible"]} for c, a in base.items()},
            "pose_bounds_effective": {k: list(v) for k, v in search_scenario.pose_bounds.items()
                                      if v != scenario.pose_bounds[k]},
            "starts_used": [s.name for s in scenario_starts],
            "starts_skipped": list(dropped),
            "baselines_patches_per_m2": args.patches_per_m2 if args.evaluator == "patch" else None,
            "baselines_rescored": ({c: {"score": a["score"], "infeasible": a["infeasible"]}
                                    for c, a in base_fine.items()} if base_fine else None),
            "baselines_rescored_patches_per_m2": args.rescore_patches_per_m2 if base_fine else None,
        }

    group_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(group_dir / "e4_summary.csv", summary_rows)
    _write_csv(group_dir / "baselines.csv", baseline_rows)
    (group_dir / "best_poses.json").write_text(
        json.dumps(explog._jsonable(poses_out), ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"{'시나리오':<12}{'best 평균':>11}{'표준편차':>10}{'B0 대비':>9}{'B1 대비':>9}{'B2 대비':>9}"
          f"{'yaw* 편차':>11}{'벌림 편차':>10}  best 시작점")
    print("-" * 100)
    for row in summary_rows:
        imps = "".join(
            f"{'불가':>9}" if row[f"{b}_infeasible"] else f"{row[f'imp_vs_{b}']:>+9.0%}"
            for b in e4.BASELINE_NAMES
        )
        print(f"{row['scenario']:<12}{row['best_mean']:>11.4f}{row['best_std']:>10.4f}{imps}"
              f"{row['pose_std_torso_yaw']:>11.1f}{row['pose_std_shoulder_abduction']:>10.1f}"
              f"  {row['best_start_counts']}")
    if rescore:
        print()
        print(f"재채점 ({args.rescore_patches_per_m2:g}/m², 같은 자세)")
        for row in summary_rows:
            imps = "".join(
                f"{'불가':>9}" if row[f"rescore_{b}_infeasible"] else f"{row[f'rescore_imp_vs_{b}']:>+9.0%}"
                for b in e4.BASELINE_NAMES
            )
            print(f"{row['scenario']:<12}{row['rescore_best_mean']:>11.4f}{row['rescore_best_std']:>10.4f}{imps}")
    print()
    print(f"저장 위치    {group_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
