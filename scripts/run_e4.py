"""E4 러너: 시나리오 × 시드마다 CMA-ES 를 돌리고 기준 자세(B0, B1, B2)와 비교한다. 소유자: C.

사용 예:
    python scripts/run_e4.py                                  # 패치판, 시나리오 3개 × 시드 0~4
    python scripts/run_e4.py --scenarios default --seeds 0 1  # 일부만
    python scripts/run_e4.py --evaluator dummy --max-evals 200 --popsize 20   # 파이프라인 확인용

결과:
    outputs/<exp_id>/                     실행마다 meta.json, history.csv, best.json (tag=e4)
    outputs/<group_id>/e4_summary.csv     시나리오별 best 평균·표준편차, B0·B1·B2 대비 개선율,
                                          시드 간 자세 편차(yaw 는 |yaw| 로 접음)
    outputs/<group_id>/best_poses.json    시드별 best 자세, 시나리오별 평균 자세, 기준선 점수
    outputs/<group_id>/baselines.csv      이번 묶음에서 평가한 B0, B1, B2

요약 파일은 실행 묶음마다 <group_id> 폴더에 따로 남긴다 (다시 돌려도 이전 요약을 덮어쓰지 않는다).
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

    body = cli.dataclass_from_json(BodyParams, args.body)
    dummy_target = cli.dataclass_from_json(PoseParams, args.dummy_target) if args.dummy_target else None
    starts = cli.parse_starts(args.starts)
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

    summary_rows: list[dict] = []
    baseline_rows: list[dict] = []
    poses_out: dict = {
        "group_id": group_id,
        "commit": commit,
        "physics_hash": physics_hash,
        "nozzle_hash": nozzle_hash,
        "args": vars(args),
        "yaw_note": "노즐·부스가 좌우 대칭이라 yaw 부호는 임의. mean_pose_folded 는 |yaw| 로 접은 평균",
        "scenarios": {},
    }

    for name in names:
        scenario = all_scenarios[name]
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
            row.update({f"removal_{p}": float(v) for p, v in zip(PART_NAMES, agg["removal_by_part"])})
            baseline_rows.append(row)

        runs: list[dict] = []
        for seed in args.seeds:
            exp_id = explog.new_exp_id(args.tag)
            result = run_cmaes(
                evaluator, body, scenario, nozzle,
                max_evals=args.max_evals, seed=seed, popsize=args.popsize, sigma0=args.sigma0,
                tol_stagnation_gens=args.tol_stagnation_gens, starts=starts,
                log_dir=args.log_dir, exp_id=exp_id,
            )
            explog.write_run(
                args.log_dir, exp_id,
                scenario=scenario, body=body, nozzle_hash=nozzle_hash, physics_hash=physics_hash,
                commit=commit, seed=seed,
                args={**vars(args), "e4_group": group_id, "scenario": name, "seed": seed,
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
        summary_rows.append({"scenario": name, **summary})
        poses_out["scenarios"][name] = {
            "runs": [{
                "exp_id": r["exp_id"], "seed": r["seed"], "best_score": r["best_score"],
                "total_removal": r["total_removal"], "best_start": e4.winning_start(r["per_start"]),
                "pose": cli.pose_dict(r["best_pose"]), "pose_folded": e4.fold_pose(r["best_pose"]),
            } for r in runs],
            "mean_pose_folded": {k: summary[f"pose_mean_{k}"] for k in e4.POSE_KEYS},
            "baselines": {c: {"score": a["score"], "infeasible": a["infeasible"]} for c, a in base.items()},
        }

    group_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(group_dir / "e4_summary.csv", summary_rows)
    _write_csv(group_dir / "baselines.csv", baseline_rows)
    (group_dir / "best_poses.json").write_text(
        json.dumps(explog._jsonable(poses_out), ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"{'시나리오':<12}{'best 평균':>11}{'표준편차':>10}{'B0 대비':>9}{'B1 대비':>9}{'B2 대비':>9}"
          f"{'|yaw| 편차':>11}{'벌림 편차':>10}  best 시작점")
    print("-" * 100)
    for row in summary_rows:
        imps = "".join(
            f"{'불가':>9}" if row[f"{b}_infeasible"] else f"{row[f'imp_vs_{b}']:>+9.0%}"
            for b in e4.BASELINE_NAMES
        )
        print(f"{row['scenario']:<12}{row['best_mean']:>11.4f}{row['best_std']:>10.4f}{imps}"
              f"{row['pose_std_torso_yaw']:>11.1f}{row['pose_std_shoulder_abduction']:>10.1f}"
              f"  {row['best_start_counts']}")
    print()
    print(f"저장 위치    {group_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
