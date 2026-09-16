"""시나리오 하나에 대해 CMA-ES 자세 최적화를 돌린다. 소유자: C.

사용 예:
    python scripts/run_optimize.py --scenario default --evaluator dummy --max-evals 3000 --seed 0
    python scripts/run_optimize.py --scenario wheelchair --evaluator patch --tag e4
    python scripts/run_optimize.py --body '{"height_m":1.6}' --popsize 100 --sigma0 0.5

결과는 outputs/<exp_id>/ 에 meta.json, history.csv, best.json 으로 남는다.
docs/tracks/C_optimize.md 단계 5.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airis.optimize import cli, explog                      # noqa: E402
from airis.optimize.cmaes_runner import run_cmaes           # noqa: E402
from airis.sim import BodyParams, PoseParams                # noqa: E402
from airis.sim.scenario import load_scenarios               # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="에어게이트 자세 CMA-ES 최적화")
    ap.add_argument("--scenario", default="default", help="configs/scenarios.yaml 의 시나리오 이름")
    ap.add_argument("--evaluator", choices=cli.EVALUATOR_CHOICES, default="patch")
    ap.add_argument("--max-evals", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--popsize", type=int, default=100)
    ap.add_argument("--sigma0", type=float, default=0.5)
    ap.add_argument("--tol-stagnation-gens", type=int, default=30)
    ap.add_argument("--patches-per-m2", type=float, default=cli.DEFAULT_PATCHES_PER_M2,
                    help="패치판 표면 패치 밀도 (--evaluator patch 에만 적용)")
    ap.add_argument("--body", default=None, help='BodyParams 덮어쓰기 JSON, 예: \'{"height_m":1.6}\'')
    ap.add_argument("--tag", default="opt", help="exp_id 접두어")
    ap.add_argument("--log-dir", default=str(ROOT / "outputs"))
    ap.add_argument(
        "--dummy-target", default=None,
        help='--evaluator dummy 의 목표 자세 JSON. 생략 시 어깨 벌림 90도, 팔꿈치 0도',
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)

    scenarios = load_scenarios()
    if args.scenario not in scenarios:
        print(f"없는 시나리오: {args.scenario} (가능: {', '.join(scenarios)})", file=sys.stderr)
        return 2
    scenario = scenarios[args.scenario]

    body = cli.dataclass_from_json(BodyParams, args.body)
    dummy_target = cli.dataclass_from_json(PoseParams, args.dummy_target) if args.dummy_target else None
    nozzle, nozzle_source = cli.resolve_nozzles()

    try:
        evaluator = cli.make_evaluator(
            args.evaluator, scenario, body=body, nozzle=nozzle, dummy_target=dummy_target,
            patches_per_m2=args.patches_per_m2,
        )
    except cli.TrackNotMerged as exc:
        print(str(exc), file=sys.stderr)
        return 3

    exp_id = explog.new_exp_id(args.tag)
    print(f"[{exp_id}] scenario={scenario.name} evaluator={args.evaluator} "
          f"seed={args.seed} popsize={args.popsize} max_evals={args.max_evals} "
          f"nozzles={nozzle.count}({nozzle_source})"
          + (f" patches_per_m2={args.patches_per_m2:g}" if args.evaluator == "patch" else ""))

    result = run_cmaes(
        evaluator, body, scenario, nozzle,
        max_evals=args.max_evals,
        seed=args.seed,
        popsize=args.popsize,
        sigma0=args.sigma0,
        tol_stagnation_gens=args.tol_stagnation_gens,
        log_dir=args.log_dir,
        exp_id=exp_id,
    )

    run_args = {
        **{k: v for k, v in vars(args).items()},
        "nozzle_source": nozzle_source,
        "free_keys": result.free_keys,
        "dim": len(result.free_keys),
        "stop_reason": result.stop_reason,
        "elapsed_s": round(result.elapsed_s, 3),
    }
    out = explog.write_run(
        args.log_dir, exp_id,
        scenario=scenario,
        body=body,
        nozzle_hash=cli.nozzle_hash(nozzle),
        physics_hash=explog.file_hash(ROOT / "configs" / "physics.yaml"),
        commit=explog.git_commit(),
        seed=args.seed,
        args=run_args,
        result=result,
    )

    print()
    print(cli.format_pose_table(result.best_pose, scenario, result.free_keys))
    print()
    print(cli.format_removal_table(result.best_result))
    print()
    print(f"best_score   {result.best_score:.6f}")
    print(f"평가 횟수    {result.n_evals} ({len(result.history)} 세대)")
    if result.n_evals:
        print(f"불가 후보    {result.n_infeasible} ({result.n_infeasible / result.n_evals:.1%})")
    print(f"종료 사유    {result.stop_reason}")
    print(f"소요 시간    {result.elapsed_s:.2f} s")
    print(f"저장 위치    {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
