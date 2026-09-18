"""E4 의 기준선 B0, B1, B2 를 모든 시나리오에서 평가한다. 소유자: C.

조건 정의와 집계 규칙은 airis/optimize/baselines.py.

사용 예:
    python scripts/run_baselines.py --evaluator patch
    python scripts/run_baselines.py --evaluator dummy          # 파이프라인 확인용

결과는 outputs/baselines_<YYYYmmdd>.csv. docs/tracks/C_optimize.md 단계 6.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airis.optimize import cli, explog                        # noqa: E402
from airis.optimize.baselines import (                        # noqa: E402,F401  (테스트가 여기서도 import)
    B1_YAWS_DEG, BASELINES, evaluate_condition, in_bounds,
)
from airis.optimize.encoding import PoseEncoder               # noqa: E402
from airis.sim import PART_NAMES, BodyParams, PoseParams      # noqa: E402
from airis.sim.scenario import load_scenarios                 # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E4 기준선(B0, B1, B2) 평가")
    ap.add_argument("--evaluator", choices=cli.EVALUATOR_CHOICES, default="patch")
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="평가할 시나리오. 생략 시 configs/scenarios.yaml 전부")
    ap.add_argument("--body", default=None, help='BodyParams 덮어쓰기 JSON')
    ap.add_argument("--log-dir", default=str(ROOT / "outputs"))
    ap.add_argument("--patches-per-m2", type=float, default=cli.DEFAULT_PATCHES_PER_M2,
                    help="패치판 표면 패치 밀도 (--evaluator patch 에만 적용)")
    ap.add_argument("--dummy-target", default=None, help="--evaluator dummy 의 목표 자세 JSON")
    return ap


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)

    all_scenarios = load_scenarios()
    names = args.scenarios or list(all_scenarios)
    missing = [n for n in names if n not in all_scenarios]
    if missing:
        print(f"없는 시나리오: {missing} (가능: {', '.join(all_scenarios)})", file=sys.stderr)
        return 2

    body = cli.dataclass_from_json(BodyParams, args.body)
    dummy_target = cli.dataclass_from_json(PoseParams, args.dummy_target) if args.dummy_target else None
    nozzle, nozzle_source = cli.resolve_nozzles()
    commit = explog.git_commit()

    rows: list[dict] = []
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

        encoder = PoseEncoder(scenario)
        for cond, poses in BASELINES.items():
            agg = evaluate_condition(evaluator, poses, encoder, nozzle, body, scenario)
            row = {
                "scenario": name,
                "condition": cond,
                "evaluator": args.evaluator,
                "nozzle_source": nozzle_source,
                "nozzle_hash": cli.nozzle_hash(nozzle),
                "patches_per_m2": args.patches_per_m2 if args.evaluator == "patch" else "",
                "commit": commit,
                "infeasible": agg["infeasible"],
                "score": agg["score"],
                "total_removal": agg["total_removal"],
                "discomfort": agg["discomfort"],
                "n_poses": agg["n_poses"],
                "n_in_bounds": agg["n_in_bounds"],
                "n_feasible": agg["n_feasible"],
                "yaws": agg["yaws"],
            }
            row.update({f"pose_{k}": v for k, v in cli.pose_dict(agg["pose"]).items()})
            row.update({
                f"removal_{part}": float(v)
                for part, v in zip(PART_NAMES, agg["removal_by_part"])
            })
            rows.append(row)

    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"baselines_{datetime.now():%Y%m%d}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    density = f", 패치 밀도 {args.patches_per_m2:g}/m²" if args.evaluator == "patch" else ""
    print(f"평가기 {args.evaluator}, 노즐 {nozzle.count}개({nozzle_source}){density}")
    print()
    print(f"{'시나리오':<14}{'조건':<6}{'score':>12}{'total_removal':>15}{'discomfort':>12}{'가능':>8}")
    print("-" * 69)
    for row in rows:
        score = "불가" if row["infeasible"] else f"{row['score']:.6f}"
        feasible = f"{row['n_feasible']}/{row['n_in_bounds']}"
        print(f"{row['scenario']:<14}{row['condition']:<6}{score:>12}"
              f"{row['total_removal']:>15.4f}{row['discomfort']:>12.4f}{feasible:>8}")
    print()
    print(f"저장 위치    {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
