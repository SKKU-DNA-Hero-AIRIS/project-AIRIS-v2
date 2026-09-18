"""E4 의 기준선 B0, B1, B2 를 모든 시나리오에서 평가한다. 소유자: C.

    B0  기본 자세 (PoseParams() 기본값)                   — 안내 없이 통과할 때
    B1  기본 자세로 서서 몸 회전 (torso_yaw 0, 30, …, 330) — 퓨리움 공식 안내
        "머문 상태에서 몸을 회전"의 정적 등가물. 가능한 yaw 의 평균.
    B2  만세 (shoulder_abduction=180, elbow_flexion=0)     — 체형에 따라 천장에 걸리면 불가

총괄 결정 ⑧′. 팔 수평 90° 조건은 새 부스에서 불가라 삭제했다.

B1 집계:
    score, removal_by_part, total_removal = 가능한 yaw 결과의 평균
    discomfort                            = 기본 자세(yaw 0) 값
    가능한 yaw 가 없으면 불가 (score 는 전체 yaw 벌점 평균)
    시나리오 pose_bounds 밖의 yaw 는 뺀다 (휠체어는 torso_yaw [-45, 45] 라 −30, 0, 30 만 남는다).

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

import numpy as np                                            # noqa: E402

from airis.optimize import cli, explog                        # noqa: E402
from airis.optimize.cmaes_runner import is_infeasible         # noqa: E402
from airis.optimize.encoding import PoseEncoder               # noqa: E402
from airis.sim import PART_NAMES, BodyParams, PoseParams, Scenario  # noqa: E402
from airis.sim.scenario import load_scenarios                 # noqa: E402

#: B1 의 몸 회전 각도 (deg). 180 초과는 pose_bounds [-180, 180] 에 맞춰 음수로 감싼다.
B1_YAWS_DEG: list[float] = [float(y) for y in range(0, 360, 30)]

#: 조건별 자세 목록. 자세가 여러 개면 가능한 자세의 평균을 낸다 (B1).
BASELINES: dict[str, list[PoseParams]] = {
    "B0": [PoseParams()],
    "B1": [PoseParams(torso_yaw=y if y <= 180 else y - 360) for y in B1_YAWS_DEG],
    "B2": [PoseParams(shoulder_abduction=180.0, elbow_flexion=0.0)],
}


def in_bounds(pose: PoseParams, scenario: Scenario) -> bool:
    """자유 변수가 전부 시나리오 pose_bounds 안인가 (fixed_pose 는 투영 때 덮어쓰므로 보지 않는다)."""
    for key, (lo, hi) in scenario.pose_bounds.items():
        if key in scenario.fixed_pose:
            continue
        if not lo <= getattr(pose, key) <= hi:
            return False
    return True


def evaluate_condition(evaluator, poses, encoder, nozzle, body, scenario) -> dict:
    """조건 하나를 평가해 집계한다. 반환 dict 는 CSV 행의 재료가 된다."""
    usable = [p for p in poses if in_bounds(p, scenario)]
    # 시나리오 제약(pose_bounds + fixed_pose)으로 투영한다. 휠체어는 여기서 고관절·무릎이 90도가 된다.
    projected = [encoder.clip_pose(p) for p in usable]
    results = [evaluator.evaluate(p, nozzle, body, scenario) for p in projected]
    feasible = [r for r in results if not is_infeasible(r)]

    pool = feasible or results                 # 전부 불가면 벌점 평균을 남긴다
    return {
        "infeasible": not feasible,
        "n_poses": len(poses),
        "n_in_bounds": len(usable),
        "n_feasible": len(feasible),
        "yaws": ";".join(f"{p.torso_yaw:g}" for p in projected) if len(poses) > 1 else "",
        "score": float(np.mean([r.score for r in pool])),
        "total_removal": float(np.mean([r.total_removal for r in feasible])) if feasible else 0.0,
        # 첫 자세 = 단일 자세 또는 회전 없는 기본 자세(yaw 0).
        "discomfort": float(results[0].discomfort),
        "removal_by_part": (np.mean([r.removal_by_part for r in feasible], axis=0)
                            if feasible else np.zeros(len(PART_NAMES))),
        "pose": projected[0],
    }


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
