"""E7 러너: 계획(자세 순서 + 구역 세기 + 시간) 조건 P0~P5 비교. 소유자: C.

docs/plan_extension.md 6절. `--evaluator patch` 는 D 의 `evaluate_plan` 이 병합된 뒤 쓸 수 있고,
그 전에는 `--evaluator dummy` 로 파이프라인을 확인한다.

    P0  기본 자세, 세기 전부 최대, 총 시간 상한      현행 운전 (안내 없음)
    P1  기본 자세로 몸 12단계 회전                  제품 안내를 순서로 평가
    P2  단일 자세 최적을 전 단계에                  자세만 (세기·시간은 P0 과 같다)
    P3  기본 자세 고정, 세기·시간만 최적
    P4  단일 자세 최적 고정, 세기·시간만 최적
    P5  자세 순서 + 세기 + 시간 전부 최적 (21차원)

P2·P4 의 단일 자세 최적은 이 스크립트가 `run_cmaes`(자세 7차원)로 먼저 구하고, P5 의 시작점으로도 쓴다.

사용 예:
    python scripts/run_e7.py --evaluator dummy --max-evals 600 --popsize 20     # 파이프라인 확인
    python scripts/run_e7.py --evaluator patch --scenarios default --seeds 0 1
    python scripts/run_e7.py --evaluator patch --energy-weights 0 0.1 0.4       # 에너지 가중 스윕

결과: outputs/<group_id>/e7_summary.csv (조건 × 시나리오 × 에너지 가중), e7_plans.json (계획 전체).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                            # noqa: E402

from airis.optimize import baselines, cli, explog, sensitivity  # noqa: E402
from airis.optimize.cmaes_runner import Start, run_cmaes, run_cmaes_plan  # noqa: E402
from airis.optimize.plan_encoding import (                    # noqa: E402
    PlanLimits, default_zone_counts, plan_kinetics_stamp, plan_physics_cfg,
)
from airis.sim import PART_NAMES, BodyParams, PoseParams      # noqa: E402
from airis.sim.scenario import load_physics, load_scenarios   # noqa: E402

CONDITIONS = ("P0", "P1", "P2", "P3", "P4", "P5")
#: 최적화 없이 한 번 평가하는 기준선. P1opt·P1down 은 기본 조건이 아니고 --conditions 로 켠다.
BASELINE_CONDITIONS = {"P0", "P1", "P2", "P1opt", "P1down"}
OPTIMIZED = {"P3", "P4", "P5"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E7: 계획 조건 P0~P5 비교")
    ap.add_argument("--evaluator", choices=cli.EVALUATOR_CHOICES, default="patch")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--energy-weights", nargs="*", type=float, default=[0.0, 0.01, 0.03, 0.1],
                    help="scoring.energy_weight 스윕 (통합·D 검토 2026-09-30). 0 이면 에너지 항 없음")
    ap.add_argument("--n-phases", type=int, default=None,
                    help="plan.n_phases 덮어쓰기 (configs 는 건드리지 않는다). E7 본 실행은 2 였고, "
                         "단계 수가 병목인지 보는 스윕에 쓴다")
    ap.add_argument("--rotation-pose", default=None,
                    help="P1down 기준선이 돌릴 자세 JSON (예: 팔 내림 봉우리). 없으면 P1down 을 건너뛴다")
    ap.add_argument("--max-evals", type=int, default=6000, help="계획 최적화 예산 (21차원)")
    ap.add_argument("--pose-max-evals", type=int, default=3000, help="단일 자세 최적(P2·P4·시작점) 예산")
    ap.add_argument("--popsize", type=int, default=100)
    ap.add_argument("--sigma0", type=float, default=0.5)
    ap.add_argument("--starts", default=cli.DEFAULT_STARTS, help="자세 최적화 시작점 (P5 시작점도 여기서 만든다)")
    ap.add_argument("--patches-per-m2", type=float, default=1500.0)
    ap.add_argument("--body", default=None)
    ap.add_argument("--tag", default="e7")
    ap.add_argument("--log-dir", default=str(ROOT / "outputs"))
    ap.add_argument("--dummy-target", default=None)
    return ap


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def with_energy_weight(base_cfg: dict, scenario, w: float):
    """scoring.energy_weight 를 w 로 둔 설정 복사본. 키가 없으면 만들어 넣는다.

    base_cfg 는 plan_physics_cfg 를 거친(= kinetics 를 켠) 설정이어야 한다.
    """
    import copy

    try:
        return sensitivity.apply_setting(base_cfg, scenario, "scoring.energy_weight", w)
    except KeyError:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("scoring", {})["energy_weight"] = w
        return cfg, scenario


def plan_summary(plan) -> dict:
    """계획을 사람이 읽는 dict 로 (JSON 저장·표시용)."""
    return {
        "duration_s": round(plan.duration_s, 2),
        "phases": [{"duration_s": round(p.duration_s, 2), **cli.pose_dict(p.pose)} for p in plan.phases],
        "zone_strengths": [round(float(v), 4) for v in plan.zone_strengths],
    }


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    known = set(CONDITIONS) | BASELINE_CONDITIONS
    bad = []
    for c in conditions:                    # P1_10 처럼 회전 단계 수를 붙인 이름도 받는다
        try:
            if baselines.parse_baseline_name(c)[0] not in known:
                bad.append(c)
        except ValueError:
            bad.append(c)
    if bad:
        print(f"알 수 없는 조건 {bad} (가능: {', '.join(sorted(known))}, 회전 기준선은 P1_10 처럼 "
              f"단계 수를 붙일 수 있다)", file=sys.stderr)
        return 2

    all_scenarios = load_scenarios()
    names = args.scenarios or list(all_scenarios)
    missing = [n for n in names if n not in all_scenarios]
    if missing:
        print(f"없는 시나리오: {missing}", file=sys.stderr)
        return 2

    # 계획 평가는 시간 의존 제거(kinetics)를 켠 설정으로 한다 (plan_extension.md 4절).
    base_cfg = plan_physics_cfg(load_physics())
    kinetics = plan_kinetics_stamp(base_cfg)
    weights = args.energy_weights or [float(base_cfg.get("scoring", {}).get("energy_weight", 0.1))]
    body = cli.dataclass_from_json(BodyParams, args.body)
    dummy_target = cli.dataclass_from_json(PoseParams, args.dummy_target) if args.dummy_target else None
    pose_starts = cli.parse_starts(args.starts)
    rotation_pose = cli.dataclass_from_json(PoseParams, args.rotation_pose) if args.rotation_pose else None
    if any(c.startswith("P1down") for c in conditions) and rotation_pose is None:
        print("P1down 에는 --rotation-pose 가 필요하다", file=sys.stderr)
        return 2
    nozzle, nozzle_source = cli.resolve_nozzles()
    nozzle_hash, commit = cli.nozzle_hash(nozzle), explog.git_commit()
    physics_hash = explog.file_hash(ROOT / "configs" / "physics.yaml")
    group_id = explog.new_exp_id(args.tag)
    group_dir = Path(args.log_dir) / group_id
    limits = PlanLimits.from_config(base_cfg)
    if args.n_phases is not None:                    # configs 는 건드리지 않고 이번 실행만
        if args.n_phases < 1:
            print("--n-phases 는 1 이상이어야 한다", file=sys.stderr)
            return 2
        limits = dataclasses.replace(limits, n_phases=int(args.n_phases))
        # 단계를 늘리면 총 시간 하한이 단계 수 × 최소 단계 시간보다 짧아질 수 있다.
        # 설정을 건드리지 않고 이번 실행의 하한만 끌어올린다 (상한을 넘으면 멈춘다).
        lo, hi = limits.duration_bounds_s
        need = limits.n_phases * limits.min_phase_s
        if need > lo:
            if need > hi:
                print(f"단계 {limits.n_phases}개 × 최소 {limits.min_phase_s:g} s = {need:g} s 가 "
                      f"총 시간 상한 {hi:g} s 보다 길다", file=sys.stderr)
                return 2
            print(f"  총 시간 하한을 {lo:g} → {need:g} s 로 올린다 "
                  f"(단계 {limits.n_phases}개 × 최소 {limits.min_phase_s:g} s)")
            limits = dataclasses.replace(limits, duration_bounds_s=(need, hi))
    print(f"[{group_id}] evaluator={args.evaluator} scenarios={names} seeds={args.seeds} "
          f"conditions={conditions} energy_weights={weights} n_phases={limits.n_phases} "
          f"nozzles={nozzle.count}({nozzle_source}) "
          f"kinetics={kinetics['kinetics_enabled']}(T_r {kinetics['time_constant_s']:g} s)")

    rows: list[dict] = []
    plans_out: dict = {"group_id": group_id, "commit": commit, "physics_hash": physics_hash,
                       "nozzle_hash": nozzle_hash, "args": vars(args), **kinetics,
                       "zone_nozzle_counts": [float(v) for v in default_zone_counts()],
                       "scenarios": {}}

    for name in names:
        scenario = all_scenarios[name]
        plans_out["scenarios"][name] = {}
        best_pose = None
        # P1opt·P1opt_10 처럼 회전 기준선도 단일 자세 최적을 쓴다 (단계 수 접미사를 떼고 본다).
        needs_pose = {"P2", "P4", "P5", "P1opt"}
        if any(baselines.parse_baseline_name(c)[0] in needs_pose for c in conditions):
            try:
                evaluator = cli.make_evaluator(args.evaluator, scenario, body=body, nozzle=nozzle,
                                               dummy_target=dummy_target,
                                               patches_per_m2=args.patches_per_m2)
            except cli.TrackNotMerged as exc:
                print(str(exc), file=sys.stderr)
                return 3
            pose_res = run_cmaes(evaluator, body, scenario, nozzle, max_evals=args.pose_max_evals,
                                 seed=args.seeds[0], popsize=args.popsize, sigma0=args.sigma0,
                                 starts=pose_starts, log_dir=args.log_dir,
                                 exp_id=explog.new_exp_id(args.tag + "pose"))
            best_pose = pose_res.best_pose
            print(f"  {name:<11} 단일 자세 최적 {pose_res.best_score:.4f} "
                  f"(벌림 {best_pose.shoulder_abduction:.1f}, yaw {best_pose.torso_yaw:.1f})")

        for w in weights:
            cfg, scen = with_energy_weight(base_cfg, scenario, float(w))
            try:
                ev = cli.make_evaluator(args.evaluator, scen, body=body, nozzle=nozzle,
                                        dummy_target=dummy_target, patches_per_m2=args.patches_per_m2,
                                        physics_cfg=cfg)
            except cli.TrackNotMerged as exc:
                print(str(exc), file=sys.stderr)
                return 3

            for cond in conditions:
                seeds = args.seeds if cond in OPTIMIZED else [args.seeds[0]]
                for seed in seeds:
                    if baselines.parse_baseline_name(cond)[0] in BASELINE_CONDITIONS:
                        plan = baselines.plan_baseline(cond, scen, limits, best_pose=best_pose,
                                                       rotation_pose=rotation_pose)
                        row = baselines.evaluate_plan_row(ev, plan, nozzle, body, scen)
                        n_evals, elapsed = 1, 0.0
                    else:
                        fixed = None
                        if cond == "P3":
                            fixed = [PoseParams()] * limits.n_phases
                        elif cond == "P4":
                            fixed = [best_pose] * limits.n_phases
                        starts = None if cond != "P5" else [
                            Start(s.name, s.pose, s.sigma0) for s in pose_starts]
                        res = run_cmaes_plan(
                            ev, body, scen, nozzle, limits=limits, starts=starts,
                            max_evals=args.max_evals, seed=seed, popsize=args.popsize,
                            sigma0=args.sigma0, log_dir=args.log_dir,
                            exp_id=explog.new_exp_id(f"{args.tag}{cond.lower()}"),
                            **({"fixed_poses": fixed} if fixed is not None else {}))
                        plan = res.best_plan
                        row = baselines.evaluate_plan_row(ev, plan, nozzle, body, scen)
                        n_evals, elapsed = res.n_evals, res.elapsed_s

                    removal = row.pop("removal_by_part")
                    row.pop("plan")
                    rows.append({"scenario": name, "condition": cond, "energy_weight": float(w),
                                 "seed": seed, **row, **kinetics, "n_evals": n_evals,
                                 "elapsed_s": round(elapsed, 2),
                                 **{f"removal_{p}": float(v) for p, v in zip(PART_NAMES, removal)}})
                    plans_out["scenarios"][name].setdefault(f"w{w:g}", {})[f"{cond}_s{seed}"] = plan_summary(plan)
                    print(f"  {name:<11} w {w:<5g} {cond} seed {seed}  score {row['score']:.4f}  "
                          f"E {row['energy']:.3f}  T {row['duration_s']:.1f} s  "
                          f"단계 {row['n_phases']}  {elapsed:.0f} s")

    group_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(group_dir / "e7_summary.csv", rows)
    (group_dir / "e7_plans.json").write_text(
        json.dumps(explog._jsonable(plans_out), ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"{'시나리오':<12}{'가중':>6}{'조건':>5}{'score':>10}{'에너지':>8}{'시간':>7}{'단계':>5}"
          f"{'P0 대비':>9}")
    base = {(r["scenario"], r["energy_weight"]): r["score"]
            for r in rows if r["condition"] == "P0" and not r["infeasible"]}
    for r in rows:
        b = base.get((r["scenario"], r["energy_weight"]))
        imp = (f"{'불가':>8}" if r["infeasible"]
               else f"{(r['score'] - b) / abs(b):+8.0%}" if b not in (None, 0) else f"{'-':>8}")
        print(f"{r['scenario']:<12}{r['energy_weight']:>6g}{r['condition']:>5}{r['score']:>10.4f}"
              f"{r['energy']:>8.3f}{r['duration_s']:>7.1f}{r['n_phases']:>5}{imp:>9}")
    print()
    print(f"저장 위치    {group_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
