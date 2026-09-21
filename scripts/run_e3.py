"""E3 민감도 러너: 설정 하나를 바꿔 E4 축소판(시드 2개)을 돌리고 최적 자세가 유지되는지 본다. 소유자: C.

사용 예:
    python scripts/run_e3.py --preset core                       # 총괄이 정한 기본 스윕 (sensitivity.CORE_PRESET)
    python scripts/run_e3.py --param adhesion.critical_shear_pa_median --factors 0.5 0.75 1.5 2
    python scripts/run_e3.py --param jet.impingement.enabled --values false
    python scripts/run_e3.py --param scenario.discomfort_weights.shoulder_abduction --factors 0.5 2

configs/ 파일은 건드리지 않는다. load_physics() 결과 dict 와 Scenario 를 복사해 바꿔 평가기에 넘긴다.

지표 (시나리오 × 설정값마다):
    best_mean / best_std       시드별 best 의 평균·표준편차
    arm_counts                 best 자세의 팔 봉우리 (hands_up: 벌림 ≥ 90°, arms_down)
    same_peak_frac             기준 설정 best 와 같은 봉우리(팔 봉우리 같고 접은 yaw 차 ≤ 30°)인 시드 비율
    peak_gap_mean              hands_up 시작점 best − default 시작점 best
    topk_spearman              기준 설정 상위 k 후보(서로 다른 자세)를 새 설정으로 다시 평가한 점수와
                               기준 점수의 Spearman 순위 상관
    regret                     1 − score(기준 best 자세 | 새 설정) / 새 설정 best
결과:
    outputs/<exp_id>/                 실행마다 (tag=e3)
    outputs/<group_id>/e3_summary.csv 위 지표
    outputs/<group_id>/e3_runs.csv    실행별 best·자세·시작점별 best
    outputs/<group_id>/top_candidates.json  기준 상위 k 후보와 설정별 재평가 점수
docs/tracks/C_optimize.md 단계 8.
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

import numpy as np                                            # noqa: E402

from airis.optimize import cli, e4, explog, sensitivity       # noqa: E402
from airis.optimize.cmaes_runner import run_cmaes             # noqa: E402
from airis.optimize.encoding import PoseEncoder               # noqa: E402
from airis.sim import BodyParams, PoseParams                  # noqa: E402
from airis.sim.scenario import load_physics, load_scenarios   # noqa: E402

REF = "(기준)"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E3: 설정 스윕에 대한 최적 자세 민감도")
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--preset", choices=["core"], help="기본 스윕 묶음")
    which.add_argument("--param", help="바꿀 설정 키 (physics 점 경로 또는 scenario.<필드>.<키>)")
    how = ap.add_mutually_exclusive_group()
    how.add_argument("--factors", nargs="+", type=float, help="기준값에 곱할 배율")
    how.add_argument("--values", nargs="+", help="그대로 넣을 값 (true/false 는 bool, 숫자는 float)")
    ap.add_argument("--evaluator", choices=cli.EVALUATOR_CHOICES, default="patch")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--max-evals", type=int, default=3000)
    ap.add_argument("--popsize", type=int, default=100)
    ap.add_argument("--sigma0", type=float, default=0.5)
    ap.add_argument("--starts", default=cli.DEFAULT_STARTS)
    ap.add_argument("--tol-stagnation-gens", type=int, default=30)
    ap.add_argument("--patches-per-m2", type=float, default=cli.DEFAULT_PATCHES_PER_M2)
    ap.add_argument("--top-k", type=int, default=10, help="순위 상관에 쓸 기준 상위 후보 수")
    ap.add_argument("--min-dist", type=float, default=0.15,
                    help="상위 후보끼리 최소 정규화 거리 (같은 자세 중복 방지)")
    ap.add_argument("--body", default=None)
    ap.add_argument("--tag", default="e3")
    ap.add_argument("--log-dir", default=str(ROOT / "outputs"))
    return ap


def _parse_value(raw: str):
    low = raw.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    return float(raw)


def _settings(args) -> list[tuple[str, str, list]]:
    if args.preset == "core":
        return list(sensitivity.CORE_PRESET)
    if args.factors:
        return [(args.param, "factor", list(args.factors))]
    if args.values:
        return [(args.param, "value", [_parse_value(v) for v in args.values])]
    raise SystemExit("--param 에는 --factors 나 --values 가 필요하다")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:                                   # 행마다 열이 조금 다를 수 있다
        fields += [k for k in row if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)
    settings = _settings(args)

    all_scenarios = load_scenarios()
    names = args.scenarios or list(all_scenarios)
    missing = [n for n in names if n not in all_scenarios]
    if missing:
        print(f"없는 시나리오: {missing}", file=sys.stderr)
        return 2
    base_cfg = load_physics()
    for key, _, _ in settings:                         # 오타는 돌리기 전에 잡는다
        try:
            for n in names:
                sensitivity.get_setting(base_cfg, all_scenarios[n], key)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    body = cli.dataclass_from_json(BodyParams, args.body)
    starts = cli.parse_starts(args.starts)
    nozzle, nozzle_source = cli.resolve_nozzles()
    nozzle_hash = cli.nozzle_hash(nozzle)
    physics_hash = explog.file_hash(ROOT / "configs" / "physics.yaml")
    commit = explog.git_commit()
    group_id = explog.new_exp_id(args.tag)
    group_dir = Path(args.log_dir) / group_id
    print(f"[{group_id}] evaluator={args.evaluator} scenarios={names} seeds={args.seeds} "
          f"settings={[(k, m, v) for k, m, v in settings]}")

    summary_rows: list[dict] = []
    run_rows: list[dict] = []
    tops_out: dict = {"group_id": group_id, "commit": commit, "physics_hash": physics_hash,
                      "nozzle_hash": nozzle_hash, "args": vars(args), "scenarios": {}}

    def run_setting(scenario, cfg, key, value, record):
        try:
            evaluator = cli.make_evaluator(args.evaluator, scenario, body=body, nozzle=nozzle,
                                           patches_per_m2=args.patches_per_m2, physics_cfg=cfg)
        except cli.TrackNotMerged as exc:
            raise SystemExit(str(exc)) from exc
        runs = []
        for seed in args.seeds:
            exp_id = explog.new_exp_id(args.tag)
            result = run_cmaes(
                evaluator, body, scenario, nozzle,
                max_evals=args.max_evals, seed=seed, popsize=args.popsize, sigma0=args.sigma0,
                tol_stagnation_gens=args.tol_stagnation_gens, starts=starts,
                record_candidates=record, log_dir=args.log_dir, exp_id=exp_id,
            )
            explog.write_run(
                args.log_dir, exp_id, scenario=scenario, body=body, nozzle_hash=nozzle_hash,
                physics_hash=physics_hash, commit=commit, seed=seed, result=result,
                args={**vars(args), "e3_group": group_id, "scenario": scenario.name, "seed": seed,
                      "setting_key": key, "setting_value": value, "nozzle_source": nozzle_source,
                      "stop_reason": result.stop_reason, "elapsed_s": round(result.elapsed_s, 3)},
            )
            runs.append((exp_id, seed, result))
            print(f"  {scenario.name:<11} {key} = {value!s:<8} seed {seed}  best {result.best_score:.4f}  "
                  f"{sensitivity.arm_class(result.best_pose):<9} "
                  f"yaw* {e4.fold_yaw(result.best_pose.torso_yaw):5.1f}  {result.elapsed_s:6.1f} s  [{exp_id}]")
        return evaluator, runs

    for name in names:
        scen0 = all_scenarios[name]
        enc = PoseEncoder(scen0)
        ev0, ref_runs = run_setting(scen0, base_cfg, REF, "", True)
        pool = [c for _, _, r in ref_runs for c in r.candidates]
        top = sensitivity.diverse_top(pool, args.top_k, args.min_dist)
        top_poses = [enc.decode(c["x"]) for c in top]
        top_ref_scores = [c["score"] for c in top]
        ref_best = max((r for _, _, r in ref_runs), key=lambda r: r.best_score)
        tops_out["scenarios"][name] = {
            "ref_best_pose": cli.pose_dict(ref_best.best_pose),
            "top": [{"pose": cli.pose_dict(p), "ref_score": s, "start": c["start"]}
                    for p, s, c in zip(top_poses, top_ref_scores, top)],
            "rescored": {},
        }

        combos = [(REF, "", base_cfg, scen0, ev0, ref_runs)]
        for key, mode, vals in settings:
            for v in vals:
                value = sensitivity.resolve_value(base_cfg, scen0, key, mode, v)
                cfg1, scen1 = sensitivity.apply_setting(base_cfg, scen0, key, value)
                label = f"x{v:g}" if mode == "factor" else str(value)
                ev1, runs1 = run_setting(scen1, cfg1, key, value, False)
                combos.append((key, label, cfg1, scen1, ev1, runs1))

        for key, label, _cfg, scen, ev, runs in combos:
            results = [r for _, _, r in runs]
            scores = np.array([r.best_score for r in results])
            new_best = float(scores.max())
            if key == REF:
                top_new = list(top_ref_scores)
                ref_under_new = ref_best.best_score
            else:
                top_new = [float(ev.evaluate(p, nozzle, body, scen).score) for p in top_poses]
                ref_under_new = float(ev.evaluate(ref_best.best_pose, nozzle, body, scen).score)
            tops_out["scenarios"][name]["rescored"][f"{key} {label}"] = top_new
            arms = [sensitivity.arm_class(r.best_pose) for r in results]
            gaps = [e4.peak_gap(r.per_start) for r in results]
            folded = [e4.fold_pose(r.best_pose) for r in results]
            row = {
                "scenario": name, "setting": key, "value": label,
                "n_seeds": len(results),
                "best_mean": float(scores.mean()),
                "best_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
                "arm_counts": ";".join(f"{a}:{arms.count(a)}" for a in sorted(set(arms))),
                "same_peak_frac": float(np.mean([sensitivity.same_peak(r.best_pose, ref_best.best_pose)
                                                 for r in results])),
                "peak_gap_mean": float(np.nanmean(gaps)) if np.isfinite(gaps).any() else float("nan"),
                "topk_n": len(top_poses),
                "topk_spearman": sensitivity.spearman(top_ref_scores, top_new),
                "ref_pose_score": ref_under_new,
                "regret": sensitivity.regret(ref_under_new, new_best),
            }
            for k in e4.POSE_KEYS:
                row[f"pose_mean_{k}"] = float(np.mean([f[k] for f in folded]))
            summary_rows.append(row)
            for exp_id, seed, r in runs:
                run_rows.append({
                    "scenario": name, "setting": key, "value": label, "seed": seed, "exp_id": exp_id,
                    "best_score": r.best_score, "arm": sensitivity.arm_class(r.best_pose),
                    "peak_gap": e4.peak_gap(r.per_start), "elapsed_s": r.elapsed_s,
                    **{f"start_best_{ps['start']}": ps["best_score"] for ps in r.per_start},
                    **{f"pose_{k}": v for k, v in cli.pose_dict(r.best_pose).items()},
                })

    group_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(group_dir / "e3_summary.csv", summary_rows)
    _write_csv(group_dir / "e3_runs.csv", run_rows)
    (group_dir / "top_candidates.json").write_text(
        json.dumps(explog._jsonable(tops_out), ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"{'시나리오':<11}{'설정':<48}{'값':>8}{'best':>9}{'팔 봉우리':>24}{'같은봉우리':>8}"
          f"{'봉우리차':>9}{'ρ top-k':>9}{'regret':>8}")
    for row in summary_rows:
        print(f"{row['scenario']:<11}{row['setting']:<48}{row['value']:>8}{row['best_mean']:>9.4f}"
              f"{row['arm_counts']:>24}{row['same_peak_frac']:>8.0%}{row['peak_gap_mean']:>+9.4f}"
              f"{row['topk_spearman']:>9.3f}{row['regret']:>+8.1%}")
    print()
    print(f"저장 위치    {group_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
