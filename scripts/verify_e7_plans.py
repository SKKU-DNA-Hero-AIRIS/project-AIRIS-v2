"""E7 계획(C의 최적화 결과)을 입자판으로 재평가해 패치판 값과 나란히 본다. 소유자: A.

`docs/e7_reference.json`의 행(계획)을 읽어 `ParticleEvaluator.evaluate_plan`으로 다시 평가하고,
파일에 적힌 패치판 점수·제거율과 함께 CSV/JSON으로 낸다. 핵심 확인은 **조건 사이의 순서**가
두 평가기에서 같은지다 (예: default P5 > P1, pregnant·wheelchair P1 > P5).

사용 예:
    python scripts/verify_e7_plans.py                       # P1·P2·P5, 전 시나리오·전 가중
    python scripts/verify_e7_plans.py --conditions P5 --particles 5000

주의
- 입자판은 제거를 "부스 밖으로 나간 시점"에 센다. 떨어진 뒤 날아 나가는 시간 때문에 단계가
  짧은 계획일수록 닫힌 식보다 낮게 나온다 (비행 시간 지연). P1(1.67 s × 12단계)처럼 단계가
  짧은 조건에 특히 불리하게 보일 수 있다. 단계가 바뀌어도 부유 입자는 상태를 유지하므로
  비행 자체는 단계를 넘어 이어진다: 같은 자세를 여러 단계로 쪼갠 계획이 한 단계 계획과
  입자 하나까지 같다 (`tests/test_particles.py::test_plan_same_pose_split_equals_single_phase`).
- 참조 파일의 번들 도장(물리·노즐 해시, kinetics)이 지금 설정과 다르면 경고만 내고 계속한다.
  해시가 다르면 "패치 점수"는 다른 물리로 뽑힌 값이라 입자판과 나란히 두는 의미가 약해진다.
  판정 결과는 `summary.json`의 `stamp`에 남는다.
- 참조 파일의 단계 시간은 소수 둘째 자리로 반올림돼 있어 합이 `plan.duration_s`와 조금
  어긋난다 (P1에서 최대 0.04 s). 기본값은 마지막 단계를 늘려/줄여 총 시간을 맞춘다
  (`--no-match-duration`이면 그대로 쓴다). 보정량은 출력에 남는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airis.optimize.plan_encoding import plan_physics_cfg            # noqa: E402
from airis.sim import BodyParams, PoseParams                         # noqa: E402
from airis.sim.human_mesh import MESH_DEFAULT_BODY                   # noqa: E402
from airis.sim.particles import ParticleEvaluator                    # noqa: E402
from airis.sim.scenario import load_nozzles, load_scenarios          # noqa: E402
from airis.sim.types import Phase, Plan                              # noqa: E402

REFERENCE = ROOT / "docs" / "e7_reference.json"


def load_reference(path: Path = REFERENCE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


#: 참조 번들의 도장 키 -> `airis.model.predict`의 도장 키 (번들은 `nozzle_hash`로 적는다)
BUNDLE_STAMP_KEYS = {"physics_hash": "physics_hash", "nozzle_hash": "nozzle_layout_hash",
                     "kinetics_enabled": "kinetics_enabled",
                     "time_constant_s": "time_constant_s"}


def stamp_report(bundle: dict) -> dict:
    """참조 번들의 설정 도장 vs 지금 설정.

    반환 ``{"bundle": {...}, "current": {...} | None, "current_error": str | None,
    "mismatched": [키 ...]}``. 지금 설정을 못 읽으면 `current`가 None이고 비교하지 않는다
    (판정 실패가 재평가를 막지는 않는다). 키 이름과 비교 규약은 `airis.model.predict`를 따른다.
    """
    from airis.model.predict import current_stamp, stamp_mismatch

    meta = {dst: bundle[src] for src, dst in BUNDLE_STAMP_KEYS.items() if src in bundle}
    try:
        now = current_stamp()
    except Exception as exc:                        # 설정을 못 읽으면 비교만 건너뛴다
        return {"bundle": meta, "current": None, "current_error": str(exc), "mismatched": []}
    return {"bundle": meta, "current": now, "current_error": None,
            "mismatched": stamp_mismatch(meta, now)}


def build_plan(row: dict, zone_names: list[str], match_duration: bool = True) -> tuple[Plan, float]:
    """참조 파일의 한 행 -> (Plan, 마지막 단계 보정량 s)."""
    raw = row["plan"]
    durations = [float(ph["duration_s"]) for ph in raw["phases"]]
    fix = 0.0
    if match_duration:
        fix = float(raw["duration_s"]) - sum(durations)
        durations[-1] += fix
    phases = [Phase(pose=PoseParams(**{k: float(v) for k, v in ph["pose"].items()}),
                    duration_s=d)
              for ph, d in zip(raw["phases"], durations)]
    zones = np.array([float(raw["zone_strengths"][z]) for z in zone_names], dtype=np.float64)
    return Plan(phases=phases, zone_strengths=zones), fix


def evaluate_rows(rows: list[dict], zone_names: list[str], body: BodyParams, particles: int,
                  match_duration: bool = True, arch: str = "gpu",
                  progress=None) -> list[dict]:
    cfg = plan_physics_cfg()
    scenarios = load_scenarios()
    nozzle = load_nozzles()
    ev = ParticleEvaluator(cfg, arch=arch, max_candidates=1, particles_per_candidate=particles)
    out = []
    try:
        for i, row in enumerate(rows):
            plan, fix = build_plan(row, zone_names, match_duration)
            result = ev.evaluate_plan(plan, nozzle, body, scenarios[row["scenario"]])
            out.append({
                "scenario": row["scenario"], "energy_weight": row["energy_weight"],
                "condition": row["condition"], "n_phases": len(plan.phases),
                "duration_s": plan.duration_s, "duration_fix_s": fix,
                "patch_score": row["score"], "particle_score": result.score,
                "patch_total": row["total_removal"], "particle_total": result.total_removal,
                "patch_energy": row["energy"], "particle_energy": result.extra["energy"],
                "discomfort": result.discomfort,
            })
            if progress is not None:
                progress(i + 1, len(rows))
    finally:
        ev.destroy()
    return out


def order_checks(rows: list[dict]) -> list[dict]:
    """시나리오·가중마다 조건 쌍의 대소가 두 평가기에서 같은지."""
    checks = []
    by_key: dict[tuple[str, float], dict[str, dict]] = {}
    for r in rows:
        by_key.setdefault((r["scenario"], r["energy_weight"]), {})[r["condition"]] = r
    for (scenario, w), got in sorted(by_key.items()):
        conditions = sorted(got)
        for i, a in enumerate(conditions):
            for b in conditions[i + 1:]:
                patch = got[a]["patch_score"] - got[b]["patch_score"]
                particle = got[a]["particle_score"] - got[b]["particle_score"]
                checks.append({"scenario": scenario, "energy_weight": w, "pair": f"{a} vs {b}",
                               "patch_diff": patch, "particle_diff": particle,
                               "same_order": bool(np.sign(patch) == np.sign(particle))})
    return checks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E7 계획을 입자판으로 재평가")
    ap.add_argument("--reference", default=str(REFERENCE))
    ap.add_argument("--conditions", nargs="*", default=["P1", "P2", "P5"])
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--energy-weights", nargs="*", type=float, default=None)
    ap.add_argument("--particles", type=int, default=20000)
    ap.add_argument("--no-match-duration", action="store_true",
                    help="단계 시간 반올림을 보정하지 않는다")
    ap.add_argument("--arch", default="gpu")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "e7_particle"))
    args = ap.parse_args(argv)

    ref = load_reference(Path(args.reference))
    rows = [r for r in ref["rows"]
            if r["condition"] in args.conditions
            and (args.scenarios is None or r["scenario"] in args.scenarios)
            and (args.energy_weights is None or r["energy_weight"] in args.energy_weights)]
    if not rows:
        print("고른 행이 없다", file=sys.stderr)
        return 2
    print(f"행 {len(rows)}개, 입자 {args.particles}, 묶음 {ref['bundle']['group_id']}", flush=True)

    stamp = stamp_report(ref["bundle"])
    if stamp["current_error"] is not None:
        print(f"  경고: 지금 설정 도장을 못 읽어 참조 번들과 비교하지 않는다 "
              f"({stamp['current_error']})", file=sys.stderr, flush=True)
    elif stamp["mismatched"]:
        diff = ", ".join(f"{k} {stamp['bundle'][k]} -> {stamp['current'][k]}"
                         for k in stamp["mismatched"])
        print(f"  경고: 참조 번들과 설정이 다르다 ({diff}). 파일의 패치 점수는 다른 물리로 뽑힌 "
              f"값이라 입자판 값과 나란히 비교하는 의미가 약하다.", file=sys.stderr, flush=True)
    else:
        print(f"  설정 도장 일치 (physics {stamp['bundle'].get('physics_hash')}, "
              f"nozzle {stamp['bundle'].get('nozzle_layout_hash')}, "
              f"T_r {stamp['bundle'].get('time_constant_s')} s)", flush=True)

    def progress(done: int, total: int) -> None:
        if done % 5 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    got = evaluate_rows(rows, ref["zone_names"], MESH_DEFAULT_BODY, args.particles,
                        match_duration=not args.no_match_duration, arch=args.arch,
                        progress=progress)
    checks = order_checks(got)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(got[0]))
        writer.writeheader()
        writer.writerows(got)
    (out / "summary.json").write_text(json.dumps(
        {"when": datetime.now().isoformat(timespec="seconds"), "args": vars(args),
         "bundle": ref["bundle"], "stamp": stamp, "rows": got, "order_checks": checks,
         "order_agreement": (float(np.mean([c["same_order"] for c in checks]))
                             if checks else None)},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'시나리오':<11}{'w':>6}{'조건':>5}{'단계':>4}{'시간':>7}"
          f"{'패치 점수':>10}{'입자 점수':>10}{'패치 R':>9}{'입자 R':>9}")
    for r in got:
        print(f"{r['scenario']:<11}{r['energy_weight']:>6}{r['condition']:>5}{r['n_phases']:>4}"
              f"{r['duration_s']:>7.1f}{r['patch_score']:>10.3f}{r['particle_score']:>10.3f}"
              f"{r['patch_total']:>9.3f}{r['particle_total']:>9.3f}")
    bad = [c for c in checks if not c["same_order"]]
    print(f"\n조건 쌍 순서 일치: {len(checks) - len(bad)}/{len(checks)}")
    for c in bad:
        print(f"  어긋남 {c['scenario']} w={c['energy_weight']} {c['pair']}: "
              f"패치 {c['patch_diff']:+.3f} vs 입자 {c['particle_diff']:+.3f}")
    print(f"출력: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
