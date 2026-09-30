"""계획(Plan) 평가의 패치판 vs 입자판 순위 일치도. 소유자: A.

`scripts/compare_evaluators.py`(D, 단일 자세 E2)의 계획판이다. 시나리오 범위 안에서 계획을
무작위로 뽑아 두 평가기로 평가하고 Spearman 순위 상관을 본다.
`docs/plan_extension.md` 6절 "검증: 계획 50개를 입자판으로 재평가(패치판 대비 순위 상관)".

사용 예:
    python scripts/compare_plan_evaluators.py --n 50
    python scripts/compare_plan_evaluators.py --n 20 --particles 5000 --scenario wheelchair

출력 (`--out`, 기본 outputs/plan_e2/):
    samples.csv   계획마다 두 평가기의 점수·제거율·에너지·시간
    summary.json  실행 설정과 Spearman 상관 (score, weighted_removal, total, 부위별)

주의
- 계획 경로 설정은 `airis.optimize.plan_encoding.plan_physics_cfg()`다 (4.6 시간 항 켬).
- 부스 밖 계획은 샘플링에서 버린다 (두 평가기가 같은 기하 벌점을 받아 비교에 정보가 없다).
- 입자판은 제거를 "부스 밖으로 나간 시점"에 세므로 짧은 계획일수록 닫힌 식보다 낮게 나온다
  (비행 시간 지연). 절대값이 아니라 순위를 본다.
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

from scipy.stats import spearmanr                                    # noqa: E402

from airis.optimize.plan_encoding import plan_physics_cfg            # noqa: E402
from airis.sim import PART_NAMES, BodyParams, PoseParams             # noqa: E402
from airis.sim.body import build_body                                # noqa: E402
from airis.sim.human_mesh import MESH_DEFAULT_BODY                   # noqa: E402
from airis.sim.particles import ParticleEvaluator                    # noqa: E402
from airis.sim.patch_baseline import PatchEvaluator, outside_booth   # noqa: E402
from airis.sim.scenario import load_nozzle_layout, load_nozzles, load_scenarios  # noqa: E402
from airis.sim.types import Phase, Plan                              # noqa: E402


def sample_plans(n: int, scenario, body: BodyParams, seed: int = 0,
                 max_phases: int = 2, duration_range: tuple[float, float] = (5.0, 20.0),
                 zero_zone_prob: float = 0.3) -> list[Plan]:
    """부스 안에 드는 계획 n개. 자세는 시나리오 범위 균등, 구역 세기는 0~1 균등."""
    booth = load_nozzle_layout()["booth"]
    rng = np.random.default_rng(seed)
    bounds = scenario.pose_bounds
    n_zones = 5
    plans: list[Plan] = []
    while len(plans) < n:
        k = int(rng.integers(1, max_phases + 1))
        total = float(rng.uniform(*duration_range))
        fracs = [1.0] if k == 1 else list(np.diff([0.0, float(rng.uniform(0.3, 0.7)), 1.0]))
        phases, feasible = [], True
        for frac in fracs:
            pose = PoseParams(**{key: float(rng.uniform(lo, hi))
                                 for key, (lo, hi) in bounds.items()})
            if outside_booth(build_body(body, pose, scenario).patch_pos, booth):
                feasible = False
                break
            phases.append(Phase(pose=pose, duration_s=total * frac))
        if not feasible:
            continue
        zones = rng.uniform(0.0, 1.0, n_zones)
        if rng.random() < zero_zone_prob:
            zones[int(rng.integers(0, n_zones))] = 0.0
        plans.append(Plan(phases=phases, zone_strengths=zones))
    return plans


def compare(plans: list[Plan], scenario, body: BodyParams, cfg: dict, nozzle,
            particles: int, arch: str = "gpu", progress=None) -> list[dict]:
    """계획마다 두 평가기 결과 한 행. 입자판은 계획 하나씩 돈다 (정확성 우선)."""
    weights = np.array([float(cfg["scoring"]["part_weights"].get(p, 0.0)) for p in PART_NAMES])
    patch = PatchEvaluator(cfg)
    particle = ParticleEvaluator(cfg, arch=arch, max_candidates=1,
                                 particles_per_candidate=particles)
    rows = []
    try:
        for i, plan in enumerate(plans):
            a = particle.evaluate_plan(plan, nozzle, body, scenario)
            b = patch.evaluate_plan(plan, nozzle, body, scenario)
            row = {"plan": i, "n_phases": len(plan.phases), "duration_s": plan.duration_s,
                   "energy": float(a.extra["energy"]),
                   "particle_score": a.score, "patch_score": b.score,
                   "particle_weighted": float(weights @ a.removal_by_part),
                   "patch_weighted": float(weights @ b.removal_by_part),
                   "particle_total": a.total_removal, "patch_total": b.total_removal}
            for j, part in enumerate(PART_NAMES):
                row[f"particle_{part}"] = float(a.removal_by_part[j])
                row[f"patch_{part}"] = float(b.removal_by_part[j])
            rows.append(row)
            if progress is not None:
                progress(i + 1, len(plans))
    finally:
        particle.destroy()
    return rows


def correlations(rows: list[dict]) -> dict[str, float]:
    def rho(a: str, b: str) -> float:
        return float(spearmanr([r[a] for r in rows], [r[b] for r in rows]).statistic)

    out = {"score": rho("particle_score", "patch_score"),
           "weighted_removal": rho("particle_weighted", "patch_weighted"),
           "total": rho("particle_total", "patch_total")}
    out.update({p: rho(f"particle_{p}", f"patch_{p}") for p in PART_NAMES})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="계획 평가 패치판 vs 입자판 순위 비교")
    ap.add_argument("--n", type=int, default=50, help="비교할 계획 수 (부스 안 계획만 센다)")
    ap.add_argument("--scenario", default="default")
    ap.add_argument("--seed", type=int, default=0, help="계획 샘플링 시드")
    ap.add_argument("--particles", type=int, default=20000)
    ap.add_argument("--max-phases", type=int, default=2)
    ap.add_argument("--duration", type=float, nargs=2, default=(5.0, 20.0), metavar=("MIN", "MAX"))
    ap.add_argument("--arch", default="gpu")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "plan_e2"))
    args = ap.parse_args(argv)

    cfg = plan_physics_cfg()
    scenario = load_scenarios()[args.scenario]
    nozzle = load_nozzles()
    body = MESH_DEFAULT_BODY
    plans = sample_plans(args.n, scenario, body, seed=args.seed,
                         max_phases=args.max_phases, duration_range=tuple(args.duration))
    print(f"계획 {len(plans)}개, 시나리오 {args.scenario}, 입자 {args.particles}, "
          f"kinetics {cfg['adhesion']['kinetics']['enabled']}", flush=True)

    def progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    rows = compare(plans, scenario, body, cfg, nozzle, args.particles, args.arch, progress)
    rho = correlations(rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(
        {"when": datetime.now().isoformat(timespec="seconds"), "args": vars(args),
         "spearman": rho,
         "particle_total_mean": float(np.mean([r["particle_total"] for r in rows])),
         "patch_total_mean": float(np.mean([r["patch_total"] for r in rows]))},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSpearman rho (patch vs particle)")
    for name, value in rho.items():
        print(f"  {name:<18} {value:.3f}")
    print(f"출력: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
