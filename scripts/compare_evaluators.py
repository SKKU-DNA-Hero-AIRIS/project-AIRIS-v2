"""E2: 패치판(PatchEvaluator)과 입자판(ParticleEvaluator)의 순위 일치도. 소유자: D.

시나리오 범위 안에서 자세를 균등 샘플링해 두 평가기로 평가하고 Spearman 순위 상관을 본다.
README 6절 E2, docs/tracks/D_patch_baseline.md 단계 7.

사용 예:
    python scripts/compare_evaluators.py --n 500 --scenario default
    # 개발용 축소 실행
    python scripts/compare_evaluators.py --n 20 --patches-per-m2 500 --particles 1000 --duration 0.5
    # 제트 상수 덮어쓰기 (configs/ 는 건드리지 않고 메모리에서만 바꾼다)
    python scripts/compare_evaluators.py --n 50 --override jet.nozzle_diameter_m=0.08 \
        --override jet.halfwidth_spread_rate=0.2

출력 (outputs/e2/ 아래, --out 으로 변경):
    samples.csv       자세마다 두 평가기의 점수·제거율·순위
    summary.json      실행 설정과 Spearman 상관 (score, weighted_removal, total, 부위별)
    scatter.png       순위 산점도 (score, weighted_removal)
    top_mismatch.md   순위 차이가 가장 큰 자세 N개의 부위별 제거율 표 (csv 도 함께)

부스 밖 자세(00_common.md 5절)는 샘플링 단계에서 버리고 다시 뽑는다. 규칙에 걸린 자세는 두
평가기 모두 같은 기하 벌점(−1 − 10·d_out)을 받아 제거율 비교에 정보가 없고, 입자판을 돌릴
이유도 없다. 버린 수는
summary.json 의 infeasible_rejected 에 남긴다.

주의: 두 평가기의 score 는 같은 불편도 항(−w·discomfort)을 공유한다. 제거율이 둘 다 0에
가까우면 score 순위가 불편도만으로 정해져 상관이 1에 가깝게 부풀려진다. 그래서 불편도를
뺀 weighted_removal(= Σ part_weights·R_부위)의 상관을 함께 보고, 차이 표도 그 순위로 정렬한다.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
import warnings
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.stats import rankdata, spearmanr                     # noqa: E402

from airis.optimize import cli                                  # noqa: E402
from airis.optimize.encoding import PoseEncoder                 # noqa: E402
from airis.sim import PART_NAMES, BodyParams, PoseParams        # noqa: E402
from airis.sim.body import build_body                           # noqa: E402
from airis.sim.patch_baseline import PatchEvaluator, outside_booth  # noqa: E402
from airis.sim.scenario import load_physics, load_scenarios     # noqa: E402

POSE_KEYS = [f.name for f in fields(PoseParams)]
EVALUATORS = ("patch", "particle")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E2 패치판 vs 입자판 순위 상관")
    ap.add_argument("--n", type=int, default=500, help="비교할 자세 수 (부스 안 자세만 센다)")
    ap.add_argument("--scenario", default="default")
    ap.add_argument("--seed", type=int, default=0, help="자세 샘플링 시드")
    ap.add_argument("--body", default=None, help='BodyParams 덮어쓰기 JSON, 예: \'{"height_m":1.6}\'')
    ap.add_argument("--patches-per-m2", type=float, default=None,
                    help="build_body 패치 밀도. 생략 시 build_body 기본값. 두 평가기에 같게 적용")
    ap.add_argument("--particles", type=int, default=None,
                    help="후보당 입자 수. 생략 시 physics.yaml particles.count_per_candidate")
    ap.add_argument("--duration", type=float, default=None,
                    help="입자판 시뮬레이션 시간(s). 생략 시 physics.yaml simulation.duration_s")
    ap.add_argument("--batch", type=int, default=100, help="입자판 한 번에 넣는 후보 수")
    ap.add_argument("--arch", default="gpu", help="Taichi 백엔드 (gpu, cpu, ...)")
    ap.add_argument("--override", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="physics.yaml 값을 메모리에서만 덮어쓴다. 여러 번 줄 수 있다")
    ap.add_argument("--top", type=int, default=10, help="차이 표에 넣을 자세 수")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "e2"))
    return ap


# ---------------------------------------------------------------- 설정
def apply_overrides(physics: dict, overrides: list[str]) -> dict:
    """'jet.nozzle_diameter_m=0.08' 목록을 적용한 사본. 없는 키는 오류로 막는다(오타 방지)."""
    cfg = copy.deepcopy(physics)
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--override 형식은 SECTION.KEY=VALUE: {item!r}")
        path, raw = item.split("=", 1)
        keys = path.strip().split(".")
        node = cfg
        for k in keys[:-1]:
            if not isinstance(node, dict) or k not in node:
                raise SystemExit(f"physics.yaml 에 없는 경로: {path}")
            node = node[k]
        if not isinstance(node, dict) or keys[-1] not in node:
            raise SystemExit(f"physics.yaml 에 없는 경로: {path}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node[keys[-1]] = value
    return cfg


def sample_poses(scenario, n: int, seed: int, is_feasible=None,
                 max_tries_per_pose: int = 50) -> tuple[list[PoseParams], int]:
    """자유 변수를 pose_bounds 안에서 독립 균등 샘플링. fixed_pose 는 PoseEncoder 가 채운다.

    `is_feasible(pose) -> bool` 이 주어지면 False 인 자세는 버리고 n 개가 찰 때까지 더 뽑는다.
    Returns: (자세 n 개, 버린 수).
    """
    enc = PoseEncoder(scenario)
    rng = np.random.default_rng(seed)
    poses: list[PoseParams] = []
    rejected = 0
    while len(poses) < n:
        if rejected > max_tries_per_pose * n:
            raise SystemExit(f"부스 안 자세를 {n}개 채우지 못했다 ({len(poses)}개, 버린 수 {rejected})")
        pose = enc.decode(rng.uniform(-1.0, 1.0, size=enc.dim))
        if is_feasible is None or is_feasible(pose):
            poses.append(pose)
        else:
            rejected += 1
    return poses, rejected


# ---------------------------------------------------------------- 평가
def weighted_removal(removal_by_part: np.ndarray, physics: dict) -> float:
    w = physics["scoring"]["part_weights"]
    return float(sum(w.get(p, 0.0) * float(r) for p, r in zip(PART_NAMES, removal_by_part)))


def run_patch(ev: PatchEvaluator, poses, nozzle, body, scenario):
    results = [ev.evaluate(p, nozzle, body, scenario) for p in poses]
    if any(r.extra.get("infeasible") for r in results):
        raise RuntimeError("샘플링에서 걸렀는데 부스 밖 자세가 남았다")
    return results


def run_particle(poses, nozzle, body, scenario, physics, args):
    # A 외 트랙은 ti.init 을 부르지 않는다. ParticleEvaluator 생성이 처리한다.
    from airis.sim.particles import ParticleEvaluator

    batch = max(1, min(args.batch, len(poses)))
    ev = ParticleEvaluator(physics, arch=args.arch, max_candidates=batch,
                           particles_per_candidate=args.particles, duration_s=args.duration)
    kw = {} if args.patches_per_m2 is None else {"patches_per_m2": args.patches_per_m2}
    results = []
    try:
        for i in range(0, len(poses), batch):
            chunk = poses[i:i + batch]
            states = [build_body(body, p, scenario, **kw) for p in chunk]
            results.extend(ev.batch_evaluate_states(states, chunk, nozzle, scenario))
            print(f"  particle {min(i + batch, len(poses))}/{len(poses)}", flush=True)
    finally:
        ev.destroy()
    return results, ev


# ---------------------------------------------------------------- 분석
def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """상수 배열이면 순위 상관이 정의되지 않으므로 None."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = spearmanr(a, b).statistic
    return None if not np.isfinite(rho) else float(rho)


def normalized_rank(x: np.ndarray) -> np.ndarray:
    """0(최저)~1(최고). 동률은 평균 순위."""
    x = np.asarray(x, float)
    if x.size < 2:
        return np.zeros_like(x)
    return (rankdata(x, method="average") - 1.0) / (x.size - 1.0)


def build_table(poses, res_patch, res_particle, physics) -> dict[str, np.ndarray]:
    t: dict[str, np.ndarray] = {k: np.array([getattr(p, k) for p in poses]) for k in POSE_KEYS}
    t["discomfort"] = np.array([r.discomfort for r in res_patch])
    for name, res in zip(EVALUATORS, (res_patch, res_particle)):
        t[f"{name}_score"] = np.array([r.score for r in res])
        t[f"{name}_weighted_removal"] = np.array([weighted_removal(r.removal_by_part, physics) for r in res])
        t[f"{name}_total"] = np.array([r.total_removal for r in res])
        for j, part in enumerate(PART_NAMES):
            t[f"{name}_{part}"] = np.array([float(r.removal_by_part[j]) for r in res])
    for metric in ("score", "weighted_removal"):
        rp = normalized_rank(t[f"patch_{metric}"])
        rq = normalized_rank(t[f"particle_{metric}"])
        t[f"patch_rank_{metric}"], t[f"particle_rank_{metric}"] = rp, rq
        t[f"rank_diff_{metric}"] = np.abs(rp - rq)
    return t


def correlations(t: dict[str, np.ndarray]) -> dict[str, float | None]:
    metrics = ["score", "weighted_removal", "total"] + PART_NAMES
    return {m: spearman(t[f"patch_{m}"], t[f"particle_{m}"]) for m in metrics}


def mismatch_order(t: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    """weighted_removal 순위 차이 기준. 그게 정의되지 않으면(제거율이 상수) score 로 대신한다."""
    for metric in ("weighted_removal", "score"):
        if np.ptp(t[f"patch_{metric}"]) > 0 and np.ptp(t[f"particle_{metric}"]) > 0:
            return np.argsort(-t[f"rank_diff_{metric}"], kind="stable"), metric
    return np.arange(len(t["discomfort"])), "none"


# ---------------------------------------------------------------- 저장
def write_csv(path: Path, t: dict[str, np.ndarray], rows: np.ndarray | None = None) -> None:
    n = len(t["discomfort"])
    idx = np.arange(n) if rows is None else rows
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", *t.keys()])
        for i in idx:
            w.writerow([int(i), *(f"{float(v[i]):.6g}" for v in t.values())])


def write_mismatch_md(path: Path, t, order: np.ndarray, metric: str, top: int) -> None:
    lines = [f"# E2 순위 차이 상위 {top}개", "",
             f"정렬 기준: `{metric}` 정규화 순위(0=최저, 1=최고)의 차이.", ""]
    for rank, i in enumerate(order[:top], start=1):
        pose = ", ".join(f"{k}={t[k][i]:.1f}" for k in POSE_KEYS)
        lines += [f"## {rank}. idx {int(i)}", "", pose, ""]
        if metric != "none":
            lines += [f"순위 patch {t[f'patch_rank_{metric}'][i]:.3f} / "
                      f"particle {t[f'particle_rank_{metric}'][i]:.3f} "
                      f"(차이 {t[f'rank_diff_{metric}'][i]:.3f})", ""]
        lines += ["| 항목 | patch | particle | particle − patch |", "|---|---:|---:|---:|"]
        for m in PART_NAMES + ["total", "weighted_removal", "score"]:
            a, b = t[f"patch_{m}"][i], t[f"particle_{m}"][i]
            lines.append(f"| {m} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_scatter(path: Path, t, rho: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, mark = "#1f1f1f", "#6b6b6b", "#e6e6e6", "#2a6fdb"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, metric, label in zip(axes, ("weighted_removal", "score"),
                                 ("가중 제거율 (불편도 제외)", "score")):
        x, y = t[f"patch_rank_{metric}"], t[f"particle_rank_{metric}"]
        ax.plot([0, 1], [0, 1], color=muted, lw=1, ls="--", zorder=1)
        ax.scatter(x, y, s=14, color=mark, alpha=0.6, edgecolors="white", linewidths=0.4, zorder=2)
        r = rho.get(metric)
        rtxt = "정의 안 됨 (상수)" if r is None else f"{r:.3f}"
        ax.set_title(f"{label}  ·  Spearman ρ = {rtxt}", color=ink, fontsize=10, loc="left")
        ax.set_xlabel("패치판 순위 (0=최저, 1=최고)", color=muted, fontsize=9)
        ax.set_ylabel("입자판 순위", color=muted, fontsize=9)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
        ax.grid(color=grid, lw=0.8)
        ax.tick_params(colors=muted, labelsize=8)
        for s in ax.spines.values():
            s.set_color(grid)
    fig.suptitle(f"E2 패치판 vs 입자판 (n={len(t['discomfort'])})", color=ink, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _set_korean_font() -> None:
    import matplotlib
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if cand in names:
            matplotlib.rcParams["font.family"] = cand
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)
    if args.n < 1:
        raise SystemExit("--n 은 1 이상")

    scenarios = load_scenarios()
    if args.scenario not in scenarios:
        print(f"없는 시나리오: {args.scenario} (가능: {', '.join(scenarios)})", file=sys.stderr)
        return 2
    scenario = scenarios[args.scenario]
    physics = apply_overrides(load_physics(), args.override)
    body = cli.dataclass_from_json(BodyParams, args.body)
    nozzle, nozzle_source = cli.resolve_nozzles()
    pev_patch = PatchEvaluator(physics, patches_per_m2=args.patches_per_m2)
    poses, rejected = sample_poses(
        scenario, args.n, args.seed,
        is_feasible=lambda p: not outside_booth(
            pev_patch.build_state(body, p, scenario).patch_pos, pev_patch.booth))

    print(f"E2: n={args.n}, scenario={args.scenario}, nozzles={nozzle_source}({nozzle.count}), "
          f"overrides={args.override or '없음'}, 부스 밖으로 버린 자세 {rejected}", flush=True)

    t0 = time.perf_counter()
    res_patch = run_patch(pev_patch, poses, nozzle, body, scenario)
    t_patch = time.perf_counter() - t0
    print(f"  patch 완료 {t_patch:.2f} s", flush=True)

    t0 = time.perf_counter()
    res_particle, pev = run_particle(poses, nozzle, body, scenario, physics, args)
    t_particle = time.perf_counter() - t0
    print(f"  particle 완료 {t_particle:.2f} s", flush=True)

    t = build_table(poses, res_patch, res_particle, physics)
    rho = correlations(t)
    order, metric = mismatch_order(t)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "samples.csv", t)
    write_csv(out / "top_mismatch.csv", t, order[:args.top])
    write_mismatch_md(out / "top_mismatch.md", t, order, metric, args.top)
    _set_korean_font()
    plot_scatter(out / "scatter.png", t, rho)

    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "nozzle_source": nozzle_source,
        "booth": pev_patch.booth,
        "infeasible_rejected": rejected,
        "nozzle_hash": cli.nozzle_hash(nozzle),
        "body": {f.name: getattr(body, f.name) for f in fields(BodyParams)},
        "particle": {"particles_per_candidate": pev.N, "duration_s": pev.duration_s,
                     "dt_s": pev.dt, "arch": str(pev.arch)},
        "jet": physics["jet"],
        "seconds": {"patch": t_patch, "particle": t_particle},
        "spearman": rho,
        "mismatch_sort_metric": metric,
        "removal_range": {
            name: {"total_min": float(t[f"{name}_total"].min()),
                   "total_max": float(t[f"{name}_total"].max())}
            for name in EVALUATORS
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                      encoding="utf-8")

    print("\nSpearman ρ (patch vs particle)")
    for k, v in rho.items():
        print(f"  {k:<18} {'정의 안 됨' if v is None else f'{v:.3f}'}")
    for name in EVALUATORS:
        rr = summary["removal_range"][name]
        print(f"  {name} total_removal 범위 [{rr['total_min']:.4g}, {rr['total_max']:.4g}]")
    if rho["weighted_removal"] is None:
        print("경고: 한쪽 이상의 제거율이 모든 자세에서 같다. score 상관은 불편도 항만 반영한다.")
    print(f"출력: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
