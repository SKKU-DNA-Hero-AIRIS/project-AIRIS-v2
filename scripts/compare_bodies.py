"""E5: 캡슐 몸 vs 메시 몸 패치판 점수 비교 (사람 메시 전환 검증). 소유자: D.

시나리오마다 두 몸 모두에서 부스 안인 자세를 균등 샘플링해, 같은 체형(`human_mesh.MESH_DEFAULT_BODY`)
으로 캡슐판(`build_body(model="capsule")`)과 메시판(`model="mesh"`)의 `PatchEvaluator` 점수를 비교한다.
통과 기준은 없고 차이를 기록하는 것이 목적이다 (`docs/mesh_transition.md` 결정 9, D 문서 단계 8).

사용 예:
    python scripts/compare_bodies.py --n 300
    python scripts/compare_bodies.py --n 300 --scenarios default wheelchair --patches-per-m2 400
    python scripts/compare_bodies.py --n 20 --patches-per-m2 400          # 동작 확인용

출력 (outputs/e5/ 아래, --out 으로 변경):
    <scenario>/samples.csv, summary.json, scatter.png, top_mismatch.md / .csv   (E2와 같은 형식)
    summary.md    시나리오별 Spearman(score·weighted_removal·total·부위별)과 제거율 범위 표
                  (docs/experiments.md 에 옮길 요약. 그 문서는 C 소유라 여기서는 쓰지 않는다)

체형: 전역 `BodyParams()`는 캡슐 시절 값이라 메시에 넣으면 팔이 길어진다. 두 몸 모두
`MESH_DEFAULT_BODY`(--body 로 덮어쓰기)를 넣어 체형 차이가 아니라 몸 모델 차이만 보이게 한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields, replace
from datetime import datetime
from functools import partial
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airis.optimize import cli                                  # noqa: E402
from airis.sim import PART_NAMES, BodyParams                    # noqa: E402
from airis.sim.body import build_body                           # noqa: E402
from airis.sim.human_mesh import MESH_DEFAULT_BODY              # noqa: E402
from airis.sim.patch_baseline import (                          # noqa: E402
    SLOT_OCCLUSION_POINTS, PatchEvaluator, outside_booth,
)
from airis.sim.scenario import load_physics, load_scenarios     # noqa: E402
from scripts.compare_evaluators import (                        # noqa: E402
    _set_korean_font, apply_overrides, build_table, correlations, mismatch_order, plot_scatter,
    sample_poses, write_csv, write_mismatch_md,
)

MODELS = ("capsule", "mesh")
METRICS = ["score", "weighted_removal", "total"] + PART_NAMES


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E5 캡슐 몸 vs 메시 몸 패치판 비교")
    ap.add_argument("--n", type=int, default=300, help="시나리오당 자세 수 (두 몸 모두 부스 안인 것만)")
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="비교할 시나리오. 생략 시 configs/scenarios.yaml 전부")
    ap.add_argument("--seed", type=int, default=0, help="자세 샘플링 시드")
    ap.add_argument("--body", default=None,
                    help='MESH_DEFAULT_BODY 덮어쓰기 JSON, 예: \'{"height_m":1.6}\'. 두 몸에 같게 넣는다')
    ap.add_argument("--patches-per-m2", type=float, default=None,
                    help="build_body 패치 밀도. 생략 시 build_body 기본값. 두 몸에 같게 적용")
    ap.add_argument("--slot-points", type=int, default=SLOT_OCCLUSION_POINTS,
                    help="슬롯 가림 점 수 (occlusion)")
    ap.add_argument("--override", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="physics.yaml 값을 메모리에서만 덮어쓴다. 여러 번 줄 수 있다")
    ap.add_argument("--top", type=int, default=10, help="차이 표에 넣을 자세 수")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "e5"))
    return ap


def _body_from_json(raw: str | None) -> BodyParams:
    """MESH_DEFAULT_BODY에 --body JSON을 덮어쓴다. 없는 키는 오류."""
    if not raw:
        return MESH_DEFAULT_BODY
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--body JSON 파싱 실패: {raw!r} ({exc})") from exc
    known = {f.name for f in fields(BodyParams)}
    if not isinstance(data, dict) or set(data) - known:
        raise SystemExit(f"--body 는 BodyParams 키만: {sorted(known)}")
    return replace(MESH_DEFAULT_BODY, **{k: float(v) for k, v in data.items()})


def evaluators(physics: dict, args) -> dict[str, PatchEvaluator]:
    return {m: PatchEvaluator(physics, build_body=partial(build_body, model=m),
                              patches_per_m2=args.patches_per_m2, slot_points=args.slot_points)
            for m in MODELS}


def run_scenario(name: str, scenario, physics: dict, body: BodyParams, nozzle, args,
                 out: Path) -> dict:
    evs = evaluators(physics, args)

    def feasible(pose) -> bool:
        return all(not outside_booth(ev.build_state(body, pose, scenario).patch_pos, ev.booth)
                   for ev in evs.values())

    poses, rejected = sample_poses(scenario, args.n, args.seed, is_feasible=feasible)
    print(f"E5 {name}: n={args.n}, 부스 밖으로 버린 자세 {rejected}", flush=True)

    results, seconds = {}, {}
    for model, ev in evs.items():
        t0 = time.perf_counter()
        results[model] = [ev.evaluate(p, nozzle, body, scenario) for p in poses]
        seconds[model] = time.perf_counter() - t0
        if any(r.extra.get("infeasible") for r in results[model]):
            raise RuntimeError(f"{model}: 샘플링에서 걸렀는데 부스 밖 자세가 남았다")
        print(f"  {model} 완료 {seconds[model]:.2f} s", flush=True)

    t = build_table(poses, results["capsule"], results["mesh"], physics, names=MODELS)
    rho = correlations(t, names=MODELS)
    order, metric = mismatch_order(t, names=MODELS)

    sub = out / name
    sub.mkdir(parents=True, exist_ok=True)
    write_csv(sub / "samples.csv", t)
    write_csv(sub / "top_mismatch.csv", t, order[:args.top])
    write_mismatch_md(sub / "top_mismatch.md", t, order, metric, args.top, names=MODELS, title=f"E5 {name}")
    plot_scatter(sub / "scatter.png", t, rho, names=MODELS, labels=("캡슐 몸", "메시 몸"),
                 title=f"E5 {name}: 캡슐 몸 vs 메시 몸 (패치판)")

    summary = {
        "scenario": name,
        "n": len(poses),
        "infeasible_rejected": rejected,
        "seconds": seconds,
        "spearman": rho,
        "mismatch_sort_metric": metric,
        "removal": {
            model: {m: {"min": float(t[f"{model}_{m}"].min()), "max": float(t[f"{model}_{m}"].max()),
                        "mean": float(t[f"{model}_{m}"].mean())}
                    for m in ["total"] + PART_NAMES}
            for model in MODELS
        },
    }
    (sub / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _fmt(v) -> str:
    return "–" if v is None else f"{v:.3f}"


def write_summary_md(path: Path, summaries: list[dict], meta: dict) -> None:
    names = [s["scenario"] for s in summaries]
    lines = [f"# E5 캡슐 몸 vs 메시 몸 (패치판, {meta['created']})", "",
             f"조건: 체형 {meta['body']}, 패치 밀도 {meta['patches_per_m2']}, 슬롯 가림 K={meta['slot_points']}, "
             f"노즐 {meta['nozzle_source']}({meta['nozzle_count']}개), 시드 {meta['seed']}, "
             f"상수 덮어쓰기 {meta['overrides'] or '없음'}. 통과 기준 없음 (차이 기록).", "",
             "## Spearman ρ (캡슐 vs 메시)", "",
             "| 지표 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for m in METRICS:
        lines.append(f"| {m} | " + " | ".join(_fmt(s["spearman"][m]) for s in summaries) + " |")
    lines += ["", "## 제거율 평균 (범위)", "",
              "| 시나리오 | 부위 | 캡슐 | 메시 | 메시 − 캡슐 |", "|---|---|---|---|---|"]
    for s in summaries:
        for m in ["total"] + PART_NAMES:
            c, h = s["removal"]["capsule"][m], s["removal"]["mesh"][m]
            lines.append(f"| {s['scenario']} | {m} | {c['mean']:.4f} ({c['min']:.4f}~{c['max']:.4f}) | "
                         f"{h['mean']:.4f} ({h['min']:.4f}~{h['max']:.4f}) | {h['mean'] - c['mean']:+.4f} |")
    lines += ["", "## 표본", "",
              "| 시나리오 | 자세 수 | 부스 밖으로 버린 수 | 캡슐 s | 메시 s |", "|---|---|---|---|---|"]
    for s in summaries:
        lines.append(f"| {s['scenario']} | {s['n']} | {s['infeasible_rejected']} | "
                     f"{s['seconds']['capsule']:.1f} | {s['seconds']['mesh']:.1f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)
    if args.n < 3:
        raise SystemExit("--n 은 3 이상 (순위 상관)")

    all_scenarios = load_scenarios()
    names = args.scenarios or list(all_scenarios)
    missing = [n for n in names if n not in all_scenarios]
    if missing:
        print(f"없는 시나리오: {missing} (가능: {', '.join(all_scenarios)})", file=sys.stderr)
        return 2

    physics = apply_overrides(load_physics(), args.override)
    body = _body_from_json(args.body)
    nozzle, nozzle_source = cli.resolve_nozzles()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _set_korean_font()

    summaries = [run_scenario(n, all_scenarios[n], physics, body, nozzle, args, out) for n in names]

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "body": {f.name: getattr(body, f.name) for f in fields(BodyParams)},
        "patches_per_m2": args.patches_per_m2 or "build_body 기본",
        "slot_points": args.slot_points,
        "nozzle_source": nozzle_source,
        "nozzle_count": nozzle.count,
        "nozzle_hash": cli.nozzle_hash(nozzle),
        "seed": args.seed,
        "overrides": args.override,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_md(out / "summary.md", summaries, meta)

    print("\nSpearman ρ (capsule vs mesh)")
    print(f"  {'':<18}" + "".join(f"{n:>12}" for n in names))
    for m in METRICS:
        print(f"  {m:<18}" + "".join(f"{_fmt(s['spearman'][m]):>12}" for s in summaries))
    print(f"출력: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
