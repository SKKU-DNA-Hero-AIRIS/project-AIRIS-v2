"""입자 덤프 → plotly 애니메이션 HTML. 소유자: E. (`docs/tracks/E_realtime.md` 단계 2)

    # A의 덤프(outputs/<exp_id>/frames/<step>.npz)가 이미 있을 때
    python scripts/render_frames.py --exp <exp_id>

    # A 병합 전·GPU 없이 확인: 합성 프레임(가짜 궤적)을 만들어서 그린다
    python scripts/render_frames.py --exp synth_demo --synthetic

    # A의 ParticleEvaluator 로 덤프를 새로 만든다 (Taichi 초기화는 A 코드가 한다)
    python scripts/render_frames.py --exp particle_demo --simulate --arch cpu --particles 5000 \
        --pose 180,3,5,0,90,1,0

결과는 기본 outputs/<exp_id>/anim.html.

자세·체형·시나리오는 다음 순서로 정한다: `--pose`/`--scenario` 인자 → `outputs/<exp_id>/render_meta.json`
(이 스크립트의 --synthetic/--simulate 가 씀) → C의 `meta.json` + `best.json` → 기본값.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from airis.sim.body import build_body                                 # noqa: E402
from airis.sim.scenario import load_nozzle_layout, load_nozzles, load_physics, load_scenarios  # noqa: E402
from airis.sim.types import BodyParams, PoseParams                    # noqa: E402
from airis.viz import anim                                            # noqa: E402


def _parse_pose(text: str) -> PoseParams:
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 7:
        raise SystemExit("--pose 는 7개 값: 벌림,굽힘,팔꿈치,pitch,yaw,hip,knee")
    return PoseParams(*vals)


def resolve_setup(exp_dir: Path, args) -> tuple[BodyParams, list[PoseParams], str]:
    """(체형, 후보별 자세 목록, 시나리오 이름)."""
    body, poses, scenario = BodyParams(), [PoseParams()], "default"
    meta_path = exp_dir / "render_meta.json"
    c_meta, c_best = exp_dir / "meta.json", exp_dir / "best.json"
    if meta_path.exists():
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        body = BodyParams(**m["body"])
        poses = [PoseParams(**p) for p in m["poses"]]
        scenario = m["scenario"]
    elif c_meta.exists() and c_best.exists():
        m = json.loads(c_meta.read_text(encoding="utf-8"))
        b = json.loads(c_best.read_text(encoding="utf-8"))
        body = BodyParams(**m["body"])
        scenario = m["scenario"]["name"] if isinstance(m["scenario"], dict) else m["scenario"]
        poses = [PoseParams(**b["best_pose"])]
    if args.pose:
        poses = [_parse_pose(args.pose)]
    if args.scenario:
        scenario = args.scenario
    return body, poses, scenario


def write_render_meta(exp_dir: Path, body: BodyParams, poses: list[PoseParams], scenario: str,
                      source: str) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "render_meta.json").write_text(json.dumps({
        "source": source, "scenario": scenario, "body": vars(body),
        "poses": [vars(p) for p in poses]}, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, help="outputs/<exp_id>")
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--out", default=None, help="기본 outputs/<exp_id>/anim.html")
    ap.add_argument("--candidate", type=int, default=0)
    ap.add_argument("--max-points", type=int, default=5000)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--pose", default=None, help="자세 7개 쉼표 구분 (deg)")
    ap.add_argument("--scenario", default=None, choices=["default", "pregnant", "wheelchair"])
    gen = ap.add_mutually_exclusive_group()
    gen.add_argument("--synthetic", action="store_true", help="합성 프레임을 만들어 그린다 (가짜 데이터)")
    gen.add_argument("--simulate", action="store_true", help="A의 ParticleEvaluator 로 덤프를 만든다")
    ap.add_argument("--arch", default="cpu", help="--simulate 의 Taichi arch (cpu | gpu)")
    ap.add_argument("--particles", type=int, default=5000, help="--simulate/--synthetic 입자 수")
    ap.add_argument("--dump-every", type=int, default=25, help="--simulate 덤프 간격 (스텝)")
    ap.add_argument("--duration", type=float, default=None, help="--simulate 시뮬레이션 길이 (s)")
    args = ap.parse_args(argv)

    exp_dir = Path(args.outputs) / args.exp
    body, poses, scenario_name = resolve_setup(exp_dir, args)
    scenarios = load_scenarios()
    scenario = scenarios[scenario_name]
    booth = load_nozzle_layout()["booth"]

    if args.synthetic:
        state = build_body(body, poses[0], scenario, patches_per_m2=400)
        frames = anim.synthetic_frames(state, booth, n_particles=args.particles)
        anim.write_frames(frames, exp_dir / "frames")
        write_render_meta(exp_dir, body, poses[:1], scenario_name, "synthetic (가짜 궤적)")
        print(f"합성 프레임 {len(frames)}개 → {exp_dir / 'frames'}")
    elif args.simulate:
        from airis.sim.particles import ParticleEvaluator     # A 소유. Taichi 초기화는 여기서 한다
        ev = ParticleEvaluator(load_physics(), arch=args.arch, max_candidates=max(1, len(poses)),
                               particles_per_candidate=args.particles, duration_s=args.duration,
                               dump_every=args.dump_every, dump_dir=exp_dir)
        states = [build_body(body, p, scenario) for p in poses]
        res = ev.batch_evaluate_states(states, poses, load_nozzles(), scenario)
        write_render_meta(exp_dir, body, poses, scenario_name, f"ParticleEvaluator arch={args.arch}")
        for i, r in enumerate(res):
            print(f"후보 {i}: score {r.score:.4f}, total_removal {r.total_removal:.4f}")

    frames = anim.load_frames(exp_dir)
    cand = args.candidate
    pose = poses[cand] if cand < len(poses) else poses[0]
    state = build_body(body, pose, scenario, patches_per_m2=400)
    fig = anim.animation(frames, state, booth, candidate=cand, max_points=args.max_points,
                         max_frames=args.max_frames, nozzle=load_nozzles(),
                         title=f"{args.exp} · {scenario_name} · 후보 {cand}")
    out = Path(args.out) if args.out else exp_dir / "anim.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn", auto_play=False)
    last = anim.status_counts(frames[-1:], booth, cand)[0]
    print(f"프레임 {len(frames)}개 (그림 {len(fig.frames)}개) → {out}")
    print(f"마지막 프레임: 부착 {last['attached']:.1%} · 부유 {last['airborne']:.1%} · 제거 {last['removed']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
