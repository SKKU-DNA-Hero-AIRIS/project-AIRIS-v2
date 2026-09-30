"""E4 묶음 → `docs/e4_reference.json` (다른 트랙이 읽는 기준 수치 한 장). 소유자: C.

E·F·A 가 스텁 자세, 회귀 테스트 기대값, 검증 대상 자세를 쓸 때 사람이 수치를 옮겨 적다가
틀리는 일을 없애려고 만든다 (2026-10-01 까지 그런 오기가 세 번 있었다). 실행 결과 파일만
읽어 만들고, 이 스크립트가 만든 JSON 이 유일한 출처다.

    python scripts/export_e4_reference.py \
        --bundle outputs/e4k14_.../ --bundle ../airis-v2-e4bound/outputs/e4down2_.../ \
        --out docs/e4_reference.json

묶음마다 `best_poses.json` 과 `e4_summary.csv` 를 읽는다. 시나리오별로

    baselines   B0·B1·B2 (탐색 밀도와 재채점 밀도를 따로)
    peaks       묶음마다 시드별 best 자세 (raw·folded, 두 밀도 점수, arm_class, 상한에 닿은 변수)

를 남긴다. `--rescore` 를 주면 접은 자세(folded)를 재채점 밀도로 한 번 더 평가해
`score_folded` 를 채운다. 좌우 거울(θ ≡ −θ)은 메시에서도 정확하지만 앞뒤 등가
(θ ≡ 180 − θ)는 근사라(상대 2e−4) raw 와 folded 점수가 다를 수 있다.
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

from airis.optimize import cli, e4, sensitivity              # noqa: E402
from airis.sim import BodyParams, PoseParams                # noqa: E402
from airis.sim.scenario import load_scenarios               # noqa: E402

POSE_KEYS = list(e4.POSE_KEYS)
BASELINES = ("B0", "B1", "B2")
#: 상한·하한에 닿았다고 볼 여유 (도).
BOUND_EPS = 0.05


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E4 묶음 → docs/e4_reference.json")
    ap.add_argument("--bundle", action="append", required=True, metavar="경로",
                    help="E4 묶음 폴더 (best_poses.json 이 있는 곳). 여러 번 쓸 수 있다")
    ap.add_argument("--out", default=str(ROOT / "docs" / "e4_reference.json"))
    ap.add_argument("--note", action="append", default=[], metavar="묶음이름=설명",
                    help="묶음에 붙일 한 줄 설명 (예: 해시 규약이 다른 이유)")
    ap.add_argument("--rescore", action="store_true",
                    help="접은 자세를 재채점 밀도로 한 번 더 평가해 score_folded 를 채운다")
    ap.add_argument("--patches-per-m2", type=float, default=2000.0,
                    help="--rescore 가 쓸 밀도 (묶음의 재채점 밀도와 같게)")
    return ap


def at_bounds(pose: dict, scenario) -> list[str]:
    """시나리오 범위의 끝에 닿은 변수 이름."""
    hit = []
    for key, (lo, hi) in scenario.pose_bounds.items():
        if key in scenario.fixed_pose or key not in pose:
            continue
        if abs(pose[key] - lo) <= BOUND_EPS or abs(pose[key] - hi) <= BOUND_EPS:
            hit.append(key)
    return hit


def bundle_path_label(path: Path) -> str:
    """저장소 기준 상대 경로. 밖(다른 worktree)이면 폴더 이름만 남긴다.

    사용자 홈 경로가 그대로 들어가면 공개 파일에 개인 경로가 남고, 다른 기계에서 쓸모도 없다.
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def read_bundle(path: Path) -> tuple[dict, dict]:
    """(best_poses.json, {시나리오: e4_summary.csv 행})."""
    poses = json.loads((path / "best_poses.json").read_text(encoding="utf-8"))
    summary = {}
    csv_path = path / "e4_summary.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as fh:
            summary = {row["scenario"]: row for row in csv.DictReader(fh)}
    return poses, summary


def baseline_block(entry: dict, row: dict | None, args_: dict) -> dict:
    """탐색 밀도·재채점 밀도 기준선. 옛 묶음은 재채점 값이 CSV 에만 있다."""
    search = {c: entry["baselines"][c]["score"] for c in BASELINES if c in entry.get("baselines", {})}
    rescored = entry.get("baselines_rescored") or None
    if rescored:
        rescored = {c: rescored[c]["score"] for c in BASELINES if c in rescored}
    elif row:
        rescored = {c: float(row[f"rescore_{c}_score"]) for c in BASELINES
                    if row.get(f"rescore_{c}_score") not in (None, "")}
    return {
        "search": {"patches_per_m2": args_.get("patches_per_m2"), "scores": search},
        "rescored": {"patches_per_m2": args_.get("rescore_patches_per_m2"),
                     "scores": rescored or None},
    }


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)
    notes = dict(n.split("=", 1) for n in args.note)
    scenarios = load_scenarios()

    evaluators: dict[str, object] = {}
    nozzle = body = None
    if args.rescore:
        nozzle, _ = cli.resolve_nozzles()
        body = BodyParams()

    out: dict = {
        "generated_by": "scripts/export_e4_reference.py",
        "pose_keys": POSE_KEYS,
        "yaw_note": ("좌우 거울(θ ≡ −θ)은 메시 몸에서도 정확하고, 앞뒤 등가(θ ≡ 180 − θ)는 근사다"
                     " (상대차 약 2e−4). score 는 pose_raw 의 점수이고, score_folded 는 pose_folded"
                     " 를 같은 밀도로 다시 평가한 값이다 (--rescore 없이 만들면 null)."),
        "bundles": [],
        "scenarios": {},
    }

    for raw in args.bundle:
        path = Path(raw)
        if not (path / "best_poses.json").exists():
            print(f"묶음이 아니다 (best_poses.json 없음): {path}", file=sys.stderr)
            return 2
        poses, summary = read_bundle(path)
        bundle_args = poses.get("args", {})
        group = poses.get("group_id", path.name)
        out["bundles"].append({
            "group_id": group,
            "path": bundle_path_label(path),
            "commit": poses.get("commit"),
            "physics_hash": poses.get("physics_hash"),
            "nozzle_hash": poses.get("nozzle_hash"),
            "patches_per_m2": bundle_args.get("patches_per_m2"),
            "rescore_patches_per_m2": bundle_args.get("rescore_patches_per_m2"),
            "starts": bundle_args.get("starts"),
            "pose_bound": bundle_args.get("pose_bound") or [],
            "note": notes.get(group),
        })

        pose_bound = bundle_args.get("pose_bound") or []
        for name, entry in poses["scenarios"].items():
            scenario = scenarios[name]
            # --pose-bound 로 좁힌 상자의 경계도 따로 본다 (시나리오 범위와 다르다).
            search_scenario = cli.narrow_scenario(scenario, pose_bound) if pose_bound else scenario
            block = out["scenarios"].setdefault(name, {"baselines": {}, "peaks": []})
            block["baselines"][group] = baseline_block(entry, summary.get(name), bundle_args)

            for run in entry["runs"]:
                pose_raw = {k: run["pose"][k] for k in POSE_KEYS}
                pose_folded = {k: run["pose_folded"][k] for k in POSE_KEYS}
                peak = {
                    "bundle": group,
                    "exp_id": run["exp_id"],
                    "seed": run["seed"],
                    "start": run.get("best_start"),
                    "arm_class": sensitivity.arm_class(PoseParams(**pose_raw)),
                    "constraint": ";".join(pose_bound) or None,
                    "pose_raw": pose_raw,
                    "pose_folded": pose_folded,
                    "at_bound": at_bounds(pose_raw, scenario),
                    "at_search_bound": at_bounds(pose_raw, search_scenario) if pose_bound else [],
                    "score": run.get("rescore_score", run["best_score"]),
                    "score_search": run["best_score"],
                    "score_folded": None,
                    "folded_rescored": bool(args.rescore),
                }
                if args.rescore:
                    ev = evaluators.get(name)
                    if ev is None:
                        ev = evaluators[name] = cli.make_evaluator(
                            "patch", scenario, body=body, nozzle=nozzle,
                            patches_per_m2=args.patches_per_m2)
                    peak["score_folded"] = float(
                        ev.evaluate(PoseParams(**pose_folded), nozzle, body, scenario).score)
                block["peaks"].append(peak)

    for name, block in out["scenarios"].items():
        block["peaks"].sort(key=lambda p: -p["score"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    n_peaks = sum(len(b["peaks"]) for b in out["scenarios"].values())
    print(f"묶음 {len(out['bundles'])}개 → 시나리오 {len(out['scenarios'])}개, 봉우리 {n_peaks}개")
    print(f"저장 위치 {out_path}")
    for name, block in out["scenarios"].items():
        top = block["peaks"][0]
        print(f"  {name:<12} 최고 {top['score']:.4f} ({top['arm_class']}, {top['bundle']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
