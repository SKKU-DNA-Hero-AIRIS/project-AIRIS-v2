"""E7 묶음 → `docs/e7_reference.json` (계획 조건 비교의 기준 수치 한 장). 소유자: C.

`export_e4_reference.py` 와 같은 뜻이다 — 총괄의 `energy_weight` 결정, A 의 입자판 재평가,
E 의 계획 화면이 사람 손을 거치지 않고 같은 수치를 읽게 한다.

    python scripts/export_e7_reference.py --bundle <e7 묶음> --out docs/e7_reference.json

묶음의 `e7_summary.csv`(지표)와 `e7_plans.json`(계획 전체)을 합친다. 조건마다

    score           목적함수 값. **w 마다 목적함수가 다르므로 w 가 다르면 비교하지 않는다**
    total_removal   제거율 R (w 와 무관)
    energy          e = Σ s³·T / (M·T_ref)
    duration_s      총 시간 T, phase_durations_s 단계별 시간
    discomfort      불편도
    plan            단계별 자세 7개 + 구역 세기 5개 (21차원 계획 그대로)

를 담는다. 조건 정의(P0~P5)는 `docs/plan_extension.md` 6절.
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

from airis.optimize import cli, e4                          # noqa: E402
from airis.sim import ZONE_NAMES                            # noqa: E402

#: e7_summary.csv 에서 그대로 옮기는 수치 열.
METRIC_COLS = ("score", "total_removal", "discomfort", "energy", "duration_s", "n_phases")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="E7 묶음 → docs/e7_reference.json")
    ap.add_argument("--bundle", action="append", required=True, metavar="경로",
                    help="E7 묶음 폴더 (e7_summary.csv 가 있는 곳). 여러 번 쓰면 한 파일로 모은다 "
                         "— 단계 수 스윕처럼 묶음이 여럿인 실험용")
    ap.add_argument("--out", default=str(ROOT / "docs" / "e7_reference.json"))
    ap.add_argument("--note", default=None, help="묶음에 붙일 한 줄 설명")
    return ap


def bundle_path_label(path: Path) -> str:
    """`outputs/<묶음>` 또는 `<worktree 이름>/outputs/<묶음>` (개인 경로를 남기지 않는다)."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        pass
    for parent in resolved.parents:
        if (parent / ".git").exists():
            return f"{parent.name}/{resolved.relative_to(parent).as_posix()}"
    return resolved.name


def plan_block(plan: dict | None) -> dict | None:
    """계획 하나를 기록용 dict 로 (단계 자세는 접은 yaw 도 같이)."""
    if not plan:
        return None
    phases = []
    for ph in plan["phases"]:
        pose = {k: ph[k] for k in e4.POSE_KEYS}
        phases.append({"duration_s": ph["duration_s"], "pose": pose,
                       "torso_yaw_folded": e4.fold_yaw(pose["torso_yaw"])})
    return {
        "duration_s": plan["duration_s"],
        "phases": phases,
        "zone_strengths": dict(zip(ZONE_NAMES, plan["zone_strengths"])),
    }


def main(argv: list[str] | None = None) -> int:
    cli.enable_utf8_stdout()
    args = build_parser().parse_args(argv)
    bundles = [Path(b) for b in args.bundle]
    for bundle in bundles:
        for name in ("e7_summary.csv", "e7_plans.json"):
            if not (bundle / name).exists():
                print(f"E7 묶음이 아니다 ({name} 없음): {bundle}", file=sys.stderr)
                return 2

    out: dict = {
        "generated_by": "scripts/export_e7_reference.py",
        "bundles": [],
        "zone_names": list(ZONE_NAMES),
        "pose_keys": list(e4.POSE_KEYS),
        "score_note": ("score = Σ 부위가중·제거율 − discomfort_weight·불편도 − w·에너지"
                       " − time_weight·T/T_ref (00_common.md 4.4). w(energy_weight)가 다르면"
                       " 목적함수가 달라 score 를 가로로 비교하지 않는다. w 를 고를 때는"
                       " total_removal · energy · duration_s · discomfort 를 본다."),
        "rows": [],
    }

    for bundle in bundles:
        plans = json.loads((bundle / "e7_plans.json").read_text(encoding="utf-8"))
        group = plans.get("group_id", bundle.name)
        bundle_args = plans.get("args", {})
        out["bundles"].append({
            "group_id": group,
            "path": bundle_path_label(bundle),
            "commit": plans.get("commit"),
            "physics_hash": plans.get("physics_hash"),
            "nozzle_hash": plans.get("nozzle_hash"),
            "kinetics_enabled": plans.get("kinetics_enabled"),
            "time_constant_s": plans.get("time_constant_s"),
            "zone_nozzle_counts": dict(zip(ZONE_NAMES, plans.get("zone_nozzle_counts", []))),
            # 단계 수 덮어쓰기(--n-phases)와 예산은 묶음마다 다를 수 있다.
            "n_phases_arg": bundle_args.get("n_phases"),
            "max_evals": bundle_args.get("max_evals"),
            "patches_per_m2": bundle_args.get("patches_per_m2"),
            "args": bundle_args,
            "note": args.note,
        })
        with (bundle / "e7_summary.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            w = float(row["energy_weight"])
            key = f"{row['condition']}_s{row['seed']}"
            plan = (plans.get("scenarios", {}).get(row["scenario"], {})
                    .get(f"w{w:g}", {}).get(key))
            entry = {
                "bundle": group,
                "scenario": row["scenario"],
                "condition": row["condition"],
                "energy_weight": w,
                "seed": int(row["seed"]),
                "infeasible": row["infeasible"] not in ("False", "false", ""),
            }
            for col in METRIC_COLS:
                value = row.get(col, "")
                entry[col] = float(value) if value not in ("", None) else None
            entry["n_phases"] = int(entry["n_phases"]) if entry["n_phases"] is not None else None
            entry["phase_durations_s"] = [float(v) for v in row["phase_durations_s"].split(";") if v]
            entry["plan"] = plan_block(plan)
            out["rows"].append(entry)

    # 묶음이 하나면 예전 모양(`bundle`)도 함께 남긴다 — A 의 scripts/verify_e7_plans.py 와
    # tests/test_particles.py 가 `ref["bundle"]` 로 도장을 읽는다 (호환 유지).
    if len(out["bundles"]) == 1:
        out["bundle"] = out["bundles"][0]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    missing = [r for r in out["rows"] if r["plan"] is None]
    print(f"묶음 {len(out['bundles'])}개 → 행 {len(out['rows'])}개 (계획 없는 행 {len(missing)}개)")
    print(f"저장 위치 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
