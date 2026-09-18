"""E4 러너와 요약 계산 테스트. 소유자: C. docs/tracks/C_optimize.md 단계 7."""
import csv
import json
import math

import pytest

from airis.optimize import e4
from airis.sim import PoseParams

pytest.importorskip("cma", reason="cma 패키지 필요 (pip install cma)")


def test_improvement():
    assert e4.improvement(0.6, 0.2) == pytest.approx(2.0)
    assert e4.improvement(0.1, 0.2) == pytest.approx(-0.5)
    # 기준선이 음수(더미)여도 부호가 뒤집히지 않는다: 더 높은 best 는 양의 개선.
    assert e4.improvement(-0.1, -0.5) == pytest.approx(0.8)
    assert math.isnan(e4.improvement(0.6, 0.2, base_infeasible=True))
    assert math.isnan(e4.improvement(0.6, 0.0))


def test_fold_pose_mirrors_yaw():
    folded = e4.fold_pose(PoseParams(torso_yaw=-97.0, shoulder_abduction=180.0))
    assert folded["torso_yaw"] == 97.0
    assert folded["shoulder_abduction"] == 180.0
    assert list(folded) == e4.POSE_KEYS


def test_winning_start_skips_nan():
    per_start = [
        {"start": "default", "best_score": 0.56},
        {"start": "hands_up", "best_score": 0.62},
        {"start": "other", "best_score": float("nan")},
    ]
    assert e4.winning_start(per_start) == "hands_up"
    assert e4.winning_start([{"start": "x", "best_score": float("nan")}]) == ""


def _run(seed, score, yaw, start="hands_up"):
    return {
        "exp_id": f"e4_{seed}", "seed": seed, "best_score": score,
        "best_pose": PoseParams(shoulder_abduction=180.0, torso_yaw=yaw),
        "per_start": [{"start": "default", "best_score": 0.5},
                      {"start": start, "best_score": score}],
        "elapsed_s": 10.0, "n_evals": 100, "n_infeasible": 5,
    }


def test_summarize_scenario_folds_yaw_and_computes_improvement():
    runs = [_run(0, 0.62, -90.0), _run(1, 0.64, 90.0), _run(2, 0.63, -90.0)]
    base = {
        "B0": {"score": 0.2, "infeasible": False},
        "B1": {"score": 0.4, "infeasible": False},
        "B2": {"score": -1.2, "infeasible": True},
    }
    row = e4.summarize_scenario(runs, base)
    assert row["n_seeds"] == 3 and row["seeds"] == "0;1;2"
    assert row["best_mean"] == pytest.approx(0.63)
    assert row["best_std"] == pytest.approx(0.01)
    assert row["imp_vs_B0"] == pytest.approx(2.15)
    assert row["imp_vs_B1"] == pytest.approx(0.575)
    assert math.isnan(row["imp_vs_B2"]) and row["B2_infeasible"]
    # ±90 이 같은 해로 접혀 편차 0.
    assert row["pose_mean_torso_yaw"] == pytest.approx(90.0)
    assert row["pose_std_torso_yaw"] == pytest.approx(0.0)
    assert row["best_start_counts"] == "hands_up:3"
    assert row["infeasible_frac"] == pytest.approx(0.05)
    # 봉우리 간 차이 = hands_up − default (0.62−0.5, 0.64−0.5, 0.63−0.5)
    assert row["peak_gap_mean"] == pytest.approx(0.13)
    assert row["peak_gap_std"] == pytest.approx(0.01)


def test_peak_gap():
    per_start = [{"start": "default", "best_score": 0.55}, {"start": "hands_up", "best_score": 0.547}]
    assert e4.peak_gap(per_start) == pytest.approx(-0.003)
    assert math.isnan(e4.peak_gap([{"start": "default", "best_score": 0.55}]))


def test_run_e4_dummy_writes_summary(tmp_path):
    """완료 기준: 시나리오 × 시드를 돌려 요약 파일 셋과 실행별 로그를 남긴다."""
    from scripts.run_e4 import main

    code = main([
        "--evaluator", "dummy", "--scenarios", "default", "wheelchair", "--seeds", "0", "1",
        "--max-evals", "80", "--popsize", "10", "--log-dir", str(tmp_path),
    ])
    assert code == 0

    (group,) = [p for p in tmp_path.iterdir() if p.is_dir() and (p / "e4_summary.csv").exists()]
    assert group.name.startswith("e4_")
    with (group / "e4_summary.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["scenario"] for r in rows] == ["default", "wheelchair"]
    for r in rows:
        assert r["n_seeds"] == "2"
        assert "rescore_best_mean" not in r, "재채점은 패치판에서만"
        for key in ("imp_vs_B0", "imp_vs_B1", "imp_vs_B2", "pose_std_torso_yaw", "best_start_counts"):
            assert key in r

    with (group / "baselines.csv").open(encoding="utf-8") as fh:
        base_rows = list(csv.DictReader(fh))
    assert {(r["scenario"], r["condition"]) for r in base_rows} == {
        (s, c) for s in ("default", "wheelchair") for c in ("B0", "B1", "B2")
    }

    poses = json.loads((group / "best_poses.json").read_text(encoding="utf-8"))
    assert poses["group_id"] == group.name
    wheelchair = poses["scenarios"]["wheelchair"]
    assert [r["seed"] for r in wheelchair["runs"]] == [0, 1]
    assert all(r["pose"]["hip_flexion"] == 90 for r in wheelchair["runs"])
    assert all(r["pose_folded"]["torso_yaw"] >= 0 for r in wheelchair["runs"])
    assert all(set(r["start_best"]) == {"default", "hands_up"} for r in wheelchair["runs"])
    assert all(r["peak_gap"] == pytest.approx(r["start_best"]["hands_up"] - r["start_best"]["default"])
               for r in wheelchair["runs"])

    # 실행 4개가 각자 폴더와 index.csv 한 줄을 남기고, meta 에 묶음 id 가 있다.
    run_ids = [r["exp_id"] for s in poses["scenarios"].values() for r in s["runs"]]
    assert len(run_ids) == 4
    for exp_id in run_ids:
        meta = json.loads((tmp_path / exp_id / "meta.json").read_text(encoding="utf-8"))
        assert meta["args"]["e4_group"] == group.name
        assert [ps["start"] for ps in meta["per_start"]] == ["default", "hands_up"]
    index = (tmp_path / "index.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(index) == 1 + 4


def test_run_e4_rejects_unknown_scenario(tmp_path):
    from scripts.run_e4 import main

    assert main(["--evaluator", "dummy", "--scenarios", "nope", "--log-dir", str(tmp_path)]) == 2


def test_run_e4_patch_rescore(tmp_path):
    """패치판은 best 자세·기준선을 촘촘한 밀도로 재채점해 rescore_* 열을 남긴다 (짧게)."""
    from scripts.run_e4 import main

    code = main([
        "--evaluator", "patch", "--scenarios", "default", "--seeds", "0",
        "--max-evals", "8", "--popsize", "4", "--starts", "default",
        "--patches-per-m2", "100", "--rescore-patches-per-m2", "400", "--log-dir", str(tmp_path),
    ])
    assert code == 0
    (group,) = [p for p in tmp_path.iterdir() if p.is_dir() and (p / "e4_summary.csv").exists()]
    with (group / "e4_summary.csv").open(encoding="utf-8") as fh:
        (row,) = list(csv.DictReader(fh))
    assert row["rescore_patches_per_m2"] == "400.0"
    for key in ("rescore_best_mean", "rescore_imp_vs_B0", "rescore_B1_score"):
        assert row[key] != ""
    poses = json.loads((group / "best_poses.json").read_text(encoding="utf-8"))
    assert poses["scenarios"]["default"]["runs"][0]["rescore_score"] is not None
