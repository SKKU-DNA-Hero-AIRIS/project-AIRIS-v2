"""E4 러너와 요약 계산 테스트. 소유자: C. docs/tracks/C_optimize.md 단계 7."""
import csv
import json
import math

import numpy as np

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


def test_fold_yaw_left_right_and_front_back():
    # 좌우 거울 θ ≡ −θ, 앞뒤 등가 θ ≡ 180° − θ → 0~90°
    assert e4.fold_yaw(72.0) == 72.0
    assert e4.fold_yaw(-72.0) == 72.0
    assert e4.fold_yaw(108.0) == pytest.approx(72.0)
    assert e4.fold_yaw(-108.0) == pytest.approx(72.0)
    assert e4.fold_yaw(180.0) == 0.0 and e4.fold_yaw(90.0) == 90.0 and e4.fold_yaw(0.0) == 0.0


def test_fold_pose_mirrors_yaw():
    folded = e4.fold_pose(PoseParams(torso_yaw=-97.0, shoulder_abduction=180.0))
    assert folded["torso_yaw"] == pytest.approx(83.0)
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


# ---------- 설정 해시 규약 (총괄 2026-09-30) ----------

def test_file_hash_ignores_comments_and_line_endings(tmp_path):
    """YAML 은 파싱한 내용으로 해시한다: 주석·줄바꿈·키 순서에 무관, 값이 다르면 다르다."""
    from airis.optimize import explog

    def write(name, text, newline="\n"):
        path = tmp_path / name
        path.write_bytes(text.replace("\n", newline).encode("utf-8"))
        return path

    body = "jet:\n  gain: 1.4\nair:\n  density: 1.2\n"
    lf = write("a.yaml", body)
    crlf = write("b.yaml", body, newline="\r\n")
    commented = write("c.yaml", "# 주석\njet:\n  gain: 1.40\nair:\n  density: 1.2\n")
    reordered = write("d.yaml", "air:\n  density: 1.2\njet:\n  gain: 1.4\n")
    changed = write("e.yaml", "jet:\n  gain: 1.0\nair:\n  density: 1.2\n")

    h = explog.file_hash(lf)
    assert h == explog.file_hash(crlf), "CRLF/LF 는 같은 해시"
    assert h == explog.file_hash(commented), "주석과 1.40/1.4 표기는 무시"
    assert h == explog.file_hash(reordered), "키 순서는 무시"
    assert h != explog.file_hash(changed), "값이 다르면 다른 해시"
    assert explog.file_hash(tmp_path / "none.yaml") == "nofile"

    # YAML 이 아닌 파일은 바이트 해시 그대로 (줄바꿈이 다르면 다른 해시).
    assert explog.file_hash(write("a.txt", "x\n")) != explog.file_hash(write("b.txt", "x\n", "\r\n"))


def test_best_poses_records_baselines_per_density(tmp_path):
    """best_poses.json 은 탐색 밀도와 재채점 밀도의 기준선을 따로 남긴다 (통합 2026-09-30).

    한 덩어리로 남기면 재채점 점수(2,000/m²)를 탐색 밀도(1,500/m²) 기준선과 비교하는
    실수가 난다 — 실제로 한 번 냈다.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "outputs"
    cmd = [sys.executable, str(root / "scripts" / "run_e4.py"),
           "--evaluator", "dummy", "--scenarios", "default", "--seeds", "0",
           "--max-evals", "60", "--popsize", "10", "--starts", "default",
           "--tag", "t", "--log-dir", str(out)]
    assert subprocess.run(cmd, cwd=root, capture_output=True).returncode == 0

    group = next(p for p in out.iterdir() if (p / "best_poses.json").exists())
    data = json.loads((group / "best_poses.json").read_text(encoding="utf-8"))["scenarios"]["default"]
    # 더미 평가기는 패치 밀도도 재채점도 없다. 키는 남고 값만 None 이다.
    assert data["baselines_patches_per_m2"] is None
    assert "baselines_rescored" in data and "baselines_rescored_patches_per_m2" in data
    assert set(data["baselines"]) == {"B0", "B1", "B2"}


def test_pose_bound_scores_on_original_scale(tmp_path):
    """--pose-bound 는 탐색 상자만 좁히고 점수는 원래 시나리오로 매긴다 (통합 2026-09-30 지적).

    scoring.discomfort 가 pose_bounds 폭으로 정규화하므로, 좁힌 시나리오로 채점하면 그 변수의
    불편도가 폭에 반비례해 커져 다른 E4 묶음과 비교할 수 없는 점수가 된다. 더미 평가기도
    정규화 공간에서 거리를 재므로 같은 실수를 드러낸다.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    from airis.optimize.cli import DEFAULT_DUMMY_TARGET
    from airis.optimize.dummy import DummyEvaluator
    from airis.sim import BodyParams, PoseParams
    from airis.sim.scenario import load_nozzles, load_scenarios

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "outputs"
    cmd = [sys.executable, str(root / "scripts" / "run_e4.py"),
           "--evaluator", "dummy", "--scenarios", "default", "--seeds", "0",
           "--max-evals", "120", "--popsize", "20", "--starts", "default",
           "--pose-bound", "shoulder_abduction=0,90", "--tag", "t", "--log-dir", str(out)]
    assert subprocess.run(cmd, cwd=root, capture_output=True).returncode == 0

    best = json.loads(next(out.glob("t_*/best.json")).read_text(encoding="utf-8"))
    pose = PoseParams(**best["best_pose"])
    assert 0.0 <= pose.shoulder_abduction <= 90.0, "탐색 상자는 좁혀져야 한다"

    full = load_scenarios()["default"]
    nozzle, _ = load_nozzles(), None
    on_full = DummyEvaluator(DEFAULT_DUMMY_TARGET, full).evaluate(
        pose, nozzle, BodyParams(), full).score
    assert best["best_score"] == pytest.approx(on_full, abs=1e-9), "원래 시나리오 척도로 채점해야 한다"

    # 좁힌 시나리오로 채점했다면 다른 값이 나온다 (이 테스트가 실제로 구분력이 있는지 확인).
    from airis.optimize.cli import narrow_scenario
    narrowed = narrow_scenario(full, ["shoulder_abduction=0,90"])
    on_narrow = DummyEvaluator(DEFAULT_DUMMY_TARGET, narrowed).evaluate(
        pose, nozzle, BodyParams(), narrowed).score
    assert abs(on_narrow - on_full) > 1e-6

    group = next(p for p in out.iterdir() if (p / "best_poses.json").exists())
    data = json.loads((group / "best_poses.json").read_text(encoding="utf-8"))["scenarios"]["default"]
    assert data["pose_bounds_effective"] == {"shoulder_abduction": [0.0, 90.0]}
    assert data["baselines_patches_per_m2"] is None, "더미 실행에 패치 밀도를 적으면 안 된다"
    rows = list(csv.DictReader((group / "e4_summary.csv").open(encoding="utf-8")))
    assert rows[0]["pose_bound"] == "shoulder_abduction=0,90"


def test_pose_bound_patch_discomfort_path_unchanged():
    """패치판에서도 좁힌 시나리오는 점수를 바꾸지 않는다 (통합 2026-10-01 지적 4).

    더미 평가기는 정규화 거리로만 차이를 드러낸다. 실제 실험이 쓰는 경로는 patch 평가기의
    scoring.score(..., discomfort) 이므로 저밀도(100/m²)로 한 번 직접 확인한다.
    같은 자세·같은 제거율인데 좁힌 시나리오로 채점하면 불편도 항이 커져 점수가 낮아진다.
    """
    from airis.optimize import cli
    from airis.sim import BodyParams, PoseParams
    from airis.sim.scenario import load_physics, load_scenarios

    scenario = load_scenarios()["default"]
    narrowed = cli.narrow_scenario(scenario, ["shoulder_abduction=0,90"])
    body, (nozzle, _) = BodyParams(), cli.resolve_nozzles()
    try:
        ev = cli.make_evaluator("patch", scenario, body=body, nozzle=nozzle, patches_per_m2=100)
    except cli.TrackNotMerged:                       # D 미병합 환경
        pytest.skip("patch 평가기 미구현")

    pose = PoseParams(shoulder_abduction=45.0, torso_yaw=70.0)
    full = ev.evaluate(pose, nozzle, body, scenario)
    narrow = ev.evaluate(pose, nozzle, body, narrowed)

    assert full.total_removal == pytest.approx(narrow.total_removal), "제거율은 범위와 무관"
    # 벌림 불편도 가중 0.3(시나리오), 폭이 180 → 90 으로 절반이라 벌림 불편도 항이 2배가 된다.
    # 점수에는 scoring.discomfort_weight(0.1)가 곱해진다 — 아래에서 설정값으로 확인한다.
    assert narrow.discomfort > full.discomfort
    assert full.score > narrow.score, "좁힌 시나리오로 채점하면 점수가 달라진다 (그래서 쓰면 안 된다)"
    gap = full.score - narrow.score
    w = float(load_physics()["scoring"]["discomfort_weight"])
    assert gap == pytest.approx(w * (narrow.discomfort - full.discomfort), rel=1e-9)


def test_jsonable_maps_non_finite_to_null():
    """nan·inf 는 null 로 쓴다 — NaN 은 표준 JSON 이 아니다 (통합 2026-10-01 지적 3)."""
    from airis.optimize import explog

    out = explog._jsonable({"peak_gap": float("nan"), "hi": float("inf"),
                            "lo": float("-inf"), "ok": 0.5, "n": 3})
    assert out == {"peak_gap": None, "hi": None, "lo": None, "ok": 0.5, "n": 3}
    json.dumps(out, allow_nan=False)                 # 표준 JSON 으로 직렬화된다
    assert explog._jsonable(np.array([float("nan"), 1.0])) == [None, 1.0]


def test_write_run_keeps_json_strict_with_nan_score(tmp_path):
    """best_score 등이 nan 이어도 meta.json·best.json 이 표준 JSON 이다 (통합 2026-10-01 지적 2).

    _jsonable 을 거치지 않던 네 필드(meta best_score, best.json 의 best_score·total_removal·
    discomfort)가 NaN 리터럴로 새던 것을 막는다.
    """
    from types import SimpleNamespace

    import numpy as np

    from airis.optimize import explog
    from airis.sim import PART_NAMES, BodyParams, PoseParams
    from airis.sim.scenario import load_scenarios

    nan = float("nan")
    result = SimpleNamespace(
        n_evals=1, n_infeasible=0, per_start=[{"start": "default", "best_score": nan}],
        best_score=nan, best_pose=PoseParams(), history=[],
        best_result=SimpleNamespace(removal_by_part=np.zeros(len(PART_NAMES)),
                                    total_removal=nan, discomfort=nan, extra={"x": nan}),
    )
    explog.write_run(tmp_path, "t_nan", scenario=load_scenarios()["default"], body=BodyParams(),
                     nozzle_hash="h", physics_hash="p", commit="c", seed=0, args={}, result=result)

    for name in ("meta.json", "best.json"):
        text = (tmp_path / "t_nan" / name).read_text(encoding="utf-8")
        assert "NaN" not in text, f"{name} 에 NaN 리터럴이 남았다"
        json.loads(text, parse_constant=_reject_constant)      # 엄격 파서로 읽힌다


def _reject_constant(name):                                    # json 의 NaN/Infinity 확장 거부
    raise AssertionError(f"표준 JSON 이 아니다: {name}")


def test_pose_bound_warns_and_records_skipped_starts(tmp_path, recwarn):
    """범위 밖 시작점은 경고를 내고 best_poses.json 에 남는다 (통합 2026-10-01 지적 3)."""
    import subprocess
    import sys
    from pathlib import Path

    from airis.optimize import cli
    from airis.sim.scenario import load_scenarios

    scenario = load_scenarios()["default"]
    narrowed = cli.narrow_scenario(scenario, ["shoulder_abduction=0,90"])
    with pytest.warns(RuntimeWarning, match="시작점을 건너뛴다"):
        kept, dropped = cli.starts_in_bounds(cli.parse_starts("default,hands_up"), narrowed)
    assert [s.name for s in kept] == ["default"] and dropped == ["hands_up"]

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "outputs"
    cmd = [sys.executable, str(root / "scripts" / "run_e4.py"),
           "--evaluator", "dummy", "--scenarios", "default", "--seeds", "0",
           "--max-evals", "60", "--popsize", "10", "--starts", "default,hands_up",
           "--pose-bound", "shoulder_abduction=0,90", "--tag", "t", "--log-dir", str(out)]
    assert subprocess.run(cmd, cwd=root, capture_output=True).returncode == 0
    data = json.loads(next(out.glob("t_*/best_poses.json")).read_text(encoding="utf-8"))
    entry = data["scenarios"]["default"]
    assert entry["starts_used"] == ["default"] and entry["starts_skipped"] == ["hands_up"]
    assert entry["runs"][0]["peak_gap"] is None, "시작점이 하나면 peak_gap 은 null"


# ---------- e4_reference.json 내보내기 ----------

def _fake_bundle(path, *, group, pose_bound=(), rescore_score=0.5):
    """작은 E4 묶음 하나 (best_poses.json + e4_summary.csv)."""
    path.mkdir(parents=True, exist_ok=True)
    pose = {"shoulder_abduction": 179.0, "shoulder_flexion": 0.0, "elbow_flexion": 1.0,
            "torso_pitch": 0.0, "torso_yaw": -100.0, "hip_flexion": 0.0, "knee_flexion": 30.0}
    folded = dict(pose, torso_yaw=80.0)
    (path / "best_poses.json").write_text(json.dumps({
        "group_id": group, "commit": "abc1234", "physics_hash": "ph", "nozzle_hash": "nh",
        "args": {"patches_per_m2": 1500.0, "rescore_patches_per_m2": 2000.0,
                 "starts": "default", "pose_bound": list(pose_bound)},
        "scenarios": {"default": {
            "runs": [{"exp_id": f"{group}_r0", "seed": 0, "best_score": 0.49,
                      "rescore_score": rescore_score, "best_start": "default",
                      "pose": pose, "pose_folded": folded}],
            "baselines": {c: {"score": v, "infeasible": False}
                          for c, v in (("B0", 0.23), ("B1", 0.38), ("B2", 0.25))},
        }},
    }, ensure_ascii=False), encoding="utf-8")
    with (path / "e4_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["scenario", "rescore_B0_score", "rescore_B1_score",
                                           "rescore_B2_score"])
        w.writeheader()
        w.writerow({"scenario": "default", "rescore_B0_score": "0.234",
                    "rescore_B1_score": "0.381", "rescore_B2_score": "0.249"})
    return path


def test_export_e4_reference(tmp_path):
    """묶음 두 개 → 기준 수치 한 장. 사람이 수치를 옮겨 적지 않게 하는 것이 목적이다."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "export_e4_reference", root / "scripts" / "export_e4_reference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    a = _fake_bundle(tmp_path / "a", group="base", rescore_score=0.62)
    b = _fake_bundle(tmp_path / "b", group="down",
                     pose_bound=("shoulder_abduction=0,90",), rescore_score=0.49)
    out = tmp_path / "e4_reference.json"
    assert mod.main(["--bundle", str(a), "--bundle", str(b), "--out", str(out),
                     "--note", "down=상한 탐색"]) == 0

    data = json.loads(out.read_text(encoding="utf-8"))
    assert [x["group_id"] for x in data["bundles"]] == ["base", "down"]
    assert data["bundles"][1]["pose_bound"] == ["shoulder_abduction=0,90"]
    assert data["bundles"][1]["note"] == "상한 탐색"

    block = data["scenarios"]["default"]
    # 기준선은 묶음별로, 탐색 밀도와 재채점 밀도를 따로 (라벨 혼동 방지).
    assert block["baselines"]["base"]["search"]["scores"]["B0"] == 0.23
    assert block["baselines"]["base"]["search"]["patches_per_m2"] == 1500.0
    assert block["baselines"]["base"]["rescored"]["scores"]["B0"] == 0.234
    assert block["baselines"]["base"]["rescored"]["patches_per_m2"] == 2000.0

    # 봉우리는 점수 내림차순, 재채점 점수를 score 로.
    assert [p["score"] for p in block["peaks"]] == [0.62, 0.49]
    top = block["peaks"][0]
    assert top["score_search"] == 0.49 and top["score_folded"] is None   # --rescore 없음
    assert top["arm_class"] == "hands_up"
    assert top["pose_raw"]["torso_yaw"] == -100.0 and top["pose_folded"]["torso_yaw"] == 80.0
    assert "knee_flexion" in top["at_bound"], "범위 끝(30°)에 닿은 변수를 표시한다"
    assert block["peaks"][1]["constraint"] == "shoulder_abduction=0,90"
    json.dumps(data, allow_nan=False)
