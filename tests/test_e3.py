"""E3 민감도 러너와 보조 함수 테스트. 소유자: C. docs/tracks/C_optimize.md 단계 8."""
import csv
import json
import math

import numpy as np
import pytest

from airis.optimize import sensitivity
from airis.sim import BodyParams, PoseParams
from airis.sim.scenario import load_nozzles, load_physics, load_scenarios

pytest.importorskip("cma", reason="cma 패키지 필요 (pip install cma)")


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios()


def test_apply_setting_physics_copies(scenarios):
    cfg = load_physics()
    scen = scenarios["default"]
    before = cfg["jet"]["impingement"]["wall_jet_gain"]     # 설정값은 바뀔 수 있다 (k 1.0 → 1.4)
    new_cfg, new_scen = sensitivity.apply_setting(cfg, scen, "jet.impingement.wall_jet_gain", 2.0)
    assert new_cfg["jet"]["impingement"]["wall_jet_gain"] == 2.0
    assert cfg["jet"]["impingement"]["wall_jet_gain"] == before, "원본은 그대로"
    assert new_scen is scen


def test_apply_setting_scenario_field(scenarios):
    cfg = load_physics()
    scen = scenarios["pregnant"]
    key = "scenario.discomfort_weights.shoulder_abduction"
    assert sensitivity.get_setting(cfg, scen, key) == 0.6
    value = sensitivity.resolve_value(cfg, scen, key, "factor", 0.5)
    new_cfg, new_scen = sensitivity.apply_setting(cfg, scen, key, value)
    assert new_scen.discomfort_weights["shoulder_abduction"] == pytest.approx(0.3)
    assert scen.discomfort_weights["shoulder_abduction"] == 0.6, "원본은 그대로"
    assert new_scen.discomfort_weights["torso_pitch"] == scen.discomfort_weights["torso_pitch"]
    assert new_cfg is cfg


def test_unknown_key_raises(scenarios):
    cfg = load_physics()
    with pytest.raises(KeyError):
        sensitivity.get_setting(cfg, scenarios["default"], "jet.impingment.wall_jet_gain")
    with pytest.raises(KeyError):
        sensitivity.get_setting(cfg, scenarios["default"], "scenario.discomfort_weights.nope")


def test_resolve_value():
    cfg = load_physics()
    scen = load_scenarios()["default"]
    rough = cfg["adhesion"]["fabric_roughness_factor"]       # 설정값은 바뀔 수 있다 (0.25 → 0.45)
    assert sensitivity.resolve_value(cfg, scen, "adhesion.fabric_roughness_factor", "factor", 2) == 2 * rough
    assert sensitivity.resolve_value(cfg, scen, "jet.impingement.enabled", "value", False) is False


def test_core_preset_keys_exist(scenarios):
    cfg = load_physics()
    for key, mode, vals in sensitivity.CORE_PRESET:
        assert mode in ("factor", "value") and vals
        keys = sensitivity.axis_keys(key)               # 쌍 축이면 키가 둘이다
        for scen in scenarios.values():
            for one in keys:
                sensitivity.get_setting(cfg, scen, one)
        for v in vals:                                  # 점마다 값 수가 키 수와 맞아야 한다
            assert len(sensitivity.axis_values(v)) == len(keys), (key, v)


def test_patch_shear_equivalences(scenarios):
    """패치판에서 friction ×k ≡ roughness ×1/k, 출구 속도 ×k ≡ roughness ×1/k² (sensitivity 모듈 설명)."""
    from airis.sim.patch_baseline import PatchEvaluator

    cfg = load_physics()
    scen = scenarios["default"]
    nozzle, body = load_nozzles(), BodyParams()
    pose = PoseParams(shoulder_abduction=180.0, elbow_flexion=0.0, torso_yaw=90.0)

    def score(key, factor):
        value = sensitivity.resolve_value(cfg, scen, key, "factor", factor)
        new_cfg, _ = sensitivity.apply_setting(cfg, scen, key, value)
        return PatchEvaluator(new_cfg, patches_per_m2=100).evaluate(pose, nozzle, body, scen).score

    rough = "adhesion.fabric_roughness_factor"
    assert score("air.friction_coeff", 2.0) == pytest.approx(score(rough, 0.5), rel=1e-6)
    assert score("jet.slot.exit_velocity_mps", 1.25) == pytest.approx(score(rough, 1 / 1.25 ** 2), rel=1e-6)


def test_arm_class_and_same_peak():
    up = PoseParams(shoulder_abduction=180.0, torso_yaw=-95.0)
    up_mirror = PoseParams(shoulder_abduction=175.0, torso_yaw=88.0)
    up_front_back = PoseParams(shoulder_abduction=180.0, torso_yaw=-78.0)   # 180 − 102 ≡ 78, 거울
    down = PoseParams(shoulder_abduction=12.0, torso_yaw=90.0)
    front = PoseParams(shoulder_abduction=180.0, torso_yaw=10.0)
    assert sensitivity.arm_class(up) == "hands_up" and sensitivity.arm_class(down) == "arms_down"
    assert sensitivity.same_peak(up_mirror, up), "yaw 부호는 무시"
    assert sensitivity.same_peak(up_front_back, PoseParams(shoulder_abduction=180.0, torso_yaw=102.0)), \
        "앞뒤 등가: 102° 와 78° 는 같은 봉우리"
    assert not sensitivity.same_peak(down, up)
    assert not sensitivity.same_peak(front, up)


def test_diverse_top_skips_near_duplicates():
    cands = [
        {"x": np.zeros(3), "score": 0.9, "infeasible": False},
        {"x": np.full(3, 0.01), "score": 0.89, "infeasible": False},   # 첫 후보와 거의 같음
        {"x": np.ones(3), "score": 0.95, "infeasible": True},         # 불가
        {"x": np.full(3, 0.5), "score": 0.8, "infeasible": False},
    ]
    top = sensitivity.diverse_top(cands, k=5, min_dist=0.15)
    assert [c["score"] for c in top] == [0.9, 0.8]


def test_spearman_and_regret():
    assert sensitivity.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert sensitivity.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert sensitivity.spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)
    assert math.isnan(sensitivity.spearman([1], [1]))
    assert math.isnan(sensitivity.spearman([1, 1, 1], [1, 2, 3]))
    assert sensitivity.regret(0.45, 0.5) == pytest.approx(0.1)


def test_run_cmaes_records_candidates(scenarios):
    from airis.optimize import cli
    from airis.optimize.cmaes_runner import run_cmaes
    from airis.optimize.dummy import DummyEvaluator

    scen = scenarios["default"]
    kw = dict(max_evals=60, popsize=10, seed=0, starts=cli.parse_starts("default,hands_up"))
    rec = run_cmaes(DummyEvaluator(PoseParams(), scen), BodyParams(), scen, load_nozzles(),
                    record_candidates=True, **kw)
    plain = run_cmaes(DummyEvaluator(PoseParams(), scen), BodyParams(), scen, load_nozzles(), **kw)
    assert len(rec.candidates) == rec.n_evals == 60
    assert {c["start"] for c in rec.candidates} == {"default", "hands_up"}
    assert plain.candidates == []
    assert rec.history == plain.history, "기록 여부가 탐색을 바꾸지 않는다"


def test_run_e3_dummy_end_to_end(tmp_path):
    from scripts.run_e3 import REF, main

    code = main([
        "--evaluator", "dummy", "--param", "scenario.discomfort_weights.shoulder_abduction",
        "--factors", "2", "--scenarios", "default", "--seeds", "0", "1",
        "--max-evals", "60", "--popsize", "10", "--top-k", "4", "--log-dir", str(tmp_path),
    ])
    assert code == 0
    (group,) = [p for p in tmp_path.iterdir() if p.is_dir() and (p / "e3_summary.csv").exists()]
    with (group / "e3_summary.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["setting"], r["value"]) for r in rows] == [
        (REF, ""), ("scenario.discomfort_weights.shoulder_abduction", "x2")]
    ref = rows[0]
    assert float(ref["topk_spearman"]) == pytest.approx(1.0)
    assert float(ref["regret"]) == pytest.approx(0.0)
    # 기준 best 를 낸 시드는 자기 자신과 같은 봉우리다 (나머지 시드는 짧은 예산이라 다를 수 있다).
    assert float(ref["same_peak_frac"]) >= 0.5
    with (group / "e3_runs.csv").open(encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 4          # 설정 2 × 시드 2
    tops = json.loads((group / "top_candidates.json").read_text(encoding="utf-8"))
    assert len(tops["scenarios"]["default"]["top"]) == 4
    assert len(tops["scenarios"]["default"]["rescored"]) == 2


def test_run_e3_rejects_unknown_key(tmp_path):
    from scripts.run_e3 import main

    assert main(["--evaluator", "dummy", "--param", "jet.nope", "--factors", "2",
                 "--log-dir", str(tmp_path)]) == 2


# ---------- 여러 키를 한 점으로 (충돌 배율 m 과 거칠기 f 쌍, 총괄 2026-10-03) ----------

def test_axis_keys_and_point_label(scenarios):
    """축 키는 문자열 하나·"a+b"·튜플을 모두 받고, 라벨은 사람이 읽는 형태다."""
    from airis.optimize import sensitivity

    single = "jet.impingement.wall_jet_gain"
    pair = ("jet.impingement.stagnation_shear_factor", "adhesion.fabric_roughness_factor")
    assert sensitivity.axis_keys(single) == (single,)
    assert sensitivity.axis_keys("a.b+c.d") == ("a.b", "c.d")
    assert sensitivity.axis_keys(pair) == pair
    assert sensitivity.axis_name(pair) == "+".join(pair)
    assert sensitivity.axis_values(3.0) == (3.0,) and sensitivity.axis_values((3.0, 0.57)) == (3.0, 0.57)
    assert sensitivity.point_label(single, "value", 1.4, (1.4,)) == "1.4"
    assert sensitivity.point_label(single, "factor", 0.5, (0.225,)) == "x0.5"
    assert sensitivity.point_label(pair, "value", (3.0, 0.57), (3.0, 0.57)) == "3,0.57"
    assert sensitivity.point_label(pair, "factor", (2.0, 0.5), (2.0, 0.225)) == "x2,x0.5"


def test_apply_point_sets_all_keys(scenarios):
    """쌍 축은 두 설정을 함께 바꾸고 원본은 건드리지 않는다."""
    from airis.optimize import sensitivity
    from airis.sim.scenario import load_physics

    cfg = load_physics()
    scen = scenarios["default"]
    before_m = cfg["jet"]["impingement"]["stagnation_shear_factor"]
    before_f = cfg["adhesion"]["fabric_roughness_factor"]
    pair = ("jet.impingement.stagnation_shear_factor", "adhesion.fabric_roughness_factor")

    values = sensitivity.resolve_point(cfg, scen, pair, "value", (3.0, 0.57))
    new_cfg, new_scen = sensitivity.apply_point(cfg, scen, pair, values)
    assert new_cfg["jet"]["impingement"]["stagnation_shear_factor"] == 3.0
    assert new_cfg["adhesion"]["fabric_roughness_factor"] == 0.57
    assert cfg["jet"]["impingement"]["stagnation_shear_factor"] == before_m, "원본 불변"
    assert cfg["adhesion"]["fabric_roughness_factor"] == before_f
    assert new_scen is scen                                  # 시나리오 키가 아니면 그대로

    # 배율 방식도 키마다 따로 곱한다.
    factored = sensitivity.resolve_point(cfg, scen, pair, "factor", (2.0, 0.5))
    assert factored == (before_m * 2.0, before_f * 0.5)

    # 키 수와 값 수가 다르면 거부한다 (조용히 한쪽만 바꾸지 않는다).
    with pytest.raises(ValueError, match="키 2개에 값 1개"):
        sensitivity.resolve_point(cfg, scen, pair, "value", 3.0)


def test_core_preset_has_recalibrated_pair_axis():
    """CORE_PRESET 의 (m, f) 쌍은 B 가 계산한 재보정 값이다 (기준 자세 R ≈ 8.0%)."""
    from airis.optimize import sensitivity

    pairs = [(sensitivity.axis_keys(k), mode, vals) for k, mode, vals in sensitivity.CORE_PRESET
             if len(sensitivity.axis_keys(k)) > 1]
    assert len(pairs) == 1, "쌍 축은 하나뿐이다"
    keys, mode, vals = pairs[0]
    assert keys == ("jet.impingement.stagnation_shear_factor", "adhesion.fabric_roughness_factor")
    assert mode == "value" and vals == [(3.0, 0.57), (6.0, 0.72)]
    # 보조 대조점: f 를 0.45 로 둔 채 m 만 6 배.
    assert ("jet.impingement.stagnation_shear_factor", "value", [6.0]) in sensitivity.CORE_PRESET
