"""계획 인코딩·계획 최적화·E7 러너 테스트. 소유자: C. docs/plan_extension.md, 00_common.md 4.7."""
import csv
import json

import numpy as np
import pytest

from airis.optimize.plan_encoding import PlanEncoder, PlanLimits, plan_keys, wrap_deg
from airis.sim import ZONE_NAMES, BodyParams, Phase, Plan, PoseParams
from airis.sim.scenario import load_nozzles, load_scenarios

pytest.importorskip("cma", reason="cma 패키지 필요 (pip install cma)")

LIMITS = PlanLimits()          # 2단계, 5~20 s, 최소 2 s, s_max 1.0, cap_ratio 1.0


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios()


def _plan(yaws, *, abd=175.0, durations=(10.0, 10.0), zones=(1.0, 1.0, 1.0, 1.0, 1.0)) -> Plan:
    return Plan([Phase(PoseParams(shoulder_abduction=abd, torso_yaw=y), t) for y, t in zip(yaws, durations)],
                np.array(zones, dtype=np.float64))


def test_keys_match_interfaces_order(scenarios):
    enc = PlanEncoder(scenarios["default"], LIMITS)
    assert enc.dim == 21
    assert enc.keys[:7] == [f"p1_{k}" for k in
                            ("shoulder_abduction", "shoulder_flexion", "elbow_flexion", "torso_pitch",
                             "torso_yaw", "hip_flexion", "knee_flexion")]
    assert enc.keys[14:] == ["duration_s", "share_1"] + [f"zone_{z}" for z in ZONE_NAMES]
    assert enc.keys == plan_keys(2)


def test_keys_and_durations_match_flow_planspace(scenarios):
    """모델(F)의 PlanSpace 와 키 순서·단계 시간 식이 같아야 한다 (interfaces.md)."""
    ps = pytest.importorskip("airis.model.flow").PlanSpace.from_scenarios(
        [scenarios[n] for n in ("default", "pregnant", "wheelchair")], n_phases=2)
    enc = PlanEncoder(scenarios["default"], LIMITS)
    assert ps.keys == enc.keys
    assert ps.durations(17.0, [0.25]) == pytest.approx(enc.durations(17.0, [0.25]))


def test_durations_keep_minimum_phase(scenarios):
    enc = PlanEncoder(scenarios["default"], LIMITS)
    for total, share in ((5.0, 0.0), (20.0, 1.0), (12.0, 0.3), (20.0, 0.5)):
        t = enc.durations(total, [share])
        assert sum(t) == pytest.approx(total)
        assert min(t) >= LIMITS.min_phase_s - 1e-9


def test_decode_clips_and_round_trip(scenarios):
    enc = PlanEncoder(scenarios["default"], LIMITS)
    rng = np.random.default_rng(0)
    for _ in range(20):
        plan = enc.decode(rng.uniform(-3.0, 3.0, enc.dim))        # 범위 밖 제안도 안전
        assert LIMITS.duration_bounds_s[0] - 1e-9 <= plan.duration_s <= LIMITS.duration_bounds_s[1] + 1e-9
        assert min(p.duration_s for p in plan.phases) >= LIMITS.min_phase_s - 1e-9
        assert (plan.zone_strengths >= 0).all() and (plan.zone_strengths <= LIMITS.s_max + 1e-9).all()
        # encode 는 대칭 정규화를 거치므로 정규화한 계획과 비교한다 (정면 동률이면 좌우 세기가 바뀐다).
        norm = enc.clip_plan(enc.normalize(plan))
        back = enc.decode(enc.encode(plan))
        assert back.duration_s == pytest.approx(norm.duration_s, abs=1e-6)
        assert np.allclose(back.zone_strengths, norm.zone_strengths, atol=1e-6)
        for a, b in zip(back.phases, norm.phases):
            assert a.duration_s == pytest.approx(b.duration_s, abs=1e-6)
            assert a.pose.shoulder_abduction == pytest.approx(b.pose.shoulder_abduction, abs=1e-6)
            # yaw 는 −180° 와 180° 가 같은 각도라 감아서 비교한다.
            assert wrap_deg(a.pose.torso_yaw) == pytest.approx(wrap_deg(b.pose.torso_yaw), abs=1e-6)


def test_wheelchair_fixed_pose_and_yaw_range(scenarios):
    enc = PlanEncoder(scenarios["wheelchair"], LIMITS)
    plan = enc.decode(np.full(enc.dim, 1.0))
    for ph in plan.phases:
        assert ph.pose.hip_flexion == 90 and ph.pose.knee_flexion == 90
        assert -45 - 1e-9 <= ph.pose.torso_yaw <= 45 + 1e-9
    assert len(enc.free_keys) == 21 - 2 * 2          # 단계마다 hip·knee 고정


def test_comfort_cap_uses_zone_keys_only(scenarios):
    """구역 이름 키만 있는 시나리오에서도 쾌적 상한이 걸린다 (옛 부위 키 torso_front 없이).

    B 가 scenarios.yaml 의 옛 키를 지워도 clip_plan 이 상한을 건너뛰지 않아야 한다 (F 발견 2026-09-30).
    """
    from dataclasses import replace

    zone_only = replace(scenarios["default"], name="zone_only",
                        nozzle_strength_cap={"chest_low": 0.6, "chest_high": 0.6})
    enc = PlanEncoder(zone_only, LIMITS)
    assert enc.clip_zone_strengths([1.0] * 5) == pytest.approx([0.6, 0.6, 1.0, 1.0, 1.0])
    plan = enc.decode(np.full(enc.dim, 1.0))                  # 벡터 상한에서도 걸린다
    assert plan.zone_strengths[:2] == pytest.approx([0.6, 0.6])


def test_comfort_cap_and_flow_cap(scenarios):
    # 임산부: 가슴 쪽 구역 0.6 (scenario.nozzle_strength_cap 의 구역 키 chest_low·chest_high)
    preg = PlanEncoder(scenarios["pregnant"], LIMITS)
    s = preg.clip_zone_strengths([1.0] * 5)
    assert s[ZONE_NAMES.index("chest_low")] == pytest.approx(0.6)
    assert s[ZONE_NAMES.index("chest_high")] == pytest.approx(0.6)
    assert s[ZONE_NAMES.index("back_low")] == pytest.approx(1.0)

    # 풍량 한도: Σ 노즐 세기가 한도를 넘으면 전 구역을 같은 비율로 줄인다.
    # 구역별 노즐 수는 B 의 zone_nozzle_counts() (기준 배치 [2, 2, 2, 2, 4]).
    from airis.sim.scenario import zone_nozzle_counts

    counts = np.asarray(zone_nozzle_counts(), dtype=float)
    tight = PlanEncoder(scenarios["default"], PlanLimits(cap_ratio=0.5))
    assert np.allclose(tight.zone_counts, counts)
    s = tight.clip_zone_strengths([1.0, 1.0, 1.0, 1.0, 1.0])
    assert s == pytest.approx([0.5] * 5)
    ratios = tight.clip_zone_strengths([1.0, 0.5, 0.5, 0.5, 0.5])
    assert ratios[0] / ratios[1] == pytest.approx(2.0)           # 비율 유지
    assert float(counts @ ratios) <= 0.5 * counts.sum() + 1e-9   # 노즐 수로 가중한 한도

    # 구역별 노즐 수를 직접 주면 그 수로 가중한다 (top 이 많으면 top 을 줄이는 효과가 크다).
    weighted = PlanEncoder(scenarios["default"], PlanLimits(cap_ratio=0.5),
                           zone_nozzle_counts=[1, 1, 1, 1, 8])
    s = weighted.clip_zone_strengths([1.0, 1.0, 1.0, 1.0, 1.0])
    assert s == pytest.approx([0.5] * 5)
    assert PlanEncoder(scenarios["default"], PlanLimits(cap_ratio=1.0),
                       zone_nozzle_counts=[1, 1, 1, 1, 8]).clip_zone_strengths(
        [1.0] * 5) == pytest.approx([1.0] * 5)                    # 합이 한도와 같으면 그대로

    # 비균일 입력: 노즐이 많은 구역(top)이 세면 같은 입력이라도 더 많이 깎인다.
    many_top = PlanEncoder(scenarios["default"], PlanLimits(cap_ratio=0.5),
                           zone_nozzle_counts=[1, 1, 1, 1, 8])
    few_top = PlanEncoder(scenarios["default"], PlanLimits(cap_ratio=0.5),
                          zone_nozzle_counts=[8, 1, 1, 1, 1])
    strengths = [0.2, 0.2, 0.2, 0.2, 1.0]                         # top 만 센 계획
    a, b = many_top.clip_zone_strengths(strengths), few_top.clip_zone_strengths(strengths)
    assert a[4] < b[4], "top 노즐이 많은 배치에서 더 많이 줄어든다"
    assert a[0] / a[4] == pytest.approx(b[0] / b[4]), "구역 사이 비율은 유지된다"


def test_normalize_mirror_front_back_and_wrap(scenarios):
    enc = PlanEncoder(scenarios["default"], LIMITS)
    ref = enc.normalize(_plan([72.0, -20.0]))
    assert [round(p.pose.torso_yaw, 6) for p in ref.phases] == [72.0, -20.0]

    # 좌우 거울: 두 단계가 함께 뒤집힌다.
    mirrored = enc.normalize(_plan([-72.0, 20.0]))
    assert [round(p.pose.torso_yaw, 6) for p in mirrored.phases] == [72.0, -20.0]

    # 앞뒤 등가 + 각도 감기: 1단계 108° → 72°, 2단계 −170° → 350° → −10°
    flipped = enc.normalize(_plan([108.0, -170.0]))
    assert flipped.phases[0].pose.torso_yaw == pytest.approx(72.0)
    assert flipped.phases[1].pose.torso_yaw == pytest.approx(-10.0)

    # 단계 관계가 유지된다 (단계마다 따로 접지 않는다).
    two_sided = enc.normalize(_plan([-80.0, 100.0]))
    assert two_sided.phases[0].pose.torso_yaw == pytest.approx(80.0)
    assert two_sided.phases[1].pose.torso_yaw == pytest.approx(-100.0)

    assert wrap_deg(350.0) == pytest.approx(-10.0)
    assert wrap_deg(-180.0) == 180.0 and wrap_deg(180.0) == 180.0


def test_normalize_skips_transform_outside_scenario_bounds(scenarios):
    """휠체어(±45°)는 앞뒤 등가를 쓰면 범위를 벗어나므로 거울만 적용한다."""
    enc = PlanEncoder(scenarios["wheelchair"], LIMITS)
    out = enc.normalize(_plan([-30.0, 40.0]))
    assert [round(p.pose.torso_yaw, 6) for p in out.phases] == [30.0, -40.0]
    keep = enc.normalize(_plan([30.0, -40.0]))
    assert [round(p.pose.torso_yaw, 6) for p in keep.phases] == [30.0, -40.0]


def test_normalize_front_tie_swaps_chest_back(scenarios):
    enc = PlanEncoder(scenarios["default"], LIMITS)
    front = enc.normalize(_plan([0.0, 30.0], zones=(0.2, 0.3, 0.9, 0.8, 0.5)))
    z = front.zone_strengths
    assert z[ZONE_NAMES.index("chest_low")] == pytest.approx(0.9)
    assert z[ZONE_NAMES.index("chest_high")] == pytest.approx(0.8)
    assert z[ZONE_NAMES.index("back_low")] == pytest.approx(0.2)
    assert z[ZONE_NAMES.index("top")] == pytest.approx(0.5)
    # 이미 chest 가 크면 그대로 둔다. 옆으로 선 계획도 건드리지 않는다.
    assert enc.normalize(_plan([0.0, 30.0], zones=(0.9, 0.8, 0.2, 0.3, 0.5))).zone_strengths[0] == pytest.approx(0.9)
    assert enc.normalize(_plan([72.0, 30.0], zones=(0.2, 0.3, 0.9, 0.8, 0.5))).zone_strengths[0] == pytest.approx(0.2)


def test_fixed_poses_only_search_time_and_strength(scenarios):
    pose = PoseParams(shoulder_abduction=179.0, torso_yaw=71.0)
    enc = PlanEncoder(scenarios["default"], LIMITS, fixed_poses=[pose, pose])
    plan = enc.decode(np.random.default_rng(1).uniform(-1, 1, enc.dim))
    for ph in plan.phases:
        assert ph.pose.shoulder_abduction == pytest.approx(179.0)
        assert ph.pose.torso_yaw == pytest.approx(71.0)
    assert enc.free_keys == ["duration_s", "share_1"] + [f"zone_{z}" for z in ZONE_NAMES]


# ---------- 계획 최적화 ----------

def test_run_cmaes_plan_optimizes_and_is_deterministic(scenarios):
    from airis.optimize.cmaes_runner import run_cmaes_plan
    from airis.optimize.dummy import DummyEvaluator

    sc = scenarios["default"]
    ev = DummyEvaluator(PoseParams(shoulder_abduction=175.0, torso_yaw=72.0), sc)
    kw = dict(max_evals=400, popsize=20, seed=0, limits=LIMITS)
    a = run_cmaes_plan(ev, BodyParams(), sc, load_nozzles(), **kw)
    b = run_cmaes_plan(ev, BodyParams(), sc, load_nozzles(), **kw)
    assert a.history == b.history and a.best_score == b.best_score
    assert a.best_plan is not None and len(a.best_plan.phases) == 2
    assert a.best_pose == a.best_plan.phases[0].pose
    assert a.best_plan.duration_s <= LIMITS.duration_bounds_s[1] + 1e-9
    # 목표 자세 쪽으로 간다 (더미는 자세 거리 점수).
    assert a.best_plan.phases[0].pose.shoulder_abduction > 120
    assert a.per_start[0]["best_pose"]["duration_s"] > 0


def test_run_cmaes_plan_with_pose_start_and_fixed_poses(scenarios):
    from airis.optimize.cmaes_runner import Start, run_cmaes_plan
    from airis.optimize.dummy import DummyEvaluator

    sc = scenarios["wheelchair"]
    target = PoseParams(shoulder_abduction=175.0, torso_yaw=36.0)
    ev = DummyEvaluator(target, sc)
    # 자세 시작점을 주면 그 자세를 모든 단계에 쓰는 계획에서 시작한다.
    res = run_cmaes_plan(ev, BodyParams(), sc, load_nozzles(), max_evals=200, popsize=20, seed=0,
                         limits=LIMITS, starts=[Start("hands_up", target)])
    assert res.best_plan is not None
    for ph in res.best_plan.phases:
        assert ph.pose.hip_flexion == 90 and ph.pose.knee_flexion == 90

    fixed = run_cmaes_plan(ev, BodyParams(), sc, load_nozzles(), max_evals=200, popsize=20, seed=0,
                           limits=LIMITS, fixed_poses=[target, target])
    for ph in fixed.best_plan.phases:
        assert ph.pose.shoulder_abduction == pytest.approx(
            PlanEncoder(sc, LIMITS).pose_encoder.clip_pose(target).shoulder_abduction)


def test_dummy_evaluate_plan_rewards_time_and_penalizes_energy(scenarios):
    from airis.optimize.dummy import DummyEvaluator

    sc = scenarios["default"]
    target = PoseParams(shoulder_abduction=175.0, torso_yaw=72.0)
    ev = DummyEvaluator(target, sc, energy_weight=0.0)
    nz = load_nozzles()
    # 목표에서 떨어진 자세여야 시간 포화 효과가 보인다 (더미 점수는 −거리²).
    away = PoseParams(shoulder_abduction=120.0, torso_yaw=40.0)
    short = Plan([Phase(away, 2.0), Phase(away, 2.0)], np.ones(5))
    long = Plan([Phase(away, 10.0), Phase(away, 10.0)], np.ones(5))
    assert ev.evaluate_plan(long, nz, BodyParams(), sc).score > ev.evaluate_plan(short, nz, BodyParams(), sc).score

    costly = DummyEvaluator(target, sc, energy_weight=1.0)
    quiet = Plan([Phase(away, 10.0), Phase(away, 10.0)], np.full(5, 0.2))
    assert costly.evaluate_plan(quiet, nz, BodyParams(), sc).score > \
        costly.evaluate_plan(long, nz, BodyParams(), sc).score
    extra = ev.evaluate_plan(long, nz, BodyParams(), sc).extra
    assert extra["duration_s"] == pytest.approx(20.0) and "energy" in extra


# ---------- 기준선과 E7 러너 ----------

def test_plan_baselines(scenarios):
    from airis.optimize.baselines import P1_PHASES, plan_baseline

    sc = scenarios["default"]
    best = PoseParams(shoulder_abduction=179.9, torso_yaw=71.0)
    p0 = plan_baseline("P0", sc, LIMITS)
    assert len(p0.phases) == 1 and p0.duration_s == pytest.approx(LIMITS.duration_bounds_s[1])
    assert p0.phases[0].pose.shoulder_abduction == PoseParams().shoulder_abduction

    p1 = plan_baseline("P1", sc, LIMITS)
    assert len(p1.phases) == P1_PHASES and p1.duration_s == pytest.approx(20.0)
    assert sorted({round(p.pose.torso_yaw) for p in p1.phases}) == [-150, -120, -90, -60, -30, 0,
                                                                    30, 60, 90, 120, 150, 180]

    p2 = plan_baseline("P2", sc, LIMITS, best_pose=best)
    assert len(p2.phases) == LIMITS.n_phases
    assert all(p.pose.shoulder_abduction == pytest.approx(179.9) for p in p2.phases)
    with pytest.raises(ValueError):
        plan_baseline("P2", sc, LIMITS)
    with pytest.raises(ValueError):
        plan_baseline("P9", sc, LIMITS)

    # 휠체어는 회전 범위(±45°) 안으로 투영된다.
    assert all(-45 <= p.pose.torso_yaw <= 45 for p in plan_baseline("P1", scenarios["wheelchair"], LIMITS).phases)


def test_run_e7_dummy_end_to_end(tmp_path):
    from scripts.run_e7 import main

    code = main(["--evaluator", "dummy", "--scenarios", "default", "--conditions", "P0,P2,P3,P5",
                 "--max-evals", "200", "--pose-max-evals", "200", "--popsize", "10",
                 "--energy-weights", "0", "0.4", "--log-dir", str(tmp_path)])
    assert code == 0
    (group,) = [p for p in tmp_path.iterdir() if p.is_dir() and (p / "e7_summary.csv").exists()]
    with (group / "e7_summary.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {(r["condition"], r["energy_weight"]) for r in rows} == {
        (c, w) for c in ("P0", "P2", "P3", "P5") for w in ("0.0", "0.4")}
    for r in rows:
        assert float(r["duration_s"]) > 0 and int(r["n_phases"]) >= 1
        assert r["infeasible"] == "False"
    # 에너지 가중이 커지면 P5 가 에너지를 줄인다.
    e = {r["energy_weight"]: float(r["energy"]) for r in rows if r["condition"] == "P5"}
    assert e["0.4"] < e["0.0"]

    plans = json.loads((group / "e7_plans.json").read_text(encoding="utf-8"))
    p5 = plans["scenarios"]["default"]["w0.4"]["P5_s0"]
    assert len(p5["phases"]) == 2 and len(p5["zone_strengths"]) == len(ZONE_NAMES)


def test_run_e7_rejects_unknown_condition(tmp_path):
    from scripts.run_e7 import main

    assert main(["--evaluator", "dummy", "--conditions", "P9", "--log-dir", str(tmp_path)]) == 2


# ---------- 계획 경로용 설정 (kinetics) ----------

def test_plan_physics_cfg_enables_kinetics_without_touching_input():
    from airis.optimize.plan_encoding import plan_kinetics_stamp, plan_physics_cfg
    from airis.sim.scenario import load_physics

    base = load_physics()
    before = bool(base.get("adhesion", {}).get("kinetics", {}).get("enabled", False))
    cfg = plan_physics_cfg(base)
    assert cfg["adhesion"]["kinetics"]["enabled"] is True
    assert bool(base["adhesion"]["kinetics"]["enabled"]) == before, "입력 설정은 그대로"
    assert cfg["adhesion"] is not base["adhesion"], "복사본이어야 한다"
    stamp = plan_kinetics_stamp(cfg)
    assert stamp["kinetics_enabled"] is True
    assert stamp["time_constant_s"] == pytest.approx(
        float(base["adhesion"]["kinetics"]["time_constant_s"]))
    # 키가 아예 없는 설정에서도 만들어 넣는다.
    assert plan_physics_cfg({})["adhesion"]["kinetics"]["enabled"] is True


def test_dummy_plan_discomfort_uses_reference_duration(scenarios):
    """더미의 discomfort 는 명세 4.4 대로 시간 가중 합 ÷ T_ref (÷T 아님)."""
    from airis.optimize.dummy import DummyEvaluator
    from airis.sim.scoring import discomfort as pose_discomfort

    sc = scenarios["default"]
    pose = PoseParams(shoulder_abduction=180.0, torso_yaw=70.0)
    ev = DummyEvaluator(pose, sc)
    half = Plan([Phase(pose, 5.0), Phase(pose, 5.0)], np.ones(5))
    full = Plan([Phase(pose, 10.0), Phase(pose, 10.0)], np.ones(5))
    d_pose = pose_discomfort(PlanEncoder(sc, LIMITS).pose_encoder.clip_pose(pose), sc)
    nz = load_nozzles()
    assert ev.evaluate_plan(half, nz, BodyParams(), sc).discomfort == pytest.approx(d_pose * 10.0 / 20.0)
    assert ev.evaluate_plan(full, nz, BodyParams(), sc).discomfort == pytest.approx(d_pose)


# ---------- 구역 노즐 수 폴백 (#98 검토 후속 3) ----------

def test_default_zone_counts_warns_on_broken_config(monkeypatch):
    """배치 파일이 없으면 조용히 1, 설정이 어긋나면 경고를 내고 1."""
    import warnings

    import numpy as np

    from airis.optimize import plan_encoding as pe

    ones = np.ones(len(pe.ZONE_NAMES))

    def missing():
        raise FileNotFoundError("nozzles.yaml")

    monkeypatch.setattr(pe, "_load_zone_counts", missing)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # 경고가 나면 실패
        assert np.array_equal(pe.default_zone_counts(), ones)

    def broken():
        raise ValueError("좌우 노즐 수가 다르다")

    monkeypatch.setattr(pe, "_load_zone_counts", broken)
    with pytest.warns(RuntimeWarning, match="구역별 노즐 수"):
        assert np.array_equal(pe.default_zone_counts(), ones)


def test_dummy_reference_duration_comes_from_config():
    """더미의 T_ref 는 설정값이고, 넘기면 그 값을 쓴다 (#98 검토 후속 2)."""
    from airis.optimize.dummy import DummyEvaluator, _config_reference_duration
    from airis.sim import PoseParams
    from airis.sim.scenario import load_physics, load_scenarios

    scenario = load_scenarios()["default"]
    expected = float(load_physics()["scoring"]["reference_duration_s"])
    assert _config_reference_duration() == expected
    assert DummyEvaluator(PoseParams(), scenario).reference_duration_s == expected
    assert DummyEvaluator(PoseParams(), scenario, reference_duration_s=7.5).reference_duration_s == 7.5


# ---------- e7_reference.json 내보내기 ----------

def test_export_e7_reference(tmp_path):
    """E7 묶음 → 기준 수치 한 장. 조건마다 지표와 21차원 계획이 함께 들어간다."""
    import csv
    import importlib.util
    import json
    from pathlib import Path

    from airis.sim import ZONE_NAMES

    root = Path(__file__).resolve().parents[1]
    bundle = tmp_path / "e7x"
    bundle.mkdir()
    pose = {"shoulder_abduction": 179.0, "shoulder_flexion": 0.0, "elbow_flexion": 1.0,
            "torso_pitch": 0.0, "torso_yaw": -100.0, "hip_flexion": 0.0, "knee_flexion": 5.0}
    plan = {"duration_s": 12.0,
            "phases": [{"duration_s": 7.0, **pose}, {"duration_s": 5.0, **pose}],
            "zone_strengths": [1.0, 0.6, 0.9, 0.9, 0.5]}
    (bundle / "e7_plans.json").write_text(json.dumps({
        "group_id": "e7x", "commit": "c", "physics_hash": "p", "nozzle_hash": "n",
        "kinetics_enabled": True, "time_constant_s": 2.0, "zone_nozzle_counts": [2, 2, 2, 2, 4],
        "args": {"max_evals": 10},
        "scenarios": {"default": {"w0.1": {"P5_s0": plan}}},
    }, ensure_ascii=False), encoding="utf-8")
    with (bundle / "e7_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["scenario", "condition", "energy_weight", "seed",
                                           "infeasible", "score", "total_removal", "discomfort",
                                           "energy", "duration_s", "n_phases",
                                           "phase_durations_s", "zone_strengths"])
        w.writeheader()
        w.writerow({"scenario": "default", "condition": "P5", "energy_weight": "0.1", "seed": "0",
                    "infeasible": "False", "score": "0.85", "total_removal": "0.27",
                    "discomfort": "0.19", "energy": "0.59", "duration_s": "12.0", "n_phases": "2",
                    "phase_durations_s": "7.0;5.0", "zone_strengths": "1.0;0.6;0.9;0.9;0.5"})

    spec = importlib.util.spec_from_file_location(
        "export_e7_reference", root / "scripts" / "export_e7_reference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "e7_reference.json"
    assert mod.main(["--bundle", str(bundle), "--out", str(out), "--note", "설명"]) == 0

    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["bundles"]) == 1                 # --bundle 을 여러 번 주면 한 파일로 모은다
    meta = data["bundles"][0]
    assert meta["kinetics_enabled"] is True and meta["note"] == "설명"
    assert meta["zone_nozzle_counts"]["top"] == 4 and meta["max_evals"] == 10
    assert not Path(meta["path"]).is_absolute()
    row = data["rows"][0]
    assert row["bundle"] == meta["group_id"]
    assert (row["scenario"], row["condition"], row["energy_weight"]) == ("default", "P5", 0.1)
    assert row["total_removal"] == 0.27 and row["energy"] == 0.59 and row["n_phases"] == 2
    assert row["phase_durations_s"] == [7.0, 5.0] and row["infeasible"] is False
    # 계획이 21차원 그대로 들어간다: 단계별 자세 7개 + 시간, 구역 세기 5개.
    assert len(row["plan"]["phases"]) == 2
    assert set(row["plan"]["phases"][0]["pose"]) == set(data["pose_keys"])
    assert row["plan"]["phases"][0]["torso_yaw_folded"] == pytest.approx(80.0)
    assert row["plan"]["zone_strengths"] == dict(zip(ZONE_NAMES, plan["zone_strengths"]))
    json.dumps(data, allow_nan=False)

    # 묶음 여러 개를 한 파일로 (단계 수 스윕처럼 묶음이 쪼개진 실험).
    import shutil
    second = tmp_path / "e7y"
    shutil.copytree(bundle, second)
    out2 = tmp_path / "merged.json"
    assert mod.main(["--bundle", str(bundle), "--bundle", str(second), "--out", str(out2)]) == 0
    merged = json.loads(out2.read_text(encoding="utf-8"))
    assert len(merged["bundles"]) == 2 and len(merged["rows"]) == 2
    assert {b["group_id"] for b in merged["bundles"]} == {"e7x"}      # group_id 는 파일 안 값
    assert "bundle" not in merged, "묶음이 여럿이면 단수 키를 쓰지 않는다"
    # 묶음이 하나면 예전 모양(bundle)도 남긴다 — A 의 verify_e7_plans.py 가 그 키를 읽는다.
    assert data["bundle"] == data["bundles"][0]


# ---------- E7 회전 기준선과 단계 수 덮어쓰기 (총괄 2026-10-03) ----------

def test_parse_baseline_name():
    """'P1_10' 처럼 회전 단계 수를 이름에 붙인다."""
    from airis.optimize.baselines import parse_baseline_name

    assert parse_baseline_name("P1") == ("P1", None)
    assert parse_baseline_name("P1_10") == ("P1", 10)
    assert parse_baseline_name("P1opt_8") == ("P1opt", 8)
    assert parse_baseline_name("P1down") == ("P1down", None)
    for bad in ("P1_0", "P1_x", "P1_-3"):
        with pytest.raises(ValueError):
            parse_baseline_name(bad)


def test_rotation_baselines(scenarios):
    """P1 계열은 자세를 유지한 채 n 방향으로 돌린다. 세기·시간은 P0 과 같다."""
    from airis.optimize import baselines
    from airis.optimize.plan_encoding import PlanEncoder, PlanLimits
    from airis.sim import PoseParams

    scen = scenarios["default"]
    limits = PlanLimits()
    enc = PlanEncoder(scen, limits)
    total = limits.duration_bounds_s[1]
    best = PoseParams(shoulder_abduction=177.0, torso_yaw=70.0)
    down = PoseParams(shoulder_abduction=19.6, torso_yaw=-77.0)

    p1 = baselines.plan_baseline("P1", scen, limits)
    assert len(p1.phases) == baselines.P1_PHASES
    assert sum(ph.duration_s for ph in p1.phases) == pytest.approx(total)
    base_abd = PoseParams().shoulder_abduction          # 기본 자세(20°)를 그대로 유지한다
    assert all(ph.pose.shoulder_abduction == pytest.approx(base_abd) for ph in p1.phases)

    # 단계 수를 바꾸면 단계 시간이 그만큼 길어진다 (20/8 = 2.5 s).
    p1_8 = baselines.plan_baseline("P1_8", scen, limits)
    assert len(p1_8.phases) == 8
    assert p1_8.phases[0].duration_s == pytest.approx(total / 8)

    # 회전 자세만 다르고 나머지는 같다.
    p1opt = baselines.plan_baseline("P1opt", scen, limits, best_pose=best)
    assert [ph.pose.shoulder_abduction for ph in p1opt.phases] == [pytest.approx(177.0)] * 12
    p1down = baselines.plan_baseline("P1down_10", scen, limits, rotation_pose=down)
    assert len(p1down.phases) == 10
    assert all(ph.pose.shoulder_abduction == pytest.approx(19.6) for ph in p1down.phases)

    # yaw 는 0°부터 균등하게 돌고 시나리오 범위 안으로 투영된다 (휠체어 ±45°).
    seated = baselines.plan_baseline("P1_12", scenarios["wheelchair"], limits)
    lo, hi = scenarios["wheelchair"].pose_bounds["torso_yaw"]
    assert all(lo - 1e-9 <= ph.pose.torso_yaw <= hi + 1e-9 for ph in seated.phases)

    # 자세가 없으면 거부한다 (조용히 기본 자세로 돌리지 않는다).
    with pytest.raises(ValueError, match="best_pose"):
        baselines.plan_baseline("P1opt", scen, limits)
    with pytest.raises(ValueError, match="rotation_pose"):
        baselines.plan_baseline("P1down", scen, limits)
    # 세기는 P0 과 같다 (쾌적 상한만 적용).
    assert np.allclose(p1.zone_strengths, enc.clip_zone_strengths(
        np.full(len(ZONE_NAMES), limits.s_max)))


def test_plan_dim_grows_with_phases(scenarios):
    """단계 수 N 의 계획 차원은 8N + 5 다 (자세 7 + 단계 시간 몫 1, 총 시간 1 + 구역 5)."""
    import dataclasses

    from airis.optimize.plan_encoding import PlanEncoder, PlanLimits

    scen = scenarios["default"]
    for n in (2, 3, 6):
        limits = dataclasses.replace(PlanLimits(), n_phases=n,
                                     duration_bounds_s=(max(5.0, n * 2.0), 20.0))
        assert PlanEncoder(scen, limits).dim == 8 * n + 5

    # 단계가 많아지면 총 시간 상한(20 s)에 막힌다: N × min_phase_s ≤ 20 → N ≤ 10.
    too_many = dataclasses.replace(PlanLimits(), n_phases=12, duration_bounds_s=(24.0, 20.0))
    with pytest.raises(ValueError):
        PlanEncoder(scen, too_many)


def test_run_e7_cli_phase_and_rotation_options(tmp_path):
    """run_e7 의 --n-phases·--rotation-pose·P1_N 조건 처리 (통합 2026-10-04 요청).

    상한을 넘는 단계 수를 조용히 넘기면 PlanEncoder 가 죽는다 — 실제로 N=10 실행이 그렇게 죽었다.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    def run(args):
        return subprocess.run([sys.executable, str(root / "scripts" / "run_e7.py"),
                               "--evaluator", "dummy", "--scenarios", "default", "--seeds", "0",
                               "--energy-weights", "0.1", "--max-evals", "60", "--pose-max-evals", "60",
                               "--popsize", "10", "--log-dir", str(tmp_path), *args],
                              cwd=root, capture_output=True, text=True, encoding="utf-8")

    # N × min_phase_s 가 총 시간 상한 이상이면 돌기 전에 멈추고 길을 알려 준다.
    out = run(["--conditions", "P5", "--n-phases", "10", "--tag", "cap"])
    assert out.returncode == 2
    assert "N × min_phase_s < 상한" in out.stderr and "9 이하" in out.stderr

    # 단계 수를 덮어쓰면 그 단계 수로 돈다 (configs 는 그대로).
    out = run(["--conditions", "P5", "--n-phases", "3", "--tag", "np3"])
    assert out.returncode == 0, out.stderr
    assert "n_phases=3" in out.stdout
    plans = json.loads(next(tmp_path.glob("np3_*/e7_plans.json")).read_text(encoding="utf-8"))
    assert plans["args"]["n_phases"] == 3
    assert len(plans["scenarios"]["default"]["w0.1"]["P5_s0"]["phases"]) == 3

    # P1_N 조건을 받고, P1down 은 자세가 없으면 돌기 전에 멈춘다.
    out = run(["--conditions", "P1_8,P1opt_10", "--tag", "p1n"])
    assert out.returncode == 0, out.stderr
    assert "P1_8" in out.stdout and "P1opt_10" in out.stdout
    out = run(["--conditions", "P1down", "--tag", "nopose"])
    assert out.returncode == 2 and "--rotation-pose" in out.stderr
    out = run(["--conditions", "P1_x", "--tag", "bad"])
    assert out.returncode == 2 and "알 수 없는 조건" in out.stderr


def test_warm_start_uses_single_pose_optimum(tmp_path):
    """--warm-start 는 P5 를 단일 자세 최적 하나에서, 작은 스텝으로 출발시킨다 (총괄 10-06).

    점수 함수·밀도는 그대로이고 **시작점과 초기 스텝만** 바뀌어야 한다.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    def run(*extra):
        return subprocess.run([sys.executable, str(root / "scripts" / "run_e7.py"),
                               "--evaluator", "dummy", "--scenarios", "default", "--seeds", "0",
                               "--energy-weights", "0.1", "--conditions", "P5",
                               "--max-evals", "200", "--pose-max-evals", "120", "--popsize", "10",
                               "--log-dir", str(tmp_path), *extra],
                              cwd=root, capture_output=True, text=True, encoding="utf-8")

    warm = run("--warm-start", "--tag", "w")
    assert warm.returncode == 0, warm.stderr
    assert "warm_start(sigma0 0.2)" in warm.stdout, "기본 sigma0 는 0.2"
    plans = json.loads(next(tmp_path.glob("w_*/e7_plans.json")).read_text(encoding="utf-8"))
    assert plans["args"]["warm_start"] == 0.2

    # 시작점 이름이 history.csv 의 start 열에 남는다 (계획 실행은 meta.json 을 쓰지 않는다).
    def starts_used(tag):
        rows = []
        for path in sorted(tmp_path.glob(f"{tag}p5_*/history.csv")):
            with path.open(encoding="utf-8") as fh:
                rows += list(csv.DictReader(fh))
        assert rows, f"{tag} 의 P5 기록이 있어야 한다"
        seen = []
        for r in rows:                                   # 순서를 지킨 중복 제거
            if r["start"] not in seen:
                seen.append(r["start"])
        return seen

    assert starts_used("w") == ["warm"]

    # 값을 직접 주면 그 값을 쓴다.
    out = run("--warm-start", "0.15", "--tag", "w15")
    assert out.returncode == 0 and "warm_start(sigma0 0.15)" in out.stdout

    # 끄면 예전처럼 시작점 두 개(default·hands_up)에서 출발한다.
    cold = run("--tag", "c")
    assert cold.returncode == 0 and "warm_start(sigma0" not in cold.stdout
    assert starts_used("c") == ["default", "hands_up"]


# ---------- 계획 데이터셋 (C·F 합의 스키마 2026-10-06) ----------

def test_plan_dataset_end_to_end(tmp_path):
    """더미 평가기로 2행 — 열 이름·단계 수·도장·한도·후보·기준선과 이어 만들기."""
    import numpy as np

    from airis.optimize.plan_dataset import PlanDatasetConfig, build_plan_dataset, plan_limits
    from airis.optimize.plan_encoding import plan_keys

    out = tmp_path / "plan.parquet"
    cfg = PlanDatasetConfig(n_phases=3, max_evals=300, pose_max_evals=150, popsize=10,
                            evaluator="dummy", candidate_k=4)
    res = build_plan_dataset(out, n_bodies=2, scenarios=["default"], cfg=cfg,
                             processes=1, flush_every=1, log=lambda *a: None)
    assert (res["n_rows"], res["n_new"], res["n_skipped"]) == (2, 2, 0)

    import pandas as pd

    df = pd.read_parquet(out)
    # 계획 열은 8N + 5 개이고 이름·순서가 interfaces.md 키 순서다.
    expected = [f"plan_{k}" for k in plan_keys(3)]
    assert [c for c in df.columns if c.startswith("plan_")] == expected
    assert len(expected) == 8 * 3 + 5

    # 도장 10개 (zone_nozzle_counts 는 문자열) + 한도 6개.
    for col in ("physics_hash", "nozzle_layout_hash", "body_model", "patches_per_m2", "commit",
                "kinetics_enabled", "time_constant_s", "zone_nozzle_counts", "n_phases",
                "energy_weight"):
        assert col in df.columns, col
        assert df[col].nunique(dropna=False) == 1, col
    assert isinstance(df["zone_nozzle_counts"].iloc[0], str)
    assert df["kinetics_enabled"].iloc[0] is True or df["kinetics_enabled"].iloc[0] == True  # noqa: E712
    limits = plan_limits(cfg)
    assert df["duration_lo_s"].iloc[0] == pytest.approx(limits.duration_bounds_s[0])
    assert df["duration_lo_s"].iloc[0] == pytest.approx(3 * limits.min_phase_s), "하한이 N×최소로 오른다"
    for col in ("duration_hi_s", "min_phase_s", "transition_s", "s_max", "cap_ratio"):
        assert col in df.columns, col

    # 후보: raw 벡터 길이가 차원과 같고 점수는 내림차순, 수가 k 이하.
    for raws, scores, n in zip(df["cand_plan_raw"], df["cand_score"], df["n_candidates"]):
        assert 0 < n <= cfg.candidate_k
        assert all(len(v) == len(expected) for v in raws)
        assert list(scores) == sorted(scores, reverse=True)

    # 선택 열: 기준선 두 개와 단일 자세 최적.
    for col in ("score_p1_10", "score_p1opt_10", "pose_shoulder_abduction", "pose_score"):
        assert col in df.columns, col
    assert np.isfinite(df["score"]).all() and np.isfinite(df["score_p1opt_10"]).all()

    # 체형은 자세 데이터셋과 같은 body_idx (body_seed 0).
    from airis.optimize.dataset import sample_bodies
    bodies = sample_bodies(2, 0, df["body_model"].iloc[0])
    assert df.sort_values("body_idx")["body_height_m"].tolist() == [
        pytest.approx(b.height_m) for b in bodies]

    # 다시 돌리면 끝난 행을 건너뛴다.
    again = build_plan_dataset(out, n_bodies=2, scenarios=["default"], cfg=cfg,
                               processes=1, flush_every=1, log=lambda *a: None)
    assert (again["n_new"], again["n_skipped"]) == (0, 2)

    # 설정이 다르면 섞지 않는다.
    with pytest.raises(ValueError, match="섞을 수 없다"):
        build_plan_dataset(out, n_bodies=2, scenarios=["default"],
                           cfg=PlanDatasetConfig(n_phases=4, max_evals=300, pose_max_evals=150,
                                                 popsize=10, evaluator="dummy", candidate_k=4),
                           processes=1, log=lambda *a: None)


def test_plan_limits_rejects_too_many_phases():
    """N × min_phase_s 가 총 시간 상한 이상이면 거부한다 (N=10 실행이 그렇게 죽었다)."""
    from airis.optimize.plan_dataset import PlanDatasetConfig, plan_limits

    assert plan_limits(PlanDatasetConfig(n_phases=9)).duration_bounds_s == (18.0, 20.0)
    with pytest.raises(ValueError, match="N ≤ 9"):
        plan_limits(PlanDatasetConfig(n_phases=10))


def test_candidate_raw_is_symmetry_normalized(scenarios):
    """후보 저장 벡터는 대칭 정규화를 거친다 — decode 는 clip 만 한다 (통합 검토 10-07).

    1단계 yaw 가 정면(0°)이면 `normalize` 가 chest/back 구역을 맞바꾼다. 그 규칙이 decode 에는
    없어서, 그냥 decode 하면 같은 계획이 구역 순서만 다른 값으로 저장된다.
    """
    import numpy as np

    from airis.optimize.plan_dataset import candidate_columns
    from airis.optimize.plan_encoding import PlanEncoder, PlanLimits, plan_keys

    enc = PlanEncoder(scenarios["default"], PlanLimits(n_phases=3, duration_bounds_s=(6.0, 20.0)))
    keys = plan_keys(3)
    x = np.zeros(enc.dim)
    x[keys.index("p1_torso_yaw")] = -1.0            # 1단계 yaw 를 범위 하한(정면 쪽)으로
    x[keys.index("zone_chest_low")] = -1.0          # 가슴 약, 등 강 → normalize 가 맞바꾼다
    x[keys.index("zone_chest_high")] = -1.0
    x[keys.index("zone_back_low")] = 1.0
    x[keys.index("zone_back_high")] = 1.0

    out = candidate_columns([{"x": x, "score": 1.0, "infeasible": False}], enc, 1.0,
                            k=4, tol=0.02, min_dist=0.05)
    raw = dict(zip(keys, out["cand_plan_raw"][0]))
    plain = dict(zip(keys, enc.raw_vector(enc.decode(x))))

    assert abs(raw["p1_torso_yaw"]) < 1e-9, "정면이면 1단계 yaw 는 0 으로 접힌다"
    # 맞바꿈이 실제로 일어나 저장값과 '그냥 decode' 값이 다르다 — 이 테스트가 구분력이 있다.
    assert raw["zone_chest_low"] > raw["zone_back_low"], raw
    assert plain["zone_chest_low"] < plain["zone_back_low"], plain
    assert raw["zone_chest_low"] == pytest.approx(plain["zone_back_low"])
    assert raw["zone_back_low"] == pytest.approx(plain["zone_chest_low"])


def test_plan_dataset_rejects_file_without_stamp_columns(tmp_path):
    """도장 열이 없는 옛 파일에는 이어 붙이지 않는다 — NaN 행이 섞이면 도장 검사가 무의미해진다."""
    import pandas as pd

    from airis.optimize.plan_dataset import PlanDatasetConfig, build_plan_dataset

    out = tmp_path / "old.parquet"
    pd.DataFrame([{"body_idx": 0, "scenario": "default", "score": 1.0}]).to_parquet(out, index=False)
    with pytest.raises(ValueError, match="도장 열이 없다"):
        build_plan_dataset(out, n_bodies=1, scenarios=["default"],
                           cfg=PlanDatasetConfig(n_phases=3, evaluator="dummy"),
                           processes=1, log=lambda *a: None)


def test_candidate_filter_compares_canonical_plans(scenarios):
    """탐색 공간에서 멀어 보여도 **같은 계획이면** 후보 하나만 남는다 (60행 실측에서 발견).

    `run_cmaes` 가 남기는 x 는 [-1, 1] 밖으로도 나가고 `decode` 가 그것을 잘라 낸다. 그래서
    x 끼리 재면 멀지만 계획은 똑같은 쌍이 생긴다(저장된 쌍거리 최소 0.097 < 임계 0.15).
    거리는 `encode(decode(x))` 정규 형태로 재야 한다.
    """
    import numpy as np

    from airis.optimize.plan_dataset import candidate_columns, plan_from_raw, refilter_candidates
    from airis.optimize.plan_encoding import PlanEncoder, PlanLimits

    enc = PlanEncoder(scenarios["default"], PlanLimits(n_phases=3, duration_bounds_s=(6.0, 20.0)))
    near = np.full(enc.dim, 1.5)                 # 범위 밖 — decode 가 1.0 으로 자른다
    far_out = np.full(enc.dim, 3.0)              # 더 밖이지만 잘리면 같은 계획
    raw_gap = float(np.linalg.norm(near - far_out)) / np.sqrt(enc.dim)
    assert raw_gap > 0.15, f"x 끼리는 멀어 보인다 ({raw_gap:.3f}) — 이 테스트의 전제"
    assert np.allclose(enc.raw_vector(enc.decode(near)), enc.raw_vector(enc.decode(far_out)))

    out = candidate_columns(
        [{"x": near, "score": 1.0, "infeasible": False},
         {"x": far_out, "score": 0.99, "infeasible": False}],
        enc, 1.0, k=4, tol=0.05, min_dist=0.15)
    assert out["n_candidates"] == 1, "같은 계획은 하나만 남아야 한다"
    assert out["cand_score"] == [1.0], "점수가 높은 쪽을 남긴다"

    # 읽는 쪽 재필터도 같은 결과 (옛 파일 보정용).
    raws = [[float(v) for v in enc.raw_vector(enc.decode(x))] for x in (near, far_out)]
    again = refilter_candidates(raws, [1.0, 0.99], enc, min_dist=0.15)
    assert again["n_candidates"] == 1 and again["cand_score"] == [1.0], again

    # 멀리 떨어진 후보는 둘 다 남는다 (과하게 거르지 않는지).
    other = enc.raw_vector(enc.decode(np.full(enc.dim, -1.0)))
    keep = refilter_candidates([raws[0], [float(v) for v in other]], [1.0, 0.9], enc, min_dist=0.15)
    assert keep["n_candidates"] == 2, keep

    # plan_from_raw 는 raw_vector 의 역이다.
    assert np.allclose(enc.raw_vector(plan_from_raw(enc, raws[0])), raws[0], atol=1e-9)

def test_refilter_candidate_frame(tmp_path):
    """DataFrame 단위 재필터 — 세 열을 함께 갱신하고 원본은 바꾸지 않는다 (F 사용 형태)."""
    import pandas as pd

    from airis.optimize.plan_dataset import (
        PlanDatasetConfig, build_plan_dataset, refilter_candidate_frame,
    )

    out = tmp_path / "plan.parquet"
    cfg = PlanDatasetConfig(n_phases=3, max_evals=300, pose_max_evals=150, popsize=10,
                            evaluator="dummy", candidate_k=8, candidate_min_dist=0.0)
    build_plan_dataset(out, n_bodies=2, scenarios=["default"], cfg=cfg,
                       processes=1, flush_every=1, log=lambda *a: None)
    df = pd.read_parquet(out)
    before = df["n_candidates"].tolist()

    tight = refilter_candidate_frame(df, min_dist=0.30)       # 빡빡하게 걸러 본다
    assert tight["n_candidates"].tolist() <= before, (tight["n_candidates"].tolist(), before)
    assert (tight["n_candidates"] > 0).all(), "후보가 0개가 되는 행은 없다"
    for _, row in tight.iterrows():
        # 세 열이 함께 갱신된다.
        assert len(row["cand_plan_raw"]) == row["n_candidates"] == len(row["cand_score"])
        assert list(row["cand_score"]) == sorted(row["cand_score"], reverse=True)
    assert df["n_candidates"].tolist() == before, "원본 DataFrame 은 바뀌지 않는다"
    assert list(tight.columns) == list(df.columns)

    # 임계 0 이면 그대로다.
    same = refilter_candidate_frame(df, min_dist=0.0)
    assert same["n_candidates"].tolist() == before


def test_encoder_for_row_reads_zone_counts_from_stamp():
    """행의 구역 노즐 수 도장을 인코더에 넘긴다 — 풍량 한도가 그 값에 걸린다."""
    import numpy as np

    from airis.optimize.plan_dataset import encoder_for_row

    row = {"scenario": "default", "n_phases": 3, "duration_lo_s": 6.0, "duration_hi_s": 20.0,
           "min_phase_s": 2.0, "transition_s": 1.5, "s_max": 1.0, "cap_ratio": 0.5,
           "zone_nozzle_counts": "9;1;1;1;1"}
    odd = encoder_for_row(row)
    normal = encoder_for_row({**row, "zone_nozzle_counts": "2;2;2;2;4"})

    z = np.array([1.0, 1.0, 1.0, 1.0, 0.2])
    a, b = odd.clip_zone_strengths(z), normal.clip_zone_strengths(z)
    assert not np.allclose(a, b), (a, b)          # 배치가 다르면 한도도 다르다
    assert a[0] < b[0], "노즐이 몰린 구역이 더 깎인다"

    # 설정 파일 기본 배치와 같은 도장이면 기본값을 쓴 것과 같다.
    from airis.optimize.plan_encoding import PlanEncoder, PlanLimits
    same = PlanEncoder(normal.scenario if hasattr(normal, "scenario") else
                       __import__("airis.sim.scenario", fromlist=["x"]).load_scenarios()["default"],
                       PlanLimits(n_phases=3, duration_bounds_s=(6.0, 20.0), cap_ratio=0.5))
    assert np.allclose(normal.clip_zone_strengths(z), same.clip_zone_strengths(z))
