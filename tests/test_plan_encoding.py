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
    assert data["bundle"]["kinetics_enabled"] is True and data["bundle"]["note"] == "설명"
    assert data["bundle"]["zone_nozzle_counts"]["top"] == 4
    assert not Path(data["bundle"]["path"]).is_absolute()
    row = data["rows"][0]
    assert (row["scenario"], row["condition"], row["energy_weight"]) == ("default", "P5", 0.1)
    assert row["total_removal"] == 0.27 and row["energy"] == 0.59 and row["n_phases"] == 2
    assert row["phase_durations_s"] == [7.0, 5.0] and row["infeasible"] is False
    # 계획이 21차원 그대로 들어간다: 단계별 자세 7개 + 시간, 구역 세기 5개.
    assert len(row["plan"]["phases"]) == 2
    assert set(row["plan"]["phases"][0]["pose"]) == set(data["pose_keys"])
    assert row["plan"]["phases"][0]["torso_yaw_folded"] == pytest.approx(80.0)
    assert row["plan"]["zone_strengths"] == dict(zip(ZONE_NAMES, plan["zone_strengths"]))
    json.dumps(data, allow_nan=False)


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
