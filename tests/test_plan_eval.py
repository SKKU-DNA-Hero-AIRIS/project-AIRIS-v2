"""계획 평가(evaluate_plan)와 계획 점수 함수. 소유자: D.

수식은 `docs/tracks/00_common.md` 4.4(계획 점수)·4.6(시간 의존 제거)·4.7(구역 세기·에너지),
계약은 `docs/interfaces.md`의 `Evaluator.evaluate_plan`이다 (`docs/plan_extension.md`).

물리 상수(k, fabric_roughness_factor, T_r 등)는 `configs/physics.yaml`에서 읽고 숫자를 박지 않는다.
시간 항은 `adhesion.kinetics.enabled`가 참일 때만 들어가므로, 시간 관련 테스트는 dict를 복사해
켠다 (`configs/`는 수정하지 않는다).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from airis.sim import scoring
from airis.sim.patch_baseline import PatchEvaluator, infeasible_score
from airis.sim.scenario import (
    apply_zone_strengths, load_nozzle_layout, load_nozzles, load_physics, load_scenarios,
)
from airis.sim.types import PART_NAMES, ZONE_NAMES, Phase, Plan, PoseParams

_PATCHES_PER_M2 = 400.0          # 계획 테스트는 최적화 루프와 같은 밀도로 돈다


@pytest.fixture(scope="module")
def physics() -> dict:
    """configs/physics.yaml 원본. 수정하지 않는다."""
    return load_physics()


@pytest.fixture(scope="module")
def kinetics_physics(physics) -> dict:
    """시간 의존 제거(4.6)를 켠 사본."""
    cfg = copy.deepcopy(physics)
    cfg["adhesion"]["kinetics"]["enabled"] = True
    return cfg


@pytest.fixture(scope="module")
def scenario():
    return load_scenarios()["default"]


@pytest.fixture(scope="module")
def nozzle():
    return load_nozzles()


def _evaluator(cfg: dict) -> PatchEvaluator:
    return PatchEvaluator(cfg, patches_per_m2=_PATCHES_PER_M2)


def _plan(phases, zone_strengths=None) -> Plan:
    z = np.ones(len(ZONE_NAMES)) if zone_strengths is None else np.asarray(zone_strengths, float)
    return Plan(phases=[Phase(pose, dur) for pose, dur in phases], zone_strengths=z)


# --- (a) 회귀: 단계 1개 + 세기 1 + 시간 항 끔 = evaluate ------------------------

@pytest.mark.parametrize("pose", [PoseParams(), PoseParams(shoulder_abduction=150.0, torso_yaw=30.0)])
def test_single_phase_full_strength_matches_evaluate(physics, scenario, nozzle, pose):
    """`interfaces.md`의 회귀 조건. 제거율이 `evaluate`와 완전히 같아야 한다."""
    assert physics["adhesion"]["kinetics"]["enabled"] is False, "기본 설정은 시간 항이 꺼져 있어야 한다"
    ev = _evaluator(physics)
    t_ref = float(physics["scoring"]["reference_duration_s"])
    plan = ev.evaluate_plan(_plan([(pose, t_ref)]), nozzle, None, scenario)
    single = ev.evaluate(pose, nozzle, None, scenario)

    assert plan.total_removal == pytest.approx(single.total_removal, rel=1e-12)
    assert np.allclose(plan.removal_by_part, single.removal_by_part, rtol=1e-12, atol=0.0)
    assert plan.discomfort == pytest.approx(single.discomfort, rel=1e-12)
    # 점수 차이는 에너지 항뿐이다 (전 노즐 s=1, T=T_ref 이면 e=1).
    assert plan.extra["energy"] == pytest.approx(1.0, rel=1e-12)
    penalty = float(physics["scoring"]["energy_weight"]) * plan.extra["energy"]
    assert plan.score == pytest.approx(single.score - penalty, rel=1e-12)
    assert plan.extra["duration_s"] == pytest.approx(t_ref)
    assert plan.extra["infeasible"] is False
    assert plan.extra["removal_by_part_per_phase"].shape == (1, len(PART_NAMES))


# --- (b) 시간이 길수록 제거율이 오르고 포화한다 (4.6) ---------------------------

def test_longer_duration_increases_removal_and_saturates(kinetics_physics, scenario, nozzle):
    ev = _evaluator(kinetics_physics)
    pose = PoseParams()
    removals = [ev.evaluate_plan(_plan([(pose, t)]), nozzle, None, scenario).total_removal
                for t in (2.0, 5.0, 10.0, 20.0, 60.0)]
    assert all(a < b for a, b in zip(removals, removals[1:])), removals

    asymptote = _evaluator(load_physics()).evaluate(pose, nozzle, None, scenario).total_removal
    assert removals[-1] < asymptote                      # 유한 시간은 점근값보다 작다
    assert removals[-1] == pytest.approx(asymptote, rel=2e-3)   # 60 s = 30·T_r 이면 사실상 포화


def test_removal_at_one_time_constant_matches_closed_form(kinetics_physics, scenario, nozzle):
    """단일 단계 R(t) = R_inf·(1 − exp(−t/T_r)) 를 패치별로 확인한다."""
    t_r = float(kinetics_physics["adhesion"]["kinetics"]["time_constant_s"])
    ev = _evaluator(kinetics_physics)
    timed = ev.evaluate_plan(_plan([(PoseParams(), t_r)]), nozzle, None, scenario)
    asymptotic = scoring.removal_fraction(timed.extra["tau"][0], kinetics_physics)
    assert np.allclose(timed.extra["removal"], asymptotic * (1.0 - np.exp(-1.0)), rtol=1e-12)


# --- (c) 같은 자세를 두 단계로 쪼개도 합친 것과 같다 (4.6) ----------------------

def test_splitting_one_pose_into_two_phases_matches_single_phase(kinetics_physics, scenario, nozzle):
    ev = _evaluator(kinetics_physics)
    pose = PoseParams(shoulder_abduction=60.0, torso_yaw=20.0)
    split = ev.evaluate_plan(_plan([(pose, 4.0), (pose, 6.0)]), nozzle, None, scenario)
    whole = ev.evaluate_plan(_plan([(pose, 10.0)]), nozzle, None, scenario)

    assert np.allclose(split.extra["removal"], whole.extra["removal"], rtol=1e-12, atol=0.0)
    assert split.total_removal == pytest.approx(whole.total_removal, rel=1e-12)
    assert split.score == pytest.approx(whole.score, rel=1e-12)
    assert split.discomfort == pytest.approx(whole.discomfort, rel=1e-12)
    assert split.extra["energy"] == pytest.approx(whole.extra["energy"], rel=1e-12)


def test_order_of_phases_does_not_change_removal(kinetics_physics, scenario, nozzle):
    """4.6은 단계 순서에 무관하다 (전단 정렬 뒤 노출 시간 합)."""
    ev = _evaluator(kinetics_physics)
    a, b = PoseParams(), PoseParams(shoulder_abduction=150.0)
    forward = ev.evaluate_plan(_plan([(a, 4.0), (b, 9.0)]), nozzle, None, scenario)
    backward = ev.evaluate_plan(_plan([(b, 9.0), (a, 4.0)]), nozzle, None, scenario)
    assert np.allclose(forward.extra["removal"], backward.extra["removal"], rtol=1e-12)
    assert forward.score == pytest.approx(backward.score, rel=1e-12)


# --- (d) 구역 세기 (4.7) -------------------------------------------------------

@pytest.mark.parametrize("zone", range(len(ZONE_NAMES)))
def test_zero_zone_strength_lowers_total_removal(physics, scenario, nozzle, zone):
    """구역 하나를 끄면 총 제거율이 떨어진다."""
    ev = _evaluator(physics)
    phases = [(PoseParams(torso_yaw=30.0), 10.0), (PoseParams(torso_yaw=30.0, shoulder_abduction=150.0), 10.0)]
    full = ev.evaluate_plan(_plan(phases), nozzle, None, scenario)
    strengths = np.ones(len(ZONE_NAMES))
    strengths[zone] = 0.0
    off = ev.evaluate_plan(_plan(phases, strengths), nozzle, None, scenario)
    assert off.total_removal < full.total_removal


def test_zero_top_zone_lowers_every_part(physics, scenario, nozzle):
    """천장 구역을 끄면 부위별 제거율도 모두 떨어진다.

    측면 구역은 그렇지 않다: 마주 보는 두 벽의 제트가 표면에서 반대 방향 접선 성분을 만들어
    상쇄되던 것이, 한쪽을 끄면 풀려서 일부 패치의 전단이 오히려 커진다 (모델 한계, PR 본문).
    """
    ev = _evaluator(physics)
    phases = [(PoseParams(torso_yaw=30.0), 20.0)]
    full = ev.evaluate_plan(_plan(phases), nozzle, None, scenario)
    strengths = np.ones(len(ZONE_NAMES))
    strengths[ZONE_NAMES.index("top")] = 0.0
    off = ev.evaluate_plan(_plan(phases, strengths), nozzle, None, scenario)
    assert (off.removal_by_part <= full.removal_by_part + 1e-12).all()
    assert off.total_removal < full.total_removal


def test_zone_strengths_are_applied_per_phase_by_chest_side(physics, scenario, nozzle):
    """구역 → 노즐 매핑은 단계마다 그 단계의 torso_yaw로 정해진다 (4.7 몸 기준 구역)."""
    ev = _evaluator(physics)
    strengths = np.array([1.0, 0.0, 0.0, 1.0, 0.5])
    plus = apply_zone_strengths(nozzle, strengths, 30.0).strengths
    minus = apply_zone_strengths(nozzle, strengths, -30.0).strengths
    assert not np.array_equal(plus, minus), "좌우 벽이 바뀌면 노즐 세기 배열도 바뀌어야 한다"

    mixed = ev.evaluate_plan(
        _plan([(PoseParams(torso_yaw=30.0), 10.0), (PoseParams(torso_yaw=-30.0), 10.0)], strengths),
        nozzle, None, scenario)
    same = ev.evaluate_plan(
        _plan([(PoseParams(torso_yaw=30.0), 10.0), (PoseParams(torso_yaw=30.0), 10.0)], strengths),
        nozzle, None, scenario)
    assert mixed.total_removal != same.total_removal


# --- (e) 에너지 (4.7) ----------------------------------------------------------

def test_energy_is_one_for_current_operation(physics):
    """전 노즐 s=1, T=T_ref 이면 e = 1 (현행 운전)."""
    t_ref = float(physics["scoring"]["reference_duration_s"])
    assert scoring.energy(np.ones(12), t_ref, physics) == pytest.approx(1.0)


def test_energy_scales_with_strength_power_and_time(physics):
    """e ∝ Σ s^p · T."""
    p = float(physics["fan"]["power_exponent"])
    t_ref = float(physics["scoring"]["reference_duration_s"])
    base = scoring.energy(np.ones(12), t_ref, physics)
    assert scoring.energy(np.full(12, 0.5), t_ref, physics) == pytest.approx(base * 0.5 ** p)
    assert scoring.energy(np.ones(12), t_ref / 2.0, physics) == pytest.approx(base / 2.0)
    half_off = np.concatenate([np.ones(6), np.zeros(6)])
    assert scoring.energy(half_off, t_ref, physics) == pytest.approx(base / 2.0)


def test_plan_energy_uses_zone_strengths_and_duration(physics, scenario, nozzle):
    ev = _evaluator(physics)
    t_ref = float(physics["scoring"]["reference_duration_s"])
    p = float(physics["fan"]["power_exponent"])
    half = ev.evaluate_plan(_plan([(PoseParams(), t_ref)], np.full(len(ZONE_NAMES), 0.5)),
                            nozzle, None, scenario)
    assert half.extra["energy"] == pytest.approx(0.5 ** p, rel=1e-12)

    zones = np.array([1.0, 0.0, 0.0, 1.0, 0.5])
    plan = _plan([(PoseParams(), 6.0), (PoseParams(shoulder_abduction=150.0), 4.0)], zones)
    result = ev.evaluate_plan(plan, nozzle, None, scenario)
    expected = sum(scoring.energy(apply_zone_strengths(nozzle, zones, ph.pose.torso_yaw).strengths,
                                  ph.duration_s, physics)
                   for ph in plan.phases)
    assert result.extra["energy"] == pytest.approx(expected, rel=1e-12)


def test_energy_weight_enters_score(physics, scenario, nozzle):
    ev = _evaluator(physics)
    plan = _plan([(PoseParams(), 10.0)], np.full(len(ZONE_NAMES), 0.5))
    result = ev.evaluate_plan(plan, nozzle, None, scenario)
    expected, _ = scoring.score_plan(result.removal_by_part, plan.phases, scenario, physics,
                                     result.extra["energy"])
    assert result.score == pytest.approx(expected, rel=1e-12)

    cheap = copy.deepcopy(physics)
    cheap["scoring"]["energy_weight"] = 0.0
    lower = _evaluator(cheap).evaluate_plan(plan, nozzle, None, scenario)
    assert lower.score > result.score


# --- (f) 대칭: yaw ±θ ----------------------------------------------------------

@pytest.mark.parametrize("yaw", [30.0, 75.0])
def test_mirror_yaw_gives_same_removal(physics, scenario, nozzle, yaw):
    """몸 기준 구역이라 yaw ±θ 는 거울 대칭이고 결과가 같아야 한다 (4.7).

    메시 정점이 float32라 정확히 같지는 않다 (상대 오차 1e-8 수준).
    """
    ev = _evaluator(physics)
    zones = np.array([1.0, 0.2, 0.6, 0.4, 0.8])
    phases = lambda y: [(PoseParams(torso_yaw=y), 8.0),
                        (PoseParams(torso_yaw=y, shoulder_abduction=150.0), 7.0)]
    plus = ev.evaluate_plan(_plan(phases(yaw), zones), nozzle, None, scenario)
    minus = ev.evaluate_plan(_plan(phases(-yaw), zones), nozzle, None, scenario)
    assert plus.total_removal == pytest.approx(minus.total_removal, rel=1e-6)
    assert np.allclose(plus.removal_by_part, minus.removal_by_part, rtol=1e-6)
    assert plus.score == pytest.approx(minus.score, rel=1e-6)


# --- 부스 밖 계획 (5절) --------------------------------------------------------

def test_plan_is_infeasible_if_any_phase_leaves_the_booth(physics, scenario, nozzle):
    """어느 단계든 부스 밖이면 계획 전체가 불가, 벌점은 단계 중 최대 d_out."""
    ev = _evaluator(physics)
    booth = load_nozzle_layout()["booth"]
    out_pose = PoseParams(shoulder_abduction=90.0, elbow_flexion=0.0)     # 팔 수평 = 벽 밖
    d_out = max(0.0, float(np.abs(ev.build_state(None, out_pose, scenario).patch_pos[:, 1]).max())
                - booth["width_m"] / 2.0)
    assert d_out > 0.0, "이 부스에서는 팔 수평이 벽 밖이어야 한다"

    result = ev.evaluate_plan(_plan([(PoseParams(), 10.0), (out_pose, 10.0)]), nozzle, None, scenario)
    assert result.extra["infeasible"] is True
    assert result.extra["d_out"] == pytest.approx(d_out, rel=1e-6)
    assert result.score == pytest.approx(infeasible_score(d_out), rel=1e-12)
    assert result.total_removal == 0.0
    assert (result.removal_by_part == 0.0).all()


def test_empty_plan_is_rejected(physics, scenario, nozzle):
    with pytest.raises(ValueError):
        _evaluator(physics).evaluate_plan(Plan(phases=[], zone_strengths=np.ones(len(ZONE_NAMES))),
                                          nozzle, None, scenario)


# --- scoring.removal_fraction_plan 단위 테스트 (4.6) ---------------------------

def test_removal_fraction_plan_without_kinetics_is_max_shear(physics):
    tau = np.array([[0.05, 0.20, 0.0], [0.15, 0.10, 0.30]])
    plan_r = scoring.removal_fraction_plan(tau, [5.0, 5.0], physics)
    assert np.allclose(plan_r, scoring.removal_fraction(tau.max(axis=0), physics), rtol=1e-12)


def test_removal_fraction_plan_single_phase_matches_closed_form(kinetics_physics):
    tau = np.array([[0.05, 0.20, 0.0]])
    t_r = float(kinetics_physics["adhesion"]["kinetics"]["time_constant_s"])
    for t in (1.0, 5.0):
        got = scoring.removal_fraction_plan(tau, [t], kinetics_physics)
        want = scoring.removal_fraction(tau[0], kinetics_physics) * (1.0 - np.exp(-t / t_r))
        assert np.allclose(got, want, rtol=1e-12)


def test_removal_fraction_plan_matches_particle_sampling(kinetics_physics):
    """닫힌 식을 입자 표본과 대조한다 (4.6의 가정: tau_ik >= tau_c 인 단계에서만 이탈)."""
    adhesion = kinetics_physics["adhesion"]
    tau_med = float(adhesion["critical_shear_pa_median"]) * float(adhesion["fabric_roughness_factor"])
    sigma = float(adhesion["critical_shear_sigma_log"])
    t_r = float(adhesion["kinetics"]["time_constant_s"])

    tau = np.array([[0.02, 0.30, 0.10], [0.25, 0.05, 0.10]])
    durations = np.array([3.0, 7.0])
    rng = np.random.default_rng(0)
    tau_c = np.exp(np.log(tau_med) + sigma * rng.standard_normal(400_000))     # 로그 정규 임계 전단
    sampled = []
    for i in range(tau.shape[1]):
        # 입자마다 "전단이 자기 임계값 이상인 단계"의 시간 합이 노출 시간이다.
        exposure = sum(durations[k] * (tau[k, i] >= tau_c) for k in range(tau.shape[0]))
        sampled.append(float(np.mean(1.0 - np.exp(-exposure / t_r))))
    got = scoring.removal_fraction_plan(tau, durations, kinetics_physics)
    assert np.allclose(got, sampled, atol=2e-3), (got, sampled)


def test_removal_fraction_plan_rejects_bad_durations(physics):
    tau = np.zeros((2, 3))
    with pytest.raises(ValueError):
        scoring.removal_fraction_plan(tau, [1.0], physics)
    with pytest.raises(ValueError):
        scoring.removal_fraction_plan(tau, [1.0, -1.0], physics)


# --- scoring.score_plan 단위 테스트 (4.4) --------------------------------------

def test_score_plan_matches_formula(physics, scenario):
    removal = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    phases = [Phase(PoseParams(), 6.0), Phase(PoseParams(shoulder_abduction=95.0), 9.0)]
    e = 0.7
    cfg = physics["scoring"]
    t_ref = float(cfg["reference_duration_s"])
    disc = sum(scoring.discomfort(ph.pose, scenario) * ph.duration_s for ph in phases) / t_ref
    expected = (sum(float(cfg["part_weights"][name]) * removal[i] for i, name in enumerate(PART_NAMES))
                - float(cfg["discomfort_weight"]) * disc
                - float(cfg["energy_weight"]) * e
                - float(cfg["time_weight"]) * 15.0 / t_ref)

    total, returned_disc = scoring.score_plan(removal, phases, scenario, physics, e)
    assert total == pytest.approx(expected)
    assert returned_disc == pytest.approx(disc)


def test_score_plan_discomfort_is_time_weighted(physics, scenario):
    """같은 자세를 T_ref 만큼 유지하면 단일 자세 불편도와 같고, 시간이 절반이면 절반이다."""
    removal = np.zeros(len(PART_NAMES))
    pose = PoseParams(shoulder_abduction=120.0)
    t_ref = float(physics["scoring"]["reference_duration_s"])
    _, full = scoring.score_plan(removal, [Phase(pose, t_ref)], scenario, physics, 0.0)
    _, half = scoring.score_plan(removal, [Phase(pose, t_ref / 2.0)], scenario, physics, 0.0)
    assert full == pytest.approx(scoring.discomfort(pose, scenario))
    assert half == pytest.approx(full / 2.0)
