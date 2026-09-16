"""정성 검증 테스트(E1) + 점수 모듈·가림 판정 단위 테스트. 소유자: D.

이 테스트들은 병합 게이트다: PR 전에 로컬에서 실행하고, 깨지면 병합하지 않는다.

B 병합 전후 자동 전환 (`docs/tracks/D_patch_baseline.md` 단계 5)
- `build_body`, `load_nozzles`가 `NotImplementedError`를 내면 `tests/fakes.py`의 가짜로
  바꾼다. B가 병합된 지금은 실제 몸과 실제 노즐 배치로 돈다. 제트는 B의
  `velocity_field_per_nozzle`이다 (가짜 제트는 B 병합 후 삭제).

테스트용 물리 상수 (E1 전용 덮어쓰기)
- `configs/physics.yaml` 그대로(노즐 지름 4 mm)면 몸 표면 풍속이 1 m/s 안팎이라 벽면
  전단이 임계값의 1/100 수준이고, 제거율이 모든 자세에서 사실상 0이다. 이 영역에서는
  부위별 대소 비교가 Φ(-6) 꼬리의 반올림 수준 차이로 정해져 방향 검증이 안 된다.
- 충돌 제트 보정(B 단계 8, 미구현)이 들어오기 전까지, 몸 표면에서 제거율이 의미 있는
  범위(총 제거율 ~15%)가 되도록 제트를 굵고 넓게 덮어쓴다. dict를 복사해 덮어쓰며
  `configs/`는 수정하지 않는다 (00_common.md 1절).
- 값은 격자 탐색으로 고르고, 두 값 각각 ±20% 이웃 9곳 중 8곳에서 E1 방향이 유지됨을
  확인했다 (PR 본문 참고).
"""
from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm

from airis.sim import scoring
from airis.sim.body import build_body
from airis.sim.jet import velocity_field_per_nozzle
from airis.sim.patch_baseline import PatchEvaluator, occlusion
from airis.sim.scenario import load_nozzle_layout, load_nozzles, load_physics, load_scenarios
from airis.sim.types import PART_NAMES, BodyParams, BodyState, NozzleConfig, PoseParams, Scenario
from tests.fakes import fake_body, fake_nozzles

# E1 테스트용 제트 덮어쓰기. 근거는 모듈 docstring.
_E1_JET_OVERRIDES = {"nozzle_diameter_m": 0.08, "halfwidth_spread_rate": 0.2}

_ARMS = PART_NAMES.index("arms")
_FRONT = PART_NAMES.index("torso_front")
_BACK = PART_NAMES.index("torso_back")


# --- B 병합 전후 전환 --------------------------------------------------------

def select_body_fn(build=build_body, scenario: Scenario | None = None):
    """(pose, scenario) -> BodyState. `build`가 미구현이면 fake_body로 바꾼다.

    반환 함수의 `uses_fake` 속성으로 어느 쪽인지 알 수 있다.
    """
    scenario = scenario or load_scenarios()["default"]
    try:
        build(BodyParams(), PoseParams(), scenario)
    except NotImplementedError:
        def fake(pose, scen):
            return fake_body(arms_up=pose.shoulder_abduction > 60, yaw_deg=pose.torso_yaw)
        fake.uses_fake = True
        return fake

    def real(pose, scen):
        return build(BodyParams(), pose, scen)
    real.uses_fake = False
    return real


def select_nozzles(load=load_nozzles) -> NozzleConfig:
    try:
        return load()
    except NotImplementedError:
        return fake_nozzles()


# --- fixture -----------------------------------------------------------------

@pytest.fixture(scope="module")
def physics() -> dict:
    """configs/physics.yaml 원본. 수정하지 않는다."""
    return load_physics()


@pytest.fixture(scope="module")
def e1_physics(physics) -> dict:
    cfg = copy.deepcopy(physics)
    cfg["jet"].update(_E1_JET_OVERRIDES)
    return cfg


@pytest.fixture(scope="module")
def scenarios() -> dict[str, Scenario]:
    return load_scenarios()


@pytest.fixture(scope="module")
def sim(e1_physics, scenarios):
    make_body = select_body_fn(scenario=scenarios["default"])
    evaluator = PatchEvaluator(
        e1_physics,
        build_body=lambda body, pose, scen: make_body(pose, scen),
        velocity_field_per_nozzle=velocity_field_per_nozzle,
    )
    return SimpleNamespace(evaluator=evaluator, nozzles=select_nozzles(),
                           scenario=scenarios["default"], uses_fake_body=make_body.uses_fake,
                           cfg=e1_physics)


def _eval(sim, pose: PoseParams, nozzle: NozzleConfig | None = None):
    return sim.evaluator.evaluate(pose, nozzle or sim.nozzles, BodyParams(), sim.scenario)


def _with_strengths(nozzle: NozzleConfig, strengths: np.ndarray) -> NozzleConfig:
    return NozzleConfig(positions=nozzle.positions, directions=nozzle.directions,
                        strengths=strengths.astype(np.float32), pulse_phase=nozzle.pulse_phase)


def _wrap_deg(angle: float) -> float:
    """[-180, 180) 으로 접는다 (scenario.pose_bounds의 torso_yaw 범위)."""
    return float((angle + 180.0) % 360.0 - 180.0)


def _yaw_facing_nozzles(nozzle: NozzleConfig) -> float:
    """몸의 정면(+x)이 노즐 뱅크를 보게 하는 torso_yaw.

    양쪽 벽에 대칭으로 노즐이 있으면 좌우 성분은 상쇄되고, 뱅크의 평균 x가 몸보다
    앞(+x)인지 뒤(-x)인지가 남는다. 실제 배치(열 0.62 / 0.82)는 몸(x = 1.0)보다
    상류에 있으므로 -180도, 즉 진행 반대 방향을 보는 것이 "노즐을 마주 본" 자세다.
    """
    booth = load_nozzle_layout()["booth"]
    body_xy = np.array([booth["length_m"] / 2.0, 0.0])
    v = nozzle.positions[:, :2].astype(np.float64).mean(axis=0) - body_xy
    assert abs(v[0]) > 1e-3, "노즐 뱅크가 몸과 같은 x에 있으면 마주 보는 방향이 정의되지 않는다"
    return _wrap_deg(math.degrees(math.atan2(v[1], v[0])))


def _move_nozzles_closer_along_axis(nozzle: NozzleConfig, wall_distance_m: float) -> NozzleConfig:
    """벽면 노즐을 자기 제트 축을 따라 앞으로 옮겨 벽과의 거리 |y|를 줄인다.

    D 문서는 "노즐 y를 ±0.6 -> ±0.4로 옮긴다"고 적는다. y만 바꾸면 25도 기울어진
    제트가 몸에 닿는 지점도 함께 옮겨가 "가까움"과 "조준 변화"가 섞인다. 축을 따라
    옮기면 조준점은 그대로이고 거리만 줄어든다 (y는 문서대로 ±0.4가 된다).
    """
    step = (np.abs(nozzle.positions[:, 1]) - wall_distance_m) / np.abs(nozzle.directions[:, 1])
    return NozzleConfig(positions=nozzle.positions + step[:, None] * nozzle.directions,
                        directions=nozzle.directions, strengths=nozzle.strengths,
                        pulse_phase=nozzle.pulse_phase)


# --- 전환 fixture 자체 -------------------------------------------------------

def test_body_fixture_falls_back_to_fake_body_when_b_is_unmerged(scenarios):
    def unmerged(*args, **kwargs):
        raise NotImplementedError("B: 1주차 구현 대상")

    fn = select_body_fn(unmerged, scenarios["default"])
    assert fn.uses_fake
    up = fn(PoseParams(shoulder_abduction=90.0, torso_yaw=30.0), scenarios["default"])
    assert np.array_equal(up.patch_pos, fake_body(arms_up=True, yaw_deg=30.0).patch_pos)


def test_nozzle_fixture_falls_back_to_fake_nozzles_when_b_is_unmerged():
    def unmerged(*args, **kwargs):
        raise NotImplementedError("B: 1주차 구현 대상")

    assert np.array_equal(select_nozzles(unmerged).positions, fake_nozzles().positions)


# --- E1: 정성 검증 -----------------------------------------------------------

def test_raising_arms_increases_armpit_removal(sim):
    """팔을 들면 겨드랑이/옆구리 제거율이 오른다.

    실제 몸은 `arms` + `torso_front` 합산, 가짜 몸은 `arms`만 비교한다 (D 문서 단계 5).
    """
    down = _eval(sim, PoseParams(shoulder_abduction=20.0))
    up = _eval(sim, PoseParams(shoulder_abduction=90.0))

    parts = [_ARMS] if sim.uses_fake_body else [_ARMS, _FRONT]
    before = down.removal_by_part[parts].sum()
    after = up.removal_by_part[parts].sum()
    assert before > 0.0, "제트가 몸에 닿지 않아 비교가 무의미하다"
    assert after > before


def test_facing_away_reduces_front_removal(sim):
    """노즐을 등지면 정면 제거율이 떨어지고 배면 제거율이 오른다.

    D 문서의 `torso_yaw` 0 vs 180 비교다. 어느 쪽이 "마주 본" 자세인지는 노즐 배치로 정한다.
    """
    facing_yaw = _yaw_facing_nozzles(sim.nozzles)
    away_yaw = _wrap_deg(facing_yaw + 180.0)
    facing = _eval(sim, PoseParams(torso_yaw=facing_yaw))
    away = _eval(sim, PoseParams(torso_yaw=away_yaw))

    assert {abs(facing_yaw), abs(away_yaw)} == {0.0, 180.0}
    assert facing.removal_by_part[_FRONT] > 0.0, "제트가 정면에 닿지 않아 비교가 무의미하다"
    assert away.removal_by_part[_FRONT] < facing.removal_by_part[_FRONT]
    assert away.removal_by_part[_BACK] > facing.removal_by_part[_BACK]


@pytest.mark.xfail(strict=True, reason=(
    "현재 제트·제거율 모델에서 성립하지 않는다 (발견 사항, PR 본문). 가까워지면 가우시안 "
    "단면 σ ∝ s 가 좁아져 제트가 덮는 면적이 줄고 로그 정규 제거율은 포화하므로, 총 "
    "제거율이 오히려 40~55% 떨어진다. 제거율이 사실상 0인 약한 제트에서만 성립하고 그 "
    "영역에서는 다른 E1 방향이 무너진다. strict: 모델이 바뀌어 통과하면 이 표시를 지워라."))
def test_closer_to_nozzle_increases_removal(sim):
    """노즐에 가까울수록 제거율이 오른다. 벽면 노즐을 |y| 0.6 -> 0.4로 옮긴다."""
    near = _move_nozzles_closer_along_axis(sim.nozzles, wall_distance_m=0.4)
    assert np.allclose(np.abs(near.positions[:, 1]), 0.4, atol=1e-6)

    base = _eval(sim, PoseParams())
    closer = _eval(sim, PoseParams(), near)
    assert base.total_removal > 0.0
    assert closer.total_removal > base.total_removal


def test_zero_strength_removes_nothing(sim):
    """노즐 세기 0이면 제거율 0."""
    off = _with_strengths(sim.nozzles, np.zeros(sim.nozzles.count))
    for pose in (PoseParams(), PoseParams(shoulder_abduction=90.0, torso_yaw=45.0)):
        result = _eval(sim, pose, off)
        assert result.total_removal == 0.0
        assert (result.removal_by_part == 0.0).all()
        assert (result.extra["tau"] == 0.0).all()
        # 제거가 0이면 점수는 불편도 페널티만 남는다.
        penalty = sim.cfg["scoring"]["discomfort_weight"] * result.discomfort
        assert result.score == pytest.approx(-penalty)


def test_deterministic(sim):
    """같은 입력 -> 같은 점수 (== 로 완전 일치)."""
    pose = PoseParams(shoulder_abduction=75.0, elbow_flexion=30.0, torso_yaw=-40.0)
    a = _eval(sim, pose)
    b = _eval(sim, pose)
    assert a.score == b.score
    assert a.total_removal == b.total_removal
    assert np.array_equal(a.removal_by_part, b.removal_by_part)
    assert np.array_equal(a.extra["removal"], b.extra["removal"])


# --- EvalResult 계약 ---------------------------------------------------------

def test_default_evaluator_runs_on_merged_b(physics, scenarios):
    """C가 쓰는 형태 그대로: `PatchEvaluator(load_physics())`, 주입 없이 B의 구현을 쓴다."""
    if select_body_fn(scenario=scenarios["default"]).uses_fake:
        pytest.skip("B 미병합: 기본 연결은 NotImplementedError를 낸다")
    for name in ("default", "pregnant", "wheelchair"):
        result = PatchEvaluator(physics).evaluate(PoseParams(), load_nozzles(), BodyParams(),
                                                  scenarios[name])
        assert np.isfinite(result.score)
        assert result.removal_by_part.shape == (len(PART_NAMES),)


def test_evaluate_fills_evalresult(sim):
    result = _eval(sim, PoseParams(shoulder_abduction=90.0))
    n = sim.evaluator._build_body(BodyParams(), PoseParams(shoulder_abduction=90.0),
                                  sim.scenario).patch_pos.shape[0]

    assert isinstance(result.score, float)
    assert isinstance(result.total_removal, float)
    assert isinstance(result.discomfort, float)
    assert result.removal_by_part.shape == (len(PART_NAMES),)
    assert ((0.0 <= result.removal_by_part) & (result.removal_by_part <= 1.0)).all()
    assert 0.0 <= result.total_removal <= 1.0

    # 3D 뷰에서 색으로 쓰는 패치별 값 (D 문서 단계 4).
    assert result.extra["tau"].shape == (n,)
    assert result.extra["removal"].shape == (n,)
    assert 0.0 <= result.extra["visible_frac"] <= 1.0

    # score는 공용 scoring.score와 같아야 한다 (A가 같은 함수를 쓴다).
    expected, disc = scoring.score(result.removal_by_part, PoseParams(shoulder_abduction=90.0),
                                   sim.scenario, sim.cfg)
    assert result.score == expected
    assert result.discomfort == disc


def test_removal_by_part_is_area_weighted(sim):
    pose = PoseParams(shoulder_abduction=90.0)
    state = sim.evaluator._build_body(BodyParams(), pose, sim.scenario)
    result = _eval(sim, pose)
    removal, area = result.extra["removal"], state.patch_area.astype(np.float64)

    for i in range(len(PART_NAMES)):
        mask = state.patch_part == i
        expected = (removal[mask] * area[mask]).sum() / area[mask].sum() if mask.any() else 0.0
        assert result.removal_by_part[i] == pytest.approx(expected, rel=1e-12, abs=1e-15)
    assert result.total_removal == pytest.approx((removal * area).sum() / area.sum(), rel=1e-12)


def test_batch_evaluate_matches_evaluate_in_order(sim):
    candidates = [(PoseParams(torso_yaw=y), sim.nozzles) for y in (0.0, 90.0, 180.0)]
    batch = sim.evaluator.batch_evaluate(candidates, BodyParams(), sim.scenario)
    single = [_eval(sim, p, n).score for p, n in candidates]
    assert batch.shape == (3,)
    assert np.allclose(batch, single)


def test_evaluator_uses_config_constants_not_literals(sim, e1_physics):
    """상수를 덮어쓰면 결과가 따라 바뀐다 -> 코드에 숫자가 박혀 있지 않다."""
    rough = copy.deepcopy(e1_physics)
    rough["adhesion"]["fabric_roughness_factor"] *= 4.0     # 임계 전단 상승 -> 제거율 하락
    ev = PatchEvaluator(rough, build_body=sim.evaluator._build_body,
                        velocity_field_per_nozzle=sim.evaluator._velocity_field_per_nozzle)
    base = _eval(sim, PoseParams())
    harder = ev.evaluate(PoseParams(), sim.nozzles, BodyParams(), sim.scenario)
    assert harder.total_removal < base.total_removal


# --- scoring.removal_fraction (00_common.md 4.3) -----------------------------

def _tau_median(cfg: dict) -> float:
    a = cfg["adhesion"]
    return a["critical_shear_pa_median"] * a["fabric_roughness_factor"]


def test_removal_zero_shear_is_zero(physics):
    assert scoring.removal_fraction(np.array([0.0]), physics)[0] == 0.0
    # 음수 전단도 물리적으로 없으므로 0.
    assert scoring.removal_fraction(np.array([-1.0]), physics)[0] == 0.0


def test_removal_at_median_shear_is_half(physics):
    r = scoring.removal_fraction(np.array([_tau_median(physics)]), physics)
    assert r[0] == pytest.approx(0.5, abs=1e-12)


def test_removal_matches_lognormal_cdf(physics):
    tau = np.array([1e-4, 0.01, 0.1, 0.3, 1.0, 3.0, 50.0])
    sigma = physics["adhesion"]["critical_shear_sigma_log"]
    expected = norm.cdf((np.log(tau) - math.log(_tau_median(physics))) / sigma)
    assert np.allclose(scoring.removal_fraction(tau, physics), expected, rtol=1e-12, atol=1e-15)


def test_removal_is_monotone_and_bounded(physics):
    tau = np.concatenate([[0.0], np.logspace(-6, 4, 200)])
    r = scoring.removal_fraction(tau, physics)
    assert ((0.0 <= r) & (r <= 1.0)).all()
    assert (np.diff(r) >= 0.0).all()


def test_removal_roughness_factor_scales_median(physics):
    rough = copy.deepcopy(physics)
    rough["adhesion"]["fabric_roughness_factor"] = 2.0
    r = scoring.removal_fraction(np.array([_tau_median(rough)]), rough)
    assert r[0] == pytest.approx(0.5, abs=1e-12)


# --- scoring.wall_shear (00_common.md 4.2) -----------------------------------

def test_wall_shear_uses_tangential_component_only(physics):
    n = np.array([[0.0, 0.0, 1.0]] * 3)
    u = np.array([[0.0, 0.0, 10.0],     # 법선 방향만 -> 전단 0
                  [3.0, 4.0, 0.0],      # 접선만 |u_t| = 5
                  [3.0, 4.0, 12.0]])    # 법선 성분은 빠지고 |u_t| = 5
    air = physics["air"]
    k = 0.5 * air["density"] * air["friction_coeff"]
    tau = scoring.wall_shear(u, n, physics)
    assert tau.shape == (3,)
    assert tau[0] == pytest.approx(0.0, abs=1e-15)
    assert tau[1] == pytest.approx(k * 25.0)
    assert tau[2] == pytest.approx(k * 25.0)


def test_wall_shear_rotation_invariant(physics):
    rng = np.random.default_rng(0)
    n = rng.normal(size=(50, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    u = rng.normal(size=(50, 3)) * 10
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    assert np.allclose(scoring.wall_shear(u, n, physics),
                       scoring.wall_shear(u @ q.T, n @ q.T, physics))


# --- scoring.discomfort / score (00_common.md 4.4) ---------------------------

def test_discomfort_default_pose_is_zero(scenarios):
    for scen in scenarios.values():
        assert scoring.discomfort(PoseParams(), scen) == 0.0


def test_discomfort_matches_formula(scenarios):
    scen = scenarios["pregnant"]
    pose = PoseParams(shoulder_abduction=80.0, torso_pitch=10.0, elbow_flexion=90.0)
    default = PoseParams()
    expected = 0.0
    for key, weight in scen.discomfort_weights.items():
        lo, hi = scen.pose_bounds[key]
        expected += weight * abs(getattr(pose, key) - getattr(default, key)) / (hi - lo)
    # elbow_flexion은 가중치가 없어 기여 0.
    assert "elbow_flexion" not in scen.discomfort_weights
    assert scoring.discomfort(pose, scen) == pytest.approx(expected)
    # 손 계산: 0.6·|80-20|/150 + 2.0·|10-0|/25
    assert scoring.discomfort(pose, scen) == pytest.approx(0.6 * 60 / 150 + 2.0 * 10 / 25)


def test_discomfort_is_symmetric_around_default(scenarios):
    scen = scenarios["default"]
    assert scoring.discomfort(PoseParams(torso_pitch=10.0), scen) == \
           scoring.discomfort(PoseParams(torso_pitch=-10.0), scen)


def test_score_matches_formula(physics, scenarios):
    scen = scenarios["default"]
    pose = PoseParams(shoulder_abduction=95.0)
    removal = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    weights = physics["scoring"]["part_weights"]
    disc = scoring.discomfort(pose, scen)
    expected = sum(weights[name] * removal[i] for i, name in enumerate(PART_NAMES)) \
        - physics["scoring"]["discomfort_weight"] * disc

    total, returned_disc = scoring.score(removal, pose, scen, physics)
    assert total == pytest.approx(expected)
    assert returned_disc == disc


def test_score_rejects_wrong_shape(physics, scenarios):
    with pytest.raises(ValueError):
        scoring.score(np.zeros(4), PoseParams(), scenarios["default"], physics)


# --- occlusion (D 문서 단계 3) -----------------------------------------------

def _single_patch_state(capsules: list[list[float]], patch_capsule: int = 0) -> BodyState:
    """원점에서 +y를 보는 패치 1개 + 주어진 캡슐들."""
    return BodyState(
        patch_pos=np.array([[0.0, 0.0, 0.0]], np.float32),
        patch_normal=np.array([[0.0, 1.0, 0.0]], np.float32),
        patch_area=np.array([1e-3], np.float32),
        patch_part=np.array([_FRONT], np.int32),
        capsules=np.array(capsules, np.float32),
        capsule_part=np.array([_FRONT] * len(capsules), np.int32),
        patch_capsule=np.array([patch_capsule], np.int32),
    )


def _nozzle_at(*positions) -> NozzleConfig:
    pos = np.array(positions, np.float32)
    return NozzleConfig(positions=pos, directions=np.tile([[0, -1, 0]], (len(pos), 1)).astype(np.float32),
                        strengths=np.ones(len(pos), np.float32))


_FAR_AWAY = [10.0, 10.0, 10.0, 10.0, 10.0, 11.0, 0.01]   # 자기 캡슐 자리 채우기용


def test_occlusion_clear_line_of_sight(physics):
    state = _single_patch_state([_FAR_AWAY])
    assert occlusion(state, _nozzle_at([0.0, 1.0, 0.0]), physics).tolist() == [[True]]


def test_occlusion_capsule_between_patch_and_nozzle_blocks(physics):
    # 패치(원점)와 노즐(0,1,0) 사이 y = 0.5에 x축으로 누운 캡슐.
    blocker = [-0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.05]
    state = _single_patch_state([_FAR_AWAY, blocker])
    assert occlusion(state, _nozzle_at([0.0, 1.0, 0.0]), physics).tolist() == [[False]]


def test_occlusion_capsule_beside_the_ray_does_not_block(physics):
    beside = [-0.5, 0.5, 0.2, 0.5, 0.5, 0.2, 0.05]   # z = 0.2로 비켜 있음
    state = _single_patch_state([_FAR_AWAY, beside])
    assert occlusion(state, _nozzle_at([0.0, 1.0, 0.0]), physics).tolist() == [[True]]


def test_occlusion_capsule_behind_nozzle_does_not_block(physics):
    behind = [-0.5, 1.5, 0.0, 0.5, 1.5, 0.0, 0.05]    # 노즐 너머
    state = _single_patch_state([_FAR_AWAY, behind])
    assert occlusion(state, _nozzle_at([0.0, 1.0, 0.0]), physics).tolist() == [[True]]


def test_occlusion_sphere_capsule_blocks(physics):
    sphere = [0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.05]     # p0 == p1
    state = _single_patch_state([_FAR_AWAY, sphere])
    assert occlusion(state, _nozzle_at([0.0, 1.0, 0.0]), physics).tolist() == [[False]]


def test_occlusion_ignores_own_capsule(physics):
    blocker = [-0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.05]
    own = _single_patch_state([_FAR_AWAY, blocker], patch_capsule=1)
    assert occlusion(own, _nozzle_at([0.0, 1.0, 0.0]), physics).tolist() == [[True]]


def test_occlusion_backface_nozzle_is_invisible(physics):
    state = _single_patch_state([_FAR_AWAY])
    vis = occlusion(state, _nozzle_at([0.0, -1.0, 0.0], [0.0, 1.0, 0.0]), physics)
    assert vis.tolist() == [[False], [True]]


def test_occlusion_backface_uses_patch_position_not_offset(physics):
    """뒷면 판정 기준은 `normal · (nozzle_pos - patch_pos)` (D 문서 단계 3).

    노즐이 접평면 바로 위(0 < 높이 < wall_offset_m)에 있으면 앞면이다. 오프셋한
    시작점 기준으로 판정하면 이 스치는 쌍을 잘못 가린다.
    """
    state = _single_patch_state([_FAR_AWAY])
    height = 0.5 * physics["air"]["wall_offset_m"]
    vis = occlusion(state, _nozzle_at([1.0, height, 0.0]), physics)
    assert vis.tolist() == [[True]]


def _occlusion_brute_force(state: BodyState, nozzle: NozzleConfig, physics: dict,
                           patches: np.ndarray, samples: int = 4001) -> np.ndarray:
    """(M, len(patches)) bool. 선분 위 점을 촘촘히 찍어 캡슐 안에 드는지 직접 본다."""
    delta = physics["air"]["wall_offset_m"]
    pos = state.patch_pos.astype(np.float64)
    normal = state.patch_normal.astype(np.float64)
    npos = nozzle.positions.astype(np.float64)
    caps = state.capsules.astype(np.float64)
    ts = np.linspace(0.0, 1.0, samples)[:, None]
    out = np.zeros((nozzle.count, len(patches)), dtype=bool)
    for m in range(nozzle.count):
        for j, i in enumerate(patches):
            if (npos[m] - pos[i]) @ normal[i] <= 0.0:          # 뒷면
                continue
            start = pos[i] + delta * normal[i]
            pts = start + ts * (npos[m] - start)
            clear = True
            for k in range(caps.shape[0]):
                if state.patch_capsule is not None and k == state.patch_capsule[i]:
                    continue
                a, ab, r = caps[k, 0:3], caps[k, 3:6] - caps[k, 0:3], caps[k, 6]
                denom = ab @ ab
                u = np.clip((pts - a) @ ab / denom, 0.0, 1.0) if denom > 0 else np.zeros(len(pts))
                if (np.linalg.norm(pts - (a + u[:, None] * ab), axis=1) < r).any():
                    clear = False
                    break
            out[m, j] = clear
    return out


@pytest.mark.parametrize("arms_up, yaw_deg", [(False, 30.0), (True, 0.0)])
def test_occlusion_matches_brute_force_on_fake_body(physics, arms_up, yaw_deg):
    """벡터화 결과를 점 샘플링 기반 느린 판정과 대조한다.

    (True, 0.0)은 든 팔 패치에서 머리 구가 실제로 광선을 가리는 경우를 포함한다.
    """
    state, nz = fake_body(arms_up=arms_up, yaw_deg=yaw_deg), fake_nozzles()
    every = np.arange(state.patch_pos.shape[0])
    assert np.array_equal(occlusion(state, nz, physics),
                          _occlusion_brute_force(state, nz, physics, every))


@pytest.mark.parametrize("scenario_name, pose", [
    ("default", PoseParams(shoulder_abduction=90.0, torso_yaw=30.0)),
    ("wheelchair", PoseParams()),     # 가림 전용 캡슐(capsule_part = -1) 포함
])
def test_occlusion_matches_brute_force_on_real_body(physics, scenarios, scenario_name, pose):
    """B의 실제 몸(길이 0 캡슐, 휠체어 프레임 포함)에서 패치 일부를 뽑아 대조한다."""
    if select_body_fn(scenario=scenarios["default"]).uses_fake:
        pytest.skip("B 미병합")
    state = build_body(BodyParams(), pose, scenarios[scenario_name])
    nz = load_nozzles()
    patches = np.random.default_rng(0).choice(state.patch_pos.shape[0], 120, replace=False)
    fast = occlusion(state, nz, physics)[:, patches]
    slow = _occlusion_brute_force(state, nz, physics, patches, samples=2001)
    assert np.array_equal(fast, slow)
