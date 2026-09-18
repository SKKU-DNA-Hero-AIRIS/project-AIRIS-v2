"""정성 검증 테스트(E1) + 점수 모듈·가림 판정 단위 테스트. 소유자: D.

이 테스트들은 병합 게이트다: PR 전에 로컬에서 실행하고, 깨지면 병합하지 않는다.

B 병합 전후 자동 전환 (`docs/tracks/D_patch_baseline.md` 단계 5)
- `build_body`, `load_nozzles`가 `NotImplementedError`를 내면 `tests/fakes.py`의 가짜로
  바꾼다. B가 병합된 지금은 실제 몸과 실제 노즐 배치로 돈다. 제트는 B의
  `velocity_field_per_nozzle`이다 (가짜 제트는 B 병합 후 삭제).

E1 노즐 배치와 물리 상수
- 기준 장비는 퓨리움 슬롯 바(`nozzles.yaml`의 `active: slot_bars`)다. 측면 바가 몸과 같은 x에
  좌우 대칭으로 있어 "노즐을 마주 본다/등진다"가 정의되지 않는다. 그래서 E1 방향 검증은
  원형 노즐 비교 배치(`load_nozzles(layout="layout")`, 벽면 2열 × 4단, 25도 경사)로 한다.
- 물리 상수는 `configs/physics.yaml` 그대로다. 이전의 제트 덮어쓰기(노즐 지름 0.08 m 등)는
  퓨리움 부스(길이 0.886 m)로 바뀐 비교 배치에서 필요 없어져 지웠다.
- 슬롯 배치에서는 배치와 무관한 항목(세기 단조, 세기 0)과 "팔 들면 겨드랑이 상승"을 따로 본다.
  마지막 항목은 충돌 제트 보정(4.2b)이 켜져야 성립한다.

테스트용 부스 (E1 전용)
- 부스 밖 자세 불가 규칙(00_common.md 5절)은 팔 90도 같은 자세를 벌점으로 끊는다. E1은
  제트·가림·제거율의 **방향**을 보는 테스트라 규칙과 섞이지 않게, E1 평가기에는 벽과 천장을
  충분히 넓힌 부스를 준다. 규칙 자체는 아래 "부스 밖 자세 불가" 절에서 따로 본다.
"""
from __future__ import annotations

import copy
import dataclasses
import math
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm

from airis.sim import scoring
from airis.sim.body import build_body
from airis.sim.jet import velocity_field_per_nozzle
from airis.sim.patch_baseline import (
    SLOT_OCCLUSION_POINTS, PatchEvaluator, booth_excess, infeasible_score, occlusion,
    occlusion_sources, outside_booth,
)
from airis.sim.scenario import load_nozzle_layout, load_nozzles, load_physics, load_scenarios
from airis.sim.types import PART_NAMES, BodyParams, BodyState, NozzleConfig, PoseParams, Scenario
from tests.fakes import fake_body, fake_mesh_body, fake_nozzles

# E1 방향 검증용 원형 노즐 비교 배치. 근거는 모듈 docstring.
_E1_LAYOUT = "layout"
# E1 전용 부스 배율. 폭·높이만 키우고 길이(마네킹 x 위치)는 그대로 둔다. 근거는 모듈 docstring.
_E1_BOOTH_SCALE = 10.0

_ARMS = PART_NAMES.index("arms")
_FRONT = PART_NAMES.index("torso_front")
_BACK = PART_NAMES.index("torso_back")

# 겨드랑이·옆구리로 보는 몸통 위끝에서의 깊이.
_FLANK_DEPTH_M = 0.25


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


def mesh_body_fn(build=build_body, scenario: Scenario | None = None):
    """(pose, scenario) -> 메시 BodyState. B의 메시 `build_body`가 없으면 None.

    체형은 `body=None`으로 넘겨 메시 기본 체형(`human_mesh.MESH_DEFAULT_BODY`)을 쓴다. 전역
    `BodyParams()`는 아직 캡슐 시절 값이라 메시에 넣으면 팔이 길어진다 (docs/mesh_transition.md).
    """
    scenario = scenario or load_scenarios()["default"]
    try:
        state = build(None, PoseParams(), scenario, model="mesh")
    except (TypeError, NotImplementedError, ImportError):
        return None
    if state.mesh_vertices is None:
        return None

    def mesh(pose, scen):
        return build(None, pose, scen, model="mesh")
    mesh.uses_fake = False
    return mesh


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
def scenarios() -> dict[str, Scenario]:
    return load_scenarios()


@pytest.fixture(scope="module", params=["capsule", "mesh"])
def sim(request, physics, scenarios):
    """E1용 평가기. 캡슐 몸과 메시 몸(B 메시 병합 후) 두 모델로 같은 테스트를 돌린다."""
    if request.param == "mesh":
        make_body = mesh_body_fn(scenario=scenarios["default"])
        if make_body is None:
            pytest.skip("B의 메시 build_body(model='mesh') 미병합")
    else:
        make_body = select_body_fn(scenario=scenarios["default"])
    booth = dict(load_nozzle_layout()["booth"])
    booth["width_m"] *= _E1_BOOTH_SCALE
    booth["height_m"] *= _E1_BOOTH_SCALE
    evaluator = PatchEvaluator(
        physics,
        build_body=lambda body, pose, scen: make_body(pose, scen),
        velocity_field_per_nozzle=velocity_field_per_nozzle,
        booth=booth,
    )
    return SimpleNamespace(evaluator=evaluator,
                           nozzles=select_nozzles(partial(load_nozzles, layout=_E1_LAYOUT)),
                           scenario=scenarios["default"], uses_fake_body=make_body.uses_fake,
                           cfg=physics, model=request.param)


def _eval(sim, pose: PoseParams, nozzle: NozzleConfig | None = None):
    return sim.evaluator.evaluate(pose, nozzle or sim.nozzles, BodyParams(), sim.scenario)


def _with_strengths(nozzle: NozzleConfig, strengths: np.ndarray) -> NozzleConfig:
    """세기만 바꾼 사본. 슬롯 필드(slot_axis, slot_length) 등 나머지는 그대로 둔다."""
    return dataclasses.replace(nozzle, strengths=np.asarray(strengths, dtype=np.float32))


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

def _armpit_flank_mask(state: BodyState) -> np.ndarray:
    """겨드랑이·옆구리 패치: 몸통 패치 중 법선이 좌우(|n_y| > 0.7)를 보고 몸통 위끝에서
    0.25 m 안에 있는 것. yaw 0 / 180에서만 y가 몸의 좌우 축이다."""
    torso = np.isin(state.patch_part, [_FRONT, _BACK])
    top = state.patch_pos[torso, 2].max()
    return (torso & (np.abs(state.patch_normal[:, 1]) > 0.7)
            & (state.patch_pos[:, 2] > top - _FLANK_DEPTH_M))



@pytest.mark.parametrize("yaw_deg", [0.0, 180.0])
def test_raising_arms_increases_armpit_removal(sim, yaw_deg):
    """팔을 들면 겨드랑이/옆구리 제거율이 오른다.

    D 문서 단계 5는 `arms` + `torso_front` 부위 합산을 비교하라고 적는다. 실제 몸에서 이
    합산은 yaw에 따라 부호가 바뀐다: 든 팔은 제트 띠(z 0.5~1.7 m) 위로 올라가 `arms`
    제거율이 떨어질 수 있고(A 입자판 관찰과 같음), 이것이 합산을 끌어내린다. E1의 물리적
    주장은 "팔에 가려 있던 옆구리가 드러난다"이므로 그 패치를 직접 비교한다 (PR 본문).
    """
    def flank_removal(abduction):
        pose = PoseParams(shoulder_abduction=abduction, torso_yaw=yaw_deg)
        state = sim.evaluator._build_body(BodyParams(), pose, sim.scenario)
        mask = _armpit_flank_mask(state)
        assert mask.sum() >= 10, "겨드랑이·옆구리 패치를 찾지 못했다"
        area = state.patch_area[mask].astype(np.float64)
        removal = _eval(sim, pose).extra["removal"][mask]
        return float((removal * area).sum() / area.sum())

    down, up = flank_removal(20.0), flank_removal(90.0)
    assert up > 0.0, "제트가 옆구리에 닿지 않아 비교가 무의미하다"
    assert up > down


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


@pytest.mark.parametrize("pose", [PoseParams(), PoseParams(shoulder_abduction=90.0, torso_yaw=45.0)])
def test_stronger_nozzles_increase_removal(sim, pose):
    """노즐 세기를 0.5 -> 1.0 -> 1.5배로 올리면 총 제거율이 단조 증가한다.

    README E1의 "노즐에 가까이 서면 상승"을 대체한 항목이다. 가까워지는 비교는 현재 자유 제트
    모델에서 성립하지 않는다 (D 문서 단계 5 표, PR #11 발견 사항 1).
    """
    removals = [_eval(sim, pose, _with_strengths(sim.nozzles, sim.nozzles.strengths * k)).total_removal
                for k in (0.5, 1.0, 1.5)]
    assert removals[0] > 0.0, "제트가 몸에 닿지 않아 비교가 무의미하다"
    assert removals[0] < removals[1] < removals[2]


@pytest.fixture(scope="module")
def slot_nozzles() -> NozzleConfig:
    nz = load_nozzles(layout="slot_bars")
    assert nz.slot_axis is not None
    return nz


def test_slot_layout_strength_monotone_and_zero(sim, slot_nozzles):
    """기준 장비(슬롯 바)에서도 세기 단조 증가와 세기 0 -> 0이 성립한다."""
    pose = PoseParams()
    removals = [_eval(sim, pose, _with_strengths(slot_nozzles, slot_nozzles.strengths * k)).total_removal
                for k in (0.5, 1.0, 1.5)]
    assert removals[0] > 0.0
    assert removals[0] < removals[1] < removals[2]
    off = _eval(sim, pose, _with_strengths(slot_nozzles, np.zeros(slot_nozzles.count)))
    assert off.total_removal == 0.0


def test_slot_layout_raising_arms_increases_armpit_removal(sim, slot_nozzles):
    """기준 장비(슬롯 바)에서도 팔을 들면 겨드랑이·옆구리 제거율이 오른다.

    비교 자세는 20도 vs 150도(만세 쪽)다. 팔 수평(90도)은 퓨리움 부스(폭 1.46 m)에서 벽 밖이라
    실제로는 평가되지 않는다. 충돌 제트 보정(4.2b) 전에는 성립하지 않았다: 측면 바가 수평으로
    쏘아 든 팔이 옆구리를 가렸다. 보정을 켜면 20도 0.021 -> 150도 0.083 (K=3, PR 본문).
    """
    real_booth = load_nozzle_layout()["booth"]
    raised = PoseParams(shoulder_abduction=150.0)
    assert not outside_booth(sim.evaluator._build_body(BodyParams(), raised, sim.scenario).patch_pos,
                             real_booth), "만세 자세가 실제 부스 밖이면 비교 자세를 다시 골라야 한다"
    down = _flank_removal(sim, PoseParams(shoulder_abduction=20.0), slot_nozzles)
    up = _flank_removal(sim, raised, slot_nozzles)
    assert down > 0.0
    assert up > down


@pytest.mark.parametrize("wall_yaw", [90.0, -90.0])
def test_slot_layout_front_facing_wall_increases_front_removal(sim, slot_nozzles, wall_yaw):
    """슬롯 배치의 "마주 봄" 항목: 정면이 측면 바(벽)를 보면(yaw ±90) 정면 제거율이 yaw 0보다 크다.

    측면 바가 몸과 같은 x에 좌우 대칭이라 원형 배치의 "등지면 하락"은 정의되지 않는다. 통합 관리
    결정으로 이 항목을 슬롯 배치의 E1으로 둔다. 충돌 보정 후 약 2.7배 (0.047 -> 0.126).
    """
    along = _eval(sim, PoseParams(torso_yaw=0.0), slot_nozzles)
    facing_wall = _eval(sim, PoseParams(torso_yaw=wall_yaw), slot_nozzles)
    assert along.removal_by_part[_FRONT] > 0.0
    assert facing_wall.removal_by_part[_FRONT] > along.removal_by_part[_FRONT]


def _flank_removal(sim, pose: PoseParams, nozzle: NozzleConfig) -> float:
    state = sim.evaluator._build_body(BodyParams(), pose, sim.scenario)
    mask = _armpit_flank_mask(state)
    area = state.patch_area[mask].astype(np.float64)
    removal = _eval(sim, pose, nozzle).extra["removal"][mask]
    return float((removal * area).sum() / area.sum())


def test_slot_occlusion_does_not_kill_flank_under_raised_arm(sim, slot_nozzles):
    """팔 90도에서도 슬롯 바 -> 옆구리 기여가 통째로 0이 되지 않는다.

    슬롯을 중심 한 점으로 보면 팔이 그 점을 가릴 때 슬롯 전체가 막혀 옆구리 R이 0이었다 (#24).
    """
    assert _flank_removal(sim, PoseParams(shoulder_abduction=90.0), slot_nozzles) > 0.0


@pytest.mark.parametrize("pose", [PoseParams(), PoseParams(shoulder_abduction=60.0, torso_yaw=30.0),
                                  PoseParams(shoulder_abduction=160.0, elbow_flexion=0.0)])
def test_slot_occlusion_converges_in_point_count(sim, slot_nozzles, pose):
    """슬롯 점 수를 늘리면 총 제거율이 K=9 결과로 수렴한다.

    기본 K=3은 15% 안, 검증용 K=5는 5% 안. 부스 안 무작위 자세 200개(400/m², 충돌 보정 켬)에서
    K=9 대비 상대 차: K=3 최대 캡슐 9.2%·메시 11.6%(p99 8.8%·8.3%, 순위 상관 0.995),
    K=5 최대 3.4%·4.4%. 팔 45~60°에 yaw 30° 부근이 메시 K=3에서 가장 크다(10~12%).
    """
    assert SLOT_OCCLUSION_POINTS == 3

    def removal_with(k):
        ev = PatchEvaluator(sim.cfg, build_body=sim.evaluator._build_body,
                            velocity_field_per_nozzle=sim.evaluator._velocity_field_per_nozzle,
                            booth=sim.evaluator.booth, slot_points=k)
        return ev.evaluate(pose, slot_nozzles, BodyParams(), sim.scenario).total_removal

    r3, r5, r9 = removal_with(3), removal_with(5), removal_with(9)
    assert r9 > 0.0
    assert abs(r3 - r9) / r9 < 0.15
    assert abs(r5 - r9) / r9 < 0.05


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


def test_evaluator_uses_config_constants_not_literals(sim, physics):
    """상수를 덮어쓰면 결과가 따라 바뀐다 -> 코드에 숫자가 박혀 있지 않다."""
    rough = copy.deepcopy(physics)
    rough["adhesion"]["fabric_roughness_factor"] *= 4.0     # 임계 전단 상승 -> 제거율 하락
    ev = PatchEvaluator(rough, build_body=sim.evaluator._build_body,
                        velocity_field_per_nozzle=sim.evaluator._velocity_field_per_nozzle)
    base = _eval(sim, PoseParams())
    harder = ev.evaluate(PoseParams(), sim.nozzles, BodyParams(), sim.scenario)
    assert harder.total_removal < base.total_removal


# --- 부스 밖 자세 불가 (00_common.md 5절) ------------------------------------

_ARMS_OUT = PoseParams(shoulder_abduction=90.0, elbow_flexion=0.0)       # 팔 수평
_HANDS_UP = PoseParams(shoulder_abduction=150.0, elbow_flexion=0.0)     # 만세 (범위 상한)


def _extent(state: BodyState) -> tuple[float, float]:
    """(max |y|, max z). 규칙이 보는 두 값을 테스트에서 따로 계산한다."""
    pos = state.patch_pos.astype(np.float64)
    return float(np.abs(pos[:, 1]).max()), float(pos[:, 2].max())


def _never_called(*args, **kwargs):
    raise AssertionError("부스 밖 자세인데 제트를 계산했다")


def test_outside_booth_checks_side_walls_and_ceiling_but_not_x():
    booth = {"length_m": 2.0, "width_m": 1.0, "height_m": 2.0}
    inside = np.array([[1.0, 0.0, 1.0]])
    assert booth_excess(inside, booth) == 0.0
    assert booth_excess(np.array([[1.0, -0.8, 2.1]]), booth) == pytest.approx(0.3)   # 벽 0.3 > 천장 0.1
    assert booth_excess(np.array([[1.0, 0.55, 2.4]]), booth) == pytest.approx(0.4)   # 천장 0.4 > 벽 0.05
    assert not outside_booth(inside, booth)
    assert not outside_booth(np.array([[1.0, 0.5, 2.0], [1.0, -0.5, 0.0]]), booth)   # 경계는 안쪽
    assert outside_booth(np.vstack([inside, [[1.0, 0.5001, 1.0]]]), booth)
    assert outside_booth(np.vstack([inside, [[1.0, -0.5001, 1.0]]]), booth)
    assert outside_booth(np.vstack([inside, [[1.0, 0.0, 2.0001]]]), booth)
    assert not outside_booth(np.array([[-5.0, 0.0, 1.0], [9.0, 0.0, 1.0]]), booth)   # x는 열린 문


def test_arms_out_is_infeasible_hands_up_is_feasible(physics, scenarios):
    """팔 90도(수평)는 옆벽 밖, 만세는 부스 안으로 판정된다.

    판정은 부스 폭에 달려 있으므로 폭을 숫자로 박지 않는다. 두 자세의 좌우 끝 사이에 벽을 두고
    높이는 실제 부스 값을 쓴다 (기준 장비가 바뀌어도 같은 주장을 검사한다).
    """
    scen = scenarios["default"]
    real = load_nozzle_layout()["booth"]
    probe = PatchEvaluator(physics, velocity_field_per_nozzle=_never_called, booth=real)
    y_out, _ = _extent(probe.build_state(BodyParams(), _ARMS_OUT, scen))
    y_up, z_up = _extent(probe.build_state(BodyParams(), _HANDS_UP, scen))
    assert y_up < y_out, "만세가 팔 수평보다 옆으로 덜 나가야 한다"
    if z_up > real["height_m"]:
        pytest.skip(f"만세 손끝 z={z_up:.3f} m가 부스 높이 {real['height_m']} m를 넘는다")

    booth = dict(real, width_m=y_up + y_out)            # 반폭 = 두 끝의 중간
    nozzle = load_nozzles()
    ev = PatchEvaluator(physics, velocity_field_per_nozzle=_never_called, booth=booth)
    out = ev.evaluate(_ARMS_OUT, nozzle, BodyParams(), scen)
    assert out.extra["infeasible"] is True
    d_out = y_out - booth["width_m"] / 2.0
    assert out.extra["d_out"] == pytest.approx(d_out, rel=1e-6)
    assert out.score == pytest.approx(-1.0 - 10.0 * d_out, rel=1e-6)
    assert out.score < -1.0
    assert out.total_removal == 0.0
    assert out.removal_by_part.shape == (len(PART_NAMES),)
    assert (out.removal_by_part == 0.0).all()

    ev_up = PatchEvaluator(physics, booth=booth)
    up = ev_up.evaluate(_HANDS_UP, nozzle, BodyParams(), scen)
    assert up.extra["infeasible"] is False
    assert np.isfinite(up.score)


@pytest.mark.parametrize("pose", [PoseParams(), _ARMS_OUT, _HANDS_UP,
                                  PoseParams(shoulder_flexion=150.0, shoulder_abduction=0.0)])
def test_infeasible_flag_matches_configured_booth(physics, scenarios, pose):
    """설정 파일의 부스로 판정한 결과가 테스트에서 직접 계산한 기하와 같다."""
    scen = scenarios["default"]
    booth = load_nozzle_layout()["booth"]
    ev = PatchEvaluator(physics)
    y_max, z_max = _extent(ev.build_state(BodyParams(), pose, scen))
    expected = y_max > booth["width_m"] / 2.0 or z_max > booth["height_m"]
    result = ev.evaluate(pose, load_nozzles(), BodyParams(), scen)
    assert result.extra["infeasible"] is expected
    if expected:
        d_out = max(y_max - booth["width_m"] / 2.0, z_max - booth["height_m"])
        assert result.score == pytest.approx(-1.0 - 10.0 * d_out, rel=1e-6)


def test_infeasible_penalty_grows_with_distance_outside(physics, scenarios):
    """벽을 많이 넘을수록 점수가 낮다 (00_common.md 5절 등급 벌점)."""
    scen, nozzle = scenarios["default"], load_nozzles()
    probe = PatchEvaluator(physics, velocity_field_per_nozzle=_never_called)
    poses = [PoseParams(shoulder_abduction=a, elbow_flexion=0.0) for a in (60.0, 75.0, 90.0)]
    reach = [_extent(probe.build_state(BodyParams(), p, scen))[0] for p in poses]
    assert reach[0] < reach[1] < reach[2]

    booth = dict(load_nozzle_layout()["booth"], width_m=2.0 * (reach[0] - 0.01))   # 셋 다 벽 밖
    ev = PatchEvaluator(physics, velocity_field_per_nozzle=_never_called, booth=booth)
    scores = [ev.evaluate(p, nozzle, BodyParams(), scen).score for p in poses]
    assert scores[0] < -1.0
    assert scores[0] > scores[1] > scores[2]


def test_infeasible_score_formula():
    assert infeasible_score(0.0) == -1.0
    assert infeasible_score(0.05) == pytest.approx(-1.5)
    assert infeasible_score(0.2) == pytest.approx(-3.0)


# --- patches_per_m2 -----------------------------------------------------------

def test_patches_per_m2_is_passed_to_build_body(physics, scenarios):
    calls = []

    def recording_build(body, pose, scen, **kwargs):
        calls.append(kwargs)
        return build_body(body, pose, scen, **kwargs)

    scen, nozzle = scenarios["default"], load_nozzles()
    PatchEvaluator(physics, build_body=recording_build).evaluate(PoseParams(), nozzle, BodyParams(), scen)
    coarse = PatchEvaluator(physics, build_body=recording_build, patches_per_m2=400.0)
    result = coarse.evaluate(PoseParams(), nozzle, BodyParams(), scen)
    assert calls == [{}, {"patches_per_m2": 400.0}]

    n_coarse = build_body(BodyParams(), PoseParams(), scen, patches_per_m2=400.0).patch_pos.shape[0]
    n_default = build_body(BodyParams(), PoseParams(), scen).patch_pos.shape[0]
    assert n_coarse < n_default
    assert result.extra["removal"].shape == (n_coarse,)


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
    # 손 계산: 0.6·|80-20|/(abduction 범위) + 2.0·|10-0|/(pitch 범위)
    span = {k: hi - lo for k, (lo, hi) in scen.pose_bounds.items()}
    assert scoring.discomfort(pose, scen) == pytest.approx(
        0.6 * 60 / span["shoulder_abduction"] + 2.0 * 10 / span["torso_pitch"])


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


def _brute_force_sources(nozzle: NozzleConfig, slot_points: int) -> list[np.ndarray]:
    """노즐별 광원 점. 구현(`occlusion_sources`)과 따로, 00_common 4.1b의 행 단위 규약대로 만든다."""
    out = []
    for m in range(nozzle.count):
        p = nozzle.positions[m].astype(np.float64)
        if nozzle.slot_axis is None or nozzle.slot_length[m] <= 0 or not np.any(nozzle.slot_axis[m]):
            out.append(p[None, :])
            continue
        e = nozzle.slot_axis[m].astype(np.float64)
        e /= np.linalg.norm(e)
        offsets = np.linspace(-0.5, 0.5, slot_points) * float(nozzle.slot_length[m])
        out.append(p + offsets[:, None] * e)
    return out


def _occlusion_brute_force(state: BodyState, nozzle: NozzleConfig, physics: dict,
                           patches: np.ndarray, samples: int = 4001,
                           slot_points: int = SLOT_OCCLUSION_POINTS) -> np.ndarray:
    """(M, len(patches)) 보이는 비율. 선분 위 점을 촘촘히 찍어 캡슐 안에 드는지 직접 본다."""
    delta = physics["air"]["wall_offset_m"]
    pos = state.patch_pos.astype(np.float64)
    normal = state.patch_normal.astype(np.float64)
    caps = state.capsules.astype(np.float64)
    ts = np.linspace(0.0, 1.0, samples)[:, None]
    sources = _brute_force_sources(nozzle, slot_points)
    out = np.zeros((nozzle.count, len(patches)))
    for m, pts_m in enumerate(sources):
        for j, i in enumerate(patches):
            clear_count = 0
            for src in pts_m:
                if (src - pos[i]) @ normal[i] <= 0.0:          # 뒷면
                    continue
                start = pos[i] + delta * normal[i]
                pts = start + ts * (src - start)
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
                clear_count += clear
            out[m, j] = clear_count / len(pts_m)
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
    patches = np.random.default_rng(0).choice(state.patch_pos.shape[0], 120, replace=False)
    for layout in ("layout", "slot_bars"):                 # 원형 0/1, 슬롯은 점 비율
        nz = load_nozzles(layout=layout)
        fast = occlusion(state, nz, physics)[:, patches]
        slow = _occlusion_brute_force(state, nz, physics, patches, samples=2001)
        assert np.allclose(fast, slow, rtol=0.0, atol=1e-12), layout
        if layout == "layout":
            assert set(np.unique(fast)) <= {0.0, 1.0}
        else:
            assert ((fast > 0.0) & (fast < 1.0)).any(), "슬롯 부분 가림이 한 번도 나오지 않았다"


def test_occlusion_sources_spread_along_slot():
    nz = load_nozzles(layout="slot_bars")
    src, owner = occlusion_sources(nz, slot_points=5)
    assert src.shape == (5 * nz.count, 3)
    for m in range(nz.count):
        pts = src[owner == m]
        e = nz.slot_axis[m] / np.linalg.norm(nz.slot_axis[m])
        assert np.allclose(pts.mean(axis=0), nz.positions[m], atol=1e-6)
        assert np.allclose(pts[-1] - pts[0], e * nz.slot_length[m], atol=1e-6)

    round_nz = load_nozzles(layout="layout")
    src, owner = occlusion_sources(round_nz)
    assert np.array_equal(owner, np.arange(round_nz.count))
    assert np.allclose(src, round_nz.positions)


# --- 메시 가림 (D 단계 8, docs/mesh_transition.md 결정 5) -----------------------

def _ray_hits_brute_force(origin: np.ndarray, target: np.ndarray, tri: np.ndarray,
                          skip_face: int | None) -> bool:
    """선분 origin -> target이 삼각형 하나라도 지나면 True (Möller–Trumbore, float64)."""
    d = target - origin
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    h = np.cross(d, e2)
    a = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(a) > 1e-14
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = origin - v0
    u = f * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = f * (q @ d)
    t = f * np.einsum("ij,ij->i", e2, q)
    hit = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 0.0) & (t < 1.0)
    if skip_face is not None:
        hit[skip_face] = False
    return bool(hit.any())


def _mesh_occlusion_brute_force(state: BodyState, nozzle: NozzleConfig, physics: dict,
                                patches: np.ndarray) -> np.ndarray:
    delta = physics["air"]["wall_offset_m"]
    pos = state.patch_pos.astype(np.float64)
    normal = state.patch_normal.astype(np.float64)
    tri = state.mesh_vertices.astype(np.float64)[state.mesh_faces]
    out = np.zeros((nozzle.count, len(patches)))
    for m, pts_m in enumerate(_brute_force_sources(nozzle, SLOT_OCCLUSION_POINTS)):
        for j, i in enumerate(patches):
            clear = 0
            for src in pts_m:
                if (src - pos[i]) @ normal[i] <= 0.0:
                    continue
                start = pos[i] + delta * normal[i]
                clear += not _ray_hits_brute_force(start, src, tri, int(state.patch_face[i]))
            out[m, j] = clear / len(pts_m)
    return out


@pytest.mark.parametrize("arms_up, yaw_deg, layout", [
    (False, 30.0, "slot_bars"), (True, 0.0, "slot_bars"), (True, 180.0, "layout"),
])
def test_mesh_occlusion_matches_brute_force(physics, arms_up, yaw_deg, layout):
    """Open3D 광선 판정을 float64 Möller–Trumbore 전수 판정과 대조한다 (가짜 메시 몸)."""
    state = fake_mesh_body(arms_up=arms_up, yaw_deg=yaw_deg)
    nz = load_nozzles(layout=layout)
    patches = np.random.default_rng(1).choice(state.patch_pos.shape[0], 150, replace=False)
    fast = occlusion(state, nz, physics)[:, patches]
    slow = _mesh_occlusion_brute_force(state, nz, physics, patches)
    step = 1.0 / SLOT_OCCLUSION_POINTS
    # 모서리를 스치는 광선은 float32(Open3D)와 float64에서 갈릴 수 있다. 광원 점 하나 차이까지 허용.
    assert np.abs(fast - slow).max() <= step + 1e-9
    assert (fast != slow).mean() < 0.01
    assert ((fast > 0.0) & (fast < 1.0)).any() or layout == "layout"
    assert (fast == 0.0).any() and (fast == 1.0).any()


def _square_mesh(center, half: float, normal_axis: int) -> tuple[np.ndarray, np.ndarray]:
    """축 normal_axis에 수직인 정사각형 (삼각형 2개). 반환 (정점, 면)."""
    c = np.asarray(center, dtype=np.float64)
    a, b = [k for k in range(3) if k != normal_axis]
    corners = []
    for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        p = c.copy()
        p[a] += du * half
        p[b] += dv * half
        corners.append(p)
    return np.array(corners), np.array([[0, 1, 2], [0, 2, 3]])


def _mesh_state(patch_pos, patch_normal, patch_face, verts, faces, capsules=None, capsule_part=None):
    n = len(patch_pos)
    return BodyState(
        patch_pos=np.asarray(patch_pos, np.float32), patch_normal=np.asarray(patch_normal, np.float32),
        patch_area=np.full(n, 1e-3, np.float32), patch_part=np.full(n, _FRONT, np.int32),
        capsules=np.zeros((0, 7), np.float32) if capsules is None else np.asarray(capsules, np.float32),
        capsule_part=None if capsule_part is None else np.asarray(capsule_part, np.int32),
        mesh_vertices=np.asarray(verts, np.float32), mesh_faces=np.asarray(faces, np.int32),
        mesh_face_part=np.full(len(faces), _FRONT, np.int32),
        patch_face=np.asarray(patch_face, np.int32))


def _floor_with_patch_and_wall(wall_center):
    """바닥 정사각형(면 0, 1)과 그 위에 뜬 작은 정사각형 벽(면 2, 3)."""
    floor_v, floor_f = _square_mesh([0.0, 0.0, 0.0], 1.0, 2)
    wall_v, wall_f = _square_mesh(wall_center, 0.1, 2)
    verts = np.vstack([floor_v, wall_v])
    faces = np.vstack([floor_f, wall_f + 4])
    return verts, faces


def test_mesh_occlusion_wall_between_blocks_and_beside_does_not(physics):
    verts, faces = _floor_with_patch_and_wall([0.0, 0.0, 0.5])
    state = _mesh_state([[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], [0], verts, faces)
    above = _nozzle_at([0.0, 0.0, 1.0])
    beside = _nozzle_at([0.5, 0.0, 1.0])            # 벽(|x| <= 0.1 at z 0.5)을 비껴가는 광선
    behind_nozzle = _nozzle_at([0.0, 0.0, 0.4])     # 벽이 노즐 너머
    assert occlusion(state, above, physics).tolist() == [[0.0]]
    assert occlusion(state, beside, physics).tolist() == [[1.0]]
    assert occlusion(state, behind_nozzle, physics).tolist() == [[1.0]]


def test_mesh_occlusion_ignores_own_face(physics):
    """광선이 자기 면(`patch_face`)을 지나면 가림으로 치지 않는다.

    평평한 면에서 법선 = 면 법선이면 광선이 자기 면에 닿을 수 없어서, 법선을 면에서 기울여
    광선이 자기 면 평면을 뚫고 나가게 만든다. 자기 면 정보를 지우면 같은 광선이 막힌다.
    """
    import dataclasses

    verts, faces = _square_mesh([0.0, 0.0, 0.0], 2.0, 2)
    tilted = np.array([0.3, 0.0, 1.0]) / np.linalg.norm([0.3, 0.0, 1.0])
    state = _mesh_state([[0.3, -0.3, 0.0]], [tilted], [0], verts, faces)
    nozzle = _nozzle_at([2.3, -0.3, -0.1])      # 기울인 법선 기준 앞면, 광선은 z = 0 을 지난다
    assert occlusion(state, nozzle, physics).tolist() == [[1.0]]
    assert occlusion(dataclasses.replace(state, patch_face=None), nozzle, physics).tolist() == [[0.0]]


def test_mesh_occlusion_uses_frame_capsules_but_not_bone_capsules(physics):
    """메시 모델: 휠체어 프레임(capsule_part -1)은 가리고, 뼈 근사 캡슐은 가리지 않는다."""
    verts, faces = _square_mesh([0.0, 0.0, 0.0], 1.0, 2)
    blocker = [-0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.05]     # 패치와 노즐 사이를 가로지르는 캡슐
    above = _nozzle_at([0.0, 0.0, 1.0])
    bone = _mesh_state([[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], [0], verts, faces,
                       capsules=[blocker], capsule_part=[_FRONT])
    frame = _mesh_state([[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], [0], verts, faces,
                        capsules=[blocker], capsule_part=[-1])
    assert occlusion(bone, above, physics).tolist() == [[1.0]]
    assert occlusion(frame, above, physics).tolist() == [[0.0]]


def test_mesh_occlusion_backface_is_invisible(physics):
    verts, faces = _square_mesh([0.0, 0.0, 0.0], 1.0, 2)
    state = _mesh_state([[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], [0], verts, faces)
    assert occlusion(state, _nozzle_at([0.0, 0.0, -1.0], [0.0, 0.0, 1.0]), physics).tolist() == [[0.0], [1.0]]


def test_evaluator_runs_on_fake_mesh_body(physics, scenarios):
    """PatchEvaluator가 메시 BodyState를 받아 EvalResult를 채운다 (B 메시 병합 전 대체)."""
    scen = scenarios["default"]
    ev = PatchEvaluator(physics, build_body=lambda body, pose, s: fake_mesh_body(
        arms_up=pose.shoulder_abduction > 60, yaw_deg=pose.torso_yaw))
    result = ev.evaluate(PoseParams(), load_nozzles(), BodyParams(), scen)
    n = fake_mesh_body().patch_pos.shape[0]
    assert not result.extra["infeasible"]
    assert result.extra["removal"].shape == (n,)
    assert 0.0 < result.extra["visible_frac"] < 1.0
    assert 0.0 < result.total_removal < 1.0
