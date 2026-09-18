"""마네킹 기하 테스트. 소유자: B. (`docs/tracks/B_body_jet.md` 단계 9)"""
from __future__ import annotations

import time

import numpy as np
import pytest

from airis.sim import body as body_mod
from airis.sim.body import (
    PART_ARMS, PART_OCCLUDER, PART_TORSO_BACK, PART_TORSO_FRONT,
    SHANK_RADIUS_M, joint_positions, surface_area,
)
from airis.sim.body import build_body as _build_body


def build_body(*args, **kwargs):
    """이 파일은 캡슐 마네킹 테스트다. 기본 모델(physics.yaml body.model)이 mesh 여도 캡슐 폴백을 검사한다."""
    kwargs.setdefault("model", "capsule")
    return _build_body(*args, **kwargs)
from airis.sim.scenario import load_scenarios
from airis.sim.types import PART_NAMES, BodyParams, BodyState, PoseParams

SCENARIOS = load_scenarios()


@pytest.fixture(scope="module")
def default_state() -> BodyState:
    return build_body(BodyParams(), PoseParams(), SCENARIOS["default"])


# ---------------------------------------------------------------------------
# 문서 단계 9 명시 항목
# ---------------------------------------------------------------------------
def test_default_top_is_height_and_bottom_is_floor(default_state):
    """최고점 z ≈ height_m (±3%), 최저점 z ≈ 0.

    발바닥은 발목 관절로 정의한다 (단계 2: 발바닥이 z=0). 문서의 몸통 길이 식이
    pelvis_z = leg_length_m 일 때 머리 꼭대기가 정확히 height_m 이 되도록 유도돼 있어
    발목 관절이 z=0 에 온다. 하퇴 캡슐 뚜껑은 그 아래로 반지름만큼 내려간다.
    """
    b = BodyParams()
    z = default_state.patch_pos[:, 2]
    assert z.max() == pytest.approx(b.height_m, rel=0.03)
    assert -SHANK_RADIUS_M - 1e-3 <= z.min() <= 1e-3

    joints, _, _ = joint_positions(b, PoseParams(), SCENARIOS["default"])
    assert min(joints["ankle_L"][2], joints["ankle_R"][2]) == pytest.approx(0.0, abs=1e-6)


def test_shoulder_abduction_90_puts_wrist_at_shoulder_height():
    """shoulder_abduction=90 이면 wrist z ≈ shoulder z. 팔꿈치 기본 굽힘(10도)이 있어도 성립해야 한다."""
    joints, _, _ = joint_positions(BodyParams(), PoseParams(shoulder_abduction=90),
                                   SCENARIOS["default"])
    for side in ("L", "R"):
        assert joints[f"wrist_{side}"][2] == pytest.approx(joints[f"shoulder_{side}"][2], abs=1e-6)
        assert joints[f"elbow_{side}"][2] == pytest.approx(joints[f"shoulder_{side}"][2], abs=1e-6)


def test_torso_yaw_180_flips_torso_front_normal_x():
    """torso_yaw=180 이면 torso_front 패치의 법선 x 성분 부호가 반전."""
    sc = SCENARIOS["default"]
    front = build_body(BodyParams(), PoseParams(torso_yaw=0), sc)
    back = build_body(BodyParams(), PoseParams(torso_yaw=180), sc)

    m0 = front.patch_part == PART_TORSO_FRONT
    m1 = back.patch_part == PART_TORSO_FRONT
    assert m0.sum() == m1.sum() > 0
    assert front.patch_normal[m0, 0].mean() > 0.0
    assert back.patch_normal[m1, 0].mean() < 0.0
    # 격자가 결정론적이므로 패치별로 정확히 부호가 뒤집힌다
    np.testing.assert_allclose(back.patch_normal[m1, 0], -front.patch_normal[m0, 0], atol=1e-5)


def test_wheelchair_seat_height_and_extra_capsules():
    """휠체어: pelvis z ≈ seat_height_m, 캡슐 수가 default보다 4개 많음."""
    sc = SCENARIOS["wheelchair"]
    joints, _, _ = joint_positions(BodyParams(), PoseParams(), sc)
    assert joints["pelvis"][2] == pytest.approx(sc.seat_height_m, abs=1e-6)

    default = build_body(BodyParams(), PoseParams(), SCENARIOS["default"])
    chair = build_body(BodyParams(), PoseParams(), sc)
    assert chair.capsules.shape[0] == default.capsules.shape[0] + 4
    assert (chair.capsule_part == PART_OCCLUDER).sum() == 4


@pytest.mark.parametrize("scenario", ["default", "pregnant", "wheelchair"])
def test_patch_area_matches_capsule_surface_area(scenario):
    """패치 면적 합 ≈ 캡슐 표면적 합 (±5%).

    몸통은 문서대로 패치 생성 때 타원 단면을 쓰므로 비교 기준도 타원 기둥 + 반타원체 뚜껑의
    해석적 면적이다 (`body.surface_area`). 가림 전용 캡슐은 패치가 없으므로 제외된다.
    """
    b, p, sc = BodyParams(), PoseParams(), SCENARIOS[scenario]
    state = build_body(b, p, sc)
    assert state.patch_area.sum() == pytest.approx(surface_area(b, p, sc), rel=0.05)


# ---------------------------------------------------------------------------
# 완료 기준 체크리스트 항목
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["default", "pregnant", "wheelchair"])
def test_build_body_contract(scenario):
    """세 시나리오 모두 예외 없이 BodyState 를 만들고 interfaces.md 계약을 지킨다."""
    state = build_body(BodyParams(), PoseParams(), SCENARIOS[scenario])
    n = state.patch_pos.shape[0]
    k = state.capsules.shape[0]

    assert isinstance(state, BodyState)
    assert state.patch_pos.shape == (n, 3) and state.patch_pos.dtype == np.float32
    assert state.patch_normal.shape == (n, 3) and state.patch_normal.dtype == np.float32
    assert state.patch_area.shape == (n,) and (state.patch_area > 0).all()
    assert state.patch_part.shape == (n,) and state.patch_part.dtype == np.int32
    assert state.capsules.shape == (k, 7) and state.capsules.dtype == np.float32
    assert state.capsule_part is not None and state.capsule_part.shape == (k,)
    assert state.patch_capsule is not None and state.patch_capsule.shape == (n,)

    # 법선은 단위 벡터
    np.testing.assert_allclose(np.linalg.norm(state.patch_normal, axis=1), 1.0, atol=1e-5)
    # 부위 ID 는 PART_NAMES 범위, 다섯 부위가 모두 존재
    assert set(np.unique(state.patch_part)) == set(range(len(PART_NAMES)))
    # 패치는 가림 전용 캡슐에 속하지 않는다
    assert (state.capsule_part[state.patch_capsule] != PART_OCCLUDER).all()


def test_normals_point_outward(default_state):
    """패치 법선은 소속 캡슐의 축에서 멀어지는 방향이다."""
    s = default_state
    caps = s.capsules[s.patch_capsule]
    p0, p1 = caps[:, :3], caps[:, 3:6]
    seg = p1 - p0
    L2 = np.maximum((seg * seg).sum(1), 1e-12)
    t = np.clip(((s.patch_pos - p0) * seg).sum(1) / L2, 0.0, 1.0)
    closest = p0 + t[:, None] * seg
    outward = s.patch_pos - closest
    assert ((outward * s.patch_normal).sum(1) > 0).all()


def test_torso_front_back_split_by_body_forward(default_state):
    """torso_front 는 몸 전방(+x) 쪽 법선, torso_back 은 반대."""
    s = default_state
    assert (s.patch_normal[s.patch_part == PART_TORSO_FRONT, 0] > 0).all()
    # 정확히 옆(θ = ±90°)인 패치는 전방 성분이 부동소수 잡음(±1e−16)이라 back 으로 둔다 (body._FRONT_BACK_TOL)
    assert (s.patch_normal[s.patch_part == PART_TORSO_BACK, 0] <= body_mod._FRONT_BACK_TOL).all()


def test_default_mannequin_stands_upright():
    """누워 있지 않다: 머리 > 골반 > 발목, 세로 길이가 가로 폭보다 훨씬 크다."""
    b = BodyParams()
    joints, _, _ = joint_positions(b, PoseParams(), SCENARIOS["default"])
    assert joints["head_center"][2] > joints["chest"][2] > joints["pelvis"][2] > joints["knee_L"][2]
    assert joints["knee_L"][2] > joints["ankle_L"][2]
    for name in ("chest", "head_center", "knee_L", "ankle_L"):
        # 직립 기본 자세에서 몸의 중심축 관절은 골반 바로 위/아래에 있다
        assert joints[name][0] == pytest.approx(joints["pelvis"][0], abs=1e-6)


def test_mannequin_centered_in_booth():
    """마네킹은 부스 중앙 (booth.length_m/2, 0) 에 선다."""
    booth = body_mod._booth()
    joints, _, _ = joint_positions(BodyParams(), PoseParams(), SCENARIOS["default"])
    assert joints["pelvis"][0] == pytest.approx(booth["length_m"] / 2.0)
    assert joints["pelvis"][1] == pytest.approx(0.0)


def test_default_arms_do_not_pass_through_torso():
    """기본 자세에서 팔꿈치·손목(팔 표면 포함)이 몸통 타원 기둥 밖에 있다."""
    b = BodyParams()
    joints, dims, _ = joint_positions(b, PoseParams(), SCENARIOS["default"])
    pelvis = joints["pelvis"]
    for side in ("L", "R"):
        for name, radius in ((f"elbow_{side}", body_mod.UPPER_ARM_RADIUS_M),
                             (f"wrist_{side}", body_mod.FOREARM_RADIUS_M)):
            dx, dy = (joints[name] - pelvis)[:2]
            # 몸통 단면 타원을 팔 반지름만큼 부풀린 영역의 바깥이어야 한다
            a_ap = dims.torso_half_ap + radius
            a_lat = dims.torso_half_lateral + radius
            assert (dx / a_ap) ** 2 + (dy / a_lat) ** 2 > 1.0, name


def test_wheelchair_is_seated():
    """휠체어: 대퇴가 수평(무릎 높이 ≈ 고관절 높이), 하퇴가 수직으로 바닥을 향한다."""
    joints, dims, _ = joint_positions(BodyParams(), PoseParams(), SCENARIOS["wheelchair"])
    for side in ("L", "R"):
        hip, knee, ankle = joints[f"hip_{side}"], joints[f"knee_{side}"], joints[f"ankle_{side}"]
        assert knee[2] == pytest.approx(hip[2], abs=1e-6)                      # 대퇴 수평
        assert knee[0] - hip[0] == pytest.approx(dims.thigh_length, abs=1e-6)  # 앞으로
        assert ankle[0] == pytest.approx(knee[0], abs=1e-6)                     # 하퇴 수직
        assert 0.0 <= ankle[2] < knee[2]


def test_fixed_pose_overrides_pose_arguments():
    """scenario.fixed_pose 가 호출자의 자세 값을 덮어쓴다."""
    sc = SCENARIOS["wheelchair"]
    a = build_body(BodyParams(), PoseParams(hip_flexion=0, knee_flexion=0), sc)
    b = build_body(BodyParams(), PoseParams(hip_flexion=20, knee_flexion=45), sc)
    np.testing.assert_array_equal(a.patch_pos, b.patch_pos)


def test_torso_pitch_moves_upper_body_only():
    """torso_pitch 는 상체에만 적용되고 다리는 그대로다. 양수는 앞으로 숙임(+x)."""
    sc = SCENARIOS["default"]
    j0, _, _ = joint_positions(BodyParams(), PoseParams(torso_pitch=0), sc)
    j1, _, _ = joint_positions(BodyParams(), PoseParams(torso_pitch=30), sc)
    assert j1["head_center"][0] > j0["head_center"][0]
    assert j1["head_center"][2] < j0["head_center"][2]
    for name in ("hip_L", "knee_L", "ankle_L"):
        np.testing.assert_allclose(j1[name] - j1["pelvis"], j0[name] - j0["pelvis"], atol=1e-6)


def test_joint_signs_follow_table():
    """단계 2 표의 양의 방향: 고관절/어깨 굴곡 = 앞으로, 무릎 = 뒤로, yaw = 왼쪽(반시계)."""
    sc = SCENARIOS["default"]

    def j(**kw):
        return joint_positions(BodyParams(), PoseParams(**kw), sc)[0]

    base = j()
    assert j(hip_flexion=30)["knee_L"][0] > base["knee_L"][0]
    bent = j(knee_flexion=60)
    assert bent["ankle_L"][0] < bent["knee_L"][0]
    assert j(shoulder_flexion=90, shoulder_abduction=0)["wrist_L"][0] > base["wrist_L"][0]
    # yaw +90: 몸 전방(+x)이 왼쪽(+y)을 향한다 → 왼쪽 어깨가 뒤(-x)로 간다
    yawed = j(torso_yaw=90)
    assert yawed["shoulder_L"][0] < yawed["pelvis"][0]
    # 만세 자세
    assert j(shoulder_abduction=180)["wrist_L"][2] > base["head_center"][2]


def test_patches_are_deterministic():
    sc = SCENARIOS["pregnant"]
    a = build_body(BodyParams(), PoseParams(), sc)
    b = build_body(BodyParams(), PoseParams(), sc)
    np.testing.assert_array_equal(a.patch_pos, b.patch_pos)
    np.testing.assert_array_equal(a.patch_normal, b.patch_normal)


def test_patch_density_scales_with_argument():
    sc = SCENARIOS["default"]
    lo = build_body(BodyParams(), PoseParams(), sc, patches_per_m2=500.0)
    hi = build_body(BodyParams(), PoseParams(), sc, patches_per_m2=2000.0)
    assert hi.patch_pos.shape[0] > 2 * lo.patch_pos.shape[0]
    # 해상도가 달라도 총 면적은 같다
    assert hi.patch_area.sum() == pytest.approx(lo.patch_area.sum(), rel=0.02)


def test_arm_patches_labelled_arms(default_state):
    s = default_state
    arm_caps = np.where(s.capsule_part == PART_ARMS)[0]
    assert len(arm_caps) == 4
    assert (s.patch_part[np.isin(s.patch_capsule, arm_caps)] == PART_ARMS).all()


def test_build_body_under_20ms():
    """완료 기준: 1회 build_body 20 ms 이하 (부하 시 기준 연산 대비 14배 이하)."""
    b, p, sc = BodyParams(), PoseParams(), SCENARIOS["default"]
    _assert_time_budget(lambda: build_body(b, p, sc), budget_s=0.020, max_ratio=14.0, label="build_body")


# ---------------------------------------------------------------------------
# 성능 판정: 절대 시간 우선, 초과하면 기준 연산 대비 비율로 판정 (CPU 경쟁에 강하게)
# ---------------------------------------------------------------------------
# 다른 세션이 CPU 를 쓰면(예: C 의 8 프로세스 스윕) 15회 최솟값도 절대 시간 기준을 넘는다.
# 같은 순간의 CPU 상태를 반영하도록, 측정마다 같은 성격(행렬곱 + exp)의 고정 numpy 연산을
# 번갈아 재고 그 비율을 본다. 부하는 둘 다 느리게 하지만, 알고리즘 퇴행은 비율을 키운다.
# 비율 상한은 부하 0·8·16 프로세스에서 관측한 최대 비율의 약 2배다 (PR 본문 표).
_REF_RNG = np.random.default_rng(12345)
_REF_A = _REF_RNG.random((16, 3600))
_REF_D = _REF_RNG.random((16, 3))
_REF_X = _REF_RNG.random((3600, 3))


def _reference_workload() -> float:
    s = _REF_D @ _REF_X.T
    return float(np.exp(-(_REF_A * _REF_A) / (1.0 + s * s)).sum())


def _assert_time_budget(fn, budget_s: float, max_ratio: float, label: str, reps: int = 15) -> None:
    """fn 의 최솟값 시간이 budget_s 이하면 통과. 넘으면 기준 연산 대비 비율이 max_ratio 이하여야 한다."""
    fn()
    _reference_workload()
    best = best_ref = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        _reference_workload()
        best_ref = min(best_ref, time.perf_counter() - t0)
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    if best < budget_s:
        return
    ratio = best / best_ref
    assert ratio < max_ratio, (
        f"{label} {best * 1e3:.2f} ms > {budget_s * 1e3:.0f} ms 이고, 기준 연산 "
        f"{best_ref * 1e3:.2f} ms 대비 {ratio:.1f}배 > {max_ratio}배 (부하가 아니라 느려졌다)"
    )


# ---------------------------------------------------------------------------
# 좌우 거울 대칭 (C 발견: 패치 격자가 y 거울 대칭이 아니어서 yaw ±θ 점수가 격자 잡음만큼 달랐다)
# ---------------------------------------------------------------------------
_MIRROR = np.array([1.0, -1.0, 1.0])
_MIRROR_POSES = [
    PoseParams(torso_yaw=45.0),
    PoseParams(torso_yaw=45.0, shoulder_abduction=35.0, shoulder_flexion=40.0, elbow_flexion=60.0,
               torso_pitch=10.0, hip_flexion=15.0, knee_flexion=20.0),
    PoseParams(torso_yaw=-30.0, shoulder_abduction=160.0),
]


def _mirror_pairs(a: BodyState, b: BodyState) -> np.ndarray:
    """a 의 각 패치에 대해 y 거울상 b 에서 가장 가까운 패치 인덱스와 거리."""
    from scipy.spatial import cKDTree
    dist, idx = cKDTree(b.patch_pos.astype(np.float64) * _MIRROR).query(a.patch_pos.astype(np.float64))
    return dist, idx


@pytest.mark.parametrize("scenario", ["default", "wheelchair"])
@pytest.mark.parametrize("patches_per_m2", [400.0, 2000.0])
@pytest.mark.parametrize("pose", _MIRROR_POSES, ids=["yaw45", "mixed", "arms_up"])
def test_patches_are_mirror_symmetric_under_yaw_flip(scenario, patches_per_m2, pose):
    """build_body(yaw = θ) 를 y 로 반사하면 build_body(yaw = −θ) 와 패치 단위로 일치한다 (1e−6 m).

    마네킹은 부스 중심선 y = 0 에 서 있고 좌우 관절 부호가 거울 관계라, 패치 격자도 거울이어야
    yaw ±θ 평가가 격자 잡음 없이 같아진다. 위치·법선·면적·부위를 짝지어 비교한다.
    """
    import dataclasses
    sc = SCENARIOS[scenario]
    a = build_body(BodyParams(), pose, sc, patches_per_m2=patches_per_m2)
    b = build_body(BodyParams(), dataclasses.replace(pose, torso_yaw=-pose.torso_yaw), sc,
                   patches_per_m2=patches_per_m2)
    assert a.patch_pos.shape == b.patch_pos.shape
    dist, idx = _mirror_pairs(a, b)
    assert dist.max() < 1e-6
    assert len(np.unique(idx)) == len(idx)                                   # 일대일
    np.testing.assert_allclose(a.patch_normal, b.patch_normal[idx] * _MIRROR, atol=1e-6)
    np.testing.assert_allclose(a.patch_area, b.patch_area[idx], rtol=1e-6)
    np.testing.assert_array_equal(a.patch_part, b.patch_part[idx])


@pytest.mark.parametrize("patches_per_m2", [400.0, 2000.0])
def test_zero_yaw_body_is_its_own_mirror(patches_per_m2):
    """yaw 0 이면 패치 집합이 y → −y 에 대해 자기 자신과 일치한다."""
    a = build_body(BodyParams(), PoseParams(), SCENARIOS["default"], patches_per_m2=patches_per_m2)
    dist, idx = _mirror_pairs(a, a)
    assert dist.max() < 1e-6
    assert len(np.unique(idx)) == len(idx)
