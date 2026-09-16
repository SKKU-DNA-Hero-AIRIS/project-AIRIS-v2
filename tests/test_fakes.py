"""가짜 입력이 BodyState / NozzleConfig 계약을 지키는지 확인한다. 소유자: D.

`docs/interfaces.md`의 배열 규약(float32 (N,3), int32 (N,), 바깥 단위 법선)과
`docs/tracks/00_common.md` 4.1(자유 제트)을 검사한다.
"""
import numpy as np
import pytest

from airis.sim.scenario import load_nozzle_layout, load_physics
from airis.sim.types import PART_NAMES, BodyState, NozzleConfig
from tests.fakes import fake_body, fake_nozzles, fake_velocity_field_per_nozzle


@pytest.fixture(scope="module")
def cfg():
    return load_physics()


@pytest.fixture(scope="module")
def booth():
    return load_nozzle_layout()["booth"]


# --- fake_body ---------------------------------------------------------------

@pytest.mark.parametrize("arms_up", [False, True])
@pytest.mark.parametrize("yaw_deg", [0.0, 90.0, 180.0])
def test_fake_body_fills_bodystate_contract(arms_up, yaw_deg):
    state = fake_body(arms_up=arms_up, yaw_deg=yaw_deg)
    assert isinstance(state, BodyState)
    n = state.patch_pos.shape[0]

    assert state.patch_pos.shape == (n, 3)
    assert state.patch_normal.shape == (n, 3)
    assert state.patch_area.shape == (n,)
    assert state.patch_part.shape == (n,)
    assert state.patch_pos.dtype == np.float32
    assert state.patch_normal.dtype == np.float32
    assert state.patch_area.dtype == np.float32
    assert state.patch_part.dtype == np.int32

    # 캡슐: (K, 7) = [p0, p1, r]. 몸통 + 머리 + 팔 2개.
    assert state.capsules.shape == (4, 7)
    assert state.capsules.dtype == np.float32
    assert (state.capsules[:, 6] > 0).all()

    # 가림 판정과 재부착 판정이 쓰는 두 인덱스 배열을 반드시 채운다.
    assert state.capsule_part is not None and state.capsule_part.shape == (4,)
    assert state.patch_capsule is not None and state.patch_capsule.shape == (n,)
    assert state.patch_capsule.dtype == np.int32
    assert state.patch_capsule.min() >= 0
    assert state.patch_capsule.max() < state.capsules.shape[0]


def test_fake_body_patch_count_in_target_range():
    """D 문서 단계 1: 총 200~300개."""
    assert 200 <= fake_body().patch_pos.shape[0] <= 300
    assert fake_body(arms_up=True).patch_pos.shape[0] == fake_body().patch_pos.shape[0]


def test_fake_body_normals_are_outward_unit_vectors():
    state = fake_body()
    assert np.allclose(np.linalg.norm(state.patch_normal, axis=1), 1.0, atol=1e-5)

    # 바깥 방향: 패치를 법선 쪽으로 밀면 소속 캡슐 축에서 멀어진다.
    caps = state.capsules[state.patch_capsule]
    p0, p1 = caps[:, 0:3], caps[:, 3:6]
    axis = p1 - p0
    length_sq = (axis * axis).sum(1)

    def dist_to_axis(points):
        t = np.where(length_sq > 0, ((points - p0) * axis).sum(1) / np.where(length_sq > 0, length_sq, 1), 0.0)
        closest = p0 + np.clip(t, 0.0, 1.0)[:, None] * axis
        return np.linalg.norm(points - closest, axis=1)

    eps = 1e-3
    assert (dist_to_axis(state.patch_pos + eps * state.patch_normal)
            > dist_to_axis(state.patch_pos)).all()


def test_fake_body_areas_positive_and_match_capsule_surface():
    state = fake_body()
    assert (state.patch_area > 0).all()
    # 몸통(캡슐 0)의 패치 면적 합 = 원통 측면적 2*pi*r*L.
    torso = state.patch_capsule == 0
    r, p0, p1 = state.capsules[0, 6], state.capsules[0, 0:3], state.capsules[0, 3:6]
    expected = 2 * np.pi * r * np.linalg.norm(p1 - p0)
    assert state.patch_area[torso].sum() == pytest.approx(expected, rel=1e-4)


def test_fake_body_parts_are_valid_indices_and_split_torso():
    state = fake_body()
    assert state.patch_part.min() >= 0
    assert state.patch_part.max() < len(PART_NAMES)

    present = {PART_NAMES[i] for i in np.unique(state.patch_part)}
    assert present == {"head", "torso_front", "torso_back", "arms"}   # 다리 없음

    # 정면 패치는 진행 방향(+x)을, 배면 패치는 반대를 본다.
    front = state.patch_part == PART_NAMES.index("torso_front")
    back = state.patch_part == PART_NAMES.index("torso_back")
    assert (state.patch_normal[front][:, 0] > 0).all()
    assert (state.patch_normal[back][:, 0] <= 0).all()


def test_fake_body_stands_at_booth_center_and_inside_booth(booth):
    state = fake_body(arms_up=True)
    pos = state.patch_pos
    assert pos[:, 0].mean() == pytest.approx(booth["length_m"] / 2.0, abs=1e-3)
    assert pos[:, 1].mean() == pytest.approx(0.0, abs=1e-3)
    # 부스 안에 들어간다. 특히 든 팔이 벽(y = ±width/2)을 뚫지 않아야 한다.
    assert np.abs(pos[:, 1]).max() < booth["width_m"] / 2.0
    assert 0.0 < pos[:, 2].min() and pos[:, 2].max() < booth["height_m"]


def test_fake_body_arms_up_moves_arms_outward_and_upward():
    down = fake_body(arms_up=False)
    up = fake_body(arms_up=True)
    arms = PART_NAMES.index("arms")
    assert np.abs(up.patch_pos[up.patch_part == arms][:, 1]).max() > \
           np.abs(down.patch_pos[down.patch_part == arms][:, 1]).max()
    assert up.patch_pos[up.patch_part == arms][:, 2].max() > \
           down.patch_pos[down.patch_part == arms][:, 2].max()


def test_fake_body_yaw_rotates_body_about_booth_center(booth):
    base = fake_body()
    turned = fake_body(yaw_deg=180.0)
    center = np.array([booth["length_m"] / 2.0, 0.0, 0.0], dtype=np.float32)

    expected = base.patch_pos - center
    expected = np.stack([-expected[:, 0], -expected[:, 1], expected[:, 2]], axis=1) + center
    assert np.allclose(turned.patch_pos, expected, atol=1e-5)
    assert np.allclose(turned.patch_normal[:, :2], -base.patch_normal[:, :2], atol=1e-5)
    # 부위 라벨은 몸에 붙어 함께 돈다.
    assert (turned.patch_part == base.patch_part).all()


def test_fake_body_is_deterministic():
    a, b = fake_body(arms_up=True, yaw_deg=30.0), fake_body(arms_up=True, yaw_deg=30.0)
    assert (a.patch_pos == b.patch_pos).all()
    assert (a.patch_normal == b.patch_normal).all()
    assert (a.patch_area == b.patch_area).all()


# --- fake_nozzles ------------------------------------------------------------

def test_fake_nozzles_fills_nozzleconfig_contract(booth):
    nz = fake_nozzles()
    layout = load_nozzle_layout()["layout"]
    assert isinstance(nz, NozzleConfig)
    m = nz.count
    assert m == 2 * len(layout["z_levels"])        # 좌우 벽 × 높이 단
    assert nz.positions.shape == (m, 3)
    assert nz.directions.shape == (m, 3)
    assert nz.strengths.shape == (m,)
    assert nz.positions.dtype == np.float32
    assert nz.directions.dtype == np.float32
    assert nz.strengths.dtype == np.float32
    assert np.allclose(np.linalg.norm(nz.directions, axis=1), 1.0, atol=1e-6)
    assert (nz.strengths == 1.0).all()

    # 좌우 벽면에 대칭, 높이는 layout의 단, x는 부스 중앙.
    wall = booth["width_m"] / 2.0
    assert sorted(np.unique(nz.positions[:, 1]).tolist()) == pytest.approx([-wall, wall])
    assert sorted(np.unique(nz.positions[:, 2]).tolist()) == pytest.approx(layout["z_levels"])
    assert np.allclose(nz.positions[:, 0], booth["length_m"] / 2.0)

    # 높이를 직접 줄 수도 있다.
    assert fake_nozzles(z_levels_m=(1.0, 1.4)).count == 4


def test_fake_nozzles_point_inward_and_downstream():
    nz = fake_nozzles()
    # 안쪽: 방향의 y 성분 부호가 위치의 y 부호와 반대.
    assert (np.sign(nz.directions[:, 1]) == -np.sign(nz.positions[:, 1])).all()
    # 진행 방향(+x)으로 기울어져 있다 (configs/nozzles.yaml layout.yaw_deg).
    assert (nz.directions[:, 0] > 0).all()


def test_fake_nozzle_axes_pass_in_front_of_the_body():
    """제트 축은 몸통 축보다 +x(정면) 쪽을 몸 가까이 지난다.

    이 비대칭이 없으면 속도장이 x에 대해 대칭이 되어 yaw 0과 180이 같은 점수를 낸다.
    """
    nz, state = fake_nozzles(), fake_body()
    torso_axis_xy = state.capsules[0, 0:2].astype(np.float64)
    r = torso_axis_xy - nz.positions[:, :2]
    d = nz.directions[:, :2] / np.linalg.norm(nz.directions[:, :2], axis=1, keepdims=True)
    s = (r * d).sum(1)
    closest = nz.positions[:, :2] + s[:, None] * d
    assert (s > 0).all()                                        # 몸이 노즐 앞쪽에 있다
    assert (closest[:, 0] > torso_axis_xy[0]).all()             # 정면(+x) 쪽을 지난다
    reach = state.capsules[0, 6] + 2 * 0.15                     # 몸통 + 제트 폭 여유
    assert (np.linalg.norm(closest - torso_axis_xy, axis=1) < reach).all()


# --- fake_velocity_field_per_nozzle ------------------------------------------

def test_velocity_field_shape_and_per_nozzle_sum(cfg):
    nz, state = fake_nozzles(), fake_body()
    u = fake_velocity_field_per_nozzle(state.patch_pos, nz, 0.0, cfg)
    assert u.shape == (nz.count, state.patch_pos.shape[0], 3)
    assert np.isfinite(u).all()


def test_velocity_is_zero_behind_the_nozzle(cfg):
    """s <= 0 이면 u = 0 (00_common.md 4.1)."""
    nz = fake_nozzles()
    behind = nz.positions - 0.5 * nz.directions      # 노즐 뒤쪽
    u = fake_velocity_field_per_nozzle(behind, nz, 0.0, cfg)
    assert (np.linalg.norm(u[np.arange(nz.count), np.arange(nz.count)], axis=-1) == 0.0).all()


def test_velocity_holds_exit_speed_in_potential_core_then_decays(cfg):
    """s <= K*D 는 U0, 그 뒤는 U0*K*D/s."""
    nz = fake_nozzles()
    jet = cfg["jet"]
    u0 = jet["exit_velocity_mps"]
    core = jet["decay_constant"] * jet["nozzle_diameter_m"]

    n0, d0 = nz.positions[0].astype(np.float64), nz.directions[0].astype(np.float64)
    for s in (0.25 * core, 0.9 * core):
        u = fake_velocity_field_per_nozzle((n0 + s * d0)[None, :], nz, 0.0, cfg)[0, 0]
        assert np.linalg.norm(u) == pytest.approx(u0, rel=1e-6)

    for s in (2.0 * core, 10.0 * core):
        u = fake_velocity_field_per_nozzle((n0 + s * d0)[None, :], nz, 0.0, cfg)[0, 0]
        assert np.linalg.norm(u) == pytest.approx(u0 * core / s, rel=1e-6)
        assert np.allclose(u / np.linalg.norm(u), d0, atol=1e-6)   # 축 방향


def test_velocity_falls_off_gaussian_away_from_axis(cfg):
    nz = fake_nozzles()
    jet = cfg["jet"]
    n0, d0 = nz.positions[0].astype(np.float64), nz.directions[0].astype(np.float64)
    s = 0.3
    perp = np.cross(d0, [0.0, 0.0, 1.0])
    perp /= np.linalg.norm(perp)
    sigma = (0.5 * jet["nozzle_diameter_m"] + jet["halfwidth_spread_rate"] * s) / 1.177

    on_axis = np.linalg.norm(
        fake_velocity_field_per_nozzle((n0 + s * d0)[None, :], nz, 0.0, cfg)[0, 0])
    off = np.linalg.norm(
        fake_velocity_field_per_nozzle((n0 + s * d0 + sigma * perp)[None, :], nz, 0.0, cfg)[0, 0])
    assert off == pytest.approx(on_axis * np.exp(-0.5), rel=1e-6)
    assert 0.0 < off < on_axis


def test_velocity_scales_with_strength(cfg):
    nz = fake_nozzles()
    half = NozzleConfig(positions=nz.positions, directions=nz.directions,
                        strengths=0.5 * nz.strengths)
    zero = NozzleConfig(positions=nz.positions, directions=nz.directions,
                        strengths=np.zeros_like(nz.strengths))
    pts = fake_body().patch_pos
    full_u = fake_velocity_field_per_nozzle(pts, nz, 0.0, cfg)
    assert np.allclose(fake_velocity_field_per_nozzle(pts, half, 0.0, cfg), 0.5 * full_u)
    assert (fake_velocity_field_per_nozzle(pts, zero, 0.0, cfg) == 0.0).all()


def test_velocity_ignores_time_when_pulse_disabled(cfg):
    nz, pts = fake_nozzles(), fake_body().patch_pos
    assert cfg["jet"]["pulse"]["enabled"] is False
    assert np.array_equal(fake_velocity_field_per_nozzle(pts, nz, 0.0, cfg),
                          fake_velocity_field_per_nozzle(pts, nz, 0.37, cfg))


def test_velocity_pulse_gate_switches_on_and_off(cfg):
    """상수를 바꿔야 하므로 dict를 복사해 덮어쓴다 (configs/는 D가 수정하지 않는다)."""
    pulsed = {**cfg, "jet": {**cfg["jet"],
                             "pulse": {"enabled": True, "period_s": 0.5, "duty": 0.5}}}
    nz, pts = fake_nozzles(), fake_body().patch_pos
    on = fake_velocity_field_per_nozzle(pts, nz, 0.1, pulsed)     # (0.1/0.5) = 0.2 < duty
    off = fake_velocity_field_per_nozzle(pts, nz, 0.4, pulsed)    # (0.4/0.5) = 0.8 >= duty
    assert np.array_equal(on, fake_velocity_field_per_nozzle(pts, nz, 0.1, cfg))
    assert (off == 0.0).all()
