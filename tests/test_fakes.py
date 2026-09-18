"""가짜 입력이 BodyState / NozzleConfig 계약을 지키는지 확인한다. 소유자: D.

`docs/interfaces.md`의 배열 규약(float32 (N,3), int32 (N,), 바깥 단위 법선)을 검사한다.
"""
import numpy as np
import pytest

from airis.sim.scenario import load_nozzle_layout
from airis.sim.types import PART_NAMES, BodyState, NozzleConfig
from tests.fakes import fake_body, fake_mesh_body, fake_nozzles


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

# --- fake_mesh_body ----------------------------------------------------------

@pytest.mark.parametrize("arms_up", [False, True])
@pytest.mark.parametrize("yaw_deg", [0.0, 30.0, 180.0])
def test_fake_mesh_body_fills_mesh_contract(arms_up, yaw_deg):
    """B가 정한 메시 필드 형식: 정점 f32 (V,3), 면 i32 (F,3), 면 부위 i32 (F,), patch_face i32 (N,)."""
    st = fake_mesh_body(arms_up=arms_up, yaw_deg=yaw_deg)
    n, f = st.patch_pos.shape[0], st.mesh_faces.shape[0]
    assert st.mesh_vertices.dtype == np.float32 and st.mesh_vertices.shape[1] == 3
    assert st.mesh_faces.dtype == np.int32 and st.mesh_faces.shape == (f, 3)
    assert st.mesh_faces.min() >= 0 and st.mesh_faces.max() < st.mesh_vertices.shape[0]
    assert st.mesh_face_part.dtype == np.int32 and st.mesh_face_part.shape == (f,)
    assert st.patch_face.dtype == np.int32 and st.patch_face.shape == (n,)
    assert st.patch_capsule is None
    assert st.capsules.shape[1] == 7 and st.capsule_part.shape == (st.capsules.shape[0],)
    assert np.array_equal(st.patch_part, st.mesh_face_part[st.patch_face])
    assert set(np.unique(st.patch_part)) == set(range(len(PART_NAMES)))


def test_fake_mesh_body_patches_lie_on_their_faces_with_outward_normals():
    st = fake_mesh_body(arms_up=True, yaw_deg=30.0)
    tri = st.mesh_vertices[st.mesh_faces[st.patch_face]].astype(np.float64)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert np.allclose(st.patch_pos, tri.mean(axis=1), atol=1e-6)
    # 정점이 float32라 면 법선을 다시 구하면 1e-5 수준으로 어긋난다.
    assert np.allclose(st.patch_normal, cross / np.linalg.norm(cross, axis=1, keepdims=True), atol=1e-4)
    assert np.allclose(np.linalg.norm(st.patch_normal, axis=1), 1.0, atol=1e-6)
    assert np.allclose(st.patch_area, 0.5 * np.linalg.norm(cross, axis=1), rtol=1e-5)
    # 닫힌 상자들의 부호 부피(발산 정리)가 양수면 면이 바깥을 향한다.
    signed_volume = float((tri.mean(axis=1) * cross).sum() / 6.0)
    assert signed_volume > 0.0


def test_fake_mesh_body_matches_fake_body_layout(booth):
    """부스 중앙에 서고, 부스 안이며, 팔을 들면 팔 패치가 위로 간다."""
    down, up = fake_mesh_body(arms_up=False), fake_mesh_body(arms_up=True)
    assert abs(float(down.patch_pos[:, 0].mean()) - booth["length_m"] / 2.0) < 1e-3
    for st in (down, up):
        assert np.abs(st.patch_pos[:, 1]).max() < booth["width_m"] / 2.0
        assert st.patch_pos[:, 2].max() < booth["height_m"]
    arms = PART_NAMES.index("arms")
    assert up.patch_pos[up.patch_part == arms, 2].mean() > down.patch_pos[down.patch_part == arms, 2].mean()


def test_fake_mesh_body_yaw_rotates_about_booth_center(booth):
    a, b = fake_mesh_body(yaw_deg=0.0), fake_mesh_body(yaw_deg=90.0)
    c = np.array([booth["length_m"] / 2.0, 0.0])
    ra = a.mesh_vertices[:, :2] - c
    rb = b.mesh_vertices[:, :2] - c
    assert np.allclose(np.stack([-ra[:, 1], ra[:, 0]], axis=1), rb, atol=1e-5)
    assert np.array_equal(a.patch_part, b.patch_part)


def test_fake_mesh_body_is_deterministic():
    a, b = fake_mesh_body(True, 45.0), fake_mesh_body(True, 45.0)
    assert np.array_equal(a.mesh_vertices, b.mesh_vertices)
    assert np.array_equal(a.patch_pos, b.patch_pos)


def test_fake_nozzles_fills_nozzleconfig_contract(booth):
    nz = fake_nozzles()
    assert isinstance(nz, NozzleConfig)
    m = nz.count
    assert m == 4                                  # 좌우 벽 각 2개
    assert nz.positions.shape == (m, 3)
    assert nz.directions.shape == (m, 3)
    assert nz.strengths.shape == (m,)
    assert nz.positions.dtype == np.float32
    assert nz.directions.dtype == np.float32
    assert nz.strengths.dtype == np.float32
    assert np.allclose(np.linalg.norm(nz.directions, axis=1), 1.0, atol=1e-6)
    assert (nz.strengths == 1.0).all()

    # 좌우 벽면에 대칭으로 2개씩, 높이 두 단 (D 문서 단계 1).
    wall = booth["width_m"] / 2.0
    assert sorted(np.unique(nz.positions[:, 1]).tolist()) == pytest.approx([-wall, wall])
    assert sorted(np.unique(nz.positions[:, 2]).tolist()) == pytest.approx([1.0, 1.4])
    assert fake_nozzles(z_levels_m=(0.5, 0.9, 1.3, 1.7)).count == 8


def test_fake_nozzles_point_inward_and_downstream():
    nz = fake_nozzles()
    # 안쪽: 방향의 y 성분 부호가 위치의 y 부호와 반대.
    assert (np.sign(nz.directions[:, 1]) == -np.sign(nz.positions[:, 1])).all()
    # 진행 방향(+x)으로 기울어져 있다 (configs/nozzles.yaml layout.yaw_deg).
    assert (nz.directions[:, 0] > 0).all()


def test_fake_nozzle_axes_pass_through_the_body():
    """노즐 축이 몸을 비껴가면 제트가 몸에 닿지 않는다."""
    nz, state = fake_nozzles(), fake_body()
    torso_axis_xy = state.capsules[0, 0:2].astype(np.float64)
    r = torso_axis_xy - nz.positions[:, :2]
    d = nz.directions[:, :2] / np.linalg.norm(nz.directions[:, :2], axis=1, keepdims=True)
    s = (r * d).sum(1)
    miss = np.linalg.norm(r - s[:, None] * d, axis=1)
    assert (s > 0).all()                        # 몸이 노즐 앞쪽에 있다
    assert (miss < 1e-3).all()                  # 축이 몸통 중심선을 지난다

