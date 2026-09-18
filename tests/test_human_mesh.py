"""사람 메시 몸 테스트. 소유자: B. (`docs/tracks/B_body_jet.md` 단계 10, `docs/mesh_transition.md`)

체형은 BodyParams 기본값과 무관하게 명시한다 (기본값 교체는 통합의 types.py PR).
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest
from scipy.spatial import cKDTree

from airis.sim import human_mesh as hm
from airis.sim.body import build_body
from airis.sim.scenario import load_nozzle_layout, load_physics, load_scenarios
from airis.sim.types import PART_NAMES, BodyParams, PoseParams

SC = load_scenarios()
BOOTH = load_nozzle_layout()["booth"]
MH = BodyParams(1.70, 0.342, 0.194, 0.463, 0.883)          # MakeHuman 기본 체형 (관절 중심 측정)
BODIES = [MH, BodyParams(1.95, 0.42, 0.26, 0.57, 1.00), BodyParams(1.50, 0.30, 0.16, 0.40, 0.76),
          BodyParams(1.70, 0.38, 0.22, 0.50, 0.85)]
FRONT, BACK = PART_NAMES.index("torso_front"), PART_NAMES.index("torso_back")
MIRROR = np.array([1.0, -1.0, 1.0])


@pytest.fixture(scope="module")
def sim():
    return hm.cached_sim_mesh()


@pytest.fixture(scope="module")
def full():
    return hm.cached_makehuman()


def _area(v, f):
    return 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)


# ---------------------------------------------------------------------------
# sim 메시 자산 (data/meshes/makehuman/sim_mesh.npz, scripts/build_sim_mesh.py)
# ---------------------------------------------------------------------------
def test_sim_mesh_asset(sim, full):
    V, F = sim.vertices, sim.faces
    assert 5000 <= len(F) <= 6000
    edges = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()                                           # 닫힌 다양체
    vol = np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6
    assert vol > 0                                                       # 면이 바깥을 향한다
    np.testing.assert_allclose(sim.weights.sum(axis=1), 1.0, atol=1e-9)
    assert sim.bone_names == full.bone_names
    np.testing.assert_allclose(sim.bone_head, full.bone_head)
    # 원본 정점의 부분집합이고 가중치를 자르지 않았다
    _, _, body_idx = hm._read_obj_body(hm.MAKEHUMAN_DIR / "base.obj")
    inv = {o: i for i, o in enumerate(body_idx)}
    idx = np.array([inv[o] for o in sim.obj_vertex])
    np.testing.assert_allclose(V, full.vertices[idx], atol=1e-12)
    np.testing.assert_allclose(sim.weights, full.weights[idx], atol=1e-12)


def test_sim_mesh_shape_close_to_original(sim, full):
    """면적 1 % 안, 부피 2 % 안, 부위별 면적 비율 0.01 안."""
    a_sim, a_full = _area(sim.vertices, sim.faces), _area(full.vertices, full.faces)
    assert a_sim.sum() == pytest.approx(a_full.sum(), rel=0.01)
    vol = lambda v, f: np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6  # noqa: E731
    assert vol(sim.vertices, sim.faces) == pytest.approx(vol(full.vertices, full.faces), rel=0.02)
    bp = np.array([hm.bone_part(n) if full.weights[:, i].any() else -1 for i, n in enumerate(full.bone_names)])
    part_full = bp[np.argmax(full.weights[full.faces].sum(axis=1), axis=1)]
    for k in set(part_full.tolist()):
        assert a_sim[sim.face_part == k].sum() / a_sim.sum() == pytest.approx(
            a_full[part_full == k].sum() / a_full.sum(), abs=0.01)


def test_sim_mesh_mirror_maps(sim):
    mv, mf, perm = sim.mirror_vertex, sim.mirror_face, sim.mirror_face_perm
    assert (mv[mv] == np.arange(len(mv))).all() and (mf[mf] == np.arange(len(mf))).all()
    assert (mf != np.arange(len(mf))).all()                              # 자기 자신이 거울인 면 없음
    np.testing.assert_allclose(sim.vertices[mv], sim.vertices * MIRROR, atol=1e-12)
    rows = np.arange(len(mf))[:, None]
    assert (sim.faces[mf] == mv[sim.faces[rows, perm]]).all()
    assert (sim.face_part[mf] == sim.face_part).all()
    mb = hm.mirror_bone_index(sim.bone_names)
    np.testing.assert_allclose(sim.weights[mv][:, mb], sim.weights, atol=1e-12)


def test_every_weighted_bone_has_a_part(full):
    for i, name in enumerate(full.bone_names):
        if full.weights[:, i].any():
            assert 0 <= hm.bone_part(name) < len(PART_NAMES)
    assert hm.bone_part("upperarm01.L") == PART_NAMES.index("arms")
    assert hm.bone_part("spine03") == FRONT
    assert hm.bone_part("head") == PART_NAMES.index("head")
    assert hm.bone_part("toe1-1.R") == PART_NAMES.index("legs")


# ---------------------------------------------------------------------------
# 체형 (BodyParams 5개, 관절 중심 정의)
# ---------------------------------------------------------------------------
def test_default_makehuman_measurements(full):
    """MakeHuman 기본 메시를 1.70 m 로 축척한 측정값 = 확정 기본값 (1.70 / 0.342 / 0.194 / 0.463 / 0.883)."""
    assert hm.MESH_DEFAULT_BODY == MH
    m = hm.measure_body(full)
    s = 1.70 / m["height_m"]
    for key, want in dataclasses.asdict(MH).items():
        assert m[key] * s == pytest.approx(want, abs=1e-3), key


@pytest.mark.parametrize("body", BODIES, ids=lambda b: f"h{b.height_m}")
def test_shape_matches_body_params_within_1cm(full, body):
    """뼈별 축척 후 5개 측정값이 목표 ±1 cm (실측 0.4 mm 이하)."""
    shaped = hm.shape_mesh(full, body)
    m = hm.measure_body(shaped)
    s = body.height_m / m["height_m"]
    for key, want in dataclasses.asdict(body).items():
        assert m[key] * s == pytest.approx(want, abs=0.01), key
        assert m[key] * s == pytest.approx(want, abs=1e-3), key


@pytest.mark.parametrize("body", BODIES[:2], ids=lambda b: f"h{b.height_m}")
def test_sim_body_is_decimated_shaped_original(sim, full, body):
    """sim 몸 = 같은 체형·자세로 맞춘 원본 몸의 부분 정점 (발바닥 정점 차이로 1 mm 이내)."""
    pose = PoseParams(torso_yaw=30.0, shoulder_abduction=60.0, elbow_flexion=30.0)
    vs = hm.posed_in_booth(sim, body, pose, SC["default"], BOOTH)
    vf = hm.posed_in_booth(full, body, pose, SC["default"], BOOTH)
    _, _, body_idx = hm._read_obj_body(hm.MAKEHUMAN_DIR / "base.obj")
    inv = {o: i for i, o in enumerate(body_idx)}
    idx = np.array([inv[o] for o in sim.obj_vertex])
    assert np.abs(vs - vf[idx]).max() < 1e-3


def test_shape_cache_returns_same_object(full):
    assert hm.shape_mesh(full, MH) is hm.shape_mesh(full, MH)


# ---------------------------------------------------------------------------
# build_body 메시 계약
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["default", "pregnant", "wheelchair"])
def test_mesh_body_contract(scenario):
    st = build_body(MH, PoseParams(), SC[scenario], 2000.0, model="mesh")
    n = len(st.patch_pos)
    assert st.patch_pos.shape == (n, 3) and st.patch_pos.dtype == np.float32
    assert st.patch_normal.shape == (n, 3) and st.patch_area.shape == (n,) and st.patch_part.shape == (n,)
    np.testing.assert_allclose(np.linalg.norm(st.patch_normal, axis=1), 1.0, atol=1e-5)
    assert (st.patch_area > 0).all()
    assert st.mesh_vertices.dtype == np.float32 and st.mesh_faces.dtype == np.int32
    assert st.mesh_face_part.shape == (len(st.mesh_faces),)
    assert st.patch_face.shape == (n,) and st.patch_face.max() < len(st.mesh_faces)
    # 패치는 자기 면 위에 있고 법선은 그 면의 바깥 법선
    tri = st.mesh_vertices[st.mesh_faces[st.patch_face]].astype(np.float64)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    np.testing.assert_allclose(st.patch_normal, fn, atol=1e-4)            # float32 정점으로 재계산한 작은 면
    assert np.abs(((st.patch_pos - tri[:, 0]) * fn).sum(axis=1)).max() < 1e-5
    assert (st.patch_part == st.mesh_face_part[st.patch_face]).all()
    # 부위 다섯 개가 모두 있고 몸통이 앞뒤로 나뉜다
    assert set(np.unique(st.patch_part)) == set(range(len(PART_NAMES)))
    # 패치 면적 합 ≈ 메시 표면적 (확률 반올림 ±5 %)
    total = _area(st.mesh_vertices.astype(np.float64), st.mesh_faces).sum()
    assert st.patch_area.sum() == pytest.approx(total, rel=0.05)
    # 캡슐 규약: 몸통 캡슐 = torso_front, 휠체어 프레임 = −1, 머리·팔·다리는 부위 번호, back 없음
    assert st.capsules.shape[1] == 7 and len(st.capsule_part) == len(st.capsules)
    assert BACK not in st.capsule_part
    assert (st.capsule_part == -1).sum() == (4 if scenario == "wheelchair" else 0)
    assert 15 <= (st.capsule_part >= 0).sum() <= 32
    assert st.patch_capsule.max() < len(st.capsules) and (st.capsule_part[st.patch_capsule] >= 0).all()


def test_torso_front_back_by_normal():
    for yaw in (0.0, 90.0, -135.0):
        st = build_body(MH, PoseParams(torso_yaw=yaw), SC["default"], 2000.0, model="mesh")
        fwd = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), 0.0])
        d = st.patch_normal @ fwd
        assert (d[st.patch_part == FRONT] > 0).all() and (d[st.patch_part == BACK] <= 1e-6).all()


def test_standing_mesh_body_placement():
    st = build_body(MH, PoseParams(), SC["default"], 400.0, model="mesh")
    v = st.mesh_vertices
    assert v[:, 2].min() == pytest.approx(0.0, abs=1e-6)
    assert v[:, 2].max() == pytest.approx(1.70, abs=0.01)
    assert v[:, 0].mean() == pytest.approx(BOOTH["length_m"] / 2.0, abs=0.05)
    assert abs(v[:, 1].mean()) < 1e-6


def test_wheelchair_hip_at_seat_height(sim):
    st = build_body(MH, PoseParams(), SC["wheelchair"], 400.0, model="mesh")
    assert st.mesh_vertices[:, 2].min() > 0.0
    assert st.mesh_vertices[:, 2].max() < 1.40                         # 앉은 키


@pytest.mark.parametrize("scenario", ["default", "wheelchair"])
@pytest.mark.parametrize("ppm", [400.0, 2000.0])
@pytest.mark.parametrize("pose", [PoseParams(torso_yaw=45.0),
                                  PoseParams(torso_yaw=-30.0, shoulder_abduction=160.0, elbow_flexion=40.0,
                                             torso_pitch=10.0, hip_flexion=15.0, knee_flexion=20.0)],
                         ids=["yaw45", "mixed"])
def test_mesh_patches_mirror_symmetric_under_yaw_flip(scenario, ppm, pose):
    """yaw ±θ 패치가 y 반사 후 위치·법선·면적·부위 일대일 (#42 규칙)."""
    a = build_body(MH, pose, SC[scenario], ppm, model="mesh")
    b = build_body(MH, dataclasses.replace(pose, torso_yaw=-pose.torso_yaw), SC[scenario], ppm, model="mesh")
    assert a.patch_pos.shape == b.patch_pos.shape
    d, i = cKDTree(b.patch_pos.astype(np.float64) * MIRROR).query(a.patch_pos.astype(np.float64))
    assert d.max() < 1e-5 and len(np.unique(i)) == len(i)
    np.testing.assert_allclose(a.patch_normal, b.patch_normal[i] * MIRROR, atol=1e-5)
    np.testing.assert_allclose(a.patch_area, b.patch_area[i], rtol=1e-4)
    assert (a.patch_part == b.patch_part[i]).all()


def test_patch_count_fits_particle_table():
    """A 가림 표 max_patches 8192 안 (최대 체형 1.95 m, 400·2000/m²). 실측은 PR 본문."""
    big = BODIES[1]
    for ppm in (400.0, 2000.0):
        for sc in ("default", "pregnant", "wheelchair"):
            n = len(build_body(big, PoseParams(), SC[sc], ppm, model="mesh").patch_pos)
            assert n < 8192
            assert n == pytest.approx(ppm * 2.2, rel=0.2)


def test_booth_feasibility_matches_pose():
    """부스(폭 1.46, 높이 2.15) 안팎: 기본·만세는 안, 어깨 90° 벌림은 벽 밖."""
    half, top = BOOTH["width_m"] / 2.0, BOOTH["height_m"]
    for pose, inside in ((PoseParams(), True), (PoseParams(shoulder_abduction=180.0, elbow_flexion=0.0), True),
                         (PoseParams(shoulder_abduction=90.0, elbow_flexion=0.0), False)):
        p = build_body(MH, pose, SC["default"], 400.0, model="mesh").patch_pos
        assert (np.abs(p[:, 1]).max() <= half and p[:, 2].max() <= top) == inside, pose


def test_mesh_body_none_uses_mesh_default_body():
    a = build_body(None, PoseParams(), SC["default"], 400.0, model="mesh")
    b = build_body(hm.MESH_DEFAULT_BODY, PoseParams(), SC["default"], 400.0, model="mesh")
    np.testing.assert_array_equal(a.mesh_vertices, b.mesh_vertices)
    v = hm.pose_mesh(None, None, PoseParams(), SC["default"])
    np.testing.assert_allclose(v, hm.posed_in_booth(hm.cached_makehuman(), hm.MESH_DEFAULT_BODY, PoseParams(),
                                                    SC["default"]))


def test_body_model_switch_defaults_to_config():
    assert load_physics()["body"]["model"] in ("capsule", "mesh")
    st = build_body(MH, PoseParams(), SC["default"], 400.0)
    assert (st.mesh_vertices is None) == (load_physics()["body"]["model"] == "capsule")
    with pytest.raises(ValueError):
        build_body(MH, PoseParams(), SC["default"], 400.0, model="sphere")


# ---------------------------------------------------------------------------
# 임산부 배 타깃 (사용자 다운로드 승인 후, 파일이 없으면 건너뛴다)
# ---------------------------------------------------------------------------
def test_target_file_maps_to_mesh_vertices(tmp_path, sim, full):
    """MakeHuman .target(base.obj 정점 번호, MakeHuman 축 dm) → 우리 축 m 변위가 원본·sim 정점에 맞게 붙는다."""
    _, _, body_idx = hm._read_obj_body(hm.MAKEHUMAN_DIR / "base.obj")
    obj = int(sim.obj_vertex[10])
    lines = ["# test", f"{obj} 0.0 0.0 0.5", "999999 1 1 1"]
    (tmp_path / "fake.target").write_text("\n".join(lines) + "\n", encoding="utf-8")
    d_sim = hm.target_displacement(sim, "fake", tmp_path)
    np.testing.assert_allclose(d_sim[10], [0.05, 0.0, 0.0])            # MakeHuman +z(앞) 0.5 dm → 우리 +x 5 cm
    assert np.count_nonzero(d_sim.any(axis=1)) == 1
    d_full = hm.target_displacement(full, "fake", tmp_path)
    np.testing.assert_allclose(d_full[list(body_idx).index(obj)], [0.05, 0.0, 0.0])
    assert hm.scenario_targets_names("pregnant") == ["stomach-pregnant-incr"]


@pytest.mark.skipif(not (hm.MAKEHUMAN_DIR / "stomach-pregnant-incr.target").exists(),
                    reason="stomach-pregnant-incr.target 미설치 (사용자 승인 대기)")
def test_pregnant_target_pushes_belly_forward():
    a = build_body(MH, PoseParams(), SC["default"], 400.0, model="mesh")
    b = build_body(MH, PoseParams(), SC["pregnant"], 400.0, model="mesh")
    assert b.mesh_vertices[:, 0].max() > a.mesh_vertices[:, 0].max() + 0.03


# ---------------------------------------------------------------------------
# 성능: 자세 + 패치 20 ms 이하 (400/m²). 부하가 있으면 기준 연산 대비 비율 (tests/test_body.py 방식)
# ---------------------------------------------------------------------------
_REF = np.random.default_rng(12345)
_REF_A, _REF_D, _REF_X = _REF.random((16, 3600)), _REF.random((16, 3)), _REF.random((3600, 3))


def _reference_workload() -> float:
    s = _REF_D @ _REF_X.T
    return float(np.exp(-(_REF_A * _REF_A) / (1.0 + s * s)).sum())


def test_mesh_build_body_under_20ms():
    fn = lambda: build_body(MH, PoseParams(torso_yaw=45.0, shoulder_abduction=120.0), SC["default"],  # noqa: E731
                            400.0, model="mesh")
    fn()
    _reference_workload()
    best = best_ref = float("inf")
    for _ in range(15):
        t0 = time.perf_counter()
        _reference_workload()
        best_ref = min(best_ref, time.perf_counter() - t0)
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    if best >= 0.020:
        assert best / best_ref < 14.0, f"메시 build_body {best * 1e3:.1f} ms, 기준 대비 {best / best_ref:.1f}배"
