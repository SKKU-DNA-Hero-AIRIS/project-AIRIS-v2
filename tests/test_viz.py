"""안내 뷰·입자 애니메이션 테스트. 소유자: E. (`docs/tracks/E_realtime.md` 단계 1~2)"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("plotly")

from airis.sim.body import build_body
from airis.sim.scenario import load_nozzle_layout, load_nozzles, load_scenarios
from airis.sim.types import BodyParams, EvalResult, PoseParams
from airis.viz import anim
from airis.viz.pose_view import (capsule_mesh, figure_compare, figure_from_pose,
                                 figure_from_state)

SCENARIOS = load_scenarios()
BOOTH = load_nozzle_layout()["booth"]

#: 대표 자세 5개 (완료 기준: 세 시나리오 × 대표 자세 5개)
REPRESENTATIVE_POSES = {
    "B0 기본": PoseParams(),
    "B2 만세": PoseParams(shoulder_abduction=180.0, elbow_flexion=0.0),
    "yaw 90": PoseParams(torso_yaw=90.0),
    "만세 + yaw 90": PoseParams(shoulder_abduction=180.0, shoulder_flexion=3.0,
                               elbow_flexion=4.6, torso_yaw=90.0),
    "팔 앞으로 + 숙임": PoseParams(shoulder_flexion=90.0, elbow_flexion=30.0, torso_pitch=15.0),
}


# ---------------------------------------------------------------------------
# 캡슐 메시
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_theta,n_cap", [(24, 6), (8, 2), (3, 1)])
def test_capsule_mesh_counts(n_theta, n_cap):
    v, f = capsule_mesh([0, 0, 0], [0, 0, 1], 0.1, n_theta=n_theta, n_cap=n_cap)
    assert v.shape == (2 + 2 * n_cap * n_theta, 3)
    assert f.shape == (4 * n_cap * n_theta, 3)
    assert f.min() == 0 and f.max() == v.shape[0] - 1
    # 모든 정점이 면에 쓰인다
    assert np.unique(f).size == v.shape[0]


def test_capsule_mesh_geometry():
    p0, p1, r = np.array([0.2, -0.1, 0.5]), np.array([0.5, 0.3, 0.9]), 0.07
    v, _ = capsule_mesh(p0, p1, r)
    # 모든 정점이 축 선분에서 정확히 r 떨어져 있다
    d = p1 - p0
    t = np.clip((v - p0) @ d / (d @ d), 0.0, 1.0)
    dist = np.linalg.norm(v - (p0 + t[:, None] * d), axis=1)
    np.testing.assert_allclose(dist, r, atol=1e-9)
    # 양 극점이 축 방향으로 r 만큼 바깥
    u = d / np.linalg.norm(d)
    np.testing.assert_allclose(v[0], p0 - r * u, atol=1e-12)
    np.testing.assert_allclose(v[-1], p1 + r * u, atol=1e-12)


def test_capsule_mesh_sphere_for_zero_length():
    v, _ = capsule_mesh([1, 1, 1], [1, 1, 1], 0.1)
    np.testing.assert_allclose(np.linalg.norm(v - 1.0, axis=1), 0.1, atol=1e-9)


def test_capsule_mesh_outward_winding():
    """면 법선(오른손 규칙)이 바깥을 향한다 (조명이 뒤집히지 않게)."""
    p0, p1 = np.zeros(3), np.array([0.0, 0.0, 1.0])
    v, f = capsule_mesh(p0, p1, 0.2, n_theta=12, n_cap=3)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    centroid = (a + b + c) / 3.0
    t = np.clip(centroid[:, 2], 0.0, 1.0)
    outward = centroid - np.stack([np.zeros_like(t), np.zeros_like(t), t], axis=1)
    area = np.linalg.norm(n, axis=1)
    ok = area > 1e-12
    assert np.all((n[ok] * outward[ok]).sum(axis=1) > 0.0)


# ---------------------------------------------------------------------------
# 안내 뷰
# ---------------------------------------------------------------------------
def _names(fig) -> set[str]:
    return {tr.name for tr in fig.data if tr.name}


@pytest.mark.parametrize("scenario", ["default", "pregnant", "wheelchair"])
@pytest.mark.parametrize("pose_name", list(REPRESENTATIVE_POSES))
def test_figure_from_pose_all_scenarios(scenario, pose_name):
    """완료 기준 1: 세 시나리오 × 대표 자세 5개가 예외 없이, 슬롯 바·부스와 함께 그려진다."""
    fig = figure_from_pose(BodyParams(), REPRESENTATIVE_POSES[pose_name], SCENARIOS[scenario],
                           title=f"{scenario} · {pose_name}")
    names = _names(fig)
    assert "부스" in names
    assert "슬롯 바 12개" in names
    assert {"head", "arms", "legs"} <= names
    if scenario == "wheelchair":
        assert "휠체어 프레임" in names
    d = fig.to_dict()                          # 직렬화 가능
    json.dumps(d, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    # 카메라 프리셋 버튼 (정면·측면·위)
    labels = {b["label"] for b in d["layout"]["updatemenus"][0]["buttons"]}
    assert {"정면", "측면", "위"} <= labels
    assert d["layout"]["scene"]["aspectmode"] == "data"


def test_slot_bars_drawn_at_nozzle_positions():
    nz = load_nozzles()
    fig = figure_from_pose(BodyParams(), PoseParams(), SCENARIOS["default"])
    bar = next(tr for tr in fig.data if tr.name == "슬롯 바 12개")
    pts = np.array([[x, y, z] for x, y, z in zip(bar.x, bar.y, bar.z) if x is not None], float)
    assert pts.shape == (24, 3)                # 바 12개 × 양 끝
    mids = (pts[0::2] + pts[1::2]) / 2.0
    np.testing.assert_allclose(mids, nz.positions, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(pts[1::2] - pts[0::2], axis=1), nz.slot_length, atol=1e-6)


def test_patch_values_color_mesh():
    """values 를 주면 부위 색 대신 intensity 메시 하나로 그리고, 길이가 다르면 ValueError."""
    sc = SCENARIOS["default"]
    state = build_body(BodyParams(), PoseParams(), sc, patches_per_m2=400)
    values = state.patch_pos[:, 2] / state.patch_pos[:, 2].max()
    fig = figure_from_state(state, values, load_nozzles())
    meshes = [tr for tr in fig.data if tr.type == "mesh3d" and tr.intensity is not None]
    assert len(meshes) == 1
    inten = np.asarray(meshes[0].intensity)
    assert inten.min() >= values.min() - 1e-9 and inten.max() <= values.max() + 1e-9
    with pytest.raises(ValueError):
        figure_from_state(state, values[:-1])


def test_torso_mesh_flattened_to_patch_ellipse():
    """몸통 메시는 패치 타원 단면(앞뒤 반축 = torso_depth/2)까지 눌린다."""
    body = BodyParams()
    state = build_body(body, PoseParams(), SCENARIOS["default"], patches_per_m2=400)
    fig = figure_from_state(state)
    torso = next(tr for tr in fig.data if tr.name == "torso_front")
    x = np.asarray(torso.x) - BOOTH["length_m"] / 2.0
    assert np.abs(x).max() == pytest.approx(body.torso_depth_m / 2.0, rel=0.05)


def test_figure_compare_titles_show_relative_improvement():
    sc = SCENARIOS["default"]
    body = BodyParams()
    poses = {"B0": PoseParams(), "추천": PoseParams(torso_yaw=90.0)}
    n = {k: build_body(body, p, sc, patches_per_m2=400).patch_pos.shape[0] for k, p in poses.items()}
    results = {
        "B0": EvalResult(0.2, np.zeros(5), 0.05, 0.0, {"removal": np.full(n["B0"], 0.1)}),
        "추천": EvalResult(0.5, np.zeros(5), 0.1, 0.0, {"removal": np.full(n["추천"], 0.3)}),
    }
    fig = figure_compare(body, poses, sc, results=results)
    titles = [a.text for a in fig.layout.annotations]
    assert any("+150%" in t for t in titles)
    assert fig.layout.scene2 is not None
    fig.to_dict()


def test_figure_compare_marks_infeasible():
    sc = SCENARIOS["default"]
    poses = {"B0": PoseParams(), "팔 수평": PoseParams(shoulder_abduction=90.0)}
    results = {"팔 수평": EvalResult(-1.5, np.zeros(5), 0.0, 0.0, {"infeasible": True})}
    fig = figure_compare(BodyParams(), poses, sc, results=results)
    assert any("불가" in a.text for a in fig.layout.annotations)


# ---------------------------------------------------------------------------
# 입자 애니메이션
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def state_default():
    return build_body(BodyParams(), PoseParams(), SCENARIOS["default"], patches_per_m2=400)


def test_synthetic_frames_roundtrip_and_animation(state_default, tmp_path):
    """완료 기준 2 (합성): 합성 프레임 → npz → 로더 → 애니메이션."""
    frames = anim.synthetic_frames(state_default, BOOTH, n_particles=500, n_frames=12,
                                   n_candidates=2)
    anim.write_frames(frames, tmp_path / "frames")
    loaded = anim.load_frames(tmp_path)        # outputs/<exp_id> 를 넘겨도 frames 를 찾는다
    assert [s for s, _ in loaded] == [s for s, _ in frames]
    assert anim.candidates_in(loaded) == [0, 1]
    for key in anim.REQUIRED_KEYS:
        np.testing.assert_array_equal(loaded[-1][1][key], frames[-1][1][key])

    fig = anim.animation(loaded, state_default, BOOTH, candidate=1, max_points=200,
                         nozzle=load_nozzles())
    assert len(fig.frames) == 12
    assert len(fig.layout.sliders[0].steps) == 12
    # 프레임마다 입자 trace 3개(부착·부유·제거)만 갱신하고, 서브샘플 상한을 지킨다
    for fr in fig.frames:
        assert len(fr.data) == 3
        assert sum(len(tr.x) for tr in fr.data) <= 200
    fig.to_dict()


def test_particle_status_inferred_without_state_key(state_default):
    """A의 `state` 키가 없어도 부착·부유·제거를 나누고, 있으면 그대로 쓴다."""
    fr = anim.synthetic_frames(state_default, BOOTH, n_particles=800, n_frames=31)
    fr_st = anim.synthetic_frames(state_default, BOOTH, n_particles=800, n_frames=31,
                                  include_state=True)
    first = anim.particle_status(fr[0][1], BOOTH)
    last = anim.particle_status(fr[-1][1], BOOTH)
    assert np.all(first == anim.ATTACHED)
    assert (last == anim.REMOVED).sum() > 0
    assert (last == anim.ATTACHED).sum() > 0
    np.testing.assert_array_equal(last, anim.particle_status(fr_st[-1][1], BOOTH))
    # 제거된 입자는 그 뒤로 위치가 변하지 않는다 ("마지막 위치")
    gone = anim.particle_status(fr[-2][1], BOOTH) == anim.REMOVED
    assert gone.any()
    np.testing.assert_array_equal(fr[-1][1]["pos"][gone], fr[-2][1]["pos"][gone])


def test_status_counts_monotone_removal(state_default):
    frames = anim.synthetic_frames(state_default, BOOTH, n_particles=600, n_frames=20)
    rows = anim.status_counts(frames, BOOTH)
    removed = [r["removed"] for r in rows]
    assert removed[0] == 0.0 and removed[-1] > 0.0
    assert all(b >= a for a, b in zip(removed, removed[1:]))
    for r in rows:
        assert r["attached"] + r["airborne"] + r["removed"] == pytest.approx(1.0)


def test_load_frames_reads_particle_evaluator_format(state_default, tmp_path):
    """A의 `_dump_frame` 형식(계약 키 + state i8, part_init, 0-d step·t_s)을 그대로 읽는다.

    ParticleEvaluator 는 Taichi 를 초기화하므로 E 테스트에서는 부르지 않고 같은 키로 흉내 낸다.
    실제 덤프로 도는지는 `scripts/render_frames.py --simulate` 로 확인했다 (PR 본문).
    """
    n = 40
    rng = np.random.default_rng(1)
    for step, state_code in ((0, 0), (25, 1), (50, 2)):
        np.savez(tmp_path / f"{step:05d}.npz",
                 pos=(rng.random((2 * n, 3)) * [0.8, 0.4, 1.8]).astype(np.float32),
                 attached=np.full(2 * n, state_code == 0),
                 part=np.zeros(2 * n, np.int32),
                 candidate=np.repeat(np.array([0, 3], np.int32), n),
                 state=np.full(2 * n, state_code, np.int8),
                 part_init=np.zeros(2 * n, np.int32),
                 step=np.int32(step), t_s=np.float32(step * 0.002),
                 particles_per_candidate=np.int32(n))
    frames = anim.load_frames(tmp_path)
    assert anim.candidates_in(frames) == [0, 3]      # 부스 밖 후보를 건너뛴 번호 그대로
    rows = anim.status_counts(frames, BOOTH, candidate=3)
    assert [r["removed"] for r in rows] == [0.0, 0.0, 1.0]
    assert rows[1]["airborne"] == 1.0 and rows[2]["t_s"] == pytest.approx(0.1)
    fig = anim.animation(frames, state_default, BOOTH, candidate=3)
    assert fig.layout.sliders[0].steps[-1].label == "0.10 s"


def test_load_frames_rejects_missing_contract_keys(tmp_path):
    np.savez(tmp_path / "00000.npz", pos=np.zeros((3, 3), np.float32),
             attached=np.ones(3, bool), part=np.zeros(3, np.int32))
    with pytest.raises(ValueError, match="candidate"):
        anim.load_frames(tmp_path)
    with pytest.raises(FileNotFoundError):
        anim.load_frames(tmp_path / "없음")


def test_animation_unknown_candidate(state_default):
    frames = anim.synthetic_frames(state_default, BOOTH, n_particles=50, n_frames=3)
    with pytest.raises(ValueError, match="후보"):
        anim.animation(frames, state_default, BOOTH, candidate=5)


def test_render_frames_script_synthetic(tmp_path):
    """scripts/render_frames.py --synthetic 이 HTML 을 만든다 (A 덤프 없이)."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "render_frames.py"
    spec = importlib.util.spec_from_file_location("render_frames", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rc = mod.main(["--exp", "t", "--outputs", str(tmp_path), "--synthetic",
                   "--particles", "300", "--pose", "180,3,5,0,90,1,0"])
    assert rc == 0
    assert (tmp_path / "t" / "anim.html").stat().st_size > 10_000
    meta = json.loads((tmp_path / "t" / "render_meta.json").read_text(encoding="utf-8"))
    assert meta["poses"][0]["torso_yaw"] == 90.0


# ---------------------------------------------------------------------------
# 사람 메시 (MakeHuman + 선형 블렌드 스키닝)
# ---------------------------------------------------------------------------
from airis.viz import human_mesh as hm  # noqa: E402

MESH_POSES = [
    PoseParams(),
    PoseParams(shoulder_abduction=180.0, elbow_flexion=0.0),
    PoseParams(torso_yaw=90.0),
    PoseParams(40.0, 30.0, 60.0, 15.0, -45.0, 20.0, 25.0),
    PoseParams(shoulder_abduction=10.0, shoulder_flexion=120.0, elbow_flexion=90.0, torso_pitch=-10.0),
]
_LEG_REST = 0.45 * np.array([0.0, 0.0889, -0.996])       # 합성 뼈대의 다리 구간 (약간 벌림)


def _stick_rig() -> hm.HumanMesh:
    """자산 없이 자세 매핑만 보는 작은 뼈대. 정점 = 관절 끝점, 각 정점은 한 뼈에 가중치 1.

    기본 자세는 MakeHuman 처럼 A자다 (팔 40° 벌림, 팔꿈치 앞으로 굽힘, 다리 약간 벌림).
    정점 순서는 쪽마다 [팔꿈치, 손목, 무릎, 발목, 어깨]."""
    names = ["root", "spine05", "upperarm01.L", "lowerarm01.L", "wrist.L",
             "upperarm01.R", "lowerarm01.R", "wrist.R",
             "upperleg01.L", "lowerleg01.L", "upperleg01.R", "lowerleg01.R"]
    parent = {"root": None, "spine05": "root", "upperarm01.L": "spine05", "lowerarm01.L": "upperarm01.L",
              "wrist.L": "lowerarm01.L", "upperarm01.R": "spine05", "lowerarm01.R": "upperarm01.R",
              "wrist.R": "lowerarm01.R", "upperleg01.L": "root", "lowerleg01.L": "upperleg01.L",
              "upperleg01.R": "root", "lowerleg01.R": "upperleg01.R"}
    a = np.deg2rad(40.0)
    head = {"root": np.array([0.0, 0.0, 0.9]), "spine05": np.array([0.0, 0.0, 0.9])}
    ankle = {}
    for s, sg in (("L", 1.0), ("R", -1.0)):
        sh = np.array([0.0, sg * 0.2, 1.4])
        el = sh + 0.30 * np.array([0.0, sg * np.sin(a), -np.cos(a)])
        wr = el + 0.25 * _unit([0.6, sg * 0.3, -0.74])
        head[f"upperarm01.{s}"], head[f"lowerarm01.{s}"], head[f"wrist.{s}"] = sh, el, wr
        hip = np.array([0.0, sg * 0.1, 0.9])
        leg = _LEG_REST * np.array([1.0, sg, 1.0])
        head[f"upperleg01.{s}"], head[f"lowerleg01.{s}"] = hip, hip + leg
        ankle[s] = hip + 2 * leg
    verts, owner = [], []
    for s in ("L", "R"):
        verts += [head[f"lowerarm01.{s}"], head[f"wrist.{s}"], head[f"lowerleg01.{s}"], ankle[s],
                  head[f"upperarm01.{s}"]]
        owner += [f"upperarm01.{s}", f"lowerarm01.{s}", f"upperleg01.{s}", f"lowerleg01.{s}", "spine05"]
    W = np.zeros((len(verts), len(names)))
    for i, o in enumerate(owner):
        W[i, names.index(o)] = 1.0
    heads = np.array([head[n] for n in names])
    return hm.HumanMesh(
        vertices=np.asarray(verts, float), faces=np.array([[0, 1, 2]]), bone_names=names,
        bone_parent=np.array([names.index(parent[n]) if parent[n] else -1 for n in names]),
        bone_head=heads, bone_tail=heads.copy(), weights=W)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("pose", MESH_POSES)
def test_mesh_pose_mapping_matches_capsule_joint_directions(pose):
    """메시 상완·전완 방향이 B의 `joint_positions`와 같다 (A자 기본 자세에서 출발해도).

    다리는 기본 자세의 벌림을 유지하고 B와 같은 회전(yaw·고관절·무릎)만 넣는다."""
    from airis.sim.body import joint_positions
    rig = _stick_rig()
    v = hm.pose_vertices(rig, pose)
    joints, _, _ = joint_positions(BodyParams(), pose, SCENARIOS["default"])
    r_yaw = hm._rot("z", pose.torso_yaw)
    for k, (s, sg) in enumerate((("L", 1.0), ("R", -1.0))):
        el, wr, kn, an, sh = v[5 * k: 5 * k + 5]
        np.testing.assert_allclose(_unit(el - sh), _unit(joints[f"elbow_{s}"] - joints[f"shoulder_{s}"]),
                                   atol=1e-9)
        np.testing.assert_allclose(_unit(wr - el), _unit(joints[f"wrist_{s}"] - joints[f"elbow_{s}"]),
                                   atol=1e-9)
        leg = _LEG_REST * np.array([1.0, sg, 1.0])
        r_thigh = r_yaw @ hm._rot("y", -pose.hip_flexion)
        hip = rig.bone_head[rig.bone(f"upperleg01.{s}")]
        np.testing.assert_allclose(kn - (r_yaw @ (hip - rig.hip_center) + rig.hip_center),
                                   r_thigh @ leg, atol=1e-9)
        np.testing.assert_allclose(an - kn, r_thigh @ hm._rot("y", pose.knee_flexion) @ leg, atol=1e-9)
    # 몸통 회전(yaw·pitch)은 어깨점을 고관절 중심 기준으로 돌린다
    R = hm._rot("z", pose.torso_yaw) @ hm._rot("y", pose.torso_pitch)
    np.testing.assert_allclose(v[4], R @ (rig.bone_head[rig.bone("upperarm01.L")] - rig.hip_center)
                               + rig.hip_center, atol=1e-9)


def test_align_rotation():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.normal(size=3), rng.normal(size=3)
        R = hm.align(a, b)
        np.testing.assert_allclose(R @ _unit(a), _unit(b), atol=1e-9)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0)
    R = hm.align([0, 0, -1], [0, 0, 1])                  # 반대 방향 (팔 내림 → 만세)
    np.testing.assert_allclose(R @ np.array([0, 0, -1]), [0, 0, 1], atol=1e-9)


def test_resolve_pose_applies_wheelchair_fixed_pose():
    p = hm.resolve_pose(PoseParams(hip_flexion=10.0), SCENARIOS["wheelchair"])
    assert p.hip_flexion == 90.0 and p.knee_flexion == 90.0
    assert hm.resolve_pose(PoseParams(hip_flexion=10.0), SCENARIOS["default"]).hip_flexion == 10.0


# 아래는 저장소에 커밋한 MakeHuman 자산(data/meshes/makehuman/, CC0)을 쓴다.
@pytest.fixture(scope="module")
def mh():
    return hm.cached_makehuman()


def test_makehuman_load(mh):
    assert mh.vertices.shape == (13380, 3) and mh.faces.shape == (26756, 3)
    assert len(mh.bone_names) == 163
    np.testing.assert_allclose(mh.weights.sum(axis=1), 1.0, atol=1e-9)
    assert mh.faces.min() == 0 and mh.faces.max() == mh.vertices.shape[0] - 1
    v, f = mh.vertices, mh.faces                          # 면이 바깥을 향한다 (부호 있는 부피 > 0)
    assert np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() > 0
    assert mh.bone_head[mh.bone("upperarm01.L"), 1] > 0 > mh.bone_head[mh.bone("upperarm01.R"), 1]


@pytest.mark.parametrize("scenario", ["default", "pregnant", "wheelchair"])
@pytest.mark.parametrize("pose", MESH_POSES[:3])
def test_makehuman_posed_in_booth(mh, scenario, pose):
    sc = SCENARIOS[scenario]
    v = hm.posed_in_booth(mh, BodyParams(), pose, sc, BOOTH)
    assert v.shape == mh.vertices.shape and np.isfinite(v).all()
    hip = hm.place_in_booth(mh.hip_center[None, :], mh, BodyParams(), BOOTH, sc)[0]
    assert hip[0] == pytest.approx(BOOTH["length_m"] / 2.0) and hip[1] == pytest.approx(0.0)
    if sc.seat_height_m is None:
        assert v[:, 2].min() == pytest.approx(0.0, abs=1e-9)          # 발바닥이 바닥
        if pose.shoulder_abduction < 90:
            assert v[:, 2].max() == pytest.approx(1.70, abs=0.02)     # 키 1.70 m
    else:
        assert hip[2] == pytest.approx(sc.seat_height_m)              # 고관절 중심 = 좌석 높이
        assert v[:, 2].min() > 0.0
    assert np.abs(v[:, 1]).max() < BOOTH["width_m"] / 2.0            # 세 자세 모두 벽 안
    assert v[:, 2].max() < BOOTH["height_m"]


def _posed_joint(mh, pose, bone: str) -> np.ndarray:
    """뼈 머리(관절)의 자세 적용 후 위치, 기본 자세 좌표계."""
    M = hm.bone_transforms(mh, pose)[mh.bone(bone)]
    return M[:3, :3] @ mh.bone_head[mh.bone(bone)] + M[:3, 3]


def test_makehuman_hands_up_raises_hands(mh):
    """만세면 손끝(손가락 뼈에 주로 붙은 정점)이 머리 위로 올라간다."""
    sc = SCENARIOS["default"]
    hand_bones = [i for i, n in enumerate(mh.bone_names) if n.startswith(("finger", "metacarpal", "wrist"))]
    hand = mh.weights[:, hand_bones].sum(axis=1) > 0.5
    down = hm.pose_mesh(None, None, PoseParams(), sc)
    up = hm.pose_mesh(None, None, PoseParams(shoulder_abduction=180.0, elbow_flexion=0.0), sc)
    assert up[hand, 2].max() > down[hand, 2].max() + 0.9            # 손끝이 1 m 가까이 올라간다
    assert up[hand, 2].max() > 1.9 > down[:, 2].max()                # 머리(1.70) 위, 천장(2.15) 아래
    assert up[:, 2].max() < BOOTH["height_m"]


@pytest.mark.parametrize("yaw", [90.0, -90.0])
def test_makehuman_yaw90_shoulder_line_along_x(mh, yaw):
    """yaw ±90 이면 어깨선(좌→우 어깨 관절)이 진행 방향 x 축과 나란하다. B: R_z(yaw) 이라 +90 이면 왼쪽 어깨가 −x."""
    pose = PoseParams(torso_yaw=yaw)
    line = _posed_joint(mh, pose, "upperarm01.L") - _posed_joint(mh, pose, "upperarm01.R")
    u = line / np.linalg.norm(line)
    assert abs(u[0]) > 0.99
    assert np.sign(u[0]) == -np.sign(yaw)
    rest = _posed_joint(mh, PoseParams(), "upperarm01.L") - _posed_joint(mh, PoseParams(), "upperarm01.R")
    assert abs(rest[1] / np.linalg.norm(rest)) > 0.99                # 기본 자세는 y 축 (좌우)


def test_pose_mesh_defaults(mh):
    """pose_mesh(asset, body, pose, scenario): None 이면 캐시 메시·키 1.70. posed_in_booth 와 같은 값."""
    assert hm.MeshAsset is hm.HumanMesh
    assert hm.load_makehuman(None).vertices.shape == mh.vertices.shape
    sc = SCENARIOS["default"]
    v = hm.pose_mesh(None, None, PoseParams(torso_yaw=30.0), sc)
    np.testing.assert_allclose(v, hm.posed_in_booth(mh, BodyParams(), PoseParams(torso_yaw=30.0), sc, BOOTH))
    tall = hm.pose_mesh(mh, BodyParams(height_m=1.85), PoseParams(), sc)
    assert tall[:, 2].max() == pytest.approx(1.85, abs=0.02)


def test_sparse_skinning_matches_dense(mh):
    """희소 블렌드 = 조밀 가중합 (가중치를 자르지 않는다)."""
    pose = PoseParams(40.0, 30.0, 60.0, 15.0, 90.0, 20.0, 25.0)
    M = hm.bone_transforms(mh, pose)
    per_bone = np.einsum("bij,vj->bvi", M[:, :3, :3], mh.vertices) + M[:, :3, 3][:, None, :]
    dense = np.einsum("vb,bvi->vi", mh.weights, per_bone)
    np.testing.assert_allclose(hm.pose_vertices(mh, pose), dense, atol=1e-12)


# 성능: 절대 시간 우선, 넘으면 같은 순간의 기준 numpy 연산 대비 비율로 본다 (B tests/test_body.py 방식).
_REF = np.random.default_rng(12345)
_REF_A, _REF_D, _REF_X = _REF.random((16, 3600)), _REF.random((16, 3)), _REF.random((3600, 3))


def _reference_workload() -> float:
    s = _REF_D @ _REF_X.T
    return float(np.exp(-(_REF_A * _REF_A) / (1.0 + s * s)).sum())


def test_makehuman_pose_speed(mh):
    """자세 1회 50 ms 이하 (실측 약 2 ms). CPU 부하로 넘으면 기준 연산 대비 40배 이하 (실측 약 2~3배)."""
    import time
    sc = SCENARIOS["default"]
    fn = lambda: hm.pose_mesh(mh, None, PoseParams(torso_yaw=90.0, shoulder_abduction=180.0), sc)  # noqa: E731
    fn()
    best = best_ref = float("inf")
    for _ in range(15):
        t0 = time.perf_counter()
        _reference_workload()
        best_ref = min(best_ref, time.perf_counter() - t0)
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    if best >= 0.050:
        assert best / best_ref < 40.0, f"pose_mesh {best * 1e3:.1f} ms, 기준 대비 {best / best_ref:.1f}배"


def test_figure_from_mesh(mh):
    v = hm.posed_in_booth(mh, BodyParams(), PoseParams(), SCENARIOS["default"], BOOTH)
    state = build_body(BodyParams(), PoseParams(), SCENARIOS["default"], patches_per_m2=400)
    fig = hm.figure_from_mesh(v, mh.faces, overlay_state=state, booth=BOOTH)
    assert {"사람 메시", "캡슐 마네킹 (반투명)", "부스", "슬롯 바 12개"} <= _names(fig)
    fig.to_dict()
