"""포즈 추정 → 체형, 추천 자세 테스트. 소유자: E. (`docs/tracks/E_realtime.md` 단계 3~4)

실제 카메라는 열지 않는다. 합성 키포인트 = 몸 모델 관절을 정면 핀홀 카메라에 투영한 것.

학습 산출물(`data/models/pose_flow.pt`·`pose_knn.parquet`)이 있든 없든 **건너뛰는 것 없이** 통과해야 한다.
표(스텁)를 기대하는 테스트는 `no_artifacts` 픽스처로 기본 경로를 없는 파일로 돌려 놓고(로컬에 산출물이 있으면
모델이 끼어든다), 모델 경로를 보는 테스트는 `_install_knn` 으로 tmp 에 작은 kNN 표를 만들어 쓴다
(torch 없이 도는 진짜 예측 경로).

전역 기본값(`BodyParams()`, `configs/physics.yaml` `body.model`)과 독립이다. 캡슐 기준 테스트는
`model="capsule"`·`profile="capsule"`과 캡슐 시절 체형(`CAPSULE_BODY`)을, 메시 기준 테스트는 `model="mesh"`와
`MESH_DEFAULT_BODY`를 명시한다 (5단계 BodyParams 기본값 교체·body.model 전환에서 깨지지 않게).
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

try:
    # 대시보드 테스트(AppTest)의 st.dataframe 이 pyarrow 를 스크립트 스레드에서 처음 import 하면,
    # 전체 스위트(test_particles 의 Taichi 초기화 뒤)에서 Windows access violation 이 났다.
    # 수집 단계(Taichi 초기화 전)에 메인 스레드에서 미리 올려 둔다.
    import pyarrow  # noqa: F401
except ImportError:
    pass

from airis.realtime import pose_estimate as pe
from airis.realtime.camera import detections_from_result, draw_pose, largest_person, read_image
from airis.realtime.recommend import (STUB_TABLE, compare_with_baselines, improvement,
                                      is_inside_booth, recommend, recommend_pose)
from airis.optimize.encoding import PoseEncoder
from airis.sim.body import build_body
from airis.sim.human_mesh import MESH_DEFAULT_BODY
from airis.sim.patch_baseline import outside_booth
from airis.sim.scenario import load_nozzle_layout, load_scenarios
from airis.sim.types import BodyParams, PoseParams

SCENARIOS = load_scenarios()
BOOTH = load_nozzle_layout()["booth"]

#: 캡슐 마네킹 시절 기본 체형 (캡슐 기준 테스트의 정답. 전역 BodyParams() 기본값과 무관하게 고정)
CAPSULE_BODY = BodyParams(1.70, 0.42, 0.22, 0.62, 0.85)
CAP = dict(model="capsule")          # synthetic_keypoints
CAPP = dict(profile="capsule")       # estimate_body / BodyEstimator

@pytest.fixture
def no_artifacts(monkeypatch, tmp_path):
    """학습 산출물이 없는 상태. 경로는 C의 `predict.py` 가 정하므로 그 기본값만 돌려 놓는다."""
    from airis.model import predict          # 저장소 안 모듈이라 import 실패는 skip 이 아니라 실패여야 한다
    monkeypatch.setattr(predict, "DEFAULT_MODEL_PATH", tmp_path / "없음_pose_flow.pt")
    monkeypatch.setattr(predict, "DEFAULT_KNN_PATH", tmp_path / "없음_pose_knn.parquet")
    return predict


BODIES = {
    "기본": CAPSULE_BODY,
    "작은 체형": BodyParams(1.55, 0.36, 0.20, 0.55, 0.74),
    "큰 체형": BodyParams(1.90, 0.48, 0.25, 0.70, 0.97),
}
FIELDS = ("height_m", "shoulder_width_m", "torso_depth_m", "arm_length_m", "leg_length_m")


def _rel_err(est: BodyParams, true: BodyParams) -> dict[str, float]:
    return {k: abs(getattr(est, k) / getattr(true, k) - 1.0) for k in FIELDS}


# ---------------------------------------------------------------------------
# 합성 키포인트 → 체형 (완료 기준 3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(BODIES))
@pytest.mark.parametrize("pose", [PoseParams(), PoseParams(shoulder_abduction=40.0, elbow_flexion=0.0)])
def test_recover_body_from_synthetic_keypoints_with_height(name, pose):
    """키 입력 보정: 체형 5개를 ±5% 안에 복원한다."""
    body = BODIES[name]
    kp, conf = pe.synthetic_keypoints(body, pose, SCENARIOS["default"], **CAP)
    est = pe.estimate_body(kp, conf, height_m=body.height_m, image_size=(1280, 720), **CAPP)
    assert est.ok, est.message
    err = _rel_err(est.body, body)
    assert max(err.values()) < 0.05, err


@pytest.mark.parametrize("name", list(BODIES))
def test_recover_body_with_marker_scale(name):
    """마커 보정: 키를 모를 때 m/px 스케일로 키까지 ±5% 안에 복원한다."""
    body = BODIES[name]
    cam = pe.PinholeCamera(distance_m=2.5, focal_px=900.0)
    kp, conf = pe.synthetic_keypoints(body, PoseParams(), SCENARIOS["default"], cam, **CAP)
    # 사람 발 옆 바닥(카메라에서 같은 거리)에 놓인 0.20 m 마커를 비스듬히 본 네 꼭짓점
    cx = BOOTH["length_m"] / 2.0
    corners3 = np.array([[cx + 0.1, 0.3, 0.0], [cx + 0.1, 0.5, 0.0],
                         [cx - 0.1, 0.5, 0.0], [cx - 0.1, 0.3, 0.0]])
    scale = pe.scale_from_marker(cam.project(corners3, cx), 0.20)
    est = pe.estimate_body(kp, conf, scale_m_per_px=scale, **CAPP)
    assert est.ok, est.message
    err = _rel_err(est.body, body)
    assert max(err.values()) < 0.05, err


def test_recover_body_robust_to_pixel_noise():
    """키포인트에 ±2 px 잡음을 넣어도 여러 프레임 중앙값은 ±5% 안."""
    body = BODIES["기본"]
    stab = pe.BodyEstimator(height_m=body.height_m, window=30, **CAPP)
    for seed in range(15):
        kp, conf = pe.synthetic_keypoints(body, PoseParams(), SCENARIOS["default"],
                                          noise_px=2.0, seed=seed, **CAP)
        stab.add(kp, conf, (1280, 720))
    est = stab.body()
    assert est is not None
    assert max(_rel_err(est, body).values()) < 0.05


def test_seated_wheelchair_uses_height_and_leg_ratio():
    body = BODIES["기본"]
    kp, conf = pe.synthetic_keypoints(body, PoseParams(), SCENARIOS["wheelchair"], **CAP)
    est = pe.estimate_body(kp, conf, height_m=body.height_m, seated=True, **CAPP)
    assert est.ok, est.message
    assert max(_rel_err(est.body, body).values()) < 0.05
    no_height = pe.estimate_body(kp, conf, scale_m_per_px=0.003, seated=True, **CAPP)
    assert not no_height.ok and "키" in no_height.message


def test_scale_required():
    kp, conf = pe.synthetic_keypoints(CAPSULE_BODY, PoseParams(), SCENARIOS["default"], **CAP)
    with pytest.raises(ValueError):
        pe.estimate_body(kp, conf, **CAPP)


def test_scale_from_marker_uses_horizontal_edges():
    # 가로 100 px, 세로(비스듬히 눌린) 40 px 사다리꼴 → 가로 변만 쓴다
    corners = np.array([[0, 0], [100, 0], [100, 40], [0, 40]], float)
    assert pe.scale_from_marker(corners, 0.2) == pytest.approx(0.002)


# ---------------------------------------------------------------------------
# 누락·신뢰도 낮은 키포인트
# ---------------------------------------------------------------------------
@pytest.fixture
def kp_default():
    return pe.synthetic_keypoints(CAPSULE_BODY, PoseParams(), SCENARIOS["default"], **CAP)


def test_low_confidence_side_replaced_by_other_side(kp_default):
    kp, conf = kp_default
    conf = conf.copy()
    conf[[pe.L_ELBOW, pe.L_WRIST, pe.R_KNEE, pe.R_ANKLE]] = 0.05     # 왼팔·오른다리 흐림
    est = pe.estimate_body(kp, conf, height_m=1.70, **CAPP)
    assert est.ok, est.message
    assert max(_rel_err(est.body, CAPSULE_BODY).values()) < 0.05
    assert "left_elbow" in est.missing


def test_foreshortened_limb_uses_longer_side(kp_default):
    """한쪽 전완이 카메라 쪽으로 굽어 화면에서 짧아져도(단축) 긴 쪽으로 팔 길이를 잰다."""
    kp, conf = kp_default
    kp = kp.copy()
    kp[pe.L_WRIST] = kp[pe.L_ELBOW] + 0.3 * (kp[pe.L_WRIST] - kp[pe.L_ELBOW])
    est = pe.estimate_body(kp, conf, height_m=1.70, **CAPP)
    assert abs(est.body.arm_length_m / CAPSULE_BODY.arm_length_m - 1.0) < 0.05


def test_nose_missing_falls_back_to_eyes(kp_default):
    kp, conf = kp_default
    conf = conf.copy()
    conf[pe.NOSE] = 0.0
    est = pe.estimate_body(kp, conf, scale_m_per_px=None, height_m=1.70, **CAPP)
    assert est.ok
    assert max(_rel_err(est.body, CAPSULE_BODY).values()) < 0.05


@pytest.mark.parametrize("drop,word", [
    ((pe.L_ANKLE, pe.R_ANKLE), "발목"),
    ((pe.L_SHOULDER,), "어깨"),
    ((pe.NOSE, pe.L_EYE, pe.R_EYE, pe.L_EAR, pe.R_EAR), "얼굴"),
    ((pe.L_WRIST, pe.R_WRIST), "팔"),
])
def test_missing_keypoints_return_none_with_message(kp_default, drop, word):
    kp, conf = kp_default
    conf = conf.copy()
    conf[list(drop)] = 0.0
    est = pe.estimate_body(kp, conf, height_m=1.70, **CAPP)
    assert est.body is None
    assert word in est.message
    assert pe.keypoints_to_body(kp, conf, height_m=1.70, **CAPP) is None


def test_offscreen_keypoints_are_missing(kp_default):
    """화면 밖(이미지 아래로 잘린) 발목은 신뢰도가 높아도 무효."""
    kp, conf = kp_default
    kp = kp.copy()
    kp[[pe.L_ANKLE, pe.R_ANKLE], 1] = 800.0            # 이미지 높이 720 밖
    est = pe.estimate_body(kp, conf, height_m=1.70, image_size=(1280, 720), **CAPP)
    assert est.body is None and "발목" in est.message
    zero = kp.copy()
    zero[[pe.L_ANKLE, pe.R_ANKLE]] = 0.0               # ultralytics 의 (0, 0) 누락 표기
    assert pe.estimate_body(zero, conf, height_m=1.70, **CAPP).body is None


def test_implausible_estimate_rejected(kp_default):
    kp, conf = kp_default
    est = pe.estimate_body(kp, conf, scale_m_per_px=0.05, **CAPP)    # 스케일 10배 → 키 수 m
    assert est.body is None and "범위" in est.message


def test_median_body_and_estimator_window():
    a, b, c = BodyParams(1.6), BodyParams(1.7), BodyParams(3.0)
    assert pe.median_body([a, b, c]).height_m == 1.7
    stab = pe.BodyEstimator(height_m=1.7, min_frames=2, **CAPP)
    kp, conf = pe.synthetic_keypoints(CAPSULE_BODY, PoseParams(), SCENARIOS["default"], **CAP)
    stab.add(kp, conf)
    assert stab.body() is None                          # 프레임 부족
    bad = conf.copy()
    bad[pe.L_SHOULDER] = 0.0
    stab.add(kp, bad)                                   # 실패 프레임은 모으지 않는다
    assert stab.body() is None and stab.n_frames == 2
    stab.add(kp, conf)
    assert stab.body() is not None


# ---------------------------------------------------------------------------
# YOLO 결과 파싱 (모델 없이 가짜 Results)
# ---------------------------------------------------------------------------
def _fake_result(n_people: int = 2):
    kp, conf = pe.synthetic_keypoints(CAPSULE_BODY, PoseParams(), SCENARIOS["default"], **CAP)
    xy = np.stack([kp * (0.5 + 0.5 * i) for i in range(n_people)])      # 뒤 사람이 더 크다
    cf = np.stack([conf] * n_people)
    boxes = np.stack([np.r_[x.min(axis=0), x.max(axis=0)] for x in xy])
    return SimpleNamespace(
        keypoints=SimpleNamespace(xy=xy, conf=cf),
        boxes=SimpleNamespace(xyxy=boxes, conf=np.full(n_people, 0.9)),
        orig_shape=(720, 1280))


def test_largest_person_from_result():
    det = largest_person(_fake_result(3))
    assert det is not None and det.image_size == (1280, 720)
    dets = detections_from_result(_fake_result(3))
    areas = [(d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) for d in dets]
    assert areas == sorted(areas, reverse=True)
    np.testing.assert_allclose(det.keypoints, dets[0].keypoints)
    empty = SimpleNamespace(keypoints=SimpleNamespace(xy=np.zeros((0, 17, 2)), conf=None),
                            boxes=None, orig_shape=(720, 1280))
    assert largest_person(empty) is None


def test_draw_pose_and_read_image(tmp_path):
    from PIL import Image
    img = np.zeros((720, 1280, 3), np.uint8)
    img[..., 2] = 255                                    # BGR 빨강
    path = tmp_path / "red.png"
    Image.fromarray(img[:, :, ::-1]).save(path)
    back = read_image(path)
    np.testing.assert_array_equal(back, img)
    np.testing.assert_array_equal(read_image(path.read_bytes()), img)
    out = draw_pose(img, largest_person(_fake_result(1)))
    assert out.shape == img.shape and not np.array_equal(out, img[:, :, ::-1])


# ---------------------------------------------------------------------------
# ④-2 회전 안내 (확장 5단계, docs/e_plan_screen_draft.md)
# ---------------------------------------------------------------------------
def test_rotation_plan_covers_a_full_turn():
    """제안 자세를 **유지한 채** 한 바퀴를 N칸으로 나눈다. 자세는 몸 방향만 달라진다."""
    from dataclasses import replace as _replace

    from airis.realtime import plan_guide as pg

    pose = STUB_TABLE["default"][0].pose
    plan = pg.rotation_plan(pose)
    assert len(plan.phases) == pg.ROTATION_STEPS == 10
    assert plan.duration_s == pytest.approx(pg.ROTATION_STEPS * pg.STEP_SECONDS) == 20.0
    assert all(ph.duration_s == pg.STEP_SECONDS for ph in plan.phases)
    # 몸 방향만 다르고 나머지 여섯 값은 그대로다 (자세를 바꾸지 않는 것이 이 모드의 핵심)
    for ph in plan.phases:
        assert _replace(ph.pose, torso_yaw=pose.torso_yaw) == pose
    # 한 바퀴를 고르게 나눈다 (접힌 각도라 차가 ±360 로 보일 수 있다)
    yaws = [ph.pose.torso_yaw for ph in plan.phases]
    steps = sorted(round((b - a) % 360.0, 6) for a, b in zip(yaws, yaws[1:]))
    assert steps == [pytest.approx(36.0)] * (len(yaws) - 1)
    assert (np.asarray(plan.zone_strengths) == 1.0).all(), "모드 A 는 바람 세기를 바꾸지 않는다"


def test_rotation_instructions_say_the_pose_once():
    """자세는 한 번만 말하고 방향은 칸으로 말한다 — 단계마다 자세를 되풀이하지 않는다."""
    from airis.realtime import plan_guide as pg

    sc = SCENARIOS["default"]
    lines = pg.rotation_instructions(STUB_TABLE["default"][0].pose, sc)
    assert any("만세" in t for t in lines)
    assert sum("만세" in t for t in lines) == 1
    assert any("10칸" in t and "36°" in t and "2초" in t for t in lines)
    assert any("왼쪽·오른쪽 어느 쪽이든" in t for t in lines), "좌우는 대칭이라 고르게 둔다"
    # 자세 문장에는 몸 방향이 들어가지 않는다 (회전 안내와 각도가 어긋나면 안 된다)
    assert not any("돌려 한쪽 벽" in t for t in lines)

    seated = pg.rotation_instructions(STUB_TABLE["wheelchair"][0].pose, SCENARIOS["wheelchair"])
    assert seated[0].startswith("휠체어") and any("돌아앉으세요" in t for t in seated)


@pytest.mark.parametrize("scenario", ["default", "pregnant", "wheelchair"])
def test_rotation_reference_reads_experiment_file(scenario):
    """화면 수치는 C 의 실험 파일에서 읽는다 (E 가 다시 계산하지 않는다)."""
    from airis.realtime import plan_guide as pg

    ref = pg.rotation_reference(scenario)
    by = {r.key: r for r in ref}
    assert {"P0", "P1_10", "P1opt_10"} <= set(by), f"없는 조건: {sorted(by)}"
    assert by["P1opt_10"].n_phases == 10 and by["P1opt_10"].duration_s == 20.0
    # 제안 자세로 도는 쪽이 제조사 안내보다, 그쪽이 안내 없이 통과보다 낫다
    assert by["P0"].total_removal < by["P1_10"].total_removal < by["P1opt_10"].total_removal
    for r in ref:
        assert "P1" not in r.label and "P0" not in r.label, f"화면 문구에 약어: {r.label}"


def test_control_json_fans_come_from_zone_helper():
    """제어 JSON 의 노즐 세기는 E 가 지어내지 않고 B 의 apply_zone_strengths 결과 그대로다."""
    import numpy as _np

    from airis.realtime import plan_guide as pg
    from airis.sim.scenario import apply_zone_strengths, load_nozzles

    sc = SCENARIOS["default"]
    plan = pg.rotation_plan(STUB_TABLE["default"][0].pose)
    doc = pg.control_json(plan, MESH_DEFAULT_BODY, sc, source="test")

    assert doc["schema"] == "airis.plan.v1" and doc["scenario"] == "default"
    assert doc["duration_s"] == 20.0 and len(doc["phases"]) == len(plan.phases)
    assert set(doc["zones"]) == set(__import__("airis.sim.types", fromlist=["ZONE_NAMES"]).ZONE_NAMES)
    nozzle = load_nozzles()
    for ph_doc, ph in zip(doc["phases"], plan.phases):
        want = apply_zone_strengths(nozzle, plan.zone_strengths, ph.pose.torso_yaw).strengths
        assert ph_doc["fans"] == pytest.approx(_np.asarray(want).reshape(-1), abs=1e-4)
        assert len(ph_doc["fans"]) == len(_np.asarray(nozzle.strengths).reshape(-1))


def test_rotation_plan_rejects_bad_arguments():
    from airis.realtime import plan_guide as pg
    with pytest.raises(ValueError):
        pg.rotation_plan(PoseParams(), steps=0)
    with pytest.raises(ValueError):
        pg.rotation_plan(PoseParams(), step_s=0.0)


# ---------------------------------------------------------------------------
# 개인정보: 화면에 원본 영상을 띄우지 않는다 (팀원 제안 3ab035e 를 받은 것)
# ---------------------------------------------------------------------------
def _edge_path():
    from pathlib import Path as P
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "render_plotly_png", P(__file__).resolve().parents[1] / "scripts" / "render_plotly_png.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, P(mod.DEFAULT_EDGE)


def test_render_plotly_png_renders_3d():
    """kaleido 대신 쓰는 경로가 실제로 3D 그림을 PNG 로 만든다.

    `fig.write_image`(kaleido)가 이 환경에서 멈추고, 헤드리스 화면 캡처는 WebGL 캔버스를 빈 채로
    가져온다. 그래서 페이지 안에서 `Plotly.toImage` 를 부르는 경로가 맞는지 여기서 확인한다.
    Edge 가 없는 환경(= 이 경로를 쓸 수 없는 환경)에서만 건너뛴다.
    """
    pytest.importorskip("websockets")
    go = pytest.importorskip("plotly.graph_objects")
    mod, edge = _edge_path()
    if not edge.exists():
        pytest.skip(f"Edge 가 없다: {edge}")

    import tempfile
    from pathlib import Path as P

    fig = go.Figure(go.Mesh3d(x=[0, 1, 0, 0], y=[0, 0, 1, 0], z=[0, 0, 0, 1],
                              i=[0, 0, 0, 1], j=[1, 2, 3, 2], k=[2, 3, 1, 3],
                              intensity=[0.0, 0.4, 0.8, 1.0]))
    fig.update_layout(paper_bgcolor="white", margin=dict(l=0, r=0, t=0, b=0))
    out = P(tempfile.gettempdir()) / "airis_render_test.png"
    out.unlink(missing_ok=True)
    try:
        mod.render_figure(fig, out, width=420, height=320, port=9407)
        assert out.exists() and out.stat().st_size > 3000
        from PIL import Image
        img = Image.open(out).convert("RGB")
        assert img.size == (420, 320)
        colors = {tuple(c) for c in np.unique(np.asarray(img).reshape(-1, 3), axis=0)}
        assert len(colors) > 3, "3D 가 그려지지 않고 빈 캔버스만 나왔다"
    finally:
        out.unlink(missing_ok=True)


def test_dashboard_warms_up_model_before_first_user():
    """기동할 때 추천을 한 번 미리 돌려 모델을 올려 둔다 (첫 사용자가 로딩을 기다리지 않게).

    소스 검사다: 예열이 실제로 걸리는지는 AppTest 가 예외 없이 도는 것으로만 확인된다
    (예열은 결과를 쓰지 않으므로 화면에 흔적이 남지 않는다).
    """
    import re
    from pathlib import Path as P
    src = (P(__file__).resolve().parents[1] / "scripts" / "run_dashboard.py").read_text(encoding="utf-8")
    assert re.search(r"@st[.]cache_resource\([^)]*\)\s*def warm_up", src), \
        "warm_up 은 서버당 한 번만 돌아야 한다 (cache_resource)"
    after = src.split("def warm_up")[1]
    assert "except Exception:" in after[:after.index("@st.cache_data")], \
        "예열이 실패해도 화면은 떠야 한다"
    body = src.split("def main(")[1]
    assert body.index("warm_up(body_model)") < body.index('st.subheader("④ 추천 자세")'), \
        "예열은 추천을 그리기 전에 끝나야 한다"


def test_display_wording_is_plain_but_identifiers_are_not():
    """화면 문구만 쉬운 말로 바꾼다. 내부 식별자(source 값·ScoreRow.name)는 그대로 둔다.

    식별자를 함께 바꾸면 C·F 와 주고받는 값이 깨지고, 문구만 바꾸면 발표에서 그대로 읽을 수 있다.
    """
    from airis.realtime.recommend import BASELINE_LABELS, SOURCE_LABELS, baseline_poses, source_text

    # 내부 식별자는 그대로
    assert set(baseline_poses()) == {"B0 기본", "B1 몸 회전", "B2 만세"}
    from airis.realtime.recommend import BASELINE_LABELS_SHORT
    for table in (BASELINE_LABELS, BASELINE_LABELS_SHORT):
        assert set(table) >= set(baseline_poses()) | {"추천"}
        for name, shown in table.items():
            assert not shown.startswith(("B0", "B1", "B2")), f"{name} 에 약어가 남았다: {shown}"
    assert all(len(v) <= 7 for v in BASELINE_LABELS_SHORT.values()), "짧은 이름이 길면 잘린다"

    # 출처 문구: 사람이 읽을 수 있고, 원문 식별자는 들어가지 않는다
    assert source_text("model: hybrid (flow)") == "AI 추천 후보 중에서 시뮬레이션으로 확인한 자세"
    assert source_text("model: hybrid (knn)").startswith("비슷한 체형 참조")
    assert source_text("model: hybrid (extra)").startswith("기본 후보표")
    assert "stub" not in source_text("stub: 메시판 k14 E4 …")
    for shown in SOURCE_LABELS.values():
        assert "flow" not in shown and "kNN" not in shown and "E4" not in shown

    # 모르는 값은 그대로 돌려준다 (새 출처가 생겨도 화면이 비지 않는다)
    assert source_text("model: 새 방식") == "model: 새 방식"


def test_pose_label_uses_plain_joint_names():
    from airis.viz.pose_view import pose_label
    text = pose_label(PoseParams(177.4, -4.6, 0.7, 0.4, 70.5, 0.6, 10.3))
    for plain in ("팔 벌림(옆으로) 177°", "팔 올림(앞으로) -5°", "팔꿈치 굽힘 1°",
                  "상체 숙임 0°", "몸 방향 70°", "고관절 굽힘 1°", "무릎 굽힘 10°"):
        assert plain in text, f"없다: {plain}"
    assert not text.startswith("벌림"), "옛 표기(벌림 … · 굽힘 … · 회전 …)가 남았다"


def test_skeleton_only_has_no_input_pixels():
    """뼈대 그림에는 입력 프레임의 픽셀이 섞일 수 없다 (프레임을 인자로 받지 않는다)."""
    import inspect

    from airis.realtime.camera import SKELETON_BACKGROUND, draw_skeleton_only

    det = largest_person(_fake_result(1))
    params = set(inspect.signature(draw_skeleton_only).parameters)
    assert params == {"det", "image_size", "min_conf", "line_width", "background", "show_box"},         "원본 프레임을 받는 인자가 생기면 안 된다"

    out = draw_skeleton_only(det)
    assert out.shape == (720, 1280, 3)
    # 화면에 나오는 색은 배경 + 그리기 색뿐이다. 사진에서 올 수 있는 그 밖의 색은 없다.
    allowed = {SKELETON_BACKGROUND, (84, 214, 199), (255, 190, 70), (160, 160, 160)}
    colors = {tuple(c) for c in np.unique(out.reshape(-1, 3), axis=0)}
    assert colors <= allowed, f"예상 밖의 색: {sorted(colors - allowed)[:5]}"
    assert len(colors) > 1, "뼈대가 그려지지 않았다"
    assert (75, 220, 120) not in colors, "상자는 사람 위치·크기를 드러내므로 그리지 않는다"
    # det 가 없으면 완전한 단색 (사람을 못 찾았다고 원본을 보여 주지 않는다)
    blank = draw_skeleton_only(None, image_size=(64, 48))
    assert blank.shape == (48, 64, 3) and (blank == np.array(SKELETON_BACKGROUND)).all()


def test_dashboard_does_not_show_original_frames():
    """대시보드는 원본 위에 겹쳐 그리는 draw_pose 를 쓰지 않고, 프레임을 세션에 남기지 않는다."""
    from pathlib import Path as P
    src = (P(__file__).resolve().parents[1] / "scripts" / "run_dashboard.py").read_text(encoding="utf-8")
    assert "draw_pose" not in src and "draw_skeleton_only" in src
    assert "local_frames" not in src


def test_uploaded_video_file_is_deleted(tmp_path):
    """업로드한 영상은 프레임을 뽑은 뒤 임시 파일을 지운다 (열지 못해 실패해도 지운다)."""
    import tempfile
    from pathlib import Path as P

    cv2 = pytest.importorskip("cv2")
    from airis.realtime.camera import frames_from_video_bytes

    def leftovers():
        return set(P(tempfile.gettempdir()).glob("airis_upload_*"))

    before = leftovers()
    src = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(12):
        writer.write(np.full((48, 64, 3), i * 10, np.uint8))
    writer.release()
    assert src.exists() and src.stat().st_size > 0

    frames = frames_from_video_bytes("up.mp4", src.read_bytes(), every_n=5, max_frames=30)
    assert frames and frames[0].shape == (48, 64, 3)
    assert leftovers() == before, "임시 파일이 남았다"

    with pytest.raises(OSError):                      # 깨진 파일이어도
        frames_from_video_bytes("bad.mp4", b"not a video")
    assert leftovers() == before, "실패했을 때 임시 파일이 남았다"


def test_dashboard_keeps_estimated_body_across_rerun(monkeypatch):
    """촬영 뒤 시나리오를 바꿔 rerun 이 나도 추정 체형이 유지된다 (프레임 대신 결과를 세션에 남긴다)."""
    pytest.importorskip("streamlit")
    from pathlib import Path as P

    from streamlit.testing.v1 import AppTest

    from airis.realtime import camera as cam

    wide = replace(MESH_DEFAULT_BODY, shoulder_width_m=0.50)      # 기본(0.342)과 확실히 다른 체형
    kp, conf = pe.synthetic_keypoints(wide, PoseParams(), SCENARIOS["default"], model="mesh")
    det = cam.PoseDetection(kp, conf, np.r_[kp.min(axis=0) - 20, kp.max(axis=0) + 20], (1280, 720))
    frames = [np.full((720, 1280, 3), 30, np.uint8)]
    monkeypatch.setattr(cam, "capture_frames", lambda *a, **k: frames)   # 실제 카메라를 열지 않는다
    monkeypatch.setattr(cam, "load_pose_model", lambda *a, **k: object())
    monkeypatch.setattr(cam, "detect_pose", lambda model, image, **k: det)

    path = P(__file__).resolve().parents[1] / "scripts" / "run_dashboard.py"
    at = AppTest.from_file(str(path), default_timeout=240)
    at.run()
    at.sidebar.radio[0].set_value("로컬 카메라 (OpenCV)").run()
    at.button[0].click().run()                                   # 촬영
    assert not at.exception, [e.value for e in at.exception]

    def shoulder():
        return next(n.value for n in at.number_input if n.key and n.key.endswith("_shoulder_width_m"))

    assert shoulder() > 0.45, "추정 체형이 반영되지 않았다"
    at.radio(key="scenario").set_value("pregnant").run()          # rerun (촬영 버튼을 다시 누르지 않는다)
    assert not at.exception, [e.value for e in at.exception]
    assert shoulder() > 0.45, "rerun 뒤 추정 체형이 기본값으로 돌아갔다"


def test_ultralytics_lazy_import():
    """ultralytics 가 설치돼 있으면 YOLO 클래스를 import 할 수 있다 (가중치는 내려받지 않는다)."""
    ultralytics = pytest.importorskip("ultralytics")
    from airis.realtime.camera import MODEL_DIR, pose_weights_path
    assert hasattr(ultralytics, "YOLO")
    assert pose_weights_path().parent == MODEL_DIR
    assert MODEL_DIR.parts[-2:] == ("data", "models")


# ---------------------------------------------------------------------------
# 추천 자세 (단계 4)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["default", "pregnant", "wheelchair"])
def test_recommend_pose_inside_booth_and_bounds(scenario, no_artifacts):
    sc = SCENARIOS[scenario]
    pose = recommend_pose(BodyParams(), sc)
    state = build_body(BodyParams(), pose, sc, patches_per_m2=400)
    assert not outside_booth(state.patch_pos, BOOTH)
    for key, (lo, hi) in sc.pose_bounds.items():
        if key not in sc.fixed_pose:
            assert lo <= getattr(pose, key) <= hi
    for key, v in sc.fixed_pose.items():
        assert getattr(pose, key) == v


@pytest.mark.parametrize("scenario", ["default", "pregnant", "wheelchair"])
def test_stub_first_entry_is_mesh_e4_best_for_default_body(scenario):
    """메시 기본 체형이면 메시판 E4 결과(첫 후보, 세 시나리오 모두 만세)를 그대로 고른다."""
    from airis.optimize.encoding import PoseEncoder
    from airis.realtime.recommend import E4_SOURCE
    sc = SCENARIOS[scenario]
    rec = recommend(MESH_DEFAULT_BODY, sc, use_model=False, model="mesh")
    first = STUB_TABLE[scenario][0]
    assert rec.source == f"stub: {E4_SOURCE}"
    assert rec.label == first.label and rec.pose == PoseEncoder(sc).clip_pose(first.pose)
    assert rec.pose.shoulder_abduction > 170.0
    assert not rec.notes


@pytest.mark.parametrize("scenario", ["default", "pregnant", "wheelchair"])
def test_mesh_baselines_match_mesh_e4_rescoring(scenario):
    """메시 몸 기준선 B0·B1·B2 가 k14 E4 재채점(2,000/m²) 값과 같다 (채점 밀도·물리 상수가 맞는지).

    기대값은 `recommend.MESH_E4_BASELINES` 한 곳에만 적는다 (C 보고값과 넷째 자리까지 같은 실측값).
    """
    from airis.realtime.recommend import (MESH_E4_BASELINES, MESH_PATCHES_PER_M2,
                                          scoring_patches_per_m2)
    assert scoring_patches_per_m2("mesh") == MESH_PATCHES_PER_M2 == 2000.0
    rows = compare_with_baselines(MESH_DEFAULT_BODY, SCENARIOS[scenario], PoseParams(), model="mesh")
    got = tuple(r.score for r in rows[1:])
    assert got == pytest.approx(MESH_E4_BASELINES[scenario], abs=5e-5)


# ---------------------------------------------------------------------------
# docs/e4_reference.json 교차 검증
#
# E의 상수(MESH_E4_BASELINES·STUB_TABLE)는 C의 E4 결과를 손으로 옮겨 적은 값이었다. 그 전달 경로에서
# 실수가 세 번 났다(자세·점수 짝이 어긋남, 밀도 라벨 오기). 이제 C가 결과를 저장소 파일로 커밋하므로
# **파일과 상수가 같은지 테스트가 확인한다**. 어긋나면 어느 쪽이 틀렸는지 사람이 보고 고친다.
# ---------------------------------------------------------------------------
E4_REFERENCE = "docs/e4_reference.json"


def _e4_reference() -> dict:
    import json
    from pathlib import Path as P
    path = P(__file__).resolve().parents[1] / E4_REFERENCE
    if not path.exists():
        pytest.skip(f"{E4_REFERENCE} 가 없다 (C 산출물)")
    return json.loads(path.read_text(encoding="utf-8"))


def _bundle_of(source: str, ref: dict) -> str:
    """스텁 출처 문자열에 적힌 묶음 ID (파일에 그 묶음이 있어야 한다)."""
    ids = [b["group_id"] for b in ref["bundles"]]
    found = [g for g in ids if g in source]
    assert len(found) == 1, f"출처 '{source}' 에서 파일의 묶음 ID 를 찾지 못했다 (파일: {ids})"
    return found[0]


def _best_peak(ref: dict, scenario: str, arm_class: str) -> dict:
    peaks = [q for q in ref["scenarios"][scenario]["peaks"] if q["arm_class"] == arm_class]
    assert peaks, f"{scenario} 에 {arm_class} 봉우리가 없다"
    return max(peaks, key=lambda q: q["score"])


def _folded(peak: dict, ref: dict) -> PoseParams:
    """봉우리의 접은 자세를 스텁 표와 같은 0.1° 단위로 반올림."""
    return PoseParams(*(round(peak["pose_folded"][k], 1) for k in ref["pose_keys"]))


@pytest.mark.parametrize("scenario", ["default", "pregnant", "wheelchair"])
def test_mesh_e4_baselines_match_reference_file(scenario):
    """기준선 상수가 docs/e4_reference.json 의 재채점(2,000/m²) 값과 같다."""
    from airis.realtime.recommend import MESH_E4_BASELINES, MESH_PATCHES_PER_M2
    ref = _e4_reference()
    group = _bundle_of(f"{STUB_TABLE[scenario][0].source}", ref)
    entry = ref["scenarios"][scenario]["baselines"][group]["rescored"]
    assert entry["patches_per_m2"] == MESH_PATCHES_PER_M2
    want = tuple(entry["scores"][b] for b in ("B0", "B1", "B2"))
    assert MESH_E4_BASELINES[scenario] == pytest.approx(want, abs=5e-6)


@pytest.mark.parametrize("scenario", ["default", "pregnant", "wheelchair"])
def test_stub_hands_up_matches_reference_peak(scenario):
    """스텁 첫 후보(만세)가 파일의 hands_up 최고 봉우리와 같다 (0.1° 단위)."""
    ref = _e4_reference()
    entry = STUB_TABLE[scenario][0]
    peak = _best_peak(ref, scenario, "hands_up")
    assert peak["bundle"] == _bundle_of(entry.source, ref)
    assert entry.pose == _folded(peak, ref)


@pytest.mark.parametrize("scenario", ["pregnant", "wheelchair"])
def test_stub_arms_down_matches_reference_peak(scenario):
    """둘째 후보(팔 내림)가 파일의 arms_down 최고와 같다. default 는 일부러 다르다(아래 테스트)."""
    ref = _e4_reference()
    entry = STUB_TABLE[scenario][1]
    peak = _best_peak(ref, scenario, "arms_down")
    assert peak["bundle"] == _bundle_of(entry.source, ref)
    assert entry.pose == _folded(peak, ref)


def test_default_arms_down_is_deliberately_not_the_search_optimum():
    """default 둘째 후보만 탐색 최적과 **일부러** 다르다 — 같다고 단언하면 안 된다.

    탐색 최적은 어깨 굽힘을 −20° 쯤 뒤로 보낸 자세다. 그런데 `pose_instructions` 는 그 구간을
    "팔은 자연스럽게 내리세요"로 읽어 준다. 그 자세를 표에 넣으면 화면이 약속하는 점수와 안내대로 선
    사람이 얻는 점수가 달라지므로, 굽힘 0 자세를 쓰고 점수도 그 자세를 직접 잰 값을 적는다.
    대신 **너무 낮지는 않은지**(≤ 1%) 와 안내 문장이 실제로 그렇게 읽히는지를 확인한다.
    """
    from airis.realtime.recommend import _nozzles, patch_evaluator, pose_instructions
    ref = _e4_reference()
    sc = SCENARIOS["default"]
    entry = STUB_TABLE["default"][1]
    peak = _best_peak(ref, "default", "arms_down")

    assert entry.pose != _folded(peak, ref), "같아지면 이 테스트의 전제가 깨진다 (주석도 고칠 것)"
    assert abs(peak["pose_folded"]["shoulder_flexion"]) > 10.0, "탐색 최적은 팔을 뒤로 보낸 자세다"
    assert abs(entry.pose.shoulder_flexion) < 5.0, "표의 자세는 굽힘 0 근처여야 한다"

    ev, nz = patch_evaluator("mesh"), _nozzles()
    enc = PoseEncoder(sc)
    ours = ev.evaluate(enc.clip_pose(entry.pose), nz, MESH_DEFAULT_BODY, sc).score
    assert ours < peak["score"], "표가 탐색 최적보다 높을 수는 없다"
    assert (peak["score"] - ours) / peak["score"] <= 0.01, "탐색 최적과 1% 넘게 벌어지면 다시 고른다"

    # 두 자세가 같은 문장으로 읽히는지 — 이것이 굽힘 0 을 쓰는 이유다
    lines = pose_instructions(_folded(peak, ref), sc)
    assert any("자연스럽게 내리세요" in t for t in lines)
    assert pose_instructions(entry.pose, sc)[-1] == lines[-1]


def test_tall_body_falls_back_to_arms_down_peak():
    """키가 커서 만세가 천장(2.15 m) 밖이면 다음 봉우리(팔 내림)로 넘어간다."""
    tall = BodyParams(1.95, 0.48, 0.25, 0.72, 1.00)
    sc = SCENARIOS["default"]
    assert not is_inside_booth(tall, STUB_TABLE["default"][0].pose, sc, "capsule")
    rec = recommend(tall, sc, use_model=False, model="capsule")
    assert rec.label == STUB_TABLE["default"][1].label
    assert rec.notes and "부스" in rec.notes[0]
    assert is_inside_booth(tall, rec.pose, sc, "capsule")


def _fake_prediction(candidates, sources, scores, infeasible=None):
    from airis.model.predict import Prediction
    infeasible = np.zeros(len(candidates), bool) if infeasible is None else np.asarray(infeasible)
    scores = np.asarray(scores, float)
    pick = np.where(infeasible, -np.inf, scores) if not infeasible.all() else scores
    i = int(np.argmax(pick))
    return Prediction(candidates[i], candidates, scores, infeasible, list(sources), sources[i])


def test_recommend_passes_stub_candidates_to_model(monkeypatch):
    """혼합 추천에 표 후보를 extra_candidates 로 넘기고, E의 평가기·노즐로 재채점하게 한다."""
    import airis.model.predict as P
    import airis.realtime.recommend as rmod
    sc = SCENARIOS["default"]
    seen = {}

    def fake_predict(body, scenario, **kw):
        seen.update(kw)
        extra = list(kw["extra_candidates"])
        return _fake_prediction([PoseParams(torso_yaw=60.0)] + extra, ["flow"] + ["extra"] * len(extra),
                                [0.1] + [0.5] * len(extra))

    monkeypatch.setattr(P, "predict", fake_predict)
    rec = recommend(CAPSULE_BODY, sc, model="capsule")
    assert [p for p in seen["extra_candidates"]] == [e.pose for e in STUB_TABLE["default"]]
    # 재채점 구성 D (F 측정 뒤 총괄 확정): 거의 같은 후보를 묶고 채점을 스레드로 나눈다
    from airis.realtime.recommend import RESCORE_DEDUP_DEG, RESCORE_THREADS
    assert seen["dedup_deg"] == RESCORE_DEDUP_DEG == 3.0
    assert seen["n_threads"] == RESCORE_THREADS == 3
    assert seen["evaluator"] is rmod.patch_evaluator("capsule") and seen["nozzle"] is rmod._nozzles()
    # 표 후보가 이겼으면 표의 이름을 쓰되 출처는 모델이다 (풀 전체에서 고른 것이라)
    assert rec.source == "model: hybrid (extra)"
    assert rec.label.startswith("모델 추천") and STUB_TABLE["default"][0].label in rec.label
    assert rec.pose == PoseEncoder(sc).clip_pose(STUB_TABLE["default"][0].pose)
    assert rec.elapsed_s > 0.0


def test_recommend_reports_source_stats(monkeypatch):
    """출처별 후보 수·부스 안 수·최고 점수를 ④ 표에 쓸 수 있게 돌려준다."""
    import airis.model.predict as P
    sc = SCENARIOS["default"]
    cands = [PoseParams(torso_yaw=y) for y in (10.0, 20.0, 30.0, 40.0)]
    monkeypatch.setattr(P, "predict", lambda b, s, **kw: _fake_prediction(
        cands, ["flow", "flow", "knn", "extra"], [0.1, 0.4, 0.9, 0.3], [False, True, False, False]))
    rec = recommend(CAPSULE_BODY, sc, model="capsule")
    stats = {s.source: s for s in rec.stats}
    assert stats["flow"].n == 2 and stats["flow"].n_feasible == 1
    assert stats["flow"].best == pytest.approx(0.1)          # 부스 밖(0.4)은 빼고 잰다
    assert stats["knn"].best == pytest.approx(0.9) and stats["extra"].n == 1
    assert rec.source == "model: hybrid (knn)" and rec.label == "모델 추천"


def test_recommend_falls_back_when_all_candidates_outside_booth(monkeypatch):
    import airis.model.predict as P
    sc = SCENARIOS["default"]
    monkeypatch.setattr(P, "predict", lambda b, s, **kw: _fake_prediction(
        [PoseParams(torso_yaw=60.0)], ["flow"], [0.1], [True]))
    rec = recommend(CAPSULE_BODY, sc, model="capsule")
    assert rec.source.startswith("stub") and "부스" in rec.notes[-1]


@pytest.mark.parametrize("exc,word", [(FileNotFoundError("없다"), "산출물"),
                                      (ImportError("torch 없음"), "torch"),
                                      (KeyError("학습에 없던 시나리오: default"), "KeyError")])
def test_recommend_falls_back_to_stub_with_note(monkeypatch, exc, word):
    """모델을 못 쓰면 표로 돌아가고, 왜 그랬는지 화면에 쓸 메모를 남긴다."""
    import airis.model.predict as P
    def boom(body, scenario, **kw):
        raise exc
    monkeypatch.setattr(P, "predict", boom)
    rec = recommend(CAPSULE_BODY, SCENARIOS["default"], model="capsule")
    assert rec.source.startswith("stub") and rec.label == STUB_TABLE["default"][0].label
    assert rec.notes and word in rec.notes[0]


def test_recommend_without_artifacts_uses_stub(no_artifacts):
    """산출물이 없는 환경(CI)에서도 표로 안내한다 — 진짜 predict 를 부른다."""
    from airis.realtime.recommend import E4_SOURCE
    rec = recommend(MESH_DEFAULT_BODY, SCENARIOS["default"], model="mesh")
    assert rec.source == f"stub: {E4_SOURCE}"
    assert rec.notes and "산출물" in rec.notes[0] and not rec.stats


def test_model_artifacts_uses_predict_status(monkeypatch):
    """산출물 상태는 C·F의 artifact_status() 를 그대로 옮긴다 (E는 판정을 다시 하지 않는다)."""
    import airis.model.predict as P
    from airis.realtime.recommend import model_artifacts

    monkeypatch.setattr(P, "artifact_status", lambda *a, **k: {
        "current": {"nozzle_layout_hash": "n", "physics_hash": "p"}, "current_error": None,
        # 도장이 없거나 일부만 찍힌 산출물은 match=None (#99). 화면에도 "확인 불가"여야 한다.
        "flow": {"path": "a.pt", "exists": True, "match": None, "mismatched": [], "error": None,
                 "stamp": {"nozzle_layout_hash": None, "physics_hash": "p", "commit": "abc1234"}},
        "knn": {"path": "b.parquet", "exists": False, "match": None, "mismatched": [], "error": None,
                "stamp": {}}})
    flow, knn = model_artifacts()
    assert flow.name == "flow" and flow.exists and flow.commit == "abc1234"
    assert flow.config_ok is None                      # "설정 일치"로 보이면 오해한다
    assert not knn.exists and knn.message == "없음" and knn.config_ok is None


def test_model_artifacts_real_call():
    """진짜 호출도 두 줄(flow·knn)을 돌려준다. 산출물이 없어도 예외가 나지 않는다."""
    from airis.realtime.recommend import model_artifacts
    rows = model_artifacts()
    assert [r.name for r in rows] == ["flow", "knn"]
    for r in rows:
        assert r.path and isinstance(r.exists, bool)


def _install_knn(monkeypatch, tmp_path, pose_of, bodies=None):
    """tmp 에 작은 kNN 표를 만들어 산출물로 꽂는다 (torch 없이도 도는 진짜 예측 경로).

    flow 는 없는 경로로 두어 kNN + 고정 후보만 쓴다. `pose_of(scenario)` 가 표에 넣을 자세다.
    """
    import pandas as pd                      # requirements 에 있다 (kNN 표가 parquet)
    import airis.model.predict as P
    from airis.model.flow import BODY_KEYS, POSE_KEYS
    from airis.model.knn import PoseKNN

    bodies = bodies or [MESH_DEFAULT_BODY, replace(MESH_DEFAULT_BODY, height_m=1.60),
                        replace(MESH_DEFAULT_BODY, height_m=1.80)]
    rows = []
    for i, b in enumerate(bodies):
        for name in SCENARIOS:
            pose = pose_of(name)
            rows.append({"body_idx": i, "scenario": name, "score": 0.0, "arm_class": "hands_up",
                         **{f"body_{k}": getattr(b, k) for k in BODY_KEYS},
                         **{f"pose_{k}": getattr(pose, k) for k in POSE_KEYS}})
    path = PoseKNN(pd.DataFrame(rows)).save(tmp_path / "pose_knn.parquet")
    monkeypatch.setattr(P, "DEFAULT_KNN_PATH", path)
    monkeypatch.setattr(P, "DEFAULT_MODEL_PATH", tmp_path / "없음_pose_flow.pt")
    return path


@pytest.mark.parametrize("scenario", ["default", "pregnant", "wheelchair"])
def test_recommend_with_artifact_keeps_table_candidate_when_it_wins(monkeypatch, tmp_path, scenario):
    """산출물이 있어도 표 후보를 같은 풀에서 재채점한다 → 표가 이기면 표가 나온다 (표보다 나빠지지 않는다)."""
    _install_knn(monkeypatch, tmp_path, lambda name: PoseParams())      # 기본 자세만 든 표 (약한 후보)
    sc = SCENARIOS[scenario]
    rec = recommend(MESH_DEFAULT_BODY, sc, model="mesh")
    assert rec.source == "model: hybrid (extra)"
    assert is_inside_booth(MESH_DEFAULT_BODY, rec.pose, sc, "mesh")
    stats = {s.source: s for s in rec.stats}
    assert set(stats) == {"knn", "extra"} and stats["extra"].n == len(STUB_TABLE[scenario])
    assert stats["extra"].best > stats["knn"].best                      # 표(E4) 가 기본 자세보다 높다
    assert max(s.best for s in rec.stats if s.best is not None) == stats["extra"].best


@pytest.mark.parametrize("scenario", ["default", "pregnant", "wheelchair"])
def test_recommend_merges_model_candidate_that_duplicates_the_table(monkeypatch, tmp_path, scenario):
    """모델 후보가 표 후보와 사실상 같으면 하나로 묶고, **표 후보 쪽**을 남긴다.

    재채점 구성 D(`RESCORE_DEDUP_DEG` 3°)가 관절 각도 차 3° 안의 후보를 묶는다. 고정 후보(표)는
    남기는 쪽이라, 모델이 표와 같은 자세를 내면 화면에는 표 후보로 보인다 — 같은 자세를 두 번 채점하지
    않으면서, 사용자에게는 이름이 붙은 쪽(표의 '만세 + 옆으로 회전')을 보여 주는 것이다.
    """
    _install_knn(monkeypatch, tmp_path, lambda name: STUB_TABLE[name][0].pose)   # 표와 같은 자세
    sc = SCENARIOS[scenario]
    rec = recommend(MESH_DEFAULT_BODY, sc, model="mesh")
    assert rec.source == "model: hybrid (extra)"
    assert rec.pose == PoseEncoder(sc).clip_pose(STUB_TABLE[scenario][0].pose)
    assert STUB_TABLE[scenario][0].label in rec.label
    stats = {s.source: s for s in rec.stats}
    assert "extra" in stats, "표 후보는 묶인 뒤에도 남아야 한다"


def test_recommend_prefers_model_candidate_that_beats_the_table(monkeypatch, tmp_path):
    """모델 후보가 표보다 **실제로 높으면** 그것을 쓴다.

    표는 메시 기본 체형의 봉우리라 캡슐 체형에서는 최적이 아니다. 회전만 바꿔 더 높은 자세를 찾아
    (테스트 안에서 직접 재서 확인) kNN 표에 넣는다.

    여유가 크지 않다 (현재 +0.0104, 약 1.9%). 물리 상수·패치 격자가 바뀌어 캡슐 체형의 봉우리가
    옮겨가면 첫 단언에서 실패한다 — 가리지 말고 그때 후보 자세를 다시 고르라는 뜻의 경보다.
    """
    from airis.realtime.recommend import _nozzles, patch_evaluator
    sc, ev, nz = SCENARIOS["default"], patch_evaluator("capsule"), _nozzles()
    enc = PoseEncoder(sc)

    def score(pose):
        return ev.evaluate(enc.clip_pose(pose), nz, CAPSULE_BODY, sc).score

    table_best = max(score(e.pose) for e in STUB_TABLE["default"])
    cands = [replace(STUB_TABLE["default"][0].pose, torso_yaw=y) for y in (30.0, 45.0, 60.0, 90.0)]
    better = max(cands, key=score)
    assert score(better) > table_best, "표보다 높은 후보를 못 찾았다 (봉우리가 바뀌었는지 확인)"

    _install_knn(monkeypatch, tmp_path, lambda name: better, bodies=[CAPSULE_BODY])
    rec = recommend(CAPSULE_BODY, sc, model="capsule")
    assert rec.source == "model: hybrid (knn)" and rec.pose == enc.clip_pose(better)
    stats = {s.source: s for s in rec.stats}
    assert stats["knn"].best > stats["extra"].best


def test_recommend_rejects_model_pose_outside_booth(monkeypatch, tmp_path):
    """평가기가 가능하다고 해도 E가 부스 안인지 다시 보고, 밖이면 표로 대체한다."""
    import airis.model.predict as P
    sc = SCENARIOS["default"]
    outside = PoseParams(shoulder_abduction=90.0)            # 캡슐 체형에서 옆벽 밖
    assert not is_inside_booth(CAPSULE_BODY, outside, sc, "capsule")
    monkeypatch.setattr(P, "predict", lambda b, s, **kw: _fake_prediction(
        [outside], ["knn"], [9.9], [False]))                 # 불가 아님이라고 우겨도
    rec = recommend(CAPSULE_BODY, sc, model="capsule")
    assert rec.source == f"stub: {STUB_TABLE['default'][0].source}"
    assert rec.label == STUB_TABLE["default"][0].label
    assert rec.notes[-1] == "모델이 고른 자세가 부스(천장·벽) 밖이라 표의 자세로 대체"
    assert is_inside_booth(CAPSULE_BODY, rec.pose, sc, "capsule")


@pytest.mark.parametrize("scenario", ["default", "wheelchair"])
def test_compare_with_baselines_matches_run_baselines(scenario):
    """기준선 정의·집계가 C의 `scripts/run_baselines.py`(E4 기준선)와 같고, 추천이 B0·B1을 이긴다.

    숫자를 박지 않는다: 물리 기준(패치 격자, 상수)이 바뀌면 experiments.md 표도 다시 재기 때문이다.
    """
    import importlib.util
    from pathlib import Path

    from airis.optimize.encoding import PoseEncoder
    from airis.realtime.recommend import _nozzles, patch_evaluator

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_baselines.py"
    spec = importlib.util.spec_from_file_location("run_baselines", path)
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)

    sc = SCENARIOS[scenario]
    rec_pose = recommend(CAPSULE_BODY, sc, use_model=False, model="capsule").pose
    rows = compare_with_baselines(CAPSULE_BODY, sc, rec_pose, model="capsule")
    by = {r.name: r for r in rows}
    for mine, theirs in (("B0 기본", "B0"), ("B1 몸 회전", "B1"), ("B2 만세", "B2")):
        ref = rb.evaluate_condition(patch_evaluator("capsule"), rb.BASELINES[theirs], PoseEncoder(sc),
                                    _nozzles(), CAPSULE_BODY, sc)
        assert by[mine].score == pytest.approx(ref["score"], abs=1e-9)
        assert by[mine].n_feasible == ref["n_feasible"]
        assert by[mine].infeasible == ref["infeasible"]
    rec = by["추천"]
    assert rec.result is not None and "removal" in rec.result.extra
    for name in ("B0 기본", "B1 몸 회전"):
        assert improvement(rec, by[name]) > 0.0


def test_improvement_none_for_infeasible_baseline():
    sc = SCENARIOS["default"]
    rows = compare_with_baselines(BodyParams(), sc, PoseParams())
    rec = rows[0]
    bad = replace(rows[1], infeasible=True)
    assert improvement(rec, bad) is None
    assert improvement(rec, rows[1]) == pytest.approx(0.0)


def test_stub_ranks_candidates_by_score_for_this_body(monkeypatch, no_artifacts):
    """표 후보가 둘 다 부스 안이면 이 체형의 패치판 점수가 높은 쪽을 고른다."""
    import airis.realtime.recommend as rmod
    sc = SCENARIOS["default"]

    class Flipped:                               # 팔 내림 봉우리가 더 높은 가상 체형
        def evaluate(self, pose, nozzle, body, scenario):
            return SimpleNamespace(score=1.0 if pose.shoulder_abduction < 90 else 0.5)

    monkeypatch.setattr(rmod, "patch_evaluator", lambda model=None: Flipped())
    rec = recommend(CAPSULE_BODY, sc, use_model=False, model="capsule")
    assert rec.label == STUB_TABLE["default"][1].label
    assert "점수가 높아" in rec.notes[0]
    assert recommend(CAPSULE_BODY, sc, use_model=False, rank_by_score=False,
                     model="capsule").label == STUB_TABLE["default"][0].label


def test_pose_instructions():
    from airis.realtime.recommend import pose_instructions
    lines = pose_instructions(STUB_TABLE["default"][0].pose, SCENARIOS["default"])
    assert any("만세" in s for s in lines) and any("약 70°" in s for s in lines)   # yaw 70.5 → 5° 단위
    lines = pose_instructions(PoseParams(torso_yaw=-92.0), SCENARIOS["default"])
    assert any("약 90°" in s and "옆으로" in s for s in lines)                  # 좌우 부호는 말하지 않는다
    lines = pose_instructions(STUB_TABLE["wheelchair"][0].pose, SCENARIOS["wheelchair"])
    assert any("약 35°" in s for s in lines) and any("만세" in s for s in lines)
    lines = pose_instructions(STUB_TABLE["wheelchair"][1].pose, SCENARIOS["wheelchair"])
    assert lines[0].startswith("휠체어") and any("45°" in s for s in lines)
    assert any("내리세요" in s for s in lines)
    lines = pose_instructions(PoseParams(torso_pitch=20.0, elbow_flexion=60.0), SCENARIOS["default"])
    assert any("숙이세요" in s for s in lines) and any("팔꿈치" in s for s in lines)


# ---------------------------------------------------------------------------
# 대시보드 (Streamlit AppTest, 카메라·YOLO 없이)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode,scenario", [("합성 마네킹 (카메라 없이)", "default"),
                                           ("체형 직접 입력", "wheelchair")])
def test_dashboard_runs_to_recommendation(mode, scenario):
    """입력 → 체형 → 시나리오 → 추천 자세·기준 비교까지 예외 없이 그려진다."""
    pytest.importorskip("streamlit")
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_dashboard.py"
    # 여러 프로세스가 CPU 를 쓰는 중에 메시 채점이 겹치면 120 초로 모자란 적이 있다 (k14 전체 실행 1회 실패).
    at = AppTest.from_file(str(path), default_timeout=240)
    at.query_params["scenario"] = scenario
    at.run()
    at.sidebar.radio[0].set_value(mode).run()
    assert not at.exception, [e.value for e in at.exception]
    scen_radio = next(r for r in at.radio if r.key == "scenario")
    assert scen_radio.value == scenario
    # 화면에는 일반인 표현을 쓴다 (내부 식별자는 ScoreRow.name 으로 따로 둔다)
    from airis.realtime.recommend import BASELINE_LABELS_SHORT
    labels = [m.label for m in at.metric]
    assert labels == [f"{BASELINE_LABELS_SHORT[n]} 대비" for n in ("B0 기본", "B1 몸 회전", "B2 만세")]
    assert not any("B0" in t or "hybrid" in t for t in labels)
    # 지표 칸이 좁아 긴 이름은 "안내 없이 서 있을 때..." 로 잘린다. 짧은 이름만 쓴다.
    assert all(len(t) <= 10 for t in labels), f"지표 라벨이 길어 잘린다: {labels}"
    at.selectbox[0].set_value("합성 프레임 미리보기 (가짜 궤적)").run()
    assert not at.exception, [e.value for e in at.exception]


def test_dashboard_capsule_model_query():
    """?model=capsule 이면 캡슐 마네킹으로 끝까지 간다 (기본은 사람 메시)."""
    pytest.importorskip("streamlit")
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_dashboard.py"
    at = AppTest.from_file(str(path), default_timeout=240)
    at.query_params["model"] = "capsule"
    at.run()
    at.sidebar.radio[0].set_value("체형 직접 입력").run()
    assert not at.exception, [e.value for e in at.exception]
    model_radio = next(r for r in at.sidebar.radio if "캡슐 마네킹" in r.options)
    assert model_radio.value == "캡슐 마네킹"
    assert len(at.metric) == 3


# ---------------------------------------------------------------------------
# 메시 몸 (관절 중심 BodyParams 정의, docs/interfaces.md). 전역 기본값은 아직 캡슐이라 model="mesh" 를 명시한다.
# ---------------------------------------------------------------------------

MESH_BODIES = {
    "메시 기본": MESH_DEFAULT_BODY,
    "작은 체형": BodyParams(1.55, 0.31, 0.18, 0.42, 0.80),
    "큰 체형": BodyParams(1.90, 0.40, 0.22, 0.52, 1.00),
    "다리 긴 체형": BodyParams(1.70, 0.36, 0.19, 0.47, 0.93),
}


def test_profiles():
    cap, mesh = pe.profile_for("capsule"), pe.profile_for("mesh")
    assert cap is pe.CAPSULE_PROFILE and mesh.name == "mesh"
    from airis.sim.body import _configured_body_model
    assert pe.profile_for(None).name == _configured_body_model()      # 설정 파일 body.model 을 따른다
    assert mesh.torso_depth_per_height == pytest.approx(0.194 / 1.70)
    assert mesh.seated_leg_per_height == pytest.approx(0.883 / 1.70)
    assert 0.08 < mesh.nose_below_top < 0.11 and 0.06 < mesh.eye_below_top < mesh.nose_below_top
    assert 0.03 < mesh.ankle_above_floor < 0.06 and 0.97 < mesh.leg_vertical_ratio <= 1.0
    with pytest.raises(ValueError):
        pe.profile_for("smplx")


def test_mesh_landmarks_on_face():
    lm = pe.mesh_landmarks()
    kp3 = pe.mesh_keypoints_3d(None, PoseParams(), SCENARIOS["default"])
    cx = BOOTH["length_m"] / 2.0
    assert kp3[pe.NOSE, 0] > kp3[pe.L_EYE, 0] > kp3[pe.L_EAR, 0]          # 코가 가장 앞
    assert abs(kp3[pe.NOSE, 1]) < 0.01                                     # 가운데
    assert kp3[pe.L_EAR, 1] > kp3[pe.L_EYE, 1] > 0 > kp3[pe.R_EYE, 1] > kp3[pe.R_EAR, 1]
    assert kp3[pe.L_SHOULDER, 1] == pytest.approx(MESH_DEFAULT_BODY.shoulder_width_m / 2, abs=0.01)
    assert abs(kp3[pe.L_HIP, 0] - cx) < 0.05 and set(lm) == {"nose", "ear_L", "ear_R"}


@pytest.mark.parametrize("name", list(MESH_BODIES))
@pytest.mark.parametrize("pose", [PoseParams(), PoseParams(shoulder_abduction=40.0, elbow_flexion=0.0)])
def test_recover_mesh_body_from_synthetic_keypoints(name, pose):
    """메시 투영 합성 키포인트 → 체형 5개 ±5% (키 입력 보정). 원근 탓에 1~3% 작게 나온다."""
    body = MESH_BODIES[name]
    kp, conf = pe.synthetic_keypoints(body, pose, SCENARIOS["default"], model="mesh")
    est = pe.estimate_body(kp, conf, height_m=body.height_m, image_size=(1280, 720), profile="mesh")
    assert est.ok, est.message
    err = _rel_err(est.body, body)
    assert max(err.values()) < 0.05, err


@pytest.mark.parametrize("name", ["메시 기본", "큰 체형"])
def test_recover_mesh_body_with_marker_scale(name):
    body = MESH_BODIES[name]
    cam = pe.PinholeCamera(distance_m=2.5, focal_px=900.0)
    kp, conf = pe.synthetic_keypoints(body, PoseParams(), SCENARIOS["default"], cam, model="mesh")
    cx = BOOTH["length_m"] / 2.0
    corners3 = np.array([[cx + 0.1, 0.3, 0.0], [cx + 0.1, 0.5, 0.0],
                         [cx - 0.1, 0.5, 0.0], [cx - 0.1, 0.3, 0.0]])
    scale = pe.scale_from_marker(cam.project(corners3, cx), 0.20)
    est = pe.estimate_body(kp, conf, scale_m_per_px=scale, profile="mesh")
    assert est.ok, est.message
    assert max(_rel_err(est.body, body).values()) < 0.05


def test_recover_mesh_body_noise_and_seated():
    stab = pe.BodyEstimator(height_m=1.70, window=30, profile="mesh")
    for seed in range(10):
        kp, conf = pe.synthetic_keypoints(None, PoseParams(), SCENARIOS["default"], noise_px=2.0,
                                          seed=seed, model="mesh")
        stab.add(kp, conf, (1280, 720))
    assert max(_rel_err(stab.body(), MESH_DEFAULT_BODY).values()) < 0.05
    kp, conf = pe.synthetic_keypoints(None, PoseParams(), SCENARIOS["wheelchair"], model="mesh")
    est = pe.estimate_body(kp, conf, height_m=1.70, seated=True, profile="mesh")
    assert est.ok and max(_rel_err(est.body, MESH_DEFAULT_BODY).values()) < 0.05


@pytest.mark.parametrize("scenario", ["default", "pregnant", "wheelchair"])
def test_recommend_and_compare_with_mesh_body(scenario, no_artifacts):
    """메시 몸: 추천은 부스 안이고, 같은 몸 모델로 채점한 B0·B1 보다 높다."""
    sc = SCENARIOS[scenario]
    rec = recommend(None, sc, model="mesh")
    assert is_inside_booth(None, rec.pose, sc, "mesh")
    rows = compare_with_baselines(None, sc, rec.pose, model="mesh")
    by = {r.name: r for r in rows}
    from airis.realtime.recommend import scoring_patches_per_m2
    n_patch = build_body(None, rec.pose, sc, patches_per_m2=scoring_patches_per_m2("mesh"),
                         model="mesh").patch_pos.shape[0]
    assert len(by["추천"].result.extra["removal"]) == n_patch             # 메시 패치로 채점
    for name in ("B0 기본", "B1 몸 회전"):
        assert improvement(by["추천"], by[name]) > 0.0
