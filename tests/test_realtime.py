"""포즈 추정 → 체형, 추천 자세 테스트. 소유자: E. (`docs/tracks/E_realtime.md` 단계 3~4)

실제 카메라는 열지 않는다. 합성 키포인트 = 몸 모델 관절을 정면 핀홀 카메라에 투영한 것.

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
def test_recommend_pose_inside_booth_and_bounds(scenario):
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


@pytest.mark.parametrize("scenario,expected", [
    ("default", (0.2469, 0.4405, 0.2463)), ("pregnant", (0.2639, 0.4409, 0.2364)),
    ("wheelchair", (0.1824, 0.2652, 0.3212))])
def test_mesh_baselines_match_mesh_e4_rescoring(scenario, expected):
    """메시 몸 기준선 B0·B1·B2 가 C 메시판 E4 재채점(2,000/m²) 값과 같다 (채점 밀도가 맞는지)."""
    from airis.realtime.recommend import MESH_PATCHES_PER_M2, scoring_patches_per_m2
    assert scoring_patches_per_m2("mesh") == MESH_PATCHES_PER_M2 == 2000.0
    rows = compare_with_baselines(MESH_DEFAULT_BODY, SCENARIOS[scenario], PoseParams(), model="mesh")
    got = tuple(r.score for r in rows[1:])
    assert got == pytest.approx(expected, abs=5e-4)


def test_tall_body_falls_back_to_arms_down_peak():
    """키가 커서 만세가 천장(2.15 m) 밖이면 다음 봉우리(팔 내림)로 넘어간다."""
    tall = BodyParams(1.95, 0.48, 0.25, 0.72, 1.00)
    sc = SCENARIOS["default"]
    assert not is_inside_booth(tall, STUB_TABLE["default"][0].pose, sc, "capsule")
    rec = recommend(tall, sc, use_model=False, model="capsule")
    assert rec.label == STUB_TABLE["default"][1].label
    assert rec.notes and "부스" in rec.notes[0]
    assert is_inside_booth(tall, rec.pose, sc, "capsule")


def test_recommend_uses_model_when_available(monkeypatch):
    """C의 predict_pose 가 생기면 그것을 쓰고, 부스 밖 예측이면 표로 대체한다."""
    import airis.realtime.recommend as rmod
    sc = SCENARIOS["default"]
    monkeypatch.setattr(rmod, "_model_predict", lambda b, s: PoseParams(torso_yaw=60.0))
    rec = recommend(CAPSULE_BODY, sc, model="capsule")
    assert rec.source == "model" and rec.pose.torso_yaw == 60.0
    # 팔 수평(90°)은 캡슐 체형(팔 0.62 m)에서 옆벽 밖이다
    monkeypatch.setattr(rmod, "_model_predict", lambda b, s: PoseParams(shoulder_abduction=90.0))
    rec = recommend(CAPSULE_BODY, sc, model="capsule")
    assert rec.source.startswith("stub") and "회귀 모델" in rec.notes[0]


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
    rec_pose = recommend(CAPSULE_BODY, sc, model="capsule").pose
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


def test_stub_ranks_candidates_by_score_for_this_body(monkeypatch):
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
    assert any("만세" in s for s in lines) and any("약 70°" in s for s in lines)   # yaw 71.0 → 5° 단위
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
    at = AppTest.from_file(str(path), default_timeout=120)
    at.query_params["scenario"] = scenario
    at.run()
    at.sidebar.radio[0].set_value(mode).run()
    assert not at.exception, [e.value for e in at.exception]
    scen_radio = next(r for r in at.radio if r.key == "scenario")
    assert scen_radio.value == scenario
    labels = [m.label for m in at.metric]
    assert labels == ["B0 기본 대비", "B1 몸 회전 대비", "B2 만세 대비"]
    at.selectbox[0].set_value("합성 프레임 미리보기 (가짜 궤적)").run()
    assert not at.exception, [e.value for e in at.exception]


def test_dashboard_capsule_model_query():
    """?model=capsule 이면 캡슐 마네킹으로 끝까지 간다 (기본은 사람 메시)."""
    pytest.importorskip("streamlit")
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_dashboard.py"
    at = AppTest.from_file(str(path), default_timeout=120)
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
def test_recommend_and_compare_with_mesh_body(scenario):
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
