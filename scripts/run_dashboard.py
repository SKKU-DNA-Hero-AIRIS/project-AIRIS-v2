"""AIRIS v2 실시간 안내 대시보드 (Streamlit + plotly). 소유자: E. (`docs/tracks/E_realtime.md` 단계 4)

    streamlit run scripts/run_dashboard.py
    streamlit run scripts/run_dashboard.py -- --image path/to/photo.jpg   # 시작 이미지 지정

화면 순서
    ① 입력 (예시 이미지 · 이미지 업로드 · 브라우저 카메라 · 영상 파일 · 로컬 카메라 · 합성 마네킹 · 직접 입력)
    ② 뼈대 그림과 추정 체형 5개 (수정 가능)
    ③ 시나리오 선택 (영상으로 판별하지 않는다)
    ④ 추천 자세 3D 마네킹 + 기준 자세(B0·B1·B2)와 점수 비교 (D PatchEvaluator, 400/m²)
       추천은 C의 혼합 모델(flow + kNN + 고정 후보표)을 부르고, 산출물이 없으면 표만 쓴다.
    ⑤ 입자 애니메이션 (A 덤프가 있으면, 없으면 합성 프레임 미리보기)

기동할 때 추천을 한 번 미리 돌려(`warm_up`) 추천 모델을 올려 둔다. 첫 호출에는 산출물 로딩이 섞여
몇 초가 걸리는데, 그 비용을 첫 사용자가 아니라 서버 기동 때 치르게 하는 것이다.

**개인정보**: 카메라·업로드 영상은 YOLO 추론에만 쓰고 **화면에는 원본을 띄우지 않는다**. ②에 보이는 것은
`camera.draw_skeleton_only` 가 빈 캔버스에 그린 뼈대뿐이다(원본 픽셀을 인자로 받지 않는다). 업로드한 영상 파일은
프레임을 뽑은 뒤 바로 지우고, 세션에는 프레임 대신 **추정 결과만** 남긴다. 팀원 제안 3ab035e 를 받은 것이다.

URL 쿼리 `?demo=1`이면 예시 이미지로 바로 시작한다 (스크린샷용). `?scenario=wheelchair` 등으로 시나리오 지정.
몸 모델은 사이드바에서 고른다 (기본 사람 메시, `?model=capsule`이면 캡슐 마네킹). 체형 추정 비율·추천·점수·
3D 그림·애니메이션이 모두 같은 몸 모델을 쓴다.
v1(project-AIRIS-MVP `realtime_vision_app.py`)의 Streamlit 틀(사이드바 입력 선택, 모델 캐시)을 따랐다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                                    # noqa: E402
import streamlit as st                                                # noqa: E402

from airis.realtime import camera, pose_estimate as pe               # noqa: E402
from airis.realtime.recommend import (MESH_E4_BASELINES, MESH_E4_HANDS_UP_SEEDS,  # noqa: E402
                                      RESPONSE_BUDGET_S, compare_with_baselines, default_body,
                                      improvement, model_artifacts, pose_instructions, recommend,
                                      scoring_patches_per_m2)
from airis.sim.body import build_body                                 # noqa: E402
from airis.sim.scenario import load_nozzle_layout, load_scenarios     # noqa: E402
from airis.sim.types import PART_NAMES, BodyParams, PoseParams       # noqa: E402
from airis.viz import anim                                            # noqa: E402
from airis.viz.pose_view import figure_from_pose, pose_label          # noqa: E402

SCENARIO_LABELS = {"default": "일반 성인", "pregnant": "임산부", "wheelchair": "휠체어 사용자"}
PART_LABELS = {"head": "머리", "torso_front": "몸통 앞", "torso_back": "몸통 뒤",
               "arms": "팔", "legs": "다리"}
BODY_LABELS = {"height_m": "키 (m)", "shoulder_width_m": "어깨 너비 (m)",
               "torso_depth_m": "몸통 두께 (m)", "arm_length_m": "팔 길이 (m)",
               "leg_length_m": "다리 길이 (m)"}
INPUT_MODES = ["예시 이미지", "이미지 업로드", "브라우저 카메라", "영상 파일", "로컬 카메라 (OpenCV)",
               "합성 마네킹 (카메라 없이)", "체형 직접 입력"]
BODY_MODELS = {"사람 메시 (MakeHuman)": "mesh", "캡슐 마네킹": "capsule"}
#: 분석 결과를 담아 두는 세션 키 (원본 프레임은 넣지 않는다)
ANALYSIS_KEY = "analysis"


def _wide_kw() -> dict:
    """streamlit 1.46+ 는 width="stretch", 그 전은 use_container_width=True."""
    major, minor = (int(x) for x in st.__version__.split(".")[:2])
    return {"width": "stretch"} if (major, minor) >= (1, 46) else {"use_container_width": True}


WIDE = _wide_kw()


def _cli_image() -> Path | None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--image", default=None)
    args, _ = ap.parse_known_args(sys.argv[1:])
    return Path(args.image) if args.image else None


def _example_image() -> Path | None:
    """시작 이미지: --image 인자, 없으면 ultralytics 에 들어 있는 예시 사진(bus.jpg)."""
    cli = _cli_image()
    if cli is not None and cli.exists():
        return cli
    try:
        from ultralytics.utils import ASSETS
    except ImportError:
        return None
    p = Path(ASSETS) / "bus.jpg"
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# 캐시
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="포즈 추정 모델(YOLO11-pose)을 불러오는 중…")
def get_pose_model():
    return camera.load_pose_model()


@st.cache_resource
def get_scenarios():
    return load_scenarios()


@st.cache_data(show_spinner=False, max_entries=32)
def detect_cached(digest: str, _image: np.ndarray):
    return camera.detect_pose(get_pose_model(), _image)


@st.cache_resource(show_spinner="추천 모델을 준비하는 중…")
def warm_up(body_model: str) -> float:
    """기동할 때 추천을 한 번 미리 돌려 모델(torch·가중치)을 올려 둔다. 결과는 쓰지 않는다.

    첫 추천 호출에는 산출물 로딩이 함께 들어가 몇 초가 걸린다(측정 약 4초). 그 비용을 첫 사용자가
    아니라 기동할 때 치르게 한다. `cache_resource` 라 서버 하나당 몸 모델별로 한 번만 돈다.
    실패해도 화면은 그대로 떠야 하므로 예외를 삼킨다 (진짜 실패는 ④에서 폴백 메모로 보인다).
    """
    t0 = time.perf_counter()
    try:
        recommend(None, get_scenarios()["default"], model=body_model)
    except Exception:
        pass
    return time.perf_counter() - t0


@st.cache_data(show_spinner=False, ttl=60)
def artifacts_cached():
    """산출물 상태. 호출마다 파일을 다시 읽으므로 rerun 마다 부르지 않게 캐시한다 (새 산출물은 60초 안에 반영)."""
    return model_artifacts()


# ttl 이 없다: 같은 입력이면 실행 내내 같은 결과를 쓴다. 산출물을 바꾸면 대시보드를 다시 시작해야 반영된다
# (사이드바 "추천 모델" 설명에 적어 두었다).
@st.cache_data(show_spinner="추천 자세를 고르는 중 (모델 후보 재채점)…", max_entries=64)
def recommend_cached(body_t: tuple, scenario: str, body_model: str):
    return recommend(BodyParams(*body_t), get_scenarios()[scenario], model=body_model)


@st.cache_data(show_spinner="기준 자세와 점수를 비교하는 중 (패치판)…", max_entries=64)
def compare_cached(body_t: tuple, scenario: str, pose_t: tuple, body_model: str):
    sc = get_scenarios()[scenario]
    return compare_with_baselines(BodyParams(*body_t), sc, PoseParams(*pose_t), model=body_model)


def _digest(arr: np.ndarray) -> str:
    return hashlib.sha1(arr.tobytes()).hexdigest() + str(arr.shape)


# ---------------------------------------------------------------------------
# ① 입력
# ---------------------------------------------------------------------------
def collect_frames(mode: str) -> tuple[list[np.ndarray], str | None]:
    """입력 방식별 BGR 프레임 목록과 설명. 체형 직접 입력이면 빈 목록."""
    if mode == "예시 이미지":
        p = _example_image()
        if p is None:
            st.warning("예시 이미지가 없습니다 (ultralytics 미설치). 다른 입력을 고르세요.")
            return [], None
        return [camera.read_image(p)], f"예시 이미지: {p.name}"
    if mode == "이미지 업로드":
        up = st.file_uploader("전신 정면 사진", type=["jpg", "jpeg", "png", "webp"])
        return ([camera.read_image(up.getvalue())], up.name) if up else ([], None)
    if mode == "브라우저 카메라":
        # 위젯 자체가 촬영 전 미리보기와 촬영한 사진을 브라우저에 보여 준다 (streamlit 위젯 특성이라
        # 없앨 수 없다). 촬영이 끝나면 접어 두어 화면에 계속 떠 있지 않게 한다.
        taken = bool(st.session_state.get("browser_shot_taken"))
        with st.expander("브라우저 카메라로 촬영", expanded=not taken):
            shot = st.camera_input("게이트 앞에서 전신이 보이게 정면으로 서서 촬영하세요")
        if (shot is not None) != taken:
            # 상태만 바꾸면 이번 그리기에는 반영되지 않아 사진이 한 번 더 보인다. 바로 다시 그린다.
            st.session_state["browser_shot_taken"] = shot is not None
            st.rerun()
        return ([camera.read_image(shot.getvalue())], "브라우저 카메라") if shot else ([], None)
    if mode == "영상 파일":
        up = st.file_uploader("영상 파일", type=["mp4", "mov", "avi", "mkv"])
        if not up:
            return [], None
        frames = camera.frames_from_video_bytes(up.name, up.getvalue())
        return frames, f"{up.name} ({len(frames)} 프레임 사용)"
    if mode == "로컬 카메라 (OpenCV)":
        st.caption("서버 컴퓨터의 카메라 0번에서 약 1초(15 프레임)를 찍습니다. "
                   "프레임은 추정이 끝나면 버리고 저장하지 않습니다.")
        if st.button("촬영", type="primary"):
            try:
                # 세션에 남기지 않는다. 이 실행 안에서 추정까지 끝내고 결과만 보관한다 (analyze_frames).
                return camera.capture_frames(0, n=15), "로컬 카메라 15 프레임"
            except IOError as e:
                st.error(str(e))
        return [], None
    return [], None


def synthetic_input(height_m: float, body_model: str) -> tuple[np.ndarray, np.ndarray]:
    """합성 마네킹: 몸 모델의 기본 비율 체형을 정면 카메라에 투영한 키포인트 (가짜 입력)."""
    base = default_body(body_model)
    k = height_m / base.height_m
    body = BodyParams(*(k * np.array(list(asdict(base).values()))))
    return pe.synthetic_keypoints(body, PoseParams(), get_scenarios()["default"], model=body_model)


@dataclass
class Analysis:
    """프레임 분석 결과. **원본 프레임을 담지 않는다** — 세션에 남겨도 영상이 남지 않는다."""
    skeleton: np.ndarray                 # 빈 캔버스에 그린 뼈대 (RGB)
    body: BodyParams | None              # 안정화한 추정 체형
    estimate: pe.BodyEstimate | None
    caption: str
    warning: str | None = None
    note: str | None = None


def analyze_frames(frames: list[np.ndarray], desc: str, *, height_m: float | None, seated: bool,
                   calib: str, marker_cm: float, body_model: str) -> Analysis:
    """프레임 → (뼈대 그림, 추정 체형). 프레임은 이 함수 밖으로 나가지 않는다."""
    stab = pe.BodyEstimator(height_m=height_m, seated=seated, window=30,
                            min_frames=1 if len(frames) == 1 else 3, profile=body_model)
    size = (frames[-1].shape[1], frames[-1].shape[0])
    last_det, estimate = None, None
    for img in frames:
        det = detect_cached(_digest(img), img)
        if calib != "키 입력" and not seated:
            scale = camera.detect_marker_scale(img, marker_cm / 100.0)
            if scale is None:
                continue
            stab.kw["scale_m_per_px"] = scale
        if det is not None:
            estimate = stab.add(det.keypoints, det.conf, det.image_size)
            last_det = det
    warning = None
    if last_det is None:
        warning = ("바닥 마커나 사람이 보이지 않습니다. 마커를 발 옆에 두거나 키 입력으로 바꾸세요."
                   if calib != "키 입력" and not seated else
                   "사람을 찾지 못했습니다. 전신이 보이게 다시 찍어 주세요.")
    return Analysis(
        skeleton=camera.draw_skeleton_only(last_det, image_size=size),
        body=stab.body(), estimate=estimate, caption=desc, warning=warning,
        note=(f"프레임 {stab.n_frames}개 중 {len(stab.bodies)}개 성공, 중앙값 사용"
              if len(frames) > 1 else None))


# ---------------------------------------------------------------------------
# 본문
# ---------------------------------------------------------------------------
def body_editor(estimated: BodyParams | None, key: str, body_model: str) -> BodyParams:
    """추정 체형을 숫자 입력으로 보여 주고 수정을 받는다. 추정이 없으면 몸 모델의 기본 체형."""
    base = estimated or default_body(body_model)
    cols = st.columns(5)
    vals = {}
    for col, f in zip(cols, fields(BodyParams)):
        vals[f.name] = col.number_input(BODY_LABELS[f.name], min_value=0.1, max_value=2.5,
                                        value=float(getattr(base, f.name)), step=0.01,
                                        format="%.2f", key=f"{key}_{f.name}")
    return BodyParams(**vals)


def main() -> None:
    st.set_page_config(page_title="AIRIS v2 자세 안내", page_icon="🌬️", layout="wide")
    q = st.query_params
    scenarios = get_scenarios()

    st.title("AIRIS v2 · 에어샤워 자세 안내")
    st.caption("퓨리움 PURIUM-10000-P 기준 부스(슬롯 바 12개). 점수는 미보정 패치판 시뮬레이터 값이라 "
               "절대 제거율이 아니라 기준 자세 대비 **상대 개선율**로만 봅니다.")

    # ---------------- 사이드바 ----------------
    with st.sidebar:
        st.header("① 입력")
        default_mode = 0 if q.get("demo") else 1
        mode = st.radio("입력 방식", INPUT_MODES, index=default_mode)
        st.divider()
        st.header("거리 보정")
        calib = st.radio("스케일", ["키 입력", "바닥 마커 (ArUco)"], horizontal=True)
        height_cm = st.number_input("키 (cm)", 100.0, 230.0, 175.0, 0.5,
                                    disabled=calib != "키 입력")
        marker_cm = st.number_input("마커 한 변 (cm)", 5.0, 100.0, 20.0, 0.5,
                                    disabled=calib == "키 입력")
        st.caption("휠체어 시나리오(앉은 자세)는 키를 입력해야 합니다.")
        st.divider()
        st.header("몸 모델")
        model_labels = list(BODY_MODELS)
        body_model = BODY_MODELS[st.radio(
            "몸 모델", model_labels, index=1 if q.get("model") == "capsule" else 0,
            label_visibility="collapsed")]
        st.caption("시뮬레이션 몸. 추천 표는 메시판 k14 E4 결과이고, 부스 안 판정·점수는 고른 몸으로 다시 잽니다 "
                   "(메시 2,000/m², 캡슐 400/m²).")
        st.divider()
        st.header("추천 모델")
        for a in artifacts_cached():
            if not a.exists:
                st.caption(f"❌ {a.name}: 없음 → 고정 후보표(E4)로 안내")
            elif a.message:
                st.caption(f"⚠️ {a.name}: 읽지 못함 ({a.message})")
            else:
                mark = {True: "✅", False: "⚠️", None: "❔"}[a.config_ok]
                note = {True: "설정 일치", False: "학습 때와 설정이 다름",
                        None: "설정 비교 불가"}[a.config_ok]
                st.caption(f"{mark} {a.name}: 학습 커밋 {a.commit or '?'} · {note}")
            st.caption(f"&nbsp;&nbsp;`{a.path}`", unsafe_allow_html=True)
        st.caption("읽기 전용입니다. 산출물은 추천 모델 담당이 만들고, 물리 설정이 바뀌면 다시 학습해야 합니다. "
                   "**산출물을 새로 설치했으면 대시보드를 다시 시작하세요** (추천 결과는 재시작 전까지 캐시됩니다).")
        st.divider()
        st.caption("**개인정보**: 화면에는 뼈대만 그립니다. 카메라·업로드 영상은 자세 추정에만 쓰고 저장하지 "
                   "않으며, 업로드한 영상 파일은 프레임을 뽑은 뒤 바로 지웁니다. 브라우저 카메라는 위젯이 "
                   "촬영 전 미리보기를 보여 주는데, 그 화면은 브라우저 안에만 있습니다. "
                   "시나리오(임산부·휠체어)는 영상으로 판별하지 않고 사용자가 직접 고릅니다.")

    # 첫 사용자가 모델 로딩을 기다리지 않게 미리 올려 둔다 (서버당 한 번, 화면에는 준비 중 표시).
    warm_up(body_model)

    scen_names = list(SCENARIO_LABELS)
    scen_default = q.get("scenario", "default")
    if "scenario" not in st.session_state:
        st.session_state["scenario"] = scen_default if scen_default in scen_names else "default"
    seated = st.session_state["scenario"] == "wheelchair"
    height_m = height_cm / 100.0 if calib == "키 입력" or seated else None

    # ---------------- ① 입력 → ② 체형 ----------------
    st.subheader("① 입력  →  ② 뼈대와 추정 체형")
    left, right = st.columns([1.1, 1.0], gap="large")
    estimate: pe.BodyEstimate | None = None
    stab_body: BodyParams | None = None
    with left:
        if mode == "체형 직접 입력":
            st.info("카메라 없이 체형 값을 직접 입력합니다 (오른쪽).")
        elif mode == "합성 마네킹 (카메라 없이)":
            kp, conf = synthetic_input(height_cm / 100.0, body_model)
            det = camera.PoseDetection(kp, conf, np.r_[kp.min(axis=0) - 20, kp.max(axis=0) + 20],
                                       (1280, 720))
            st.image(camera.draw_skeleton_only(det),
                     caption="합성 마네킹 키포인트 (몸 모델 관절을 정면 카메라에 투영한 가짜 입력)", **WIDE)
            estimate = pe.estimate_body(kp, conf, height_m=height_cm / 100.0, seated=False,
                                        profile=body_model)
            stab_body = estimate.body
        else:
            frames, desc = collect_frames(mode)
            if frames:
                try:
                    pose_model = get_pose_model()
                except Exception as e:                     # ultralytics 미설치·다운로드 실패
                    st.error(f"포즈 모델을 불러오지 못했습니다: {e}. '체형 직접 입력'을 쓰세요.")
                    pose_model = None
                if pose_model is not None:
                    # 프레임은 여기서 끝난다. 세션에는 결과(Analysis)만 남는다.
                    st.session_state[ANALYSIS_KEY] = (mode, analyze_frames(
                        frames, desc, height_m=height_m, seated=seated, calib=calib,
                        marker_cm=marker_cm, body_model=body_model))
            saved = st.session_state.get(ANALYSIS_KEY)
            done: Analysis | None = saved[1] if saved and saved[0] == mode else None
            if done is not None:
                st.image(done.skeleton, caption=done.caption, **WIDE)
                st.caption("화면에 띄우는 것은 뼈대뿐입니다. 원본 영상은 추정에만 쓰고 저장하지 않습니다.")
                if done.warning:
                    st.warning(done.warning)
                if done.note:
                    st.caption(done.note)
                estimate, stab_body = done.estimate, done.body
            else:
                st.info("입력을 기다리는 중입니다.")
    with right:
        if estimate is not None:
            if estimate.ok and stab_body is not None:
                st.success(f"체형 추정 완료 (스케일 {estimate.scale_m_per_px * 1000:.2f} mm/px). "
                           "틀린 값은 아래에서 고치세요.")
            else:
                st.warning(estimate.message)
            if estimate.missing:
                st.caption("안 보이는 키포인트: " + ", ".join(estimate.missing))
        # 추정값이 바뀌면 입력칸을 새 값으로 다시 채운다 (키에 추정값을 넣는다)
        key = f"body_{body_model}_" + ("_".join(f"{v:.4f}" for v in asdict(stab_body).values())
                                       if stab_body else "none")
        body = body_editor(stab_body, key, body_model)
        ratio = pe.profile_for(body_model).torso_depth_per_height
        st.caption(f"몸통 두께는 정면 사진으로 잴 수 없어 키 × {ratio:.3f}로 둡니다. "
                   "어깨 너비·팔·다리는 관절 중심 기준(어깨 관절 간격, 상완+전완, 고관절 높이)입니다.")

    # ---------------- ③ 시나리오 ----------------
    st.subheader("③ 시나리오")
    scen = st.radio("시나리오", scen_names, format_func=SCENARIO_LABELS.get, horizontal=True,
                    key="scenario", label_visibility="collapsed")
    scenario = scenarios[scen]
    try:
        build_body(body, PoseParams(), scenario, patches_per_m2=100, model=body_model)
    except ValueError as e:
        st.error(f"이 체형으로는 마네킹을 만들 수 없습니다: {e}")
        st.stop()

    # ---------------- ④ 추천 자세 ----------------
    st.subheader("④ 추천 자세")
    rec = recommend_cached(tuple(asdict(body).values()), scen, body_model)
    rows = compare_cached(tuple(asdict(body).values()), scen, tuple(asdict(rec.pose).values()),
                          body_model)
    by = {r.name: r for r in rows}
    rec_row = by["추천"]

    guide, metrics = st.columns([1.2, 1.0], gap="large")
    with guide:
        st.markdown(f"#### {rec.label}")
        for line in pose_instructions(rec.pose, scenario):
            st.markdown(f"- {line}")
        st.caption(f"자세 값: {pose_label(rec.pose)}")
        st.caption(f"출처: {rec.source}" + (" (추천 모델 산출물이 없을 때 쓰는 고정 후보표)"
                                              if rec.source.startswith("stub") else ""))
        # 수치는 recommend 의 상수에서 만든다 (물리가 바뀌어 기준선을 다시 재면 문구도 따라 바뀐다).
        b0_p, _, b2_p = MESH_E4_BASELINES["pregnant"]
        st.caption(f"메시판에서는 세 시나리오 모두 **옆으로 돌아선 만세**가 최적입니다"
                   f"(임산부 포함, {MESH_E4_HANDS_UP_SEEDS[0]}/{MESH_E4_HANDS_UP_SEEDS[1]} 시드). "
                   f"휠체어는 약 36° 회전. 다만 임산부는 *정면* 만세라면 기본 자세보다 낮습니다"
                   f"(기준선 {b2_p:.3f} < {b0_p:.3f}) — 이득은 만세 자체가 아니라 몸을 옆으로 돌리는 데서 옵니다.")
        for note in rec.notes:
            st.info(note)
        over = rec.elapsed_s > RESPONSE_BUDGET_S
        st.caption(("⚠️ " if over else "") + f"응답 시간 {rec.elapsed_s:.2f} s "
                   f"(목표 {RESPONSE_BUDGET_S:.1f} s 이내" +
                   (", 후보 재채점이 많아 넘었습니다" if over else "") + "). 같은 체형·시나리오는 캐시합니다.")
        if rec.stats:
            with st.expander(f"모델 후보 {sum(s.n for s in rec.stats)}개 (출처별 최고 점수)"):
                st.dataframe([{"출처": s.label, "후보 수": s.n, "부스 안": s.n_feasible,
                               "최고 점수": None if s.best is None else round(s.best, 4)}
                              for s in rec.stats], **WIDE, hide_index=True)
                st.caption("모든 후보를 이 체형·몸 모델의 패치판으로 다시 채점해 가장 높은 것을 고릅니다. "
                           "고정 후보표(E4)도 같은 풀에 있어 추천이 표보다 나빠지지 않습니다.")
    with metrics:
        cols = st.columns(3)
        for col, name in zip(cols, ("B0 기본", "B1 몸 회전", "B2 만세")):
            imp = improvement(rec_row, by[name])
            col.metric(f"{name} 대비", "불가" if imp is None else f"{imp:+.0%}",
                       help=f"{name} 점수 {by[name].score:.3f} → 추천 {rec_row.score:.3f}")
        st.caption("B0 = 안내 없이 통과, B1 = 제조사 안내 '몸 회전'(yaw 0~330° 평균), B2 = 만세. "
                   "개선율 = 추천 점수 / 기준 점수 − 1.")

    booth = load_nozzle_layout()["booth"]
    density = scoring_patches_per_m2(body_model)            # 채점과 같은 밀도로 그려야 패치 색이 맞는다
    cmax = max(float(np.max(r.result.extra["removal"])) for r in (rec_row, by["B0 기본"])
               if r.result is not None and not r.infeasible)
    f1, f2 = st.columns(2)
    with f1:
        b0 = by["B0 기본"]
        fig = figure_from_pose(body, PoseParams(), scenario, result=b0.result, booth=booth,
                               title=f"B0 기본 자세 · 점수 {b0.score:.3f}", cmin=0.0, cmax=cmax,
                               height=560, model=body_model, patches_per_m2=density)
        st.plotly_chart(fig, **WIDE)
    with f2:
        fig = figure_from_pose(body, rec.pose, scenario, result=rec_row.result, booth=booth,
                               title=f"추천: {rec.label} · 점수 {rec_row.score:.3f}", cmin=0.0,
                               cmax=cmax, height=560, model=body_model, patches_per_m2=density)
        st.plotly_chart(fig, **WIDE)
    st.caption("색 = 패치별 제거율 (두 그림 같은 색 범위). 파란 선 = 퓨리움 슬롯 바 12개, 점선 = 분사 방향.")

    table = []
    for r in rows:
        imp = improvement(rec_row, r) if r.name != "추천" else None
        table.append({
            "조건": r.name,
            "점수": None if r.infeasible else round(r.score, 4),
            "추천의 개선율": "" if r.name == "추천" else ("불가" if imp is None else f"{imp:+.0%}"),
            "불편도": round(r.discomfort, 3),
            **{PART_LABELS[p]: round(float(v), 3) for p, v in zip(PART_NAMES, r.removal_by_part)},
            "평가 자세 수": f"{r.n_feasible}/{r.n_poses}",
        })
    with st.expander("점수 표 (부위별 값은 시뮬레이터 내부 값, 순위 비교용)"):
        st.dataframe(table, **WIDE, hide_index=True)

    # ---------------- ⑤ 입자 애니메이션 ----------------
    st.subheader("⑤ 입자 애니메이션")
    out_dir = ROOT / "outputs"
    dumps = sorted(p.parent.name for p in out_dir.glob("*/frames") if any(p.glob("*.npz")))
    choice = st.selectbox("덤프", ["(보지 않음)", "합성 프레임 미리보기 (가짜 궤적)"] + dumps)
    if choice == "합성 프레임 미리보기 (가짜 궤적)":
        state = build_body(body, rec.pose, scenario, patches_per_m2=400, model=body_model)
        frames = anim.synthetic_frames(state, booth, n_particles=1500)
        st.plotly_chart(anim.animation(frames, state, booth, title="합성 프레임 (물리 계산 아님)"),
                        **WIDE)
    elif choice in dumps:
        frames = anim.load_frames(out_dir / choice)
        cands = anim.candidates_in(frames)
        cand = st.selectbox("후보", cands) if len(cands) > 1 else cands[0]
        meta = out_dir / choice / "render_meta.json"
        if meta.exists():                      # render_frames.py --simulate 가 남긴 체형·자세
            m = json.loads(meta.read_text(encoding="utf-8"))
            poses = [PoseParams(**p) for p in m["poses"]]
            state = build_body(BodyParams(**m["body"]) if m.get("body") else None,
                               poses[cand] if cand < len(poses) else poses[0],
                               scenarios[m["scenario"]], patches_per_m2=400, model=m.get("model"))
            st.caption(f"덤프 출처: {m.get('source', '?')} · 시나리오 {m['scenario']} · "
                       f"몸 모델 {m.get('model') or '설정 파일 기본값'}")
        else:
            st.caption("덤프를 만든 자세 정보(render_meta.json)가 없어 몸은 현재 체형·추천 자세로 그립니다.")
            state = build_body(body, rec.pose, scenario, patches_per_m2=400, model=body_model)
        st.plotly_chart(anim.animation(frames, state, booth, candidate=cand, title=choice),
                        **WIDE)


main()
