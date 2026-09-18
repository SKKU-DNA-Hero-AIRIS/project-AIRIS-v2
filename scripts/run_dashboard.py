"""AIRIS v2 실시간 안내 대시보드 (Streamlit + plotly). 소유자: E. (`docs/tracks/E_realtime.md` 단계 4)

    streamlit run scripts/run_dashboard.py
    streamlit run scripts/run_dashboard.py -- --image path/to/photo.jpg   # 시작 이미지 지정

화면 순서
    ① 입력 (예시 이미지 · 이미지 업로드 · 브라우저 카메라 · 영상 파일 · 로컬 카메라 · 합성 마네킹 · 직접 입력)
    ② 키포인트 오버레이와 추정 체형 5개 (수정 가능)
    ③ 시나리오 선택 (영상으로 판별하지 않는다)
    ④ 추천 자세 3D 마네킹 + 기준 자세(B0·B1·B2)와 점수 비교 (D PatchEvaluator, 400/m²)
    ⑤ 입자 애니메이션 (A 덤프가 있으면, 없으면 합성 프레임 미리보기)

URL 쿼리 `?demo=1`이면 예시 이미지로 바로 시작한다 (스크린샷용). `?scenario=wheelchair` 등으로 시나리오 지정.
v1(project-AIRIS-MVP `realtime_vision_app.py`)의 Streamlit 틀(사이드바 입력 선택, 모델 캐시)을 따랐다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                                    # noqa: E402
import streamlit as st                                                # noqa: E402

from airis.realtime import camera, pose_estimate as pe               # noqa: E402
from airis.realtime.recommend import (compare_with_baselines, improvement,  # noqa: E402
                                      pose_instructions, recommend)
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


@st.cache_data(show_spinner="기준 자세와 점수를 비교하는 중 (패치판)…", max_entries=64)
def compare_cached(body_t: tuple, scenario: str, pose_t: tuple):
    sc = get_scenarios()[scenario]
    return compare_with_baselines(BodyParams(*body_t), sc, PoseParams(*pose_t))


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
        shot = st.camera_input("게이트 앞에서 전신이 보이게 정면으로 서서 촬영하세요")
        return ([camera.read_image(shot.getvalue())], "브라우저 카메라") if shot else ([], None)
    if mode == "영상 파일":
        up = st.file_uploader("영상 파일", type=["mp4", "mov", "avi", "mkv"])
        if not up:
            return [], None
        tmp = ROOT / "outputs" / "dashboard_upload" / up.name
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(up.getvalue())
        frames = list(camera.iter_video_frames(tmp, every_n=5, max_frames=30))
        return frames, f"{up.name} ({len(frames)} 프레임 사용)"
    if mode == "로컬 카메라 (OpenCV)":
        st.caption("서버 컴퓨터의 카메라 0번에서 약 1초(15 프레임)를 찍습니다. 영상은 저장하지 않습니다.")
        if st.button("촬영", type="primary"):
            try:
                st.session_state["local_frames"] = camera.capture_frames(0, n=15)
            except IOError as e:
                st.error(str(e))
        frames = st.session_state.get("local_frames", [])
        return frames, (f"로컬 카메라 {len(frames)} 프레임" if frames else None)
    return [], None


def synthetic_input(height_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """합성 마네킹: 기본 비율 체형을 정면 카메라에 투영한 키포인트 (가짜 입력)."""
    s = height_m / 1.70
    body = BodyParams(height_m, 0.42 * s, 0.22 * s, 0.62 * s, 0.85 * s)
    kp, conf = pe.synthetic_keypoints(body, PoseParams(), get_scenarios()["default"])
    canvas = np.full((720, 1280, 3), 245, np.uint8)
    return canvas, kp, conf


# ---------------------------------------------------------------------------
# 본문
# ---------------------------------------------------------------------------
def body_editor(estimated: BodyParams | None, key: str) -> BodyParams:
    """추정 체형을 숫자 입력으로 보여 주고 수정을 받는다."""
    base = estimated or BodyParams()
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
        st.caption("카메라 영상은 저장하지 않습니다. 시나리오(임산부·휠체어)는 영상으로 판별하지 않고 "
                   "사용자가 직접 고릅니다.")

    scen_names = list(SCENARIO_LABELS)
    scen_default = q.get("scenario", "default")
    if "scenario" not in st.session_state:
        st.session_state["scenario"] = scen_default if scen_default in scen_names else "default"
    seated = st.session_state["scenario"] == "wheelchair"
    height_m = height_cm / 100.0 if calib == "키 입력" or seated else None

    # ---------------- ① 입력 → ② 체형 ----------------
    st.subheader("① 입력  →  ② 키포인트와 추정 체형")
    left, right = st.columns([1.1, 1.0], gap="large")
    estimate: pe.BodyEstimate | None = None
    stab_body: BodyParams | None = None
    with left:
        if mode == "체형 직접 입력":
            st.info("카메라 없이 체형 값을 직접 입력합니다 (오른쪽).")
        elif mode == "합성 마네킹 (카메라 없이)":
            canvas, kp, conf = synthetic_input(height_cm / 100.0)
            det = camera.PoseDetection(kp, conf, np.r_[kp.min(axis=0) - 20, kp.max(axis=0) + 20],
                                       (canvas.shape[1], canvas.shape[0]))
            st.image(camera.draw_pose(canvas, det), caption="합성 마네킹 키포인트 (B 관절을 정면 카메라에 투영한 가짜 입력)",
                     **WIDE)
            estimate = pe.estimate_body(kp, conf, height_m=height_cm / 100.0, seated=False)
            stab_body = estimate.body
        else:
            frames, desc = collect_frames(mode)
            if frames:
                try:
                    model = get_pose_model()
                except Exception as e:                     # ultralytics 미설치·다운로드 실패
                    st.error(f"포즈 모델을 불러오지 못했습니다: {e}. '체형 직접 입력'을 쓰세요.")
                    model = None
                if model is not None:
                    stab = pe.BodyEstimator(height_m=height_m, seated=seated, window=30,
                                            min_frames=1 if len(frames) == 1 else 3)
                    last_img, last_det = frames[-1], None
                    for img in frames:
                        det = detect_cached(_digest(img), img)
                        scale = None
                        if calib != "키 입력" and not seated:
                            scale = camera.detect_marker_scale(img, marker_cm / 100.0)
                            if scale is None:
                                continue
                            stab.kw["scale_m_per_px"] = scale
                        if det is not None:
                            estimate = stab.add(det.keypoints, det.conf, det.image_size)
                            last_img, last_det = img, det
                    st.image(camera.draw_pose(last_img, last_det), caption=desc,
                             **WIDE)
                    if last_det is None:
                        if calib != "키 입력" and not seated:
                            st.warning("바닥 마커나 사람이 보이지 않습니다. 마커를 발 옆에 두거나 키 입력으로 바꾸세요.")
                        else:
                            st.warning("사람을 찾지 못했습니다. 전신이 보이게 다시 찍어 주세요.")
                    stab_body = stab.body()
                    if len(frames) > 1:
                        st.caption(f"프레임 {stab.n_frames}개 중 {len(stab.bodies)}개 성공, 중앙값 사용")
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
        key = "body_" + ("_".join(f"{v:.4f}" for v in asdict(stab_body).values())
                         if stab_body else "none")
        body = body_editor(stab_body, key)
        st.caption("몸통 두께는 정면 사진으로 잴 수 없어 키 × 0.13으로 둡니다.")

    # ---------------- ③ 시나리오 ----------------
    st.subheader("③ 시나리오")
    scen = st.radio("시나리오", scen_names, format_func=SCENARIO_LABELS.get, horizontal=True,
                    key="scenario", label_visibility="collapsed")
    scenario = scenarios[scen]
    try:
        build_body(body, PoseParams(), scenario, patches_per_m2=100)
    except ValueError as e:
        st.error(f"이 체형으로는 마네킹을 만들 수 없습니다: {e}")
        st.stop()

    # ---------------- ④ 추천 자세 ----------------
    st.subheader("④ 추천 자세")
    rec = recommend(body, scenario)
    rows = compare_cached(tuple(asdict(body).values()), scen, tuple(asdict(rec.pose).values()))
    by = {r.name: r for r in rows}
    rec_row = by["추천"]

    guide, metrics = st.columns([1.2, 1.0], gap="large")
    with guide:
        st.markdown(f"#### {rec.label}")
        for line in pose_instructions(rec.pose, scenario):
            st.markdown(f"- {line}")
        st.caption(f"자세 값: {pose_label(rec.pose)}")
        st.caption(f"출처: {rec.source}" + (" (회귀 모델이 준비되기 전의 임시 표)" if rec.source.startswith("stub") else ""))
        for note in rec.notes:
            st.info(note)
    with metrics:
        cols = st.columns(3)
        for col, name in zip(cols, ("B0 기본", "B1 몸 회전", "B2 만세")):
            imp = improvement(rec_row, by[name])
            col.metric(f"{name} 대비", "불가" if imp is None else f"{imp:+.0%}",
                       help=f"{name} 점수 {by[name].score:.3f} → 추천 {rec_row.score:.3f}")
        st.caption("B0 = 안내 없이 통과, B1 = 제조사 안내 '몸 회전'(yaw 0~330° 평균), B2 = 만세. "
                   "개선율 = 추천 점수 / 기준 점수 − 1.")

    booth = load_nozzle_layout()["booth"]
    cmax = max(float(np.max(r.result.extra["removal"])) for r in (rec_row, by["B0 기본"])
               if r.result is not None and not r.infeasible)
    f1, f2 = st.columns(2)
    with f1:
        b0 = by["B0 기본"]
        fig = figure_from_pose(body, PoseParams(), scenario, result=b0.result, booth=booth,
                               title=f"B0 기본 자세 · 점수 {b0.score:.3f}", cmin=0.0, cmax=cmax,
                               height=560)
        st.plotly_chart(fig, **WIDE)
    with f2:
        fig = figure_from_pose(body, rec.pose, scenario, result=rec_row.result, booth=booth,
                               title=f"추천: {rec.label} · 점수 {rec_row.score:.3f}", cmin=0.0,
                               cmax=cmax, height=560)
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
        state = build_body(body, rec.pose, scenario, patches_per_m2=400)
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
            state = build_body(BodyParams(**m["body"]), poses[cand] if cand < len(poses) else poses[0],
                               scenarios[m["scenario"]], patches_per_m2=400)
            st.caption(f"덤프 출처: {m.get('source', '?')} · 시나리오 {m['scenario']}")
        else:
            st.caption("덤프를 만든 자세 정보(render_meta.json)가 없어 몸은 현재 체형·추천 자세로 그립니다.")
            state = build_body(body, rec.pose, scenario, patches_per_m2=400)
        st.plotly_chart(anim.animation(frames, state, booth, candidate=cand, title=choice),
                        **WIDE)


main()
