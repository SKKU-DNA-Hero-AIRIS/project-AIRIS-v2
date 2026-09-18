"""COCO 17 키포인트 → 체형 5개 (`BodyParams`). 소유자: E. (`docs/tracks/E_realtime.md` 단계 3)

정면 카메라 한 대로 선 사람을 찍은 키포인트(px)에서 체형을 추정한다. YOLO 호출은
`camera.py`에 있고 여기는 순수 기하 계산이라 테스트에서 카메라·모델 없이 돈다.

    est = estimate_body(kp, conf, height_m=1.72)          # 키 입력으로 스케일 보정 (1차)
    est = estimate_body(kp, conf, scale_m_per_px=s)       # 바닥 마커로 스케일 보정 (2차)
    est.body     # BodyParams 또는 None (est.message 에 안내 문구)

측정 정의 (몸 모델의 BodyParams 정의와 같은 의미가 되게 맞췄다. 관절 중심 기준, docs/interfaces.md)
- 키 = 머리 꼭대기 − 바닥. 코(없으면 눈·귀)와 발목 키포인트 사이 거리에 몸 모델 비율
  보정(머리 꼭대기→코, 발목 키포인트→바닥)을 더한다.
- 어깨 너비 = 좌우 어깨 키포인트 거리 (어깨 관절 간격. 메시: 좌우 upperarm01 머리).
- 팔 길이 = 어깨–팔꿈치 + 팔꿈치–손목 (상완 + 전완 구간 합, 자세 불변).
- 다리 길이 = (엉덩이–무릎 + 무릎–발목) × 다리 수직 비 + 발목 높이 (고관절 높이. 메시 휴지 자세는 다리를
  약간 벌려 구간 합이 수직 높이보다 1.2% 길다).
- 몸통 두께는 정면으로 못 재므로 `비율 × 키` (메시 0.114 = 0.194/1.70, 캡슐 0.13).
- 구간(상완·전완·허벅지·정강이)마다 좌우 중 긴 쪽 (단축은 길이를 줄이기만 한다). 한쪽 신뢰도가
  낮으면 반대쪽만 쓴다. 필요한 점이 양쪽 다 없으면 None + 안내 문구.
- 앉은 자세(`seated=True`, 휠체어)는 다리가 카메라 쪽으로 접혀 길이를 못 재므로 다리 길이는
  `SEATED_LEG_PER_HEIGHT × 키`, 스케일은 코–엉덩이 거리로 잡는다. 키는 직접 입력해야 한다.

시나리오(임산부·휠체어)는 영상으로 판별하지 않는다. 사용자가 고른다 (README 1절).

몸 모델별 비율 (`KeypointProfile`):
- `capsule`: 인체 측정 평균에서 잡은 추정값 (`CAPSULE_PROFILE`, 아래 상수). 캡슐 마네킹 관절에 코·눈·귀·
  발목 키포인트를 이 비율로 붙여 합성 키포인트를 만든다.
- `mesh`: MakeHuman 메시(메시 기본 체형, 키 1.70)에서 잰 값 (`mesh_profile()`). 코끝·귀 정점, 눈·발목·
  고관절 뼈 머리. 합성 키포인트도 체형·자세를 입힌 메시에서 같은 점을 투영한다.
- 기본값은 `configs/physics.yaml` `body.model` (지금 capsule. 전역 기본값 전환 5단계에서 mesh).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from functools import lru_cache

import numpy as np

from ..sim.body import HEAD_RADIUS_PER_HEIGHT, joint_positions
from ..sim.types import BodyParams, PoseParams, Scenario

# ---------------------------------------------------------------------------
# COCO 키포인트 번호 (left/right 는 찍힌 사람 기준)
# ---------------------------------------------------------------------------
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
)
#: COCO 뼈대 (project-AIRIS-MVP `posture_analysis.py` SKELETON 에서 가져옴)
SKELETON = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
)

# ---------------------------------------------------------------------------
# 인체 비율 (키 대비). 추정값, 실측 보정 대상.
# ---------------------------------------------------------------------------
#: 머리 꼭대기 → 코끝 (코끝 높이 ≈ 0.91 H)
NOSE_BELOW_TOP_PER_HEIGHT = 0.09
#: 머리 꼭대기 → 눈 (눈 높이 ≈ 0.935 H). B 마네킹에서는 머리 구 중심 높이와 같다.
EYE_BELOW_TOP_PER_HEIGHT = HEAD_RADIUS_PER_HEIGHT
#: 머리 꼭대기 → 귓구멍 (≈ 0.925 H)
EAR_BELOW_TOP_PER_HEIGHT = 0.075
#: 바닥 → 발목 키포인트(복사뼈) (≈ 0.039 H). B 마네킹의 발목 관절은 바닥(z = 0)이다.
ANKLE_ABOVE_FLOOR_PER_HEIGHT = 0.039
#: 몸통 두께 / 키. BodyParams 기본값 0.22 / 1.70 과 같다 (C 샘플러와 같은 비율).
TORSO_DEPTH_PER_HEIGHT = 0.13
#: 앉은 자세 다리 길이 / 키 (BodyParams 기본값 0.85 / 1.70).
SEATED_LEG_PER_HEIGHT = 0.5



@dataclass(frozen=True)
class KeypointProfile:
    """몸 모델별 키포인트 ↔ 체형 비율 (모두 키 대비)."""
    name: str
    nose_below_top: float          # 머리 꼭대기 → 코끝
    eye_below_top: float           # 머리 꼭대기 → 눈
    ear_below_top: float           # 머리 꼭대기 → 귀
    ankle_above_floor: float       # 바닥 → 발목 키포인트
    torso_depth_per_height: float  # 가슴 두께 기본값 (정면 사진으로 못 잰다)
    seated_leg_per_height: float   # 고관절 높이 (앉은 자세 다리 길이 기본값)
    leg_vertical_ratio: float = 1.0  # 고관절→발목 수직 / (허벅지 + 정강이) (휴지 자세 다리 벌림)


CAPSULE_PROFILE = KeypointProfile(
    "capsule", NOSE_BELOW_TOP_PER_HEIGHT, EYE_BELOW_TOP_PER_HEIGHT, EAR_BELOW_TOP_PER_HEIGHT,
    ANKLE_ABOVE_FLOOR_PER_HEIGHT, TORSO_DEPTH_PER_HEIGHT, SEATED_LEG_PER_HEIGHT, 1.0)

#: 메시 귀 정점을 찾는 눈 높이 기준 띠 (m, 키 1.70 축척)
_EAR_BAND_M = (-0.06, 0.01)


@lru_cache(maxsize=1)
def mesh_landmarks() -> dict[str, int]:
    """MakeHuman 원본 메시의 얼굴 기준 정점 번호 (체형·자세를 입혀도 같은 번호).

    코끝 = 머리 부위·가운데(|y| < 1 cm) 정점 중 가장 앞(x 최대). 귀 = 머리 부위 중 눈 높이 −6 ~ +1 cm 띠에서
    가장 바깥(y 최대·최소) 정점. 눈은 뼈 머리(eye.L/R)를 쓴다.
    """
    from ..sim import human_mesh as hm
    from ..sim.types import PART_NAMES

    mesh = hm.cached_makehuman()
    s = hm.MESH_DEFAULT_BODY.height_m / mesh.height_m
    v = mesh.vertices * s
    head = np.flatnonzero(hm.vertex_parts(mesh) == PART_NAMES.index("head"))
    mid = head[np.abs(v[head, 1]) < 0.01]
    eye_z = mesh.bone_head[mesh.bone("eye.L"), 2] * s
    band = head[(v[head, 2] > eye_z + _EAR_BAND_M[0]) & (v[head, 2] < eye_z + _EAR_BAND_M[1])]
    return {"nose": int(mid[np.argmax(v[mid, 0])]),
            "ear_L": int(band[np.argmax(v[band, 1])]), "ear_R": int(band[np.argmin(v[band, 1])])}


@lru_cache(maxsize=1)
def mesh_profile() -> KeypointProfile:
    """MakeHuman 메시(메시 기본 체형, 휴지 자세)에서 잰 비율. 약 0.095 / 0.0725 / 0.075 / 0.043 / 0.114 / 0.52."""
    from ..sim import human_mesh as hm

    mesh = hm.cached_makehuman()
    body = hm.MESH_DEFAULT_BODY
    s = body.height_m / mesh.height_m
    v = mesh.vertices * s
    top, floor = v[:, 2].max(), v[:, 2].min()
    H = top - floor
    lm = mesh_landmarks()
    head = lambda name: mesh.bone_head[mesh.bone(name)] * s        # noqa: E731
    hip, knee, ankle = head("upperleg01.L"), head("lowerleg01.L"), head("foot.L")
    seg = np.linalg.norm(knee - hip) + np.linalg.norm(ankle - knee)
    return KeypointProfile(
        "mesh",
        nose_below_top=float((top - v[lm["nose"], 2]) / H),
        eye_below_top=float((top - head("eye.L")[2]) / H),
        ear_below_top=float((top - 0.5 * (v[lm["ear_L"], 2] + v[lm["ear_R"], 2])) / H),
        ankle_above_floor=float((ankle[2] - floor) / H),
        torso_depth_per_height=float(body.torso_depth_m / body.height_m),
        seated_leg_per_height=float(body.leg_length_m / body.height_m),
        leg_vertical_ratio=float((hip[2] - ankle[2]) / seg),
    )


def profile_for(model: str | KeypointProfile | None = None) -> KeypointProfile:
    """몸 모델 이름(또는 프로파일) → 프로파일. None 이면 `configs/physics.yaml` `body.model`."""
    if isinstance(model, KeypointProfile):
        return model
    if model is None:
        from ..sim.body import _configured_body_model
        model = _configured_body_model()
    if model == "capsule":
        return CAPSULE_PROFILE
    if model == "mesh":
        return mesh_profile()
    raise ValueError(f"몸 모델은 capsule | mesh: {model!r}")


#: 신뢰도 기본 문턱 (v1 `posture_analysis._visible` 의 0.35 와 같은 수준)
MIN_CONF = 0.35

#: 추정값이 이 범위를 벗어나면 키포인트 오류로 보고 None (대시보드에서 직접 수정)
PLAUSIBLE_RANGE_M: dict[str, tuple[float, float]] = {
    "height_m": (0.9, 2.3),
    "shoulder_width_m": (0.2, 0.7),
    "torso_depth_m": (0.1, 0.4),
    "arm_length_m": (0.3, 1.0),
    "leg_length_m": (0.35, 1.3),
}


@dataclass
class BodyEstimate:
    """추정 결과. `body`가 None이면 `message`가 사용자 안내 문구다."""
    body: BodyParams | None
    message: str
    scale_m_per_px: float | None = None
    measured_px: dict[str, float] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.body is not None


# ---------------------------------------------------------------------------
# 키포인트 유효성
# ---------------------------------------------------------------------------
def valid_mask(kp: np.ndarray, conf: np.ndarray, image_size: tuple[int, int] | None = None,
               min_conf: float = MIN_CONF) -> np.ndarray:
    """(17,) bool. 신뢰도 ≥ min_conf 이고, 화면 안(image_size = (w, h))이며, (0,0)이 아닌 점.

    ultralytics 는 안 보이는 점을 (0, 0)·낮은 신뢰도로 준다.
    """
    kp = np.asarray(kp, dtype=np.float64).reshape(17, 2)
    conf = np.asarray(conf, dtype=np.float64).reshape(17)
    ok = (conf >= min_conf) & np.all(np.isfinite(kp), axis=1) & ~np.all(kp == 0.0, axis=1)
    if image_size is not None:
        w, h = image_size
        ok &= (kp[:, 0] >= 0) & (kp[:, 0] < w) & (kp[:, 1] >= 0) & (kp[:, 1] < h)
    return ok


def _dist(kp: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(kp[a] - kp[b]))


def _limb_length(kp: np.ndarray, ok: np.ndarray,
                 segments: Sequence[Sequence[tuple[int, int]]]) -> float | None:
    """구간별로 좌우 중 긴 쪽의 합 (px). 예: 팔 = max(좌·우 상완) + max(좌·우 전완).

    카메라 쪽으로 기운 팔다리는 화면에서 짧아지기만 하므로(단축) 긴 쪽이 실제 길이에 가깝다.
    구간마다 보이는 쪽만 쓰므로 왼팔꿈치와 오른손목이 흐려도 잴 수 있다. 어느 구간이 양쪽 다
    안 보이면 None.
    """
    total = 0.0
    for pairs in segments:
        lengths = [_dist(kp, a, b) for a, b in pairs if ok[a] and ok[b]]
        seg = max(lengths) if lengths else None
        if seg is None:
            return None
        total += seg
    return total


ARM_SEGMENTS = (((L_SHOULDER, L_ELBOW), (R_SHOULDER, R_ELBOW)),
                ((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)))
LEG_SEGMENTS = (((L_HIP, L_KNEE), (R_HIP, R_KNEE)),
                ((L_KNEE, L_ANKLE), (R_KNEE, R_ANKLE)))


def _head_reference(kp: np.ndarray, ok: np.ndarray,
                    profile: KeypointProfile = CAPSULE_PROFILE) -> tuple[np.ndarray, float] | None:
    """(머리 기준점 px, 그 점의 머리 꼭대기 아래 비율). 코 → 두 눈 평균 → 두 귀 평균 순."""
    if ok[NOSE]:
        return kp[NOSE], profile.nose_below_top
    for pair, below in (((L_EYE, R_EYE), profile.eye_below_top),
                        ((L_EAR, R_EAR), profile.ear_below_top)):
        pts = [kp[i] for i in pair if ok[i]]
        if pts:
            return np.mean(pts, axis=0), below
    return None


def _pair_mean_point(kp: np.ndarray, ok: np.ndarray, a: int, b: int) -> np.ndarray | None:
    pts = [kp[i] for i in (a, b) if ok[i]]
    return np.mean(pts, axis=0) if pts else None


# ---------------------------------------------------------------------------
# 추정
# ---------------------------------------------------------------------------
def estimate_body(kp: np.ndarray, conf: np.ndarray, scale_m_per_px: float | None = None, *,
                  height_m: float | None = None, image_size: tuple[int, int] | None = None,
                  seated: bool = False, min_conf: float = MIN_CONF,
                  profile: str | KeypointProfile | None = None) -> BodyEstimate:
    """키포인트 (17,2) px, 신뢰도 (17,) → `BodyEstimate`.

    스케일: `height_m`(사용자 입력 키)이 있으면 그 키로 px→m 를 잡는다(1차). 없으면
    `scale_m_per_px`(바닥 마커, `scale_from_marker`)를 쓴다(2차). 둘 다 없으면 ValueError.
    둘 다 있으면 키 입력을 믿고 스케일을 다시 잡는다.
    `profile`: 몸 모델 (`"mesh"` | `"capsule"` | `KeypointProfile` | None = 설정 파일 `body.model`).
    """
    if height_m is None and scale_m_per_px is None:
        raise ValueError("height_m(키 입력) 또는 scale_m_per_px(마커) 중 하나는 필요하다")
    prof = profile_for(profile)
    kp = np.asarray(kp, dtype=np.float64).reshape(17, 2)
    ok = valid_mask(kp, conf, image_size, min_conf)
    missing = [KEYPOINT_NAMES[i] for i in range(17) if not ok[i]]

    head = _head_reference(kp, ok, prof)
    if head is None:
        return BodyEstimate(None, "얼굴(코·눈·귀)이 보이지 않습니다. 카메라를 정면으로 보고 서 주세요.",
                            missing=missing)
    head_pt, head_below = head
    if not (ok[L_SHOULDER] and ok[R_SHOULDER]):
        return BodyEstimate(None, "양쪽 어깨가 다 보이도록 정면으로 서 주세요.", missing=missing)

    shoulder_px = _dist(kp, L_SHOULDER, R_SHOULDER)
    arm_px = _limb_length(kp, ok, ARM_SEGMENTS)
    if arm_px is None:
        return BodyEstimate(None, "팔(팔꿈치·손목)이 보이도록 팔을 몸에서 조금 떼 주세요.",
                            missing=missing)
    measured = {"shoulder_px": shoulder_px, "arm_px": arm_px}

    if seated:
        hip_pt = _pair_mean_point(kp, ok, L_HIP, R_HIP)
        if hip_pt is None:
            return BodyEstimate(None, "엉덩이(골반)가 보이도록 카메라를 낮춰 주세요.", missing=missing)
        if height_m is None:
            return BodyEstimate(None, "앉은 자세는 키를 직접 입력해야 합니다.", missing=missing)
        span_px = float(np.linalg.norm(hip_pt - head_pt))
        span_ratio = 1.0 - head_below - prof.seated_leg_per_height     # 코 ↔ 골반 관절 / 키
        scale = height_m * span_ratio / span_px
        height = float(height_m)
        leg = prof.seated_leg_per_height * height
        measured["head_hip_px"] = span_px
    else:
        ankle_pt = _pair_mean_point(kp, ok, L_ANKLE, R_ANKLE)
        if ankle_pt is None:
            return BodyEstimate(None, "발목이 화면에 들어오도록 한 걸음 뒤로 물러나 주세요.",
                                missing=missing)
        leg_px = _limb_length(kp, ok, LEG_SEGMENTS)
        if leg_px is None:
            return BodyEstimate(None, "다리(엉덩이·무릎)가 보이도록 서 주세요.", missing=missing)
        span_px = float(np.linalg.norm(ankle_pt - head_pt))
        span_ratio = 1.0 - head_below - prof.ankle_above_floor        # 머리 기준점 ↔ 발목 / 키
        if height_m is not None:
            scale = height_m * span_ratio / span_px
            height = float(height_m)
        else:
            scale = float(scale_m_per_px)
            height = span_px * scale / span_ratio
        leg = leg_px * scale * prof.leg_vertical_ratio + prof.ankle_above_floor * height
        measured.update(head_ankle_px=span_px, leg_px=leg_px)

    body = BodyParams(
        height_m=round(height, 4),
        shoulder_width_m=round(shoulder_px * scale, 4),
        torso_depth_m=round(prof.torso_depth_per_height * height, 4),
        arm_length_m=round(arm_px * scale, 4),
        leg_length_m=round(leg, 4),
    )
    bad = check_plausible(body, prof)
    if bad:
        return BodyEstimate(None, f"추정값이 사람 범위를 벗어났습니다 ({', '.join(bad)}). "
                                  "자세를 바로 하거나 값을 직접 입력해 주세요.",
                            scale, measured, missing)
    return BodyEstimate(body, "추정 완료", scale, measured, missing)


def keypoints_to_body(kp: np.ndarray, conf: np.ndarray, scale_m_per_px: float | None = None,
                      **kw) -> BodyParams | None:
    """`E_realtime.md` 단계 3 시그니처. 실패하면 None (문구가 필요하면 `estimate_body`)."""
    return estimate_body(kp, conf, scale_m_per_px, **kw).body


def check_plausible(body: BodyParams, profile: KeypointProfile | None = None) -> list[str]:
    """범위를 벗어난 필드 이름 목록. 캡슐이면 B의 몸통 길이 조건(다리 < 키 × 0.83)도 본다."""
    bad = [f"{k}={getattr(body, k):.2f}" for k, (lo, hi) in PLAUSIBLE_RANGE_M.items()
           if not lo <= getattr(body, k) <= hi]
    # body_dims: torso_length = H − leg − 2·r_head − neck > 0  (r_head 0.065 H, neck 0.04 H)
    capsule = profile is None or profile.name == "capsule"
    if capsule and body.leg_length_m >= body.height_m * (1.0 - 2 * HEAD_RADIUS_PER_HEIGHT - 0.04):
        bad.append("다리가 키에 비해 너무 김")
    return bad


def median_body(bodies: Sequence[BodyParams]) -> BodyParams:
    """여러 프레임 추정의 필드별 중앙값."""
    if not bodies:
        raise ValueError("추정값이 없다")
    names = [f.name for f in fields(BodyParams)]
    arr = np.array([[getattr(b, n) for n in names] for b in bodies], dtype=np.float64)
    return BodyParams(**{n: round(float(v), 4) for n, v in zip(names, np.median(arr, axis=0))})


class BodyEstimator:
    """프레임별 추정을 모아 중앙값으로 안정화한다 (예: 1초분 = 15~30 프레임).

        est = BodyEstimator(height_m=1.72, window=30)
        for det in detections: est.add(det.keypoints, det.conf, det.image_size)
        body = est.body()        # 유효 프레임 min_frames 개 미만이면 None
    """

    def __init__(self, *, height_m: float | None = None, scale_m_per_px: float | None = None,
                 seated: bool = False, window: int = 30, min_frames: int = 3,
                 min_conf: float = MIN_CONF, profile: str | KeypointProfile | None = None):
        self.kw = dict(height_m=height_m, scale_m_per_px=scale_m_per_px, seated=seated,
                       min_conf=min_conf, profile=profile_for(profile))
        self.window = int(window)
        self.min_frames = int(min_frames)
        self.bodies: list[BodyParams] = []
        self.last: BodyEstimate | None = None
        self.n_frames = 0

    def add(self, kp: np.ndarray, conf: np.ndarray,
            image_size: tuple[int, int] | None = None) -> BodyEstimate:
        est = estimate_body(kp, conf, self.kw["scale_m_per_px"], height_m=self.kw["height_m"],
                            image_size=image_size, seated=self.kw["seated"],
                            min_conf=self.kw["min_conf"], profile=self.kw["profile"])
        self.n_frames += 1
        self.last = est
        if est.body is not None:
            self.bodies.append(est.body)
            self.bodies = self.bodies[-self.window:]
        return est

    def body(self) -> BodyParams | None:
        return median_body(self.bodies) if len(self.bodies) >= self.min_frames else None


# ---------------------------------------------------------------------------
# 스케일 보정: 바닥 기준 마커 (2차, 선택)
# ---------------------------------------------------------------------------
def scale_from_marker(corners_px: np.ndarray, marker_size_m: float) -> float:
    """바닥에 놓인 정사각 마커의 네 꼭짓점 (4,2) px → m/px.

    바닥 마커는 비스듬히 보여 세로 변이 짧아지므로, 화면에서 가장 수평에 가까운 두 변
    (카메라와 같은 거리의 변)의 평균 길이만 쓴다. 마커는 사람이 서는 자리(발 옆)에 둔다.
    """
    c = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    edges = [c[(i + 1) % 4] - c[i] for i in range(4)]
    horiz = sorted(edges, key=lambda e: abs(e[1]) / max(np.linalg.norm(e), 1e-9))[:2]
    length = float(np.mean([np.linalg.norm(e) for e in horiz]))
    if length <= 0:
        raise ValueError("마커 변 길이가 0")
    return float(marker_size_m) / length


# ---------------------------------------------------------------------------
# 합성 키포인트 (테스트·데모용): 몸 모델 관절을 정면 핀홀 카메라에 투영
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PinholeCamera:
    """-x 방향을 보는 정면 카메라 (사람은 +x 를 보고 선다 → 카메라를 마주 본다).

    이미지 u = 오른쪽(= 월드 +y, 찍힌 사람의 왼쪽), v = 아래(= 월드 −z).
    `distance_m`은 부스 중심(마네킹 골반 x)에서 카메라까지의 x 거리.
    """
    distance_m: float = 3.0
    height_m: float = 1.0
    focal_px: float = 1000.0
    image_size: tuple[int, int] = (1280, 720)

    def project(self, pts: np.ndarray, center_x: float) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        cam = np.array([center_x + self.distance_m, 0.0, self.height_m])
        depth = cam[0] - pts[:, 0]
        if np.any(depth <= 0.05):
            raise ValueError("카메라 뒤에 점이 있다")
        w, h = self.image_size
        u = w / 2.0 + self.focal_px * (pts[:, 1] - cam[1]) / depth
        v = h / 2.0 - self.focal_px * (pts[:, 2] - cam[2]) / depth
        return np.stack([u, v], axis=1)


def keypoints_3d(body: BodyParams, pose: PoseParams, scenario: Scenario) -> np.ndarray:
    """B의 `joint_positions` + 인체 비율 → COCO 17 키포인트의 3D 위치 (17,3) m.

    코·눈·귀는 머리 구(반지름 `HEAD_RADIUS_PER_HEIGHT × 키`) 앞면·옆면에, 발목 키포인트는
    발목 관절(바닥) 위 `ANKLE_ABOVE_FLOOR_PER_HEIGHT × 키`에 둔다.
    """
    joints, dims, r_body = joint_positions(body, pose, scenario)
    fwd = r_body.apply([1.0, 0.0, 0.0])
    left = r_body.apply([0.0, 1.0, 0.0])
    up = r_body.apply([0.0, 0.0, 1.0])
    hc, r, H = joints["head_center"], dims.head_radius, body.height_m
    top = hc + up * r                                  # B 머리 꼭대기
    kp = np.zeros((17, 3))
    kp[NOSE] = top - up * NOSE_BELOW_TOP_PER_HEIGHT * H + fwd * r
    for i, s in ((L_EYE, 1.0), (R_EYE, -1.0)):
        kp[i] = top - up * EYE_BELOW_TOP_PER_HEIGHT * H + fwd * 0.8 * r + left * s * 0.35 * r
    for i, s in ((L_EAR, 1.0), (R_EAR, -1.0)):
        kp[i] = top - up * EAR_BELOW_TOP_PER_HEIGHT * H + left * s * r
    names = {L_SHOULDER: "shoulder_L", R_SHOULDER: "shoulder_R", L_ELBOW: "elbow_L",
             R_ELBOW: "elbow_R", L_WRIST: "wrist_L", R_WRIST: "wrist_R", L_HIP: "hip_L",
             R_HIP: "hip_R", L_KNEE: "knee_L", R_KNEE: "knee_R"}
    for i, n in names.items():
        kp[i] = joints[n]
    kp[L_ANKLE] = joints["ankle_L"] + np.array([0.0, 0.0, ANKLE_ABOVE_FLOOR_PER_HEIGHT * H])
    kp[R_ANKLE] = joints["ankle_R"] + np.array([0.0, 0.0, ANKLE_ABOVE_FLOOR_PER_HEIGHT * H])
    return kp


#: COCO 관절 키포인트 ↔ MakeHuman 뼈 머리 (관절 중심)
MESH_JOINT_BONES = {
    L_EYE: "eye.L", R_EYE: "eye.R",
    L_SHOULDER: "upperarm01.L", R_SHOULDER: "upperarm01.R", L_ELBOW: "lowerarm01.L",
    R_ELBOW: "lowerarm01.R", L_WRIST: "wrist.L", R_WRIST: "wrist.R", L_HIP: "upperleg01.L",
    R_HIP: "upperleg01.R", L_KNEE: "lowerleg01.L", R_KNEE: "lowerleg01.R", L_ANKLE: "foot.L",
    R_ANKLE: "foot.R",
}


def mesh_keypoints_3d(body: BodyParams | None, pose: PoseParams, scenario: Scenario) -> np.ndarray:
    """체형·자세를 입힌 MakeHuman 메시의 COCO 17 키포인트 3D 위치 (17,3), 부스 좌표 m.

    관절은 뼈 머리(`MESH_JOINT_BONES`), 코·귀는 `mesh_landmarks()` 정점. 부스 배치는 B의 `place_in_booth`
    와 같다 (키 축척, 고관절 중심을 부스 중앙, 발바닥 z = 0 또는 좌석 높이).
    """
    from ..sim import human_mesh as hm

    body = body if body is not None else hm.MESH_DEFAULT_BODY
    shaped = hm.shape_mesh(hm.cached_makehuman(), body, scenario)
    p = hm.resolve_pose(pose, scenario)
    M = hm.bone_transforms(shaped, p)
    verts = hm.pose_vertices(shaped, p)
    lm = mesh_landmarks()
    pts = np.zeros((17, 3))
    for i, name in MESH_JOINT_BONES.items():
        b = shaped.bone(name)
        pts[i] = M[b, :3, :3] @ shaped.bone_head[b] + M[b, :3, 3]
    pts[NOSE], pts[L_EAR], pts[R_EAR] = verts[lm["nose"]], verts[lm["ear_L"]], verts[lm["ear_R"]]
    placed = hm.place_in_booth(np.vstack([verts, pts]), shaped, body, hm._default_booth(), scenario)
    return placed[-17:]


def synthetic_keypoints(body: BodyParams | None, pose: PoseParams, scenario: Scenario,
                        camera: PinholeCamera | None = None, *, noise_px: float = 0.0,
                        seed: int = 0, model: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """합성 키포인트 (17,2) px, 신뢰도 (17,) (전부 0.9). 테스트와 대시보드 "합성 입력"용 가짜 데이터.

    `model`: `"mesh"`(메시 관절·얼굴 정점 투영, `body=None` 이면 `MESH_DEFAULT_BODY`) | `"capsule"`
    (캡슐 관절 + 캡슐 비율) | None(= 설정 파일 `body.model`).
    """
    camera = camera or PinholeCamera()
    if profile_for(model).name == "mesh":
        from ..sim.human_mesh import _default_booth
        kp3 = mesh_keypoints_3d(body, pose, scenario)
        center_x = float(_default_booth()["length_m"]) / 2.0
    else:
        body = body if body is not None else BodyParams()
        joints, _, _ = joint_positions(body, pose, scenario)
        kp3 = keypoints_3d(body, pose, scenario)
        center_x = float(joints["pelvis"][0])
    kp = camera.project(kp3, center_x=center_x)
    if noise_px > 0:
        kp = kp + np.random.default_rng(seed).normal(0.0, noise_px, kp.shape)
    return kp, np.full(17, 0.9)
