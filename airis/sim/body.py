"""마네킹 기하. 소유자: B.

`docs/tracks/B_body_jet.md` 단계 1~4 구현.

- 단계 1: 체형 5개 값 → 골격 치수 (아래 인체 측정 비율 상수)
- 단계 2: 자세 7개 각도 → 관절 17개 위치 (순기구학)
- 단계 3: 관절 위치 → 캡슐 (K, 7)
- 단계 4: 캡슐 표면 → 패치 (위치/법선/면적/부위), 난수 없는 결정론적 격자

좌표계는 `docs/tracks/00_common.md` 5절: x=진행 방향(앞), y=좌우(왼쪽이 +), z=상하(바닥 0).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .types import PART_NAMES, BodyParams, BodyState, PoseParams, Scenario
from .scenario import load_nozzle_layout

# ---------------------------------------------------------------------------
# 부위 인덱스 (PART_NAMES 순서에 의존하지 않도록 이름으로 찾는다)
# ---------------------------------------------------------------------------
PART_HEAD = PART_NAMES.index("head")
PART_TORSO_FRONT = PART_NAMES.index("torso_front")
PART_TORSO_BACK = PART_NAMES.index("torso_back")
PART_ARMS = PART_NAMES.index("arms")
PART_LEGS = PART_NAMES.index("legs")
PART_OCCLUDER = -1  # 패치 없음 (휠체어 프레임 등)

# ---------------------------------------------------------------------------
# 단계 1. 인체 측정 비율 (B_body_jet.md 단계 1 "체형 → 치수" 표)
# ---------------------------------------------------------------------------
HEAD_RADIUS_PER_HEIGHT = 0.065
NECK_LENGTH_PER_HEIGHT = 0.04
TORSO_LATERAL_PER_SHOULDER = 0.85   # 몸통 좌우 반축 = shoulder_width/2 × 0.85
UPPER_ARM_PER_ARM = 0.45
FOREARM_PER_ARM = 0.55
THIGH_PER_LEG = 0.5
SHANK_PER_LEG = 0.5
UPPER_ARM_RADIUS_M = 0.045
FOREARM_RADIUS_M = 0.038
THIGH_RADIUS_M = 0.075
SHANK_RADIUS_M = 0.055
HIP_WIDTH_PER_SHOULDER = 0.4

# 휠체어 가림 캡슐 치수 (골반 원점 기준 local frame, m)
WHEELCHAIR_SEAT_RADIUS_M = 0.08
WHEELCHAIR_BACKREST_RADIUS_M = 0.06
WHEELCHAIR_WHEEL_RADIUS_M = 0.30
WHEELCHAIR_WHEEL_HALF_WIDTH_M = 0.03
WHEELCHAIR_WHEEL_Y_M = 0.31

JOINT_NAMES = (
    "pelvis", "spine", "chest", "neck", "head_center",
    "shoulder_L", "elbow_L", "wrist_L",
    "shoulder_R", "elbow_R", "wrist_R",
    "hip_L", "knee_L", "ankle_L",
    "hip_R", "knee_R", "ankle_R",
)

_DOWN = np.array([0.0, 0.0, -1.0])
_FORWARD = np.array([1.0, 0.0, 0.0])
_LEFT = np.array([0.0, 1.0, 0.0])
_UP = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# 단계 1. 치수
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BodyDims:
    """BodyParams에서 유도한 골격 치수 (m)."""
    head_radius: float
    neck_length: float
    torso_length: float
    torso_half_lateral: float   # 좌우 반축 (캡슐 반지름으로도 사용)
    torso_half_ap: float        # 앞뒤 반축
    upper_arm_length: float
    forearm_length: float
    thigh_length: float
    shank_length: float
    shoulder_half_width: float
    hip_half_width: float


def body_dims(body: BodyParams) -> BodyDims:
    head_radius = HEAD_RADIUS_PER_HEIGHT * body.height_m
    neck_length = NECK_LENGTH_PER_HEIGHT * body.height_m
    # 골반→가슴. 직립 시 머리 꼭대기가 정확히 height_m 이 되도록 유도된 식이다.
    torso_length = body.height_m - body.leg_length_m - 2.0 * head_radius - neck_length
    if torso_length <= 0.0:
        raise ValueError(
            f"몸통 길이가 0 이하다 (height={body.height_m}, leg={body.leg_length_m}). "
            "체형 값을 확인하라."
        )
    return BodyDims(
        head_radius=head_radius,
        neck_length=neck_length,
        torso_length=torso_length,
        torso_half_lateral=body.shoulder_width_m / 2.0 * TORSO_LATERAL_PER_SHOULDER,
        torso_half_ap=body.torso_depth_m / 2.0,
        upper_arm_length=UPPER_ARM_PER_ARM * body.arm_length_m,
        forearm_length=FOREARM_PER_ARM * body.arm_length_m,
        thigh_length=THIGH_PER_LEG * body.leg_length_m,
        shank_length=SHANK_PER_LEG * body.leg_length_m,
        shoulder_half_width=body.shoulder_width_m / 2.0,
        hip_half_width=HIP_WIDTH_PER_SHOULDER * body.shoulder_width_m / 2.0,
    )


# ---------------------------------------------------------------------------
# 단계 2. 순기구학
# ---------------------------------------------------------------------------
# 회전 부호 결정 (B_body_jet.md 단계 2 표 + 오른손 좌표계 x=앞, y=왼쪽, z=위):
#
#   표는 "축"과 "해부학적 양의 방향"만 정하고 부호는 정하지 않으므로, 각 분절이
#   향하는 방향에 맞춰 회전 벡터의 부호를 아래와 같이 확정한다. 같은 "앞으로"라도
#   위를 향한 분절(몸통)과 아래를 향한 분절(대퇴/상완)은 전역 회전 벡터가 반대다.
#
#   torso_pitch  앞으로 숙임  : +z(몸통)가 +x로  →  (+z)×(+x) = +y  →  R_y(+θ)
#   hip_flexion  앞으로 들기  : -z(대퇴)가 +x로  →  (-z)×(+x) = -y  →  R_y(-θ)
#   knee_flexion 뒤로 굽힘    : -z(하퇴)가 -x로  →  (-z)×(-x) = +y  →  R_y(+θ) (대퇴 국소계)
#   shoulder_flexion 앞으로 들기 : 대퇴와 동일   →  R_y(-θ)
#   shoulder_abduction 옆으로 벌림 : 왼팔은 -z가 +y로 → (-z)×(+y) = +x → R_x(+θ)
#                                    오른팔은 부호 반전 → R_x(-θ)  (좌우 대칭)
#   torso_yaw    왼쪽으로 회전 : 위에서 볼 때 반시계 → R_z(+θ)
#
# 어깨 합성 순서: 문서가 "abduction 먼저, flexion 다음"이라고 못 박았으므로
#   R_shoulder = R_flexion ∘ R_abduction  (벡터에 abduction이 먼저 곱해진다).
#
# 팔꿈치 축: "상완에 수직인 축". 어깨 회전을 적용하기 전 국소계에서 -y를 쓰고
#   (= 어깨 굴곡과 같은 축 규약, 팔을 내린 자세에서 손목이 앞으로 간다),
#   실제 축은 이를 어깨 회전으로 옮긴 R_shoulder·(0,-1,0)이다. 이 선택 덕분에
#   abduction=90(팔 수평)에서 팔꿈치를 굽히면 손목이 앞으로만 가고 높이가
#   유지된다 (tests/test_body.py 의 wrist z ≈ shoulder z).


def _resolve_pose(pose: PoseParams, scenario: Scenario) -> PoseParams:
    """scenario.fixed_pose 가 있으면 해당 각도를 덮어쓴다."""
    if not scenario.fixed_pose:
        return pose
    overrides = {}
    for key, value in scenario.fixed_pose.items():
        if not hasattr(pose, key):
            raise ValueError(f"알 수 없는 자세 변수: {key}")
        overrides[key] = float(value)
    return replace(pose, **overrides)


def _rot(axis: str, degrees: float) -> Rotation:
    return Rotation.from_euler(axis, degrees, degrees=True)


def joint_positions(
    body: BodyParams, pose: PoseParams, scenario: Scenario
) -> tuple[dict[str, np.ndarray], BodyDims, Rotation]:
    """관절 17개의 월드 좌표.

    반환: (관절 위치 dict, 치수, 몸통 자세 회전).
    몸통 자세 회전은 패치 타원 단면을 몸 기준으로 세울 때 쓴다 (전방=R·+x, 좌=R·+y).
    """
    d = body_dims(body)
    p = _resolve_pose(pose, scenario)

    # --- 상체: 골반 원점 local frame (torso_pitch 적용 전) ---
    chest_l = _UP * d.torso_length
    local = {
        "pelvis": np.zeros(3),
        "spine": _UP * (d.torso_length / 2.0),
        "chest": chest_l,
        "neck": chest_l + _UP * d.neck_length,
        "head_center": chest_l + _UP * (d.neck_length + d.head_radius),
        "shoulder_L": chest_l + _LEFT * d.shoulder_half_width,
        "shoulder_R": chest_l - _LEFT * d.shoulder_half_width,
    }

    # --- 팔: 어깨 기준 ---
    r_flex = _rot("y", -p.shoulder_flexion)
    for side, sign in (("L", 1.0), ("R", -1.0)):
        r_shoulder = r_flex * _rot("x", sign * p.shoulder_abduction)
        upper_dir = r_shoulder.apply(_DOWN)
        elbow = local[f"shoulder_{side}"] + upper_dir * d.upper_arm_length
        elbow_axis = r_shoulder.apply(-_LEFT)
        r_elbow = Rotation.from_rotvec(elbow_axis * np.deg2rad(p.elbow_flexion))
        forearm_dir = r_elbow.apply(upper_dir)
        local[f"elbow_{side}"] = elbow
        local[f"wrist_{side}"] = elbow + forearm_dir * d.forearm_length

    # torso_pitch 는 pelvis 기준으로 spine 이상(상체 전체)에만 적용된다.
    r_pitch = _rot("y", p.torso_pitch)
    for name in list(local):
        if name != "pelvis":
            local[name] = r_pitch.apply(local[name])

    # --- 다리: 골반 기준, torso_pitch 영향 없음 ---
    r_hip = _rot("y", -p.hip_flexion)
    thigh_dir = r_hip.apply(_DOWN)
    shank_dir = r_hip.apply(_rot("y", p.knee_flexion).apply(_DOWN))
    for side, sign in (("L", 1.0), ("R", -1.0)):
        hip = _LEFT * (sign * d.hip_half_width)
        knee = hip + thigh_dir * d.thigh_length
        local[f"hip_{side}"] = hip
        local[f"knee_{side}"] = knee
        local[f"ankle_{side}"] = knee + shank_dir * d.shank_length

    # --- torso_yaw: 골반 기준 전신 ---
    r_yaw = _rot("z", p.torso_yaw)
    for name in local:
        local[name] = r_yaw.apply(local[name])

    # --- 월드 배치: 부스 중앙, 골반 높이 ---
    booth = _booth()
    if scenario.seat_height_m is not None:
        pelvis_z = float(scenario.seat_height_m)
    else:
        # 발바닥(= 발목 관절)이 z=0 에 오도록. 직립 기본 자세에서 pelvis_z = leg_length_m 이고
        # 이때 머리 꼭대기가 정확히 height_m 이 된다.
        pelvis_z = -min(local["ankle_L"][2], local["ankle_R"][2])
    origin = np.array([booth["length_m"] / 2.0, 0.0, pelvis_z])

    joints = {name: origin + pos for name, pos in local.items()}
    # 몸통 자세 회전: yaw ∘ pitch. 전방 = r_body·(1,0,0), 좌 = r_body·(0,1,0).
    return joints, d, r_yaw * r_pitch


@lru_cache(maxsize=4)
def _booth(path: str | None = None) -> dict:
    layout = load_nozzle_layout(Path(path) if path else None)
    return layout["booth"]


# ---------------------------------------------------------------------------
# 단계 3. 캡슐
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Segment:
    """패치 생성을 위한 캡슐 1개의 기하. a1/a2 는 단면 반축, a3 는 양 끝 뚜껑의 축 방향 반축."""
    p0: np.ndarray
    p1: np.ndarray
    e1: np.ndarray   # 단면 기준축 1 (몸통은 전방)
    e2: np.ndarray   # 단면 기준축 2 (몸통은 좌)
    a1: float
    a2: float
    a3: float
    part: int
    split_front_back: bool = False   # 법선의 e1 성분 부호로 torso_front/back 을 나눈다


# torso_front/back 판정에서 "정확히 옆"으로 보는 법선 전방 성분의 허용치 (부동소수 잡음 흡수, 무차원).
_FRONT_BACK_TOL = 1e-9


def _perp_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """축에 수직인 결정론적 정규 직교 기저. 축이 +x 에 가까우면 +z 를, 아니면 +x 를 참조로 쓴다.

    e1 = 참조축을 축에 수직인 평면으로 투영한 것, e2 = axis × e1.
    참조축(+x, +z)은 y 거울 M = diag(1, −1, 1) 에 대해 불변이라 e1(M·axis) = M·e1 이고
    e2(M·axis) = −M·e2 다. 몸통 단면(e1 = 전방, e2 = 좌)과 같은 변환이라, 둘레 격자
    θ_k = 2π(k + 0.5)/n 이 거울에서 θ → −θ 로 자기 자신에 겹친다 → 좌우 캡슐의 패치가 정확히
    거울 대칭이다. (이전 e1 = ref × axis 는 거울에서 −M·e1 이 되어 θ → π − θ 가 되고, n 이
    홀수면 격자가 반 칸 어긋났다. C 발견, 400/m² 에서 최대 2.6 cm.)
    """
    ref = _UP if abs(float(axis @ _FORWARD)) > 0.9 else _FORWARD
    e1 = ref - float(ref @ axis) * axis
    n = np.linalg.norm(e1)
    if n < 1e-9:                     # 축과 참조가 평행 (길이 0 캡슐 등)
        e1 = _FORWARD.copy()
        e2 = _LEFT.copy()
        return e1, e2
    e1 = e1 / n
    e2 = np.cross(axis, e1)
    return e1, e2 / np.linalg.norm(e2)


def _limb(p0: np.ndarray, p1: np.ndarray, radius: float, part: int) -> _Segment:
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    axis = axis / length if length > 1e-9 else _UP
    e1, e2 = _perp_basis(axis)
    return _Segment(p0=p0, p1=p1, e1=e1, e2=e2, a1=radius, a2=radius, a3=radius, part=part)


def _segments(joints: dict, d: BodyDims, r_body: Rotation, scenario: Scenario) -> list[_Segment]:
    """캡슐 목록. 순서 고정 (머리, 몸통, 상완 L/R, 전완 L/R, 대퇴 L/R, 하퇴 L/R, [휠체어 4개])."""
    head = joints["head_center"]
    segs: list[_Segment] = [
        # 머리: 길이 0 캡슐 = 구
        _limb(head, head.copy(), d.head_radius, PART_HEAD),
        # 몸통: 타원 단면. e1=전방, e2=좌. 뚜껑은 앞뒤 반축으로 둥글린다.
        _Segment(
            p0=joints["pelvis"], p1=joints["chest"],
            e1=r_body.apply(_FORWARD), e2=r_body.apply(_LEFT),
            a1=d.torso_half_ap, a2=d.torso_half_lateral, a3=d.torso_half_ap,
            part=PART_TORSO_FRONT, split_front_back=True,
        ),
    ]
    for side in ("L", "R"):
        segs.append(_limb(joints[f"shoulder_{side}"], joints[f"elbow_{side}"],
                          UPPER_ARM_RADIUS_M, PART_ARMS))
    for side in ("L", "R"):
        segs.append(_limb(joints[f"elbow_{side}"], joints[f"wrist_{side}"],
                          FOREARM_RADIUS_M, PART_ARMS))
    for side in ("L", "R"):
        segs.append(_limb(joints[f"hip_{side}"], joints[f"knee_{side}"],
                          THIGH_RADIUS_M, PART_LEGS))
    for side in ("L", "R"):
        segs.append(_limb(joints[f"knee_{side}"], joints[f"ankle_{side}"],
                          SHANK_RADIUS_M, PART_LEGS))

    if scenario.name == "wheelchair":
        segs.extend(_wheelchair_segments(joints, r_body))
    return segs


def _wheelchair_segments(joints: dict, r_body: Rotation) -> list[_Segment]:
    """휠체어 프레임. 가림 전용(part=-1, 패치 없음). 골반 기준으로 놓고 몸통 회전을 따른다."""
    pelvis = joints["pelvis"]
    fwd = r_body.apply(_FORWARD)
    left = r_body.apply(_LEFT)
    up = _UP

    def at(f: float, l: float, u: float) -> np.ndarray:
        return pelvis + fwd * f + left * l + up * u

    floor_u = -pelvis[2] + WHEELCHAIR_WHEEL_RADIUS_M   # 바퀴 중심이 바닥에서 반지름만큼 뜨도록
    return [
        # 좌석: 짧고 굵은 수평 캡슐, 허벅지 아래
        _limb(at(-0.10, 0.0, -0.10), at(0.35, 0.0, -0.10), WHEELCHAIR_SEAT_RADIUS_M, PART_OCCLUDER),
        # 등받이: 수직 캡슐, 골반 뒤
        _limb(at(-0.15, 0.0, -0.05), at(-0.15, 0.0, 0.45), WHEELCHAIR_BACKREST_RADIUS_M, PART_OCCLUDER),
        # 바퀴 2개: y 축 방향 짧은 캡슐, 반지름 0.3
        _limb(at(0.0, WHEELCHAIR_WHEEL_Y_M - WHEELCHAIR_WHEEL_HALF_WIDTH_M, floor_u),
              at(0.0, WHEELCHAIR_WHEEL_Y_M + WHEELCHAIR_WHEEL_HALF_WIDTH_M, floor_u),
              WHEELCHAIR_WHEEL_RADIUS_M, PART_OCCLUDER),
        _limb(at(0.0, -WHEELCHAIR_WHEEL_Y_M - WHEELCHAIR_WHEEL_HALF_WIDTH_M, floor_u),
              at(0.0, -WHEELCHAIR_WHEEL_Y_M + WHEELCHAIR_WHEEL_HALF_WIDTH_M, floor_u),
              WHEELCHAIR_WHEEL_RADIUS_M, PART_OCCLUDER),
    ]


def _capsule_array(segs: list[_Segment]) -> np.ndarray:
    """(K, 7) = [x0,y0,z0, x1,y1,z1, r]. 몸통은 원형 근사(좌우 반축)로 넣는다."""
    rows = [np.concatenate([s.p0, s.p1, [max(s.a1, s.a2)]]) for s in segs]
    return np.asarray(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# 단계 4. 패치 샘플링 (난수 없음)
# ---------------------------------------------------------------------------
def _ellipse_perimeter(a: float, b: float) -> float:
    """Ramanujan 근사. a=b 이면 2πa 로 정확히 환원된다."""
    if a <= 0.0 and b <= 0.0:
        return 0.0
    h3 = 3.0 * (a - b) ** 2 / ((a + b) ** 2) if (a + b) > 0 else 0.0
    return np.pi * (a + b) * (1.0 + h3 / (10.0 + np.sqrt(4.0 - h3)))


def _sample_segment(seg: _Segment, patches_per_m2: float) -> tuple[np.ndarray, ...]:
    """캡슐 1개 → (pos, normal, area, part). 원통 측면 + 양 끝 반타원체 뚜껑."""
    axis_vec = seg.p1 - seg.p0
    length = float(np.linalg.norm(axis_vec))
    axis = axis_vec / length if length > 1e-9 else np.cross(seg.e1, seg.e2)

    h = 1.0 / np.sqrt(patches_per_m2)          # 목표 패치 한 변 길이
    perimeter = _ellipse_perimeter(seg.a1, seg.a2)
    n_theta = max(6, int(round(perimeter / h)))
    theta = 2.0 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    pos_parts, nrm_parts, area_parts = [], [], []

    # --- 원통(타원 기둥) 측면 ---
    if length > 1e-9:
        n_len = max(1, int(round(length / h)))
        t = (np.arange(n_len) + 0.5) / n_len
        # 단면 둘레 방향 호 길이 요소: |d/dθ (a1 cosθ, a2 sinθ)| dθ
        ds = np.sqrt((seg.a1 * sin_t) ** 2 + (seg.a2 * cos_t) ** 2) * (2.0 * np.pi / n_theta)
        ring = seg.a1 * cos_t[:, None] * seg.e1 + seg.a2 * sin_t[:, None] * seg.e2   # (T,3)
        nrm_ring = cos_t[:, None] / seg.a1 * seg.e1 + sin_t[:, None] / seg.a2 * seg.e2
        nrm_ring /= np.linalg.norm(nrm_ring, axis=1, keepdims=True)

        centers = seg.p0 + axis * (t[:, None] * length)                              # (L,3)
        pos_parts.append((centers[:, None, :] + ring[None, :, :]).reshape(-1, 3))
        nrm_parts.append(np.broadcast_to(nrm_ring[None, :, :], (n_len, n_theta, 3)).reshape(-1, 3))
        area_parts.append(np.broadcast_to(ds[None, :] * (length / n_len), (n_len, n_theta)).ravel())

    # --- 양 끝 반타원체 뚜껑 (등입체각 격자 + 야코비안 면적) ---
    arc = 0.5 * np.pi * ((seg.a1 + seg.a2) / 2.0 + seg.a3) / 2.0
    n_lat = max(2, int(round(arc / h)))
    cos_p = 1.0 - (np.arange(n_lat) + 0.5) / n_lat          # 밴드별 등입체각 중심
    sin_p = np.sqrt(np.maximum(0.0, 1.0 - cos_p ** 2))
    d_omega = 2.0 * np.pi / (n_lat * n_theta)

    for cap_center, cap_axis in ((seg.p1, axis), (seg.p0, -axis)):
        # 단위 구 방향 (n1, n2, n3) → 타원체 위 점/법선/면적
        n1 = sin_p[:, None] * cos_t[None, :]
        n2 = sin_p[:, None] * sin_t[None, :]
        n3 = np.broadcast_to(cos_p[:, None], (n_lat, n_theta))
        pos = (cap_center
               + (seg.a1 * n1)[..., None] * seg.e1
               + (seg.a2 * n2)[..., None] * seg.e2
               + (seg.a3 * n3)[..., None] * cap_axis)
        nrm = ((n1 / seg.a1)[..., None] * seg.e1
               + (n2 / seg.a2)[..., None] * seg.e2
               + (n3 / seg.a3)[..., None] * cap_axis)
        scale = np.linalg.norm(nrm, axis=-1)
        nrm = nrm / scale[..., None]
        area = seg.a1 * seg.a2 * seg.a3 * scale * d_omega
        pos_parts.append(pos.reshape(-1, 3))
        nrm_parts.append(nrm.reshape(-1, 3))
        area_parts.append(area.ravel())

    pos = np.concatenate(pos_parts)
    nrm = np.concatenate(nrm_parts)
    area = np.concatenate(area_parts)

    if seg.split_front_back:
        # 몸 전방 성분이 양이면 torso_front, 아니면 torso_back. 정확히 옆(θ = ±90°)을 보는 패치는
        # 전방 성분이 부동소수 잡음(cos(π/2) ≈ +6e−17, cos(3π/2) ≈ −2e−16)이라 좌우에서 부호가 갈렸다.
        # 허용치 이하를 둘 다 back 으로 보내 좌우 거울 대칭을 지킨다.
        part = np.where(nrm @ seg.e1 > _FRONT_BACK_TOL, PART_TORSO_FRONT, PART_TORSO_BACK)
    else:
        part = np.full(pos.shape[0], seg.part)
    return pos, nrm, area, part.astype(np.int32)


def surface_area(body: BodyParams, pose: PoseParams, scenario: Scenario) -> float:
    """모델이 표현하는 캡슐 표면적의 해석적 합 (패치 없는 부위 제외). 테스트용."""
    joints, d, r_body = joint_positions(body, pose, scenario)
    total = 0.0
    for seg in _segments(joints, d, r_body, scenario):
        if seg.part == PART_OCCLUDER:
            continue
        length = float(np.linalg.norm(seg.p1 - seg.p0))
        total += _ellipse_perimeter(seg.a1, seg.a2) * length            # 측면
        total += _ellipsoid_area(seg.a1, seg.a2, seg.a3)                # 뚜껑 2개 = 타원체 1개
    return total


def _ellipsoid_area(a: float, b: float, c: float) -> float:
    """Knud Thomsen 근사 (p=1.6). 오차 1% 이내, 구에서는 정확."""
    p = 1.6
    return 4.0 * np.pi * (((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3.0) ** (1.0 / p)


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------
def build_body(body: BodyParams, pose: PoseParams, scenario: Scenario,
               patches_per_m2: float = 2000.0) -> BodyState:
    """체형 + 자세 + 시나리오 → 패치와 캡슐로 표현된 마네킹.

    `docs/interfaces.md` 의 build_body 계약을 따른다.
    """
    if patches_per_m2 <= 0.0:
        raise ValueError("patches_per_m2 는 양수여야 한다")

    joints, dims, r_body = joint_positions(body, pose, scenario)
    segs = _segments(joints, dims, r_body, scenario)

    pos_l, nrm_l, area_l, part_l, cap_l = [], [], [], [], []
    for idx, seg in enumerate(segs):
        if seg.part == PART_OCCLUDER:
            continue                       # 가림 전용 캡슐은 패치를 만들지 않는다
        pos, nrm, area, part = _sample_segment(seg, patches_per_m2)
        pos_l.append(pos)
        nrm_l.append(nrm)
        area_l.append(area)
        part_l.append(part)
        cap_l.append(np.full(pos.shape[0], idx, dtype=np.int32))

    return BodyState(
        patch_pos=np.concatenate(pos_l).astype(np.float32),
        patch_normal=np.concatenate(nrm_l).astype(np.float32),
        patch_area=np.concatenate(area_l).astype(np.float32),
        patch_part=np.concatenate(part_l).astype(np.int32),
        capsules=_capsule_array(segs),
        capsule_part=np.array([s.part for s in segs], dtype=np.int32),
        patch_capsule=np.concatenate(cap_l).astype(np.int32),
    )
