"""팀 공용 데이터 타입. 이 파일의 시그니처 변경은 반드시 PR + 전원 리뷰.

배열 형태 규약
- 위치/방향: float32, (N, 3), 단위 m. 좌표계: x=게이트 진행 방향, y=좌우, z=상하(바닥=0).
- 부위 ID: int32, (N,). PART_NAMES 인덱스.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PART_NAMES = ["head", "torso_front", "torso_back", "arms", "legs"]


@dataclass
class BodyParams:
    """체형. 포즈 추정에서 추정하거나 데이터셋 생성 시 샘플링.

    메시 몸(docs/mesh_transition.md) 기준 정의. 모두 관절 중심 기준 m.
    - height_m: 키 (정수리 − 발바닥)
    - shoulder_width_m: 좌우 upperarm01(상완골 관절) 머리 간격
    - torso_depth_m: breast 뼈 머리 높이(유두선) ±1 cm 몸통 단면의 앞뒤(x) 범위
    - arm_length_m: 상완 + 전완 구간 합 (upperarm01→lowerarm01→wrist 머리). 자세 불변
    - leg_length_m: 고관절 중심 높이
    기본값 = MakeHuman 기본 체형을 키 1.70 m 로 축척한 측정값 (= human_mesh.MESH_DEFAULT_BODY,
    2026-09-18 확정). 캡슐 마네킹 시절 기본값은 1.70 / 0.42 / 0.22 / 0.62 / 0.85 였다.
    """
    height_m: float = 1.70
    shoulder_width_m: float = 0.342
    torso_depth_m: float = 0.194
    arm_length_m: float = 0.463
    leg_length_m: float = 0.883


@dataclass
class PoseParams:
    """자세 변수 (degree). 시나리오의 pose_bounds 안에 있어야 함."""
    shoulder_abduction: float = 20.0
    shoulder_flexion: float = 0.0
    elbow_flexion: float = 10.0
    torso_pitch: float = 0.0
    torso_yaw: float = 0.0
    hip_flexion: float = 0.0
    knee_flexion: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.array(list(vars(self).values()), dtype=np.float32)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "PoseParams":
        return cls(*[float(x) for x in v])


@dataclass
class NozzleConfig:
    """노즐 M개. 소유자: B."""
    positions: np.ndarray            # (M, 3) m
    directions: np.ndarray           # (M, 3) 단위 벡터
    strengths: np.ndarray            # (M,) exit_velocity 배수 (강/중/약 모드, 팬 병렬 수 반영. 1을 넘을 수 있음)
    pulse_phase: np.ndarray | None = None  # (M,) 0~1
    # 슬롯(평면) 제트: 슬롯 길이 방향 단위 벡터와 길이. None 이면 전부 원형 노즐.
    # 배열이면 행 단위: slot_length[m] > 0 이고 slot_axis[m] 이 0벡터가 아니면 슬롯(4.1b), 아니면 원형(4.1).
    slot_axis: np.ndarray | None = None    # (M, 3), 분사 방향에 수직
    slot_length: np.ndarray | None = None  # (M,) m

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])


@dataclass
class Scenario:
    """시나리오 제약. configs/scenarios.yaml에서 로드."""
    name: str
    pose_bounds: dict[str, tuple[float, float]]
    discomfort_weights: dict[str, float] = field(default_factory=dict)
    fixed_pose: dict[str, float] = field(default_factory=dict)
    nozzle_height_range_m: tuple[float, float] = (0.3, 2.0)
    nozzle_strength_cap: dict[str, float] = field(default_factory=dict)
    seat_height_m: float | None = None


@dataclass
class BodyState:
    """마네킹 기하. B가 생성, A/D가 소비."""
    patch_pos: np.ndarray            # (N, 3)
    patch_normal: np.ndarray         # (N, 3)
    patch_area: np.ndarray           # (N,) m^2
    patch_part: np.ndarray           # (N,) int, PART_NAMES 인덱스
    # 충돌 프리미티브: 캡슐 (p0, p1, r) 목록. 입자 충돌과 가림 판정에 사용.
    capsules: np.ndarray             # (K, 7) = [x0,y0,z0, x1,y1,z1, r]
    # 캡슐 부위. -1 = 가림 전용 (휠체어 프레임 등, 패치 없음). A의 재부착 부위 판정에 사용.
    capsule_part: np.ndarray | None = None   # (K,) int
    # 패치가 속한 캡슐 인덱스. D의 가림 판정이 자기 캡슐을 제외할 때 사용 (캡슐 모델).
    patch_capsule: np.ndarray | None = None  # (N,) int
    # --- 메시 몸 (docs/mesh_transition.md). physics.yaml body.model == "mesh" 일 때 B가 채운다. ---
    # 시뮬레이션용 데시메이션 메시 (약 5~6k 삼각형). 정점은 자세가 적용된 부스 좌표.
    mesh_vertices: np.ndarray | None = None   # (V, 3) float32
    mesh_faces: np.ndarray | None = None      # (F, 3) int32, 바깥 방향 반시계
    mesh_face_part: np.ndarray | None = None  # (F,) int, PART_NAMES 인덱스
    # 패치가 놓인 면 인덱스. D의 메시 가림 판정이 자기 면을 제외할 때 사용.
    patch_face: np.ndarray | None = None      # (N,) int
    # 메시 모델에서 `capsules` 는 뼈에 맞춘 근사 캡슐(A 입자 충돌·D 폴백용)이다. 휠체어 프레임 포함.


@dataclass
class EvalResult:
    """evaluate()의 반환. 최적화는 score만, 분석은 나머지도 사용."""
    score: float
    removal_by_part: np.ndarray      # (len(PART_NAMES),) 0~1
    total_removal: float
    discomfort: float
    extra: dict = field(default_factory=dict)
