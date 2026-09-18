"""가짜 입력. 소유자: D. 다른 트랙(A, C)도 여기서 import한다.

B의 `build_body` / `load_nozzles`가 병합되기 전까지의 대체재였고, B 병합 후에는
단위 테스트용으로 남긴다 (`docs/tracks/00_common.md` 3절).

제공하는 것
- `fake_body(arms_up, yaw_deg)`   -> BodyState : 몸통/머리/팔 4개 캡슐, 패치 280개
- `fake_mesh_body(arms_up, yaw_deg)` -> BodyState : 상자 6개(몸통·머리·팔·다리) 삼각형 메시.
  메시 필드(`mesh_vertices` 등)를 채운 메시 모델 대체재 (`docs/mesh_transition.md`, D 단계 8)
- `fake_nozzles()`                -> NozzleConfig : 좌우 벽 × 높이 2단 = 4개

`fake_velocity_field_per_nozzle`는 B 병합 후 삭제했다 (D 문서 단계 1).
제트는 `airis.sim.jet.velocity_field_per_nozzle`를 쓴다.

배열 규약은 `airis/sim/types.py`를 따른다 (위치/방향 float32 (N,3), 부위 ID int32).
"""
from __future__ import annotations

import functools

import numpy as np

from airis.sim.scenario import load_nozzle_layout
from airis.sim.types import PART_NAMES, BodyState, NozzleConfig

# --- 가짜 마네킹 치수 (docs/tracks/D_patch_baseline.md 단계 1) -----------------
_TORSO_LENGTH_M = 0.6
_TORSO_RADIUS_M = 0.15
_HEAD_RADIUS_M = 0.1
_ARM_LENGTH_M = 0.6
_ARM_RADIUS_M = 0.04

_TORSO_BOTTOM_Z = 0.9       # 몸통 캡슐 아래끝. 위끝 = 1.5
_HEAD_GAP_M = 0.05          # 몸통 위끝과 머리 구 아래끝 사이 목 간격
_SHOULDER_DROP_M = 0.05     # 어깨 관절은 몸통 위끝보다 조금 아래
_SHOULDER_OFFSET_M = 0.20   # 어깨 관절의 y. 몸통(0.15) + 팔 반지름(0.04)에 살짝 여유

# 팔 방향. 어깨에서 본 각도로, 0 = 바로 위(+z), 180 = 바로 아래(-z).
# 내린 팔은 거의 수직이라 몸통이 안쪽 면을 가린다.
_ARM_POLAR_DOWN_DEG = 175.0
# 든 팔은 "수평"이 이상적이지만 (D 문서 단계 1) 부스 폭이 1.2 m라 팔 길이 0.6 m를
# 수평으로 뻗으면 팔 끝이 y = 0.80으로 벽(y = ±0.6)을 뚫는다. 팔 끝이 벽 안쪽에
# 남는 범위에서 최대한 눕힌 각도를 쓴다. 36도 -> 팔 끝 y = ±0.553, z = 1.94.
_ARM_POLAR_UP_DEG = 36.0

# 패치 격자 해상도. 합계 24*7 + 2*(8*4) + 6*8 = 280개 (목표 200~300).
_TORSO_GRID = (24, 7)       # (둘레 방향, 축 방향)
_ARM_GRID = (8, 4)
_HEAD_GRID = (6, 8)         # (위도 밴드, 경도)

# --- 가짜 메시 몸 (docs/mesh_transition.md, D 단계 8) --------------------------
# 상자 치수 (m). 가로 = 몸 좌우(y), 깊이 = 몸 앞뒤(x). 몸통·머리·팔·어깨 높이는 fake_body와 같다.
_MESH_TORSO_HALF = (0.10, 0.15)     # (깊이/2, 가로/2)
_MESH_HEAD_HALF = 0.10              # 정육면체 반변
_MESH_ARM_HALF = 0.04               # 팔 단면 반변
_MESH_LEG_HALF = 0.06               # 다리 단면 반변
_MESH_LEG_Y = 0.08                  # 다리 중심 y. 두 다리 사이 틈 0.04
_MESH_LEG_BOTTOM_Z = 0.05
# 면 분할 목표 크기. 면마다 ceil(변 길이 / 이 값)칸으로 나눈다 (삼각형 약 1,000개).
_MESH_CELL_M = 0.05


def _box_mesh(p0: np.ndarray, p1: np.ndarray, half_a: float, half_b: float,
              e_a: np.ndarray, e_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """축 p0 -> p1, 단면 반변 (half_a along e_a, half_b along e_b)인 직육면체 삼각형 메시.

    (e_a, e_b, 축 방향)은 오른손 정규 직교 기저여야 한다.

    면마다 격자로 나눠 삼각형 두 개씩 만든다. 면끼리 정점을 공유하지 않는다(광선 판정에는 무관).
    삼각형은 바깥에서 보아 반시계. Returns (vertices (V,3), faces (F,3)).
    """
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    e_c = axis / length
    assert np.dot(np.cross(e_a, e_b), e_c) > 0.999, "오른손 기저가 아니다"
    center = 0.5 * (p0 + p1)
    half = {"a": half_a, "b": half_b, "c": 0.5 * length}
    basis = {"a": e_a, "b": e_b, "c": e_c}
    verts, faces = [], []
    # 면 = (법선 축, 부호, 면 위 두 축). u × v = 법선 방향이 되게 고른다.
    for n_key, u_key, v_key in (("a", "b", "c"), ("b", "c", "a"), ("c", "a", "b")):
        for sign in (1.0, -1.0):
            u_axis = basis[u_key] * sign     # 부호를 u에 실어 u × v 가 바깥을 보게 한다
            v_axis = basis[v_key]
            nu = max(1, int(np.ceil(2 * half[u_key] / _MESH_CELL_M)))
            nv = max(1, int(np.ceil(2 * half[v_key] / _MESH_CELL_M)))
            us = np.linspace(-half[u_key], half[u_key], nu + 1)
            vs = np.linspace(-half[v_key], half[v_key], nv + 1)
            grid = (center + sign * half[n_key] * basis[n_key]
                    + us[:, None, None] * u_axis + vs[None, :, None] * v_axis)   # (nu+1, nv+1, 3)
            base = sum(len(v) for v in verts)
            verts.append(grid.reshape(-1, 3))
            idx = base + np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
            a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
            c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
            faces.append(np.stack([a, b, c], axis=1))
            faces.append(np.stack([a, c, d], axis=1))
    return np.concatenate(verts), np.concatenate(faces)


def fake_mesh_body(arms_up: bool = False, yaw_deg: float = 0.0) -> BodyState:
    """상자 6개(몸통·머리·팔 2·다리 2)로 만든 가짜 메시 몸. 부스 중앙에 선다.

    B의 메시 `build_body`가 병합되기 전 D의 메시 가림 판정을 시험하는 대체재다.
    - 패치 = 삼각형마다 1개(무게중심), 법선 = 면 법선, 면적 = 삼각형 면적, `patch_face` = 그 면.
    - 부위: 몸통은 회전 전 법선의 x 성분이 양수면 `torso_front`, 아니면 `torso_back`.
    - `capsules`: 상자마다 근사 캡슐 1개 (A 입자 충돌·캡슐 폴백용). `patch_capsule`은 None.
    - `arms_up`, `yaw_deg`는 `fake_body`와 같은 뜻이다.
    """
    center = np.array(_booth_center())
    cx = center[0]
    ex, ey, ez = np.eye(3)
    torso_top_z = _TORSO_BOTTOM_Z + _TORSO_LENGTH_M
    shoulder_z = torso_top_z - _SHOULDER_DROP_M
    polar = np.radians(_ARM_POLAR_UP_DEG if arms_up else _ARM_POLAR_DOWN_DEG)
    head_bottom = torso_top_z + _HEAD_GAP_M

    # (p0, p1, half_a, half_b, e_a, e_b, part, capsule_radius)
    boxes = [
        (np.array([cx, 0.0, _TORSO_BOTTOM_Z]), np.array([cx, 0.0, torso_top_z]),
         *_MESH_TORSO_HALF, ex, ey, "torso", _MESH_TORSO_HALF[1]),
        (np.array([cx, 0.0, head_bottom]), np.array([cx, 0.0, head_bottom + 2 * _MESH_HEAD_HALF]),
         _MESH_HEAD_HALF, _MESH_HEAD_HALF, ex, ey, "head", _MESH_HEAD_HALF),
    ]
    for sign in (-1.0, 1.0):
        shoulder = np.array([cx, sign * _SHOULDER_OFFSET_M, shoulder_z])
        direction = np.array([0.0, sign * np.sin(polar), np.cos(polar)])
        e_b = _unit(np.cross(direction, ex))     # (ex, e_b, direction) 오른손 기저
        boxes.append((shoulder, shoulder + _ARM_LENGTH_M * direction, _MESH_ARM_HALF,
                      _MESH_ARM_HALF, ex, e_b, "arms", _MESH_ARM_HALF))
        boxes.append((np.array([cx, sign * _MESH_LEG_Y, _MESH_LEG_BOTTOM_Z]),
                      np.array([cx, sign * _MESH_LEG_Y, _TORSO_BOTTOM_Z]),
                      _MESH_LEG_HALF, _MESH_LEG_HALF, ex, ey, "legs", _MESH_LEG_HALF))

    verts, faces, face_part, capsules, capsule_part = [], [], [], [], []
    for p0, p1, half_a, half_b, e_a, e_b, part, radius in boxes:
        v, f = _box_mesh(p0, p1, half_a, half_b, e_a, e_b)
        tri = v[f]
        n = _unit(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
        if part == "torso":
            ids = np.where(n[:, 0] > 1e-9, PART_NAMES.index("torso_front"),
                           PART_NAMES.index("torso_back"))
        else:
            ids = np.full(len(f), PART_NAMES.index(part))
        faces.append(f + sum(len(x) for x in verts))
        verts.append(v)
        face_part.append(ids)
        capsules.append(np.concatenate([p0, p1, [radius]]))
        capsule_part.append(PART_NAMES.index("torso_front" if part == "torso" else part))

    verts = _rotate_z(np.concatenate(verts), yaw_deg, center)
    faces = np.concatenate(faces)
    tri = verts[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    caps = np.stack(capsules)
    caps = np.concatenate([_rotate_z(caps[:, 0:3], yaw_deg, center),
                           _rotate_z(caps[:, 3:6], yaw_deg, center), caps[:, 6:7]], axis=1)
    face_part = np.concatenate(face_part)
    return BodyState(
        patch_pos=tri.mean(axis=1).astype(np.float32),
        patch_normal=_unit(cross).astype(np.float32),
        patch_area=(0.5 * np.linalg.norm(cross, axis=1)).astype(np.float32),
        patch_part=face_part.astype(np.int32),
        capsules=caps.astype(np.float32),
        capsule_part=np.array(capsule_part, dtype=np.int32),
        patch_capsule=None,
        mesh_vertices=verts.astype(np.float32),
        mesh_faces=faces.astype(np.int32),
        mesh_face_part=face_part.astype(np.int32),
        patch_face=np.arange(len(faces), dtype=np.int32),
    )


# --- 가짜 노즐 ----------------------------------------------------------------
# 높이는 D 문서 단계 1이 지정한 두 단. 벽 y, yaw, pitch, 세기는 configs/nozzles.yaml
# layout에서 읽는다 (D는 configs를 수정하지 않는다. 읽기만 한다).
_NOZZLE_Z_LEVELS_M = (1.0, 1.4)


def _unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.where(n == 0.0, 1.0, n)


def _frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """축에 수직인 정규 직교 기저 두 개."""
    seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(axis, seed))
    return e1, np.cross(axis, e1)


def _cylinder_patches(p0: np.ndarray, p1: np.ndarray, radius: float,
                      grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """캡슐의 원통부만 격자 샘플링 -> (위치, 바깥 법선, 면적)."""
    n_theta, n_axial = grid
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    d = axis / length
    e1, e2 = _frame(d)

    theta = (np.arange(n_theta) + 0.5) * (2.0 * np.pi / n_theta)
    s = (np.arange(n_axial) + 0.5) * (length / n_axial)

    normal = np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2      # (n_theta, 3)
    pos = p0 + s[:, None] * d                                              # (n_axial, 3)
    pos = pos[None, :, :] + radius * normal[:, None, :]                    # (n_theta, n_axial, 3)
    normal = np.ascontiguousarray(np.broadcast_to(normal[:, None, :], pos.shape))

    area = (2.0 * np.pi * radius / n_theta) * (length / n_axial)
    return (pos.reshape(-1, 3), normal.reshape(-1, 3),
            np.full(n_theta * n_axial, area))


def _sphere_patches(center: np.ndarray, radius: float,
                    grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """구 표면을 위도/경도 격자로 샘플링. 면적은 밴드별 정확값."""
    n_lat, n_lon = grid
    edges = np.linspace(0.0, np.pi, n_lat + 1)
    polar = 0.5 * (edges[:-1] + edges[1:])
    azim = (np.arange(n_lon) + 0.5) * (2.0 * np.pi / n_lon)

    sin_p, cos_p = np.sin(polar)[:, None], np.cos(polar)[:, None]
    ones = np.ones((n_lat, n_lon))
    normal = np.stack([sin_p * np.cos(azim) * ones,
                       sin_p * np.sin(azim) * ones,
                       cos_p * ones], axis=-1)

    band_area = radius**2 * (np.cos(edges[:-1]) - np.cos(edges[1:])) * (2.0 * np.pi / n_lon)
    area = (band_area[:, None] * np.ones((n_lat, n_lon))).reshape(-1)
    normal = normal.reshape(-1, 3)
    return center + radius * normal, normal, area


def _rotate_z(v: np.ndarray, yaw_deg: float, center: np.ndarray | None = None) -> np.ndarray:
    """z축 회전. center가 주어지면 그 점을 중심으로 (위치), 없으면 원점 기준 (방향)."""
    if yaw_deg == 0.0:
        return v
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if center is None:
        return v @ rot.T
    return (v - center) @ rot.T + center


@functools.lru_cache(maxsize=1)
def _booth_center() -> tuple[float, float, float]:
    """부스 중앙 바닥점. fake_body가 평가마다 불리므로 YAML은 한 번만 읽는다."""
    booth = load_nozzle_layout()["booth"]
    return (float(booth["length_m"]) / 2.0, 0.0, 0.0)


def fake_body(arms_up: bool = False, yaw_deg: float = 0.0) -> BodyState:
    """몸통 캡슐 + 머리 구 + 팔 캡슐 2개짜리 가짜 마네킹. 부스 중앙에 선다.

    - `arms_up=False`: 팔이 몸통 옆에 붙어 안쪽 면이 몸통에 가린다.
    - `arms_up=True`: 팔이 옆위로 벌어져 벽면 노즐에 가까워진다.
    - `yaw_deg`: 몸 전체를 z축으로 회전. 0이면 몸의 정면이 +x(진행 방향)를 본다.
      B의 `build_body`와 같은 규약이다.

    부위는 회전 전 몸 기준 좌표에서 정한다. 몸통 패치는 법선의 x 성분 부호로
    `torso_front` / `torso_back`을 나누므로, 등지면(yaw 180) 정면 패치가 함께 돈다.
    다리는 없다 -> `removal_by_part[legs]`는 항상 0 (패치 0개).
    """
    center = np.array(_booth_center())
    cx = center[0]

    torso_top_z = _TORSO_BOTTOM_Z + _TORSO_LENGTH_M
    torso_p0 = np.array([cx, 0.0, _TORSO_BOTTOM_Z])
    torso_p1 = np.array([cx, 0.0, torso_top_z])
    head_c = np.array([cx, 0.0, torso_top_z + _HEAD_GAP_M + _HEAD_RADIUS_M])

    polar = np.radians(_ARM_POLAR_UP_DEG if arms_up else _ARM_POLAR_DOWN_DEG)
    shoulder_z = torso_top_z - _SHOULDER_DROP_M

    capsules: list[np.ndarray] = []
    capsule_part: list[int] = []
    pos_all: list[np.ndarray] = []
    normal_all: list[np.ndarray] = []
    area_all: list[np.ndarray] = []
    part_all: list[np.ndarray] = []
    patch_capsule: list[np.ndarray] = []

    def add(p0, p1, radius, patches, part_ids, part_of_capsule):
        idx = len(capsules)
        capsules.append(np.concatenate([p0, p1, [radius]]))
        capsule_part.append(part_of_capsule)
        pos, normal, area = patches
        pos_all.append(pos)
        normal_all.append(normal)
        area_all.append(area)
        part_all.append(part_ids)
        patch_capsule.append(np.full(len(pos), idx, dtype=np.int32))

    # 몸통: 법선의 +x 성분으로 앞/뒤를 가른다.
    torso = _cylinder_patches(torso_p0, torso_p1, _TORSO_RADIUS_M, _TORSO_GRID)
    torso_part = np.where(torso[1][:, 0] > 0.0,
                          PART_NAMES.index("torso_front"),
                          PART_NAMES.index("torso_back")).astype(np.int32)
    add(torso_p0, torso_p1, _TORSO_RADIUS_M, torso, torso_part,
        PART_NAMES.index("torso_front"))

    # 머리: 구 (캡슐 p0 == p1).
    head = _sphere_patches(head_c, _HEAD_RADIUS_M, _HEAD_GRID)
    add(head_c, head_c, _HEAD_RADIUS_M, head,
        np.full(len(head[0]), PART_NAMES.index("head"), dtype=np.int32),
        PART_NAMES.index("head"))

    # 팔 2개.
    for sign in (-1.0, 1.0):
        shoulder = np.array([cx, sign * _SHOULDER_OFFSET_M, shoulder_z])
        direction = np.array([0.0, sign * np.sin(polar), np.cos(polar)])
        elbow = shoulder + _ARM_LENGTH_M * direction
        arm = _cylinder_patches(shoulder, elbow, _ARM_RADIUS_M, _ARM_GRID)
        add(shoulder, elbow, _ARM_RADIUS_M, arm,
            np.full(len(arm[0]), PART_NAMES.index("arms"), dtype=np.int32),
            PART_NAMES.index("arms"))

    patch_pos = _rotate_z(np.concatenate(pos_all), yaw_deg, center)
    patch_normal = _rotate_z(np.concatenate(normal_all), yaw_deg)
    caps = np.stack(capsules)
    caps = np.concatenate([
        _rotate_z(caps[:, 0:3], yaw_deg, center),
        _rotate_z(caps[:, 3:6], yaw_deg, center),
        caps[:, 6:7],
    ], axis=1)

    return BodyState(
        patch_pos=patch_pos.astype(np.float32),
        patch_normal=_unit(patch_normal).astype(np.float32),
        patch_area=np.concatenate(area_all).astype(np.float32),
        patch_part=np.concatenate(part_all).astype(np.int32),
        capsules=caps.astype(np.float32),
        capsule_part=np.array(capsule_part, dtype=np.int32),
        patch_capsule=np.concatenate(patch_capsule).astype(np.int32),
    )


def fake_nozzles(z_levels_m: tuple[float, ...] = _NOZZLE_Z_LEVELS_M) -> NozzleConfig:
    """좌우 벽에 각 `len(z_levels_m)`개. 안쪽을 보되 진행 방향(+x)으로 기울어져 있다.

    x는 제트 축이 몸 중심선(y = 0)을 부스 중앙 x에서 지나도록 둔다:
    `x = length_m/2 - |wall_y| · tan(yaw)` (= 1.0 - 0.6·tan25° ≈ 0.72).
    B의 `configs/nozzles.yaml`이 실제 두 열(0.62, 0.82)을 고른 근거와 같은 식이고,
    실제 두 열의 평균과 일치한다. D 문서의 x = 1.0(부스 중앙)에 두면 25도 기울기 때문에
    제트 축이 몸통을 0.25 m 차이로 비껴간다.
    """
    nozzle_file = load_nozzle_layout()
    layout = nozzle_file["layout"]
    center_x = float(nozzle_file["booth"]["length_m"]) / 2.0
    yaw, pitch = np.radians(layout["yaw_deg"]), np.radians(layout["pitch_deg"])

    positions, directions = [], []
    for y in layout["wall_y"]:
        y = float(y)
        inward = -np.sign(y)  # y = +0.6 이면 -y 방향이 안쪽
        x = center_x - abs(y) * np.tan(yaw)
        for z in z_levels_m:
            positions.append([x, y, float(z)])
            # 안쪽 법선에서 진행 방향(+x)으로 yaw, 아래로 pitch.
            directions.append([np.sin(yaw) * np.cos(pitch),
                               inward * np.cos(yaw) * np.cos(pitch),
                               -np.sin(pitch)])

    return NozzleConfig(
        positions=np.array(positions, dtype=np.float32),
        directions=_unit(np.array(directions)).astype(np.float32),
        strengths=np.full(len(positions), float(layout["strength"]), dtype=np.float32),
    )
