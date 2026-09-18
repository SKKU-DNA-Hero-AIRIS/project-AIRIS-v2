"""사람 메시 (MakeHuman CC0 기본 메시 + 선형 블렌드 스키닝). 소유자: B (E 가 작성, 단계 10 에서 이관).

시뮬레이션 몸을 캡슐 마네킹에서 사람 메시로 바꾸는 작업(③)의 첫 부품이다. 메시를 불러와 우리 자세
변수(`PoseParams` 7개)를 뼈 회전으로 바꾸고 정점을 움직인다. numpy 만 쓴다 (Taichi·torch 없음).
다음 단계에서 B가 `airis/sim/human_mesh.py`로 옮겨 소유한다(체형 매핑·패치 샘플링 추가). 그 뒤 E의
시각화 코드는 sim 쪽을 import 한다. 그때까지는 이 위치에 둔다.

공개 API
--------
    mesh = load_makehuman()                   # HumanMesh (기본 자세, 우리 좌표, m)
    mesh.vertices   (V,3) float64             # 기본 자세 정점. V = 13,380
    mesh.faces      (F,3) int64               # 삼각형 (정점 인덱스, 바깥에서 보아 반시계). F = 26,756
    mesh.bone_names list[str]  (B = 163)      # MakeHuman 기본 뼈대 이름
    mesh.bone_parent (B,) int                 # 부모 인덱스, 뿌리 −1. 부모가 항상 먼저 온다
    mesh.bone_head / bone_tail (B,3)          # 기본 자세 뼈 머리·꼬리
    mesh.weights    (V,B) float64             # 스키닝 가중치, 행 합 1

    M = bone_transforms(mesh, pose)           # (B,4,4) 뼈별 월드 변환: 기본 자세 점 x → M_b·x
    v = pose_vertices(mesh, pose)             # (V,3) 기본 자세 좌표계에서 자세를 입힌 정점
    v = posed_in_booth(mesh, body, pose, scenario)   # (V,3) 부스 좌표 (아래 규약)
    v = pose_mesh(asset, body, pose, scenario)       # 같은 것. docs/interfaces.md 시그니처 (asset=None 이면
                                                     # 캐시한 기본 메시, body=None 이면 키 1.70)
    MeshAsset = HumanMesh                            # docs/interfaces.md 의 이름

좌표 규약 (`docs/tracks/00_common.md` 5절, B의 `build_body`와 같음)
- x = 게이트 진행 방향(사람이 보는 앞), y = 왼쪽(+), z = 위. 단위 m.
- `posed_in_booth`: 키를 `body.height_m`로 균일 축척하고, 고관절 중심(B 마네킹의 골반 원점)을
  부스 중앙 (length/2, 0)에 둔다. 선 자세는 발바닥(메시 최저점)을 z = 0에 둔다. 좌석 시나리오
  (`scenario.seat_height_m`, 휠체어)는 고관절 중심을 그 높이에 둔다. `scenario.fixed_pose`(휠체어
  고관절·무릎 90°)를 먼저 덮어쓴다.
- 체형은 지금 키만 맞춘다. 어깨 폭·팔·다리 비율은 메시 그대로다 (1.70 m 축척 시 어깨 관절 간격 0.34,
  팔 0.46, 고관절 높이 0.88, 가슴 두께 0.22 m. BodyParams 기본값 0.42 / 0.62 / 0.85 / 0.22 와 다르다).
  BodyParams 5개로 뼈별 축척하는 일은 B 몫이다.

자세 규약 (B의 `joint_positions`와 같은 식)
- torso_yaw: `root` 뼈에 R_z, 중심은 고관절 중심 (전신).
- torso_pitch: `spine05`(상체 전체)에 R_y, 중심은 고관절 중심.
- 팔: 상완 방향 = R_y(−굽힘)·R_x(±벌림)·(0,0,−1), 전완 = 팔꿈치 축(R_shoulder·(0,−1,0))으로 회전.
  MakeHuman 기본 자세는 팔을 약 40° 벌리고 팔꿈치를 앞으로 굽힌 A자라, `upperarm01`·`lowerarm01`
  뼈에 기본 방향을 목표 방향으로 옮기는 최소 회전을 넣는다 (뼈 축 비틀림은 맞추지 않는다).
- 다리: `upperleg01`에 R_y(−고관절 굴곡), `lowerleg01`에 R_y(+무릎 굴곡). 기본 자세의 다리 벌림은 그대로.

성능 (Windows, CPU): 로드 약 0.2 s (1회). 자세 한 번(`pose_mesh`)은 뼈 변환 계산 + 희소 블렌드로
수 ms다. 정확한 값은 PR 본문 표 참고 (다른 트랙 계산 부하에 따라 달라진다).
- 스키닝: 가중치를 CSR 희소 행렬(정점당 뼈 최대 9개, 평균 3개, nnz 약 4만)로 두고 W (V,B) × 뼈 변환
  (B,12) 한 번으로 정점별 3×4 행렬을 만든 뒤 곱한다. 가중치를 자르지 않아 조밀 계산과 결과가 같다(1e-15).
  "정점당 상위 4개 뼈"로 자르면 일부 정점에서 가중치 최대 30%가 사라져 쓰지 않았다.
- 처음 비교 렌더 때의 "뼈마다 전 정점 변환 후 가중합"은 0.3 s였다.

자산 (`data/meshes/makehuman/`에 커밋, CC0 1.0. 출처·커밋·blob 해시·라이선스 전문은 같은 폴더 LICENSE.txt.
upstream 과 바이트 단위로 같게 두려고 폴더 .gitattributes 에서 줄바꿈 변환을 끈다)
    https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/
        3dobjs/base.obj              1,749,303 B   기본 메시 hm08 (y 위, z 앞, +x = 캐릭터 왼쪽, 단위 dm)
        rigs/default.mhskel            117,790 B   기본 뼈대 (관절 = 정점 묶음의 평균)
        rigs/default_weights.mhw       897,521 B   스키닝 가중치
body 그룹 면만 쓴다 (눈·이빨·속눈썹·치마 같은 헬퍼 형상 제외). 사각형은 삼각형 둘로 나눈다.

비교 렌더: `python -m airis.viz.human_mesh` → `outputs/mesh_preview/makehuman_*.png`
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from .types import BodyParams, PoseParams, Scenario

ROOT = Path(__file__).resolve().parents[2]
MAKEHUMAN_DIR = ROOT / "data" / "meshes" / "makehuman"
MAKEHUMAN_FILES = ("base.obj", "default.mhskel", "default_weights.mhw")

# MakeHuman (x 왼쪽, y 위, z 앞) → 우리 (x 앞, y 왼쪽, z 위). 순환 치환이라 회전(행렬식 +1)이다.
_MH_TO_OURS = np.array([[0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0]])
_DM_TO_M = 0.1


@dataclass
class HumanMesh:
    """리깅된 사람 메시. 좌표는 우리 축(x 앞, y 왼쪽, z 위), 단위 m, 기본 자세 (머리말 "공개 API")."""
    vertices: np.ndarray            # (V,3)
    faces: np.ndarray               # (F,3) 삼각형
    bone_names: list[str]
    bone_parent: np.ndarray         # (B,)
    bone_head: np.ndarray           # (B,3)
    bone_tail: np.ndarray           # (B,3)
    weights: np.ndarray             # (V,B) 행 합 1
    source: str = ""

    def __post_init__(self) -> None:
        from scipy.sparse import csr_matrix

        self._index = {n: i for i, n in enumerate(self.bone_names)}
        # 희소 가중치 (CSR). 정점당 가중치가 있는 뼈는 최대 9개, 평균 3개 (nnz 약 4만).
        self._weights_csr = csr_matrix(np.asarray(self.weights, dtype=np.float64))

    def bone(self, name: str) -> int:
        return self._index[name]

    @property
    def height_m(self) -> float:
        """기본 자세 키 (정점 z 범위)."""
        return float(np.ptp(self.vertices[:, 2]))

    @property
    def hip_center(self) -> np.ndarray:
        """기본 자세 고관절 중심 (B 마네킹의 골반 원점에 해당)."""
        return 0.5 * (self.bone_head[self.bone("upperleg01.L")] + self.bone_head[self.bone("upperleg01.R")])

    def posed(self, pose: PoseParams) -> np.ndarray:
        return pose_vertices(self, pose)


# ---------------------------------------------------------------------------
# 로더
# ---------------------------------------------------------------------------
def _read_obj_body(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(전체 정점 (N,3), body 그룹 삼각형 (F,3) 전체 인덱스, body 정점 인덱스). 사각형은 둘로 나눈다."""
    verts, faces = [], []
    group = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("g "):
                group = line.split()[1]
            elif line.startswith("f ") and group == "body":
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                for k in range(1, len(idx) - 1):          # 부채꼴 삼각분할
                    faces.append([idx[0], idx[k], idx[k + 1]])
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    return verts, faces, np.unique(faces)


def load_makehuman(directory: Path | str | None = None) -> HumanMesh:
    """MakeHuman 기본 메시 + 기본 뼈대 + 가중치 → `HumanMesh` (body 그룹만, 헬퍼 제외).

    `directory` 가 None 이면 저장소의 `data/meshes/makehuman/`."""
    d = Path(directory) if directory is not None else MAKEHUMAN_DIR
    missing = [n for n in MAKEHUMAN_FILES if not (d / n).exists()]
    if missing:
        raise FileNotFoundError(f"MakeHuman 자산이 없다: {d} / {missing} "
                                "(airis/viz/human_mesh.py 머리말의 주소에서 받는다)")
    all_v, faces_all, body_idx = _read_obj_body(d / "base.obj")
    skel = json.loads((d / "default.mhskel").read_text(encoding="utf-8"))
    wts = json.loads((d / "default_weights.mhw").read_text(encoding="utf-8"))["weights"]

    joint = {k: all_v[v].mean(axis=0) for k, v in skel["joints"].items()}
    names = _topological(skel["bones"])
    index = {n: i for i, n in enumerate(names)}
    parent = np.array([index.get(skel["bones"][n]["parent"], -1) for n in names])
    head = np.array([joint[skel["bones"][n]["head"]] for n in names])
    tail = np.array([joint[skel["bones"][n]["tail"]] for n in names])

    remap = np.full(all_v.shape[0], -1)
    remap[body_idx] = np.arange(body_idx.size)
    W = np.zeros((body_idx.size, len(names)))
    for bname, pairs in wts.items():
        if bname not in index:
            continue
        arr = np.asarray(pairs, dtype=np.float64)
        vi = remap[arr[:, 0].astype(int)]
        ok = vi >= 0
        W[vi[ok], index[bname]] += arr[ok, 1]
    W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-12)

    to_ours = lambda p: (p @ _MH_TO_OURS.T) * _DM_TO_M          # noqa: E731
    return HumanMesh(
        vertices=to_ours(all_v[body_idx]),
        faces=remap[faces_all],
        bone_names=names,
        bone_parent=parent,
        bone_head=to_ours(head),
        bone_tail=to_ours(tail),
        weights=W,
        source="MakeHuman hm08 base mesh + default skeleton (CC0)",
    )


def _topological(bones: Mapping[str, dict]) -> list[str]:
    order, seen = [], set()

    def visit(n: str) -> None:
        if n in seen:
            return
        p = bones[n]["parent"]
        if p:
            visit(p)
        seen.add(n)
        order.append(n)

    for n in bones:
        visit(n)
    return order


@lru_cache(maxsize=2)
def cached_makehuman(directory: str = str(MAKEHUMAN_DIR)) -> HumanMesh:
    """프로세스당 한 번만 읽는다 (최적화 루프·대시보드용)."""
    return load_makehuman(directory)


# ---------------------------------------------------------------------------
# 회전
# ---------------------------------------------------------------------------
def _rot(axis: str, deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _axis_angle(axis: np.ndarray, deg: float) -> np.ndarray:
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    t = np.deg2rad(deg)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """단위 벡터 a 를 b 로 옮기는 최소 회전 (Rodrigues). 반대 방향이면 a 에 수직인 축으로 180°."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-9:
        perp = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        return _axis_angle(perp, 180.0)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def arm_targets(pose: PoseParams) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """B의 `joint_positions`와 같은 상완·전완 단위 방향 (몸통 기준, pitch·yaw 적용 전). {"L"|"R": (상완, 전완)}"""
    out = {}
    down = np.array([0.0, 0.0, -1.0])
    r_flex = _rot("y", -pose.shoulder_flexion)
    for side, sign in (("L", 1.0), ("R", -1.0)):
        r_sh = r_flex @ _rot("x", sign * pose.shoulder_abduction)
        upper = r_sh @ down
        forearm = _axis_angle(r_sh @ np.array([0.0, -1.0, 0.0]), pose.elbow_flexion) @ upper
        out[side] = (upper, forearm)
    return out


# ---------------------------------------------------------------------------
# 자세 → 뼈 변환 → 선형 블렌드 스키닝
# ---------------------------------------------------------------------------
def bone_transforms(mesh: HumanMesh, pose: PoseParams) -> np.ndarray:
    """뼈별 4×4 월드 변환 (B,4,4). 기본 자세 점 x → M_b·x (기본 자세 좌표계, 부스 배치 전).

    M_b = M_parent · T(pivot) · D_b · T(−pivot). D_b 는 머리말 "자세 규약"의 국소 회전이고,
    나머지 뼈는 D = I 라 부모를 그대로 따른다.
    """
    nb = len(mesh.bone_names)
    idx = mesh.bone
    hip_center = mesh.hip_center
    r_torso = _rot("z", pose.torso_yaw) @ _rot("y", pose.torso_pitch)
    targets = arm_targets(pose)

    local: dict[int, tuple[np.ndarray, np.ndarray]] = {
        idx("root"): (_rot("z", pose.torso_yaw), hip_center),
        idx("spine05"): (_rot("y", pose.torso_pitch), hip_center),
    }
    for side in ("L", "R"):
        local[idx(f"upperleg01.{side}")] = (_rot("y", -pose.hip_flexion),
                                           mesh.bone_head[idx(f"upperleg01.{side}")])
        local[idx(f"lowerleg01.{side}")] = (_rot("y", pose.knee_flexion),
                                           mesh.bone_head[idx(f"lowerleg01.{side}")])
    # 팔: (뼈, 관절 구간 시작, 끝, 목표 방향). 상완 = 어깨→팔꿈치, 전완 = 팔꿈치→손목
    arm: dict[int, tuple[int, int, np.ndarray]] = {}
    for side in ("L", "R"):
        upper_t, fore_t = targets[side]
        sh, el, wr = idx(f"upperarm01.{side}"), idx(f"lowerarm01.{side}"), idx(f"wrist.{side}")
        arm[sh] = (sh, el, r_torso @ upper_t)
        arm[el] = (el, wr, r_torso @ fore_t)

    M = np.zeros((nb, 4, 4))
    for b in range(nb):
        p = mesh.bone_parent[b]
        Mp = M[p] if p >= 0 else np.eye(4)
        if b in arm:
            a, c, want = arm[b]
            rest = mesh.bone_head[c] - mesh.bone_head[a]
            D, pivot = align(rest, Mp[:3, :3].T @ want), mesh.bone_head[a]
        elif b in local:
            D, pivot = local[b]
        else:
            M[b] = Mp
            continue
        T = np.eye(4)
        T[:3, :3] = D
        T[:3, 3] = pivot - D @ pivot
        M[b] = Mp @ T
    return M


def pose_vertices(mesh: HumanMesh, pose: PoseParams) -> np.ndarray:
    """선형 블렌드 스키닝 v' = (Σ_b w_vb · M_b) · v → (V,3), 기본 자세 좌표계.

    희소 가중치 행렬 (V,B) × 뼈 변환 (B,12) 으로 정점별 3×4 행렬을 만든 뒤 한 번 곱한다.
    """
    M = bone_transforms(mesh, pose)[:, :3, :].reshape(-1, 12)     # (B,12)
    A = np.asarray(mesh._weights_csr @ M).reshape(-1, 3, 4)       # (V,3,4)
    return np.einsum("vij,vj->vi", A[:, :, :3], mesh.vertices) + A[:, :, 3]


def resolve_pose(pose: PoseParams, scenario: Scenario | None) -> PoseParams:
    """`scenario.fixed_pose`(휠체어 고관절·무릎 90°)를 덮어쓴다. B의 `_resolve_pose`와 같다."""
    if scenario is None or not scenario.fixed_pose:
        return pose
    return replace(pose, **{k: float(v) for k, v in scenario.fixed_pose.items()})


def place_in_booth(verts: np.ndarray, mesh: HumanMesh, body: BodyParams, booth: Mapping,
                   scenario: Scenario | None = None) -> np.ndarray:
    """기본 자세 좌표계 정점 → 부스 좌표 (머리말 "좌표 규약"). 키를 `body.height_m`로 균일 축척한다."""
    s = body.height_m / mesh.height_m
    out = (np.asarray(verts, np.float64) - mesh.hip_center) * s
    out[:, 0] += float(booth["length_m"]) / 2.0
    if scenario is not None and scenario.seat_height_m is not None:
        out[:, 2] += float(scenario.seat_height_m)
    else:
        out[:, 2] -= out[:, 2].min()
    return out


@lru_cache(maxsize=1)
def _default_booth() -> dict:
    from .scenario import load_nozzle_layout
    return load_nozzle_layout()["booth"]


def posed_in_booth(mesh: HumanMesh, body: BodyParams, pose: PoseParams,
                   scenario: Scenario | None = None, booth: Mapping | None = None) -> np.ndarray:
    """자세를 입힌 정점 (V,3), 부스 좌표. 면은 `mesh.faces` 그대로. 시뮬레이션·안내 뷰의 진입점.

    체형 5개(`BodyParams`)를 뼈별 축척(`shape_mesh`)으로 맞춘 뒤 자세를 입히고 부스에 놓는다.
    """
    if booth is None:
        booth = _default_booth()
    shaped = shape_mesh(mesh, body, scenario)
    return place_in_booth(pose_vertices(shaped, resolve_pose(pose, scenario)), shaped, body, booth,
                          scenario)


def pose_mesh(asset: HumanMesh | None, body: BodyParams | None, pose: PoseParams,
              scenario: Scenario | None = None, booth: Mapping | None = None) -> np.ndarray:
    """자세를 입힌 정점 (V,3), 부스 좌표. `docs/interfaces.md` "airis/sim/human_mesh.py" 시그니처와 같다.

    `asset` 이 None 이면 캐시한 MakeHuman 기본 메시, `body` 가 None 이면 `MESH_DEFAULT_BODY`(메시 기본 체형).
    체형 5개를 뼈별 축척으로 맞춘다 (`shape_mesh`). 면은 `asset.faces`."""
    return posed_in_booth(asset if asset is not None else cached_makehuman(),
                          body if body is not None else MESH_DEFAULT_BODY, pose, scenario, booth)


#: MakeHuman 기본 메시를 키 1.70 m 로 축척해 잰 체형 (관절 중심 정의, 2026-09-18 확정). BodyParams 기본값을
#: 이 값으로 바꾸는 것은 통합의 types.py PR 이다. 그 전에는 BodyParams() 가 캡슐 시절 값(0.42/0.62/0.85)이다.
MESH_DEFAULT_BODY = BodyParams(height_m=1.70, shoulder_width_m=0.342, torso_depth_m=0.194,
                               arm_length_m=0.463, leg_length_m=0.883)


#: `docs/interfaces.md` 의 이름
MeshAsset = HumanMesh


# ---------------------------------------------------------------------------
# 뼈 → 부위 (docs/mesh_transition.md "패치 샘플링 규약", B 설계 승인 2026-09-18)
# ---------------------------------------------------------------------------
_HEAD_PREFIXES = ("head", "jaw", "neck", "eye", "oculi", "orbicularis", "oris", "levator", "risorius",
                  "tongue", "temporalis", "special")
_TORSO_PREFIXES = ("root", "spine", "pelvis", "breast", "clavicle")
_ARM_PREFIXES = ("shoulder", "upperarm", "lowerarm", "wrist", "metacarpal", "finger")
_LEG_PREFIXES = ("upperleg", "lowerleg", "foot", "toe")


def bone_part(name: str) -> int:
    """MakeHuman 뼈 이름 → PART_NAMES 인덱스. 몸통은 torso_front (앞뒤는 법선으로 나중에 나눈다).

    shoulder01(삼각근 윗부분)은 캡슐판 경계(어깨 관절부터 팔)와 맞춰 arms 다.
    """
    from .types import PART_NAMES
    base = name.split(".")[0]
    for prefixes, part in ((_HEAD_PREFIXES, "head"), (_TORSO_PREFIXES, "torso_front"),
                           (_ARM_PREFIXES, "arms"), (_LEG_PREFIXES, "legs")):
        if base.startswith(prefixes):
            return PART_NAMES.index(part)
    raise KeyError(f"부위를 정하지 않은 뼈: {name}")


def mirror_bone_index(names: list[str]) -> np.ndarray:
    """(B,) 좌우 짝 뼈 인덱스 (.L ↔ .R, 가운데 뼈는 자기 자신)."""
    index = {n: i for i, n in enumerate(names)}

    def swap(n: str) -> str:
        if n.endswith(".L"):
            return n[:-2] + ".R"
        if n.endswith(".R"):
            return n[:-2] + ".L"
        return n
    return np.array([index[swap(n)] for n in names])



# ---------------------------------------------------------------------------
# 체형 측정 (BodyParams 정의, docs/mesh_transition.md 8, 2026-09-18 확정)
# ---------------------------------------------------------------------------
#: 가슴 두께 단면 반높이 (m, 휴지 자세). "breast 뼈 머리 높이(유두선) ±1 cm 몸통 단면의 x 범위".
CHEST_SLICE_HALF_M = 0.01


def vertex_parts(mesh: HumanMesh) -> np.ndarray:
    """(V,) 정점 부위 = 가중치 최댓값 뼈의 부위 (몸통은 torso_front)."""
    W = mesh.weights
    parts = np.array([bone_part(n) if W[:, i].any() else -1 for i, n in enumerate(mesh.bone_names)])
    return parts[np.argmax(W, axis=1)]


def measure_body(mesh: HumanMesh) -> dict[str, float]:
    """휴지 자세 메시의 BodyParams 5개 (메시 단위 m, 관절 중심 기준).

    - height_m: 정점 z 범위 (정수리 − 발바닥)
    - shoulder_width_m: 좌우 upperarm01 머리(상완골 관절) 간격
    - torso_depth_m: breast 뼈 머리 높이 ±1 cm 에 있는 몸통 부위 정점의 x 범위 (가슴 두께)
    - arm_length_m: 상완 + 전완 구간 합 (upperarm01 머리 → lowerarm01 머리 → wrist 머리, 왼쪽). 자세 불변이다
      (직선 거리는 MakeHuman A자 휴지 자세의 팔꿈치 굽힘에 따라 바뀐다). 캡슐 정의·포즈 추정과 같다.
    - leg_length_m: 고관절 중심 z − 발바닥 z
    """
    from .types import PART_NAMES
    h = mesh.bone_head
    v = mesh.vertices
    torso = vertex_parts(mesh) == PART_NAMES.index("torso_front")
    zb = h[mesh.bone("breast.L"), 2]
    band = torso & (np.abs(v[:, 2] - zb) <= CHEST_SLICE_HALF_M)
    return {
        "height_m": float(np.ptp(v[:, 2])),
        "shoulder_width_m": float(np.linalg.norm(h[mesh.bone("upperarm01.L")] - h[mesh.bone("upperarm01.R")])),
        "torso_depth_m": float(np.ptp(v[band, 0])),
        "arm_length_m": float(np.linalg.norm(h[mesh.bone("lowerarm01.L")] - h[mesh.bone("upperarm01.L")])
                              + np.linalg.norm(h[mesh.bone("wrist.L")] - h[mesh.bone("lowerarm01.L")])),
        "leg_length_m": float(mesh.hip_center[2] - v[:, 2].min()),
    }


def _chest_center_x(mesh: HumanMesh) -> float:
    from .types import PART_NAMES
    v = mesh.vertices
    torso = vertex_parts(mesh) == PART_NAMES.index("torso_front")
    band = torso & (np.abs(v[:, 2] - mesh.bone_head[mesh.bone("breast.L"), 2]) <= CHEST_SLICE_HALF_M)
    return float(0.5 * (v[band, 0].min() + v[band, 0].max()))


# ---------------------------------------------------------------------------
# 체형 맞춤: 휴지 자세 뼈별 아핀 (B 설계, 2026-09-18 승인)
# ---------------------------------------------------------------------------
def _affine(A: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = A
    M[:3, 3] = t
    return M


def _bone_group(name: str) -> str:
    base, _, side = name.partition(".")
    if base.startswith(("upperleg", "lowerleg", "foot", "toe")):
        return "leg"
    if base.startswith("clavicle"):
        return "clavicle"
    if base.startswith("shoulder"):
        return "shoulder"
    if base.startswith("upperarm"):
        return "upperarm"
    if base.startswith("lowerarm"):
        return "lowerarm"
    if base.startswith(("wrist", "metacarpal", "finger")):
        return "hand"
    if base.startswith(("root", "spine", "pelvis", "breast")):
        return "torso"
    return "head"                                             # 목·머리·얼굴


def shape_affines(mesh: HumanMesh, body: BodyParams) -> np.ndarray:
    """(B,4,4) 휴지 자세 뼈별 아핀 S_b. 체형 정점 v' = (Σ_b w_vb S_b) v.

    휴지 메시 단위(키 H0)에서 목표값을 s = body.height_m / H0 로 나눠 맞춘 뒤, 부스 배치가 s 로 균일 축척한다.
      1. 다리: 고관절 높이 z 로 kL 배 (upperleg·lowerleg·foot·toe)
      2. 몸통: 가슴 중심 x 기준 전후 kD 배(가슴 두께), 고관절 z 기준 상하 kT 배.
         kT = 1 + (1 − kL)·LL0 / (z_neck − z_hip) 이라 다리가 길어진 만큼 몸통이 줄어 키가 H0 로 유지된다.
      3. 목·머리: 몸통이 옮긴 목 밑(neck01 머리)만큼 평행이동 (모양 유지)
      4. 쇄골: 몸통 변환 + 좌우 ky 배 (어깨 관절 간격)
      5. 팔: 어깨 관절이 옮긴 만큼 평행이동 + 상완·전완을 각자 휴지 축 방향으로 같은 kA 배 (구간 합이 kA 배),
         손(wrist·metacarpal·finger)은 평행이동만
    """
    ref = _reference_mesh(mesh)                               # 측정 정의는 원본(고해상도) 메시 기준
    m0 = measure_body(ref)
    H0 = m0["height_m"]
    s = body.height_m / H0
    kL = (body.leg_length_m / s) / m0["leg_length_m"]
    kD = (body.torso_depth_m / s) / m0["torso_depth_m"]
    ky = (body.shoulder_width_m / s) / m0["shoulder_width_m"]
    kA = (body.arm_length_m / s) / m0["arm_length_m"]
    h = mesh.bone_head
    z_hip = float(mesh.hip_center[2])
    z_neck = float(h[mesh.bone("neck01"), 2])
    kT = 1.0 + (1.0 - kL) * m0["leg_length_m"] / (z_neck - z_hip)
    xc = _chest_center_x(ref)

    torso = _affine(np.diag([kD, 1.0, kT]), np.array([xc * (1 - kD), 0.0, z_hip * (1 - kT)]))
    clavicle = _affine(np.diag([kD, ky, kT]), torso[:3, 3])
    leg = _affine(np.diag([1.0, 1.0, kL]), np.array([0.0, 0.0, z_hip * (1 - kL)]))
    neck = h[mesh.bone("neck01")]
    head = _affine(np.eye(3), torso[:3, :3] @ neck + torso[:3, 3] - neck)

    arm: dict[str, dict[str, np.ndarray]] = {}
    for side in ("L", "R"):
        sh, el, wr = (h[mesh.bone(f"{b}.{side}")] for b in ("upperarm01", "lowerarm01", "wrist"))
        a = (el - sh) / np.linalg.norm(el - sh)
        b = (wr - el) / np.linalg.norm(wr - el)
        d_sh = clavicle[:3, :3] @ sh + clavicle[:3, 3] - sh           # 어깨 관절 이동
        Pa, Pb = (kA - 1.0) * np.outer(a, a), (kA - 1.0) * np.outer(b, b)
        d_el = Pa @ (el - sh)
        d_wr = d_el + Pb @ (wr - el)
        arm[side] = {
            "shoulder": _affine(np.eye(3), d_sh),
            "upperarm": _affine(np.eye(3) + Pa, d_sh - Pa @ sh),
            "lowerarm": _affine(np.eye(3) + Pb, d_sh + d_el - Pb @ el),
            "hand": _affine(np.eye(3), d_sh + d_wr),
        }

    S = np.empty((len(mesh.bone_names), 4, 4))
    for i, name in enumerate(mesh.bone_names):
        g = _bone_group(name)
        side = name.rpartition(".")[2]
        if g in ("shoulder", "upperarm", "lowerarm", "hand"):
            S[i] = arm[side][g]
        else:
            S[i] = {"torso": torso, "clavicle": clavicle, "leg": leg, "head": head}[g]
    return S


def _reference_mesh(mesh: HumanMesh) -> HumanMesh:
    """체형 측정 기준 메시. sim 메시는 원본의 부분 정점이라 단면 측정(가슴 두께)이 달라지므로
    원본 기본 메시로 잰 비율을 같은 뼈대에 그대로 적용한다 → sim 몸 = 맞춘 원본 몸의 데시메이션."""
    return cached_makehuman() if isinstance(mesh, SimMesh) else mesh


def _apply_affines(mesh: HumanMesh, S: np.ndarray) -> np.ndarray:
    A = np.asarray(mesh._weights_csr @ S[:, :3, :].reshape(-1, 12)).reshape(-1, 3, 4)
    return np.einsum("vij,vj->vi", A[:, :, :3], mesh.vertices) + A[:, :, 3]


def shape_mesh(mesh: HumanMesh, body: BodyParams, scenario: Scenario | None = None) -> HumanMesh:
    """체형을 맞춘 휴지 자세 메시 (같은 클래스, 새 인스턴스). 키는 휴지 키 H0 그대로 두고 부스 배치가 축척한다.

    뼈 머리·꼬리도 각 뼈의 아핀으로 옮긴다 (자세 회전의 중심). 같은 (메시, 체형) 은 캐시한다.
    """
    targets = _scenario_targets(scenario)
    key = (body.height_m, body.shoulder_width_m, body.torso_depth_m, body.arm_length_m, body.leg_length_m,
           tuple(sorted(targets.items())))
    cache = mesh.__dict__.setdefault("_shape_cache", {})
    if key in cache:
        return cache[key]
    S = shape_affines(mesh, body)
    head = np.einsum("bij,bj->bi", S[:, :3, :3], mesh.bone_head) + S[:, :3, 3]
    tail = np.einsum("bij,bj->bi", S[:, :3, :3], mesh.bone_tail) + S[:, :3, 3]
    fields = {f: getattr(mesh, f) for f in mesh.__dataclass_fields__}
    base = fields["vertices"]
    if targets:                                     # 모디파이어 타깃은 체형 축척 전 휴지 정점에 더한다
        base = base + sum(w * target_displacement(mesh, name) for name, w in targets.items())
    fields.update(vertices=base, bone_head=head, bone_tail=tail)
    fields.update(vertices=_apply_affines(type(mesh)(**fields), S))
    shaped = type(mesh)(**fields)
    shaped.__dict__["_shape_cache"] = {key: shaped}           # 이미 맞춘 메시를 다시 맞추지 않게
    if len(cache) > 64:
        cache.clear()
    cache[key] = shaped
    return shaped


# ---------------------------------------------------------------------------
# sim 메시 (scripts/build_sim_mesh.py 가 만든 약 5.6k 삼각형, 좌우 대칭)
# ---------------------------------------------------------------------------
SIM_MESH_FILE = MAKEHUMAN_DIR / "sim_mesh.npz"


@dataclass
class SimMesh(HumanMesh):
    """데시메이션한 시뮬레이션 메시. `HumanMesh` 연산(자세·체형·부스 배치)을 그대로 쓴다."""
    face_part: np.ndarray | None = None          # (F,) PART_NAMES 인덱스 (몸통은 torso_front)
    mirror_vertex: np.ndarray | None = None      # (V,)
    mirror_face: np.ndarray | None = None        # (F,)
    mirror_face_perm: np.ndarray | None = None   # (F,3)
    obj_vertex: np.ndarray | None = None         # (V,) base.obj 정점 번호

    @property
    def height_m(self) -> float:
        """원본 기본 메시의 휴지 키 H0. 체형 맞춤은 키를 H0 로 유지하고, sim 은 원본의 부분 정점이라
        정수리·발바닥 정점이 조금 다를 수 있으므로 부스 축척을 원본과 같게 둔다."""
        return cached_makehuman().height_m


def load_sim_mesh(path: Path | str | None = None) -> SimMesh:
    d = np.load(Path(path) if path is not None else SIM_MESH_FILE, allow_pickle=False)
    nv, nb = d["vertices"].shape[0], d["bone_names"].shape[0]
    W = np.zeros((nv, nb))
    rows = np.repeat(np.arange(nv), d["weight_bone"].shape[1])
    np.add.at(W, (rows, d["weight_bone"].ravel().astype(np.int64)), d["weight_val"].ravel())
    return SimMesh(
        vertices=d["vertices"].astype(np.float64), faces=d["faces"].astype(np.int64),
        bone_names=[str(n) for n in d["bone_names"]], bone_parent=d["bone_parent"].astype(np.int64),
        bone_head=d["bone_head"], bone_tail=d["bone_tail"], weights=W, source=str(d["source"]),
        face_part=d["face_part"].astype(np.int64), mirror_vertex=d["mirror_vertex"].astype(np.int64),
        mirror_face=d["mirror_face"].astype(np.int64), mirror_face_perm=d["mirror_face_perm"].astype(np.int64),
        obj_vertex=d["obj_vertex"].astype(np.int64),
    )


@lru_cache(maxsize=2)
def cached_sim_mesh(path: str = str(SIM_MESH_FILE)) -> SimMesh:
    return load_sim_mesh(path)



# ---------------------------------------------------------------------------
# 메시 몸 → BodyState (docs/mesh_transition.md "패치 샘플링 규약", interfaces.md build_body 메시 계약)
# ---------------------------------------------------------------------------
#: 면별 확률 반올림용 난수 씨앗 (결정론). 거울 짝 면은 같은 값을 쓴다.
_PATCH_SEED = 20260918
#: torso_front/back 판정 허용치 (body.py `_FRONT_BACK_TOL` 과 같은 규칙, #42)
_FRONT_BACK_TOL = 1e-9


def _halton(i: np.ndarray, base: int) -> np.ndarray:
    f = np.ones_like(i, dtype=np.float64)
    r = np.zeros_like(i, dtype=np.float64)
    i = i.astype(np.int64).copy()
    while (i > 0).any():
        f = f / base
        r = r + f * (i % base)
        i = i // base
    return r


def _canonical_faces(mesh: SimMesh) -> np.ndarray:
    """(F,) 거울 짝 중 작은 번호. 짝 면은 같은 표본 수·같은 무게중심 좌표(순서만 치환)를 쓴다."""
    return np.minimum(np.arange(len(mesh.faces)), mesh.mirror_face)


def _face_areas(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cr = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    a2 = np.linalg.norm(cr, axis=1)
    return 0.5 * a2, cr / np.maximum(a2, 1e-30)[:, None]


def _patch_counts(shaped: SimMesh, scale: float, patches_per_m2: float) -> tuple[np.ndarray, np.ndarray]:
    """(F,) 면별 패치 수와 기대 패치 수. 체형 맞춘 휴지 면적(부스 축척) × 밀도를 확률 반올림한다.

    자세와 무관하다 (같은 체형이면 자세가 바뀌어도 패치 수·면 배정이 같다). 거울 짝 면은 면적 평균과
    같은 난수를 써서 수가 정확히 같다.
    """
    area, _ = _face_areas(shaped.vertices, shaped.faces)
    area = 0.5 * (area + area[shaped.mirror_face]) * scale * scale
    expect = area * patches_per_m2
    base = np.floor(expect)
    u = np.random.default_rng(_PATCH_SEED).random(len(area))[_canonical_faces(shaped)]
    return (base + (u < expect - base)).astype(np.int64), expect


def _barycentric(counts: np.ndarray, mesh: SimMesh) -> tuple[np.ndarray, np.ndarray]:
    """(N,) 면 번호와 (N,3) 무게중심 좌표. 면 안 k 번째 점은 (k + ½)/n 층화 + Halton(3) (결정론).

    거울 짝 면(정점 순서 [0,2,1])은 같은 좌표를 [0,2,1] 로 치환해 정확히 거울 위치에 놓는다.
    """
    face = np.repeat(np.arange(len(counts)), counts)
    start = np.repeat(np.cumsum(counts) - counts, counts)
    k = np.arange(len(face)) - start
    n = counts[face]
    r1 = (k + 0.5) / n
    r2 = _halton(k + 1, 3)
    sq = np.sqrt(r1)
    bary = np.stack([1.0 - sq, sq * (1.0 - r2), sq * r2], axis=1)
    flip = _canonical_faces(mesh)[face] != face
    bary[flip] = bary[flip][:, [0, 2, 1]]
    return face, bary


# --- 근사 캡슐 (A 입자 충돌·D 캡슐 폴백). 휴지 체형에서 맞추고 자세는 주 뼈 변환으로 강체 이동 ---
_TORSO_BANDS = 5
_CAPSULE_RADIUS_PERCENTILE = 75.0


def _capsule_groups(mesh: HumanMesh) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """(V,) 정점별 캡슐 번호와 [(이름, 부위)] 목록. 몸통은 고관절~목 밑을 높이 띠로 나눈다."""
    from .types import PART_NAMES
    W = mesh.weights
    names = mesh.bone_names
    arg = np.argmax(W, axis=1)
    part = vertex_parts(mesh)
    z = mesh.vertices[:, 2]
    z0, z1 = float(mesh.hip_center[2]) - 0.12, float(mesh.bone_head[mesh.bone("neck01"), 2])
    groups: list[tuple[str, int]] = []
    gid = np.full(len(z), -1)

    def add(label: str, part_name: str, mask: np.ndarray) -> None:
        gid[mask & (gid < 0)] = len(groups)
        groups.append((label, PART_NAMES.index(part_name)))

    base = np.array([names[b].split(".")[0] for b in arg])
    side = np.array([names[b].rpartition(".")[2] if "." in names[b] else "" for b in arg])
    add("neck", "head", np.char.startswith(base, "neck"))
    add("head", "head", part == PART_NAMES.index("head"))
    for sd in ("L", "R"):
        add(f"upperarm.{sd}", "arms", (side == sd) & (np.char.startswith(base, "upperarm")
                                                        | np.char.startswith(base, "shoulder")))
        add(f"lowerarm.{sd}", "arms", (side == sd) & np.char.startswith(base, "lowerarm"))
        add(f"hand.{sd}", "arms", (side == sd) & (part == PART_NAMES.index("arms")))
        add(f"upperleg.{sd}", "legs", (side == sd) & np.char.startswith(base, "upperleg"))
        add(f"lowerleg.{sd}", "legs", (side == sd) & np.char.startswith(base, "lowerleg"))
        add(f"foot.{sd}", "legs", (side == sd) & (part == PART_NAMES.index("legs")))
    torso = part == PART_NAMES.index("torso_front")
    edges = np.linspace(z0, z1, _TORSO_BANDS + 1)
    band = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, _TORSO_BANDS - 1)
    for k in range(_TORSO_BANDS):
        add(f"torso{k}", "torso_front", torso & (band == k))
    assert (gid >= 0).all()
    return gid, groups


def _fit_capsules(shaped: HumanMesh) -> dict:
    """휴지(체형 맞춤) 좌표의 캡슐 (K,7), 부위 (K,), 주 뼈 (K,), 정점별 캡슐 번호 (V,).

    주성분 축 위 투영 범위 [t_min + r, t_max − r], 반지름 r = 축까지 거리의 75 백분위.
    좌우 짝 캡슐은 왼쪽을 맞춘 뒤 y 반사로 만들어 정확히 대칭이다.
    """
    gid, groups = _capsule_groups(shaped)
    v = shaped.vertices
    arg = np.argmax(shaped.weights, axis=1)
    label_index = {g[0]: i for i, g in enumerate(groups)}
    mb = mirror_bone_index(shaped.bone_names)
    caps = np.zeros((len(groups), 7))
    driver = np.zeros(len(groups), dtype=np.int64)
    M = np.array([1.0, -1.0, 1.0])
    for i, (label, _) in enumerate(groups):
        if label.endswith(".R"):
            continue
        pts = v[gid == i]
        c = pts.mean(axis=0)
        if not label.endswith(".L"):                          # 가운데 캡슐은 축을 좌우 대칭으로 맞춘다
            c[1] = 0.0
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        a = vt[0]
        if label.startswith("torso"):                         # 몸통 띠: 좌우(y) 축 캡슐
            a = np.array([0.0, 1.0, 0.0])
        elif not label.endswith(".L"):                        # 목·머리: 대칭면(xz) 안의 주축
            _, _, vt2 = np.linalg.svd((pts - c)[:, [0, 2]], full_matrices=False)
            a = np.array([vt2[0, 0], 0.0, vt2[0, 1]])
        t = (pts - c) @ a
        radial = np.linalg.norm((pts - c) - t[:, None] * a, axis=1)
        r = float(np.percentile(radial, _CAPSULE_RADIUS_PERCENTILE))
        lo, hi = t.min() + r, t.max() - r
        if hi < lo:
            lo = hi = 0.5 * (t.min() + t.max())
        caps[i] = np.concatenate([c + lo * a, c + hi * a, [r]])
        bones, counts = np.unique(arg[gid == i], return_counts=True)
        driver[i] = bones[np.argmax(counts)]
        if label.endswith(".L"):
            j = label_index[label[:-2] + ".R"]
            caps[j] = np.concatenate([caps[i, :3] * M, caps[i, 3:6] * M, [r]])
            driver[j] = mb[driver[i]]
    parts = np.array([g[1] for g in groups], dtype=np.int32)
    return {"caps": caps, "part": parts, "driver": driver, "vertex_capsule": gid}


def build_mesh_body(body: BodyParams | None, pose: PoseParams, scenario: Scenario,
                    patches_per_m2: float = 2000.0, mesh: SimMesh | None = None) -> "BodyState":
    """MakeHuman sim 메시 몸 → `BodyState` (interfaces.md build_body 메시 계약).

    - mesh_vertices/mesh_faces/mesh_face_part: 자세·체형 적용, 부스 좌표. 몸통 면은 법선으로 앞뒤.
    - 패치: 면 위 점, 면적 비례(체형 휴지 면적 기준 확률 반올림), 법선 = 면 법선, 부위 = 면 부위.
      y → −y 거울 짝이 정확하다 (#42 규칙).
    - capsules/capsule_part: 뼈에 맞춘 근사 캡슐 (몸통 여러 개 = torso_front) + 휠체어 프레임(−1).
    - patch_capsule: 패치가 놓인 면의 캡슐 (캡슐 폴백용).
    """
    from scipy.spatial.transform import Rotation

    from .body import _wheelchair_segments, _capsule_array
    from .types import PART_NAMES, BodyState

    if patches_per_m2 <= 0.0:
        raise ValueError("patches_per_m2 는 양수여야 한다")
    body = body if body is not None else MESH_DEFAULT_BODY           # 메시 기본 체형 (뼈 축척 항등)
    mesh = mesh if mesh is not None else cached_sim_mesh()
    shaped = shape_mesh(mesh, body, scenario)
    booth = _default_booth()
    p = resolve_pose(pose, scenario)
    M = bone_transforms(shaped, p)
    A = np.asarray(shaped._weights_csr @ M[:, :3, :].reshape(-1, 12)).reshape(-1, 3, 4)
    rest_posed = np.einsum("vij,vj->vi", A[:, :, :3], shaped.vertices) + A[:, :, 3]

    # 부스 배치 (place_in_booth 와 같은 식을 캡슐에도 쓰려고 풀어 쓴다)
    scale = body.height_m / shaped.height_m
    hip = shaped.hip_center
    verts = (rest_posed - hip) * scale
    offset = np.array([float(booth["length_m"]) / 2.0, 0.0, 0.0])
    if scenario is not None and scenario.seat_height_m is not None:
        offset[2] = float(scenario.seat_height_m)
    else:
        offset[2] = -verts[:, 2].min()
    verts = verts + offset

    faces = shaped.faces
    f_area, f_normal = _face_areas(verts, faces)
    r_body = Rotation.from_euler("z", p.torso_yaw, degrees=True) * Rotation.from_euler("y", p.torso_pitch,
                                                                                      degrees=True)
    forward = r_body.apply([1.0, 0.0, 0.0])
    front, back = PART_NAMES.index("torso_front"), PART_NAMES.index("torso_back")
    face_part = np.where((shaped.face_part == front) & (f_normal @ forward <= _FRONT_BACK_TOL),
                         back, shaped.face_part)

    counts, expect = _patch_counts(shaped, scale, patches_per_m2)
    pf, bary = _barycentric(counts, shaped)
    tri = verts[faces[pf]]                                                 # (N,3,3)
    patch_pos = np.einsum("nk,nkd->nd", bary, tri)
    # 패치 면적 = 면 면적 / 기대 패치 수 (≈ 1/밀도). "면 면적 / 실제 패치 수" 로 하면 패치가 없는 면의
    # 면적이 빠지고(400/m² 에서 표면적의 약 30% 만 남음), 뽑힌 면이 면적² 에 비례해 가중돼 면적 가중
    # 평균이 큰 면 쪽으로 치우친다. 기대값으로 나누면 면마다 기대 합이 그 면 면적이라 편향이 없다.
    patch_area = f_area[pf] / expect[pf]

    fit = shaped.__dict__.get("_capsule_fit")
    if fit is None:
        fit = _fit_capsules(shaped)
        shaped.__dict__["_capsule_fit"] = fit
    caps = fit["caps"].copy()
    Md = M[fit["driver"]]
    for k in (0, 3):
        pts = caps[:, k:k + 3]
        caps[:, k:k + 3] = (np.einsum("kij,kj->ki", Md[:, :3, :3], pts) + Md[:, :3, 3] - hip) * scale + offset
    caps[:, 6] *= scale
    cap_part = fit["part"]
    if scenario is not None and scenario.name == "wheelchair":
        hip_booth = (np.einsum("ij,j->i", M[shaped.bone("root"), :3, :3], hip)
                     + M[shaped.bone("root"), :3, 3] - hip) * scale + offset
        wc = _capsule_array(_wheelchair_segments({"pelvis": hip_booth}, r_body)).astype(np.float64)
        caps = np.concatenate([caps, wc])
        cap_part = np.concatenate([cap_part, np.full(len(wc), -1, dtype=np.int32)])
    vcap = fit["vertex_capsule"][faces]                                    # (F,3)
    face_cap = np.where(vcap[:, 1] == vcap[:, 2], vcap[:, 1], vcap[:, 0])  # 세 정점 중 다수

    return BodyState(
        patch_pos=patch_pos.astype(np.float32),
        patch_normal=f_normal[pf].astype(np.float32),
        patch_area=patch_area.astype(np.float32),
        patch_part=face_part[pf].astype(np.int32),
        capsules=caps.astype(np.float32),
        capsule_part=cap_part.astype(np.int32),
        patch_capsule=face_cap[pf].astype(np.int32),
        mesh_vertices=verts.astype(np.float32),
        mesh_faces=faces.astype(np.int32),
        mesh_face_part=face_part.astype(np.int32),
        patch_face=pf.astype(np.int32),
    )



# ---------------------------------------------------------------------------
# MakeHuman 모디파이어 타깃 (임산부 배 등). 파일 형식: 줄마다 "base.obj 정점 번호 dx dy dz" (MakeHuman 축, dm)
# ---------------------------------------------------------------------------
def _scenario_targets(scenario: Scenario | None) -> dict[str, float]:
    """시나리오의 메시 타깃 중 파일이 있는 것만 {이름: 가중치}."""
    if scenario is None:
        return {}
    from .scenario import scenario_mesh_targets
    return {n: w for n, w in scenario_mesh_targets(scenario.name).items()
            if (MAKEHUMAN_DIR / f"{n}.target").exists()}


def scenario_targets_names(name: str) -> list[str]:
    """설정(scenarios.yaml mesh_targets)에 적힌 타깃 이름 (파일 유무와 무관)."""
    from .scenario import scenario_mesh_targets
    return sorted(scenario_mesh_targets(name))


def read_target(path: Path | str) -> dict[int, np.ndarray]:
    """MakeHuman .target → {base.obj 정점 번호: 변위 (우리 축, m)}."""
    out: dict[int, np.ndarray] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        idx, dx, dy, dz = line.split()[:4]
        out[int(idx)] = (_MH_TO_OURS @ np.array([float(dx), float(dy), float(dz)])) * _DM_TO_M
    return out


def target_displacement(mesh: HumanMesh, name: str, directory: Path | None = None) -> np.ndarray:
    """(V,3) 이 메시 정점에 대응하는 타깃 변위. 원본 메시는 body 정점 순서, sim 메시는 obj_vertex 로 찾는다."""
    d = Path(directory) if directory is not None else MAKEHUMAN_DIR
    cache = mesh.__dict__.setdefault("_target_cache", {})
    key = (str(d), name)
    if key not in cache:
        tgt = read_target(d / f"{name}.target")
        if isinstance(mesh, SimMesh) and mesh.obj_vertex is not None:
            obj = mesh.obj_vertex
        else:
            _, _, obj = _read_obj_body(MAKEHUMAN_DIR / "base.obj")
        disp = np.zeros((len(obj), 3))
        for i, o in enumerate(obj):
            if int(o) in tgt:
                disp[i] = tgt[int(o)]
        cache[key] = disp
    return cache[key]
