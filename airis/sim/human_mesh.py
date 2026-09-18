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
    """자세를 입힌 정점 (V,3), 부스 좌표. 면은 `mesh.faces` 그대로. 시뮬레이션·안내 뷰의 진입점."""
    if booth is None:
        booth = _default_booth()
    return place_in_booth(pose_vertices(mesh, resolve_pose(pose, scenario)), mesh, body, booth,
                          scenario)


def pose_mesh(asset: HumanMesh | None, body: BodyParams | None, pose: PoseParams,
              scenario: Scenario | None = None, booth: Mapping | None = None) -> np.ndarray:
    """자세를 입힌 정점 (V,3), 부스 좌표. `docs/interfaces.md` "airis/sim/human_mesh.py" 시그니처와 같다.

    `asset` 이 None 이면 캐시한 MakeHuman 기본 메시, `body` 가 None 이면 `BodyParams()`(키 1.70 m 균일 축척).
    면은 `asset.faces`. 체형은 아직 키만 맞춘다 (뼈별 축척·타깃은 B 이관 때)."""
    return posed_in_booth(asset if asset is not None else cached_makehuman(),
                          body if body is not None else BodyParams(), pose, scenario, booth)


#: `docs/interfaces.md` 의 이름
MeshAsset = HumanMesh
