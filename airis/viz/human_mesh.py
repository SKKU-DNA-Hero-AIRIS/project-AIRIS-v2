"""사람 메시 (MakeHuman CC0 기본 메시 + 선형 블렌드 스키닝). 소유자: E.

시뮬레이션 몸을 캡슐 마네킹에서 사람 메시로 바꾸는 작업(③)의 첫 부품이다. 메시를 불러와 우리 자세
변수(`PoseParams` 7개)를 뼈 회전으로 바꾸고 정점을 움직인다. numpy 만 쓴다 (Taichi·torch 없음).
B가 체형 맞춤·패치 샘플링에서, E가 3D 안내 뷰에서 같이 쓴다.

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
    v = posed_in_booth(mesh, body, pose, scenario)   # (V,3) 부스 좌표 (아래 규약). 시뮬레이션은 이것을 쓴다

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

성능 (Windows, CPU, 다른 트랙 계산 8 프로세스가 도는 상태): 로드 약 0.2 s (1회), 자세 한 번
(`posed_in_booth`) 약 16 ms. 가중치로 뼈 변환을 정점별 3×4 행렬로 먼저 섞은 뒤 한 번 곱한다.
처음 비교 렌더 때 쓴 "뼈마다 전 정점 변환 후 가중합"은 0.3 s였고 결과는 1e-15 안에서 같다.

자산 (git 밖 `data/meshes/makehuman/`, CC0 1.0. 저장소 LICENSE.md "C. The license for the bundled assets")
    https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/
        3dobjs/base.obj              1,749,303 B   기본 메시 hm08 (y 위, z 앞, +x = 캐릭터 왼쪽, 단위 dm)
        rigs/default.mhskel            117,790 B   기본 뼈대 (관절 = 정점 묶음의 평균)
        rigs/default_weights.mhw       897,521 B   스키닝 가중치
body 그룹 면만 쓴다 (눈·이빨·속눈썹·치마 같은 헬퍼 형상 제외). 사각형은 삼각형 둘로 나눈다.

비교 렌더: `python -m airis.viz.human_mesh` → `outputs/mesh_preview/makehuman_*.png`
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..sim.types import BodyParams, PoseParams, Scenario

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
        self._index = {n: i for i, n in enumerate(self.bone_names)}
        self._used = np.flatnonzero(self.weights.any(axis=0))      # 가중치가 있는 뼈만 섞는다

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


def load_makehuman(directory: Path | str = MAKEHUMAN_DIR) -> HumanMesh:
    """MakeHuman 기본 메시 + 기본 뼈대 + 가중치 → `HumanMesh` (body 그룹만, 헬퍼 제외)."""
    d = Path(directory)
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

    뼈 변환 (U,3,4)를 가중치로 정점별 (V,3,4)로 먼저 섞은 뒤 한 번 곱한다 (행렬곱 1번).
    """
    M = bone_transforms(mesh, pose)[mesh._used, :3, :]            # (U,3,4)
    A = (mesh.weights[:, mesh._used] @ M.reshape(len(mesh._used), 12)).reshape(-1, 3, 4)
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


def posed_in_booth(mesh: HumanMesh, body: BodyParams, pose: PoseParams,
                   scenario: Scenario | None = None, booth: Mapping | None = None) -> np.ndarray:
    """자세를 입힌 정점 (V,3), 부스 좌표. 면은 `mesh.faces` 그대로. 시뮬레이션·안내 뷰의 진입점."""
    if booth is None:
        from ..sim.scenario import load_nozzle_layout
        booth = load_nozzle_layout()["booth"]
    return place_in_booth(pose_vertices(mesh, resolve_pose(pose, scenario)), mesh, body, booth,
                          scenario)


# ---------------------------------------------------------------------------
# 그림 (비교 렌더)
# ---------------------------------------------------------------------------
SKIN_COLOR = "#e0b49a"


def figure_from_mesh(verts: np.ndarray, faces: np.ndarray, *, overlay_state=None,
                     booth: Mapping | None = None, nozzle=None, title: str = "",
                     camera: str | None = None, height: int = 640):
    """사람 메시를 부스·슬롯 바와 함께 그린다 (`pose_view`와 같은 장면·카메라). `overlay_state`
    (캡슐 `BodyState`)를 주면 반투명으로 겹친다 (치수 비교용)."""
    import plotly.graph_objects as go

    from ..sim.scenario import load_nozzle_layout, load_nozzles
    from . import pose_view as pv

    booth = booth if booth is not None else load_nozzle_layout()["booth"]
    nozzle = nozzle if nozzle is not None else load_nozzles()
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=SKIN_COLOR, opacity=1.0, flatshading=False, name="사람 메시", showlegend=True,
        hoverinfo="skip",
        lighting=dict(ambient=0.45, diffuse=0.8, specular=0.15, roughness=0.6, fresnel=0.1),
        lightposition=dict(x=2000, y=1000, z=3000)))
    if overlay_state is not None:
        for tr in pv.body_traces(overlay_state, opacity=0.3):
            tr.showlegend = tr.name == "head"
            tr.name = "캡슐 마네킹 (반투명)" if tr.showlegend else f"캡슐 {tr.name}"
            fig.add_trace(tr)
    for tr in pv.booth_traces(booth) + pv.nozzle_traces(nozzle):
        fig.add_trace(tr)
    fig.update_layout(
        title=dict(text=title, x=0.5, y=0.98) if title else None,
        scene=pv._scene_layout(booth, camera or pv.DEFAULT_CAMERA),
        updatemenus=pv._camera_buttons(["scene"]),
        margin=dict(l=0, r=0, t=60 if title else 40, b=0),
        legend=dict(orientation="h", y=0.0, x=0.5, xanchor="center", yanchor="top"),
        height=height,
    )
    return fig


PREVIEW_POSES: dict[str, PoseParams] = {
    "a_default": PoseParams(),
    "b_hands_up": PoseParams(shoulder_abduction=180.0, elbow_flexion=0.0),
    "c_yaw90": PoseParams(torso_yaw=90.0),
}
PREVIEW_TITLES = {"a_default": "(a) 기본 자세 PoseParams()",
                  "b_hands_up": "(b) 만세 (벌림 180°, 팔꿈치 0°)",
                  "c_yaw90": "(c) 옆으로 서기 (yaw 90°)"}


def render_preview(out_dir: Path | str, mesh_dir: Path | str = MAKEHUMAN_DIR,
                   width: int = 900, height: int = 760) -> dict:
    """3자세 + 캡슐 겹침 1장을 PNG 로 저장한다 (kaleido, 단일 프로세스). 반환: 수치 요약."""
    from ..sim.body import build_body
    from ..sim.scenario import load_nozzle_layout, load_scenarios

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    mesh = load_makehuman(mesh_dir)
    t_load = time.perf_counter() - t0
    booth = load_nozzle_layout()["booth"]
    sc = load_scenarios()["default"]
    body = BodyParams()
    info = {"vertices": int(mesh.vertices.shape[0]), "triangles": int(mesh.faces.shape[0]),
            "bones": len(mesh.bone_names), "rest_height_m": round(mesh.height_m, 3),
            "load_s": round(t_load, 2), "files": {}}
    for key, pose in PREVIEW_POSES.items():
        t1 = time.perf_counter()
        v = posed_in_booth(mesh, body, pose, sc, booth)
        info.setdefault("pose_s", {})[key] = round(time.perf_counter() - t1, 4)
        info.setdefault("top_z_m", {})[key] = round(float(v[:, 2].max()), 3)
        info.setdefault("max_abs_y_m", {})[key] = round(float(np.abs(v[:, 1]).max()), 3)
        fig = figure_from_mesh(v, mesh.faces, booth=booth, title=f"MakeHuman · {PREVIEW_TITLES[key]}")
        path = out / f"makehuman_{key}.png"
        fig.write_image(path, width=width, height=height)
        info["files"][key] = str(path)
        if key == "a_default":
            state = build_body(body, pose, sc, patches_per_m2=400)
            fig = figure_from_mesh(v, mesh.faces, overlay_state=state, booth=booth,
                                   title="MakeHuman + 캡슐 마네킹 (반투명) · 기본 자세", camera="정면")
            path = out / "makehuman_overlay_capsule.png"
            fig.write_image(path, width=width, height=height)
            info["files"]["overlay"] = str(path)
    info["total_s"] = round(time.perf_counter() - t0, 1)
    (out / "makehuman_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="사람 메시(MakeHuman) 비교 렌더")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "mesh_preview"))
    ap.add_argument("--mesh-dir", default=str(MAKEHUMAN_DIR))
    args = ap.parse_args(argv)
    info = render_preview(args.out, args.mesh_dir)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
