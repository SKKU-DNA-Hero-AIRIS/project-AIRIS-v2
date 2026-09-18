"""안내용 3D 마네킹 뷰 (plotly). 소유자: E. (`docs/tracks/E_realtime.md` 단계 1)

B의 디버그 뷰(`debug3d.py`, matplotlib 산점도)와 달리 사람에게 "이 자세를 취하세요"를 보여주는
그림이다. 캡슐을 메시로 그리고 부스·슬롯 바를 함께 그려 대시보드에 그대로 넣는다.

    from airis.viz.pose_view import figure_from_pose, figure_compare
    fig = figure_from_pose(BodyParams(), PoseParams(torso_yaw=90), scenarios["default"])
    fig = figure_compare(body, {"B0 기본": PoseParams(), "추천": rec}, scenario, results=scores)

물리 계산은 하지 않는다. 패치 색(`values`)은 호출자가 넘기는 값(D의 `EvalResult.extra["removal"]`)이다.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree

from ..sim.body import build_body
from ..sim.jet import slot_mask
from ..sim.scenario import load_nozzle_layout, load_nozzles
from ..sim.types import PART_NAMES, BodyParams, BodyState, EvalResult, NozzleConfig, PoseParams, Scenario

#: 부위별 색. PART_NAMES 순서. B의 debug3d 와 같은 색상 계열이다.
PART_COLORS: dict[str, str] = {
    "head": "#d95f02",
    "torso_front": "#1b9e77",
    "torso_back": "#7570b3",
    "arms": "#e7298a",
    "legs": "#66a61e",
}
OCCLUDER_COLOR = "#9e9e9e"
BOOTH_COLOR = "#607d8b"
SLOT_COLOR = "#1e88e5"
#: 패치 제거율 색 (0 → 1). 제거율은 순위·상대 비교용이라 색 범위는 호출자가 정할 수 있다.
REMOVAL_COLORSCALE = "Viridis"

#: 카메라 프리셋. plotly scene.camera. 사람은 +x(진행 방향)를 보고 선다.
CAMERA_PRESETS: dict[str, dict] = {
    "정면": dict(eye=dict(x=2.0, y=0.0, z=0.25), up=dict(x=0, y=0, z=1),
               center=dict(x=0, y=0, z=0)),
    "측면": dict(eye=dict(x=0.0, y=2.0, z=0.25), up=dict(x=0, y=0, z=1),
               center=dict(x=0, y=0, z=0)),
    "위": dict(eye=dict(x=0.0, y=0.0, z=2.4), up=dict(x=1, y=0, z=0),
              center=dict(x=0, y=0, z=0)),
    "사선": dict(eye=dict(x=1.5, y=1.2, z=0.6), up=dict(x=0, y=0, z=1),
               center=dict(x=0, y=0, z=0)),
}
DEFAULT_CAMERA = "사선"


# ---------------------------------------------------------------------------
# 캡슐 메시
# ---------------------------------------------------------------------------
def _perp_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(ref, axis)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return e1, e2 / np.linalg.norm(e2)


def capsule_mesh(p0, p1, r: float, n_theta: int = 24, n_cap: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """캡슐(원통 + 반구 2개) 삼각형 메시.

    정점 = 남극 1 + 위도 고리 2·n_cap개 × n_theta + 북극 1, 면 = 4·n_cap·n_theta.
    아래 반구 고리는 p0 중심, 위 반구 고리는 p1 중심이라 둘 사이 띠가 원통 측면이 된다.
    길이 0 캡슐(머리 구)도 같은 식으로 구가 된다.

    반환: vertices (V,3) float64, faces (F,3) int64 (정점 인덱스, 바깥에서 보아 반시계).
    """
    if n_theta < 3 or n_cap < 1:
        raise ValueError("n_theta >= 3, n_cap >= 1")
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    axis = axis / length if length > 1e-9 else np.array([0.0, 0.0, 1.0])
    e1, e2 = _perp_basis(axis)

    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    circle = np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2          # (T,3)
    # 위도: 아래 반구 -90° 초과 ~ 0° (p0), 위 반구 0° ~ 90° 미만 (p1)
    lat_lo = -0.5 * np.pi + 0.5 * np.pi * np.arange(1, n_cap + 1) / n_cap
    lat_hi = 0.5 * np.pi * np.arange(0, n_cap) / n_cap
    rings = []
    for lat in lat_lo:
        rings.append(p0 + r * (np.cos(lat) * circle + np.sin(lat) * axis))
    for lat in lat_hi:
        rings.append(p1 + r * (np.cos(lat) * circle + np.sin(lat) * axis))
    south = p0 - r * axis
    north = p1 + r * axis
    vertices = np.vstack([south[None, :], *rings, north[None, :]])

    n_rings = 2 * n_cap
    ring_start = lambda k: 1 + k * n_theta                                          # noqa: E731
    j = np.arange(n_theta)
    jn = (j + 1) % n_theta
    faces = [np.stack([np.zeros(n_theta, int), ring_start(0) + jn, ring_start(0) + j], axis=1)]
    for k in range(n_rings - 1):
        a, b = ring_start(k), ring_start(k + 1)
        faces.append(np.stack([a + j, a + jn, b + jn], axis=1))
        faces.append(np.stack([a + j, b + jn, b + j], axis=1))
    top = vertices.shape[0] - 1
    last = ring_start(n_rings - 1)
    faces.append(np.stack([last + j, last + jn, np.full(n_theta, top)], axis=1))
    return vertices, np.vstack(faces).astype(np.int64)


def _merge_meshes(meshes: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """메시 여러 개 → 하나. 반환 (V,3), (F,3), 정점별 원래 메시 번호 (V,)."""
    verts, faces, owner = [], [], []
    offset = 0
    for idx, (v, f) in enumerate(meshes):
        verts.append(v)
        faces.append(f + offset)
        owner.append(np.full(v.shape[0], idx))
        offset += v.shape[0]
    return np.vstack(verts), np.vstack(faces), np.concatenate(owner)


# ---------------------------------------------------------------------------
# 부스·노즐
# ---------------------------------------------------------------------------
def booth_traces(booth: Mapping) -> list[go.Scatter3d]:
    """부스 상자 모서리 12개 (한 trace) + 바닥 윤곽."""
    lx, w, h = float(booth["length_m"]), float(booth["width_m"]), float(booth["height_m"])
    xs, ys, zs = (0.0, lx), (-w / 2.0, w / 2.0), (0.0, h)
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
             if np.count_nonzero(corners[a] != corners[b]) == 1]
    pts = []
    for a, b in edges:
        pts.extend([corners[a], corners[b], [None, None, None]])
    pts = np.array(pts, dtype=object)
    frame = go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
        line=dict(color=BOOTH_COLOR, width=3), name="부스", hoverinfo="skip",
        legendgroup="booth")
    floor = go.Mesh3d(
        x=[0, lx, lx, 0], y=[-w / 2, -w / 2, w / 2, w / 2], z=[0, 0, 0, 0],
        i=[0, 0], j=[1, 2], k=[2, 3], color=BOOTH_COLOR, opacity=0.08,
        name="바닥", hoverinfo="skip", showlegend=False, legendgroup="booth")
    return [frame, floor]


def nozzle_traces(nozzle: NozzleConfig, jet_arrow_m: float = 0.15) -> list[go.Scatter3d]:
    """슬롯 바는 굵은 선분(`slot_axis`·`slot_length`), 원형 노즐은 점. 분사 방향은 짧은 선."""
    pos = np.asarray(nozzle.positions, dtype=np.float64).reshape(-1, 3)
    dirs = np.asarray(nozzle.directions, dtype=np.float64).reshape(-1, 3)
    is_slot = slot_mask(nozzle)
    traces: list[go.Scatter3d] = []

    if is_slot.any():
        axis = np.asarray(nozzle.slot_axis, dtype=np.float64).reshape(-1, 3)
        half = 0.5 * np.asarray(nozzle.slot_length, dtype=np.float64).reshape(-1)
        seg = []
        for m in np.flatnonzero(is_slot):
            e = axis[m] / np.linalg.norm(axis[m])
            seg.extend([pos[m] - half[m] * e, pos[m] + half[m] * e, [None] * 3])
        seg = np.array(seg, dtype=object)
        traces.append(go.Scatter3d(
            x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines",
            line=dict(color=SLOT_COLOR, width=9), name=f"슬롯 바 {int(is_slot.sum())}개",
            hoverinfo="skip", legendgroup="nozzle"))
    if (~is_slot).any():
        p = pos[~is_slot]
        traces.append(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
            marker=dict(color=SLOT_COLOR, size=4), name=f"노즐 {int((~is_slot).sum())}개",
            hoverinfo="skip", legendgroup="nozzle"))

    arrows = []
    for m in range(pos.shape[0]):
        arrows.extend([pos[m], pos[m] + jet_arrow_m * dirs[m], [None] * 3])
    arrows = np.array(arrows, dtype=object)
    traces.append(go.Scatter3d(
        x=arrows[:, 0], y=arrows[:, 1], z=arrows[:, 2], mode="lines",
        line=dict(color=SLOT_COLOR, width=3, dash="dot"), name="분사 방향",
        hoverinfo="skip", showlegend=False, legendgroup="nozzle"))
    return traces


# ---------------------------------------------------------------------------
# 몸
# ---------------------------------------------------------------------------
def body_traces(state: BodyState, values: np.ndarray | None = None, *,
                cmin: float | None = None, cmax: float | None = None,
                colorbar_title: str = "제거율", showscale: bool = True,
                opacity: float = 1.0, n_theta: int = 24, n_cap: int = 6) -> list[go.Mesh3d]:
    """캡슐을 메시로. `values (N,)`가 없으면 부위 색, 있으면 정점마다 가장 가까운 자기 캡슐 패치 값.

    `capsule_part == -1`(휠체어 프레임 등 가림 전용)은 회색 반투명.
    """
    caps = np.asarray(state.capsules, dtype=np.float64)
    cap_part = (np.asarray(state.capsule_part) if state.capsule_part is not None
                else np.zeros(caps.shape[0], dtype=int))
    meshes = [capsule_mesh(c[0:3], c[3:6], float(c[6]), n_theta, n_cap) for c in caps]
    for k in range(caps.shape[0]):
        meshes[k] = (_flatten_torso(state, k, meshes[k][0]), meshes[k][1])
    traces: list[go.Mesh3d] = []

    occluders = [k for k in range(caps.shape[0]) if cap_part[k] < 0]
    if occluders:
        v, f, _ = _merge_meshes([meshes[k] for k in occluders])
        traces.append(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=OCCLUDER_COLOR, opacity=0.35, name="휠체어 프레임", hoverinfo="skip",
            showlegend=True, flatshading=False))

    body_caps = [k for k in range(caps.shape[0]) if cap_part[k] >= 0]
    if values is None:
        # 부위별 trace. 몸통 캡슐은 torso_front 색으로 그린다 (앞뒤는 패치에서만 갈린다).
        for part_id, name in enumerate(PART_NAMES):
            ks = [k for k in body_caps if cap_part[k] == part_id]
            if not ks:
                continue
            v, f, _ = _merge_meshes([meshes[k] for k in ks])
            traces.append(go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color=PART_COLORS[name], opacity=opacity, name=name, hoverinfo="name",
                showlegend=True, lighting=dict(ambient=0.55, diffuse=0.7, specular=0.2)))
        return traces

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] != state.patch_pos.shape[0]:
        raise ValueError(f"values 길이 {values.shape[0]} != 패치 수 {state.patch_pos.shape[0]}")
    v, f, owner = _merge_meshes([meshes[k] for k in body_caps])
    intensity = _vertex_values(state, values, v, np.asarray(body_caps)[owner])
    lo = float(np.nanmin(values)) if cmin is None else cmin
    hi = float(np.nanmax(values)) if cmax is None else cmax
    traces.append(go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
        intensity=intensity, colorscale=REMOVAL_COLORSCALE, cmin=lo, cmax=max(hi, lo + 1e-9),
        showscale=showscale, colorbar=dict(title=colorbar_title, len=0.6),
        opacity=opacity, name=colorbar_title, hovertemplate=f"{colorbar_title} %{{intensity:.3f}}<extra></extra>",
        lighting=dict(ambient=0.6, diffuse=0.6, specular=0.1)))
    return traces


def _flatten_torso(state: BodyState, k: int, verts: np.ndarray) -> np.ndarray:
    """몸통 캡슐 메시를 패치가 놓인 타원 단면에 맞춰 앞뒤로 눌러 준다.

    `BodyState.capsules`는 몸통을 원형(좌우 반축)으로 근사하지만 패치는 타원(앞뒤 반축이 더 짧다)
    위에 있다. 그대로 그리면 옆에서 본 몸통이 실제보다 두껍다. 앞 방향은 torso_front 패치
    법선 평균 − torso_back 평균, 앞뒤 반축은 그 방향으로 잰 패치의 최대 거리다. 부위가 앞뒤로
    갈리지 않는 캡슐은 그대로 둔다.
    """
    if state.patch_capsule is None:
        return verts
    pm = np.asarray(state.patch_capsule) == k
    parts = np.asarray(state.patch_part)[pm]
    front, back = PART_NAMES.index("torso_front"), PART_NAMES.index("torso_back")
    if not ((parts == front).any() and (parts == back).any()):
        return verts
    cap = np.asarray(state.capsules[k], dtype=np.float64)
    p0, axis = cap[0:3], cap[3:6] - cap[0:3]
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    nrm = np.asarray(state.patch_normal, dtype=np.float64)[pm]
    fwd = nrm[parts == front].mean(axis=0) - nrm[parts == back].mean(axis=0)
    fwd -= (fwd @ axis) * axis
    if np.linalg.norm(fwd) < 1e-6:
        return verts
    fwd /= np.linalg.norm(fwd)
    rel = np.asarray(state.patch_pos, dtype=np.float64)[pm] - p0
    half_ap = float(np.abs(rel @ fwd).max())
    r = float(cap[6])
    if not 0.0 < half_ap < r:
        return verts
    along = (verts - p0) @ fwd
    return verts - np.outer(along * (1.0 - half_ap / r), fwd)


def _vertex_values(state: BodyState, values: np.ndarray, verts: np.ndarray,
                   vert_capsule: np.ndarray) -> np.ndarray:
    """정점마다 가장 가까운 패치 값. `patch_capsule`이 있으면 같은 캡슐의 패치에서만 찾는다."""
    pos = np.asarray(state.patch_pos, dtype=np.float64)
    out = np.empty(verts.shape[0])
    if state.patch_capsule is None:
        _, idx = cKDTree(pos).query(verts)
        return values[idx]
    patch_cap = np.asarray(state.patch_capsule)
    for k in np.unique(vert_capsule):
        vm = vert_capsule == k
        pm = np.flatnonzero(patch_cap == k)
        if pm.size == 0:                       # 패치 없는 캡슐: 전체에서 찾는다
            _, idx = cKDTree(pos).query(verts[vm])
            out[vm] = values[idx]
            continue
        _, idx = cKDTree(pos[pm]).query(verts[vm])
        out[vm] = values[pm[idx]]
    return out


# ---------------------------------------------------------------------------
# 장면 설정
# ---------------------------------------------------------------------------
def _scene_layout(booth: Mapping, camera: str = DEFAULT_CAMERA) -> dict:
    """축 비율 동일(aspectmode=data), 범위는 부스 + 여유."""
    lx, w, h = float(booth["length_m"]), float(booth["width_m"]), float(booth["height_m"])
    pad = 0.15
    return dict(
        xaxis=dict(range=[-pad - 0.25, lx + pad + 0.25], title="x 진행 (m)"),
        yaxis=dict(range=[-w / 2 - pad, w / 2 + pad], title="y 좌우 (m)"),
        zaxis=dict(range=[0.0, h + pad], title="z (m)"),
        aspectmode="data",
        camera=CAMERA_PRESETS[camera],
    )


def _camera_buttons(scene_names: list[str]) -> list[dict]:
    """정면·측면·위·사선 버튼. 모든 scene의 카메라를 같이 바꾼다."""
    buttons = []
    for label, cam in CAMERA_PRESETS.items():
        args = {f"{s}.camera": cam for s in scene_names}
        buttons.append(dict(label=label, method="relayout", args=[args]))
    return [dict(type="buttons", direction="left", buttons=buttons, x=0.0, y=1.08,
                 xanchor="left", yanchor="bottom", pad=dict(r=4, t=0), showactive=False)]


def _resolve_booth(booth: Mapping | None) -> Mapping:
    return booth if booth is not None else load_nozzle_layout()["booth"]


def figure_from_state(state: BodyState, values: np.ndarray | None = None,
                      nozzle: NozzleConfig | None = None, booth: Mapping | None = None,
                      title: str = "", *, camera: str = DEFAULT_CAMERA,
                      cmin: float | None = None, cmax: float | None = None,
                      show_booth: bool = True, height: int = 640) -> go.Figure:
    """BodyState 하나를 그린다. `booth`가 None이면 `configs/nozzles.yaml`의 booth, `nozzle`이 None이면 노즐 생략."""
    booth = _resolve_booth(booth)
    fig = go.Figure()
    for tr in body_traces(state, values, cmin=cmin, cmax=cmax):
        fig.add_trace(tr)
    if show_booth:
        for tr in booth_traces(booth):
            fig.add_trace(tr)
    if nozzle is not None:
        for tr in nozzle_traces(nozzle):
            fig.add_trace(tr)
    fig.update_layout(
        title=dict(text=title, x=0.5, y=0.98) if title else None,
        scene=_scene_layout(booth, camera),
        updatemenus=_camera_buttons(["scene"]),
        margin=dict(l=0, r=0, t=60 if title else 40, b=0),
        legend=dict(orientation="h", y=0.0, x=0.5, xanchor="center", yanchor="top"),
        height=height,
    )
    return fig


def figure_from_pose(body: BodyParams, pose: PoseParams, scenario: Scenario, *,
                     values: np.ndarray | None = None, result: EvalResult | None = None,
                     nozzle: NozzleConfig | None = None, booth: Mapping | None = None,
                     patches_per_m2: float = 400.0, title: str = "", **kw) -> go.Figure:
    """체형 + 자세 + 시나리오 → 그림. 슬롯 바(`load_nozzles()`)와 부스를 항상 함께 그린다.

    패치 색은 `values` 또는 `result.extra["removal"]`. 둘 다 D의 `PatchEvaluator`와 **같은
    `patches_per_m2`**로 만든 몸이어야 길이가 맞는다 (기본 400 = 최적화 루프 밀도).
    """
    state = build_body(body, pose, scenario, patches_per_m2=patches_per_m2)
    if values is None and result is not None and "removal" in result.extra:
        values = result.extra["removal"]
    return figure_from_state(state, values, nozzle if nozzle is not None else load_nozzles(),
                             booth, title, **kw)


def pose_label(pose: PoseParams) -> str:
    """자세 7개를 짧은 한 줄로 (도 단위, 반올림)."""
    return (f"벌림 {pose.shoulder_abduction:.0f}° · 굽힘 {pose.shoulder_flexion:.0f}° · "
            f"팔꿈치 {pose.elbow_flexion:.0f}° · 숙임 {pose.torso_pitch:.0f}° · "
            f"회전 {pose.torso_yaw:.0f}° · 고관절 {pose.hip_flexion:.0f}° · 무릎 {pose.knee_flexion:.0f}°")


def figure_compare(body: BodyParams, poses: Mapping[str, PoseParams], scenario: Scenario, *,
                   results: Mapping[str, EvalResult] | None = None,
                   nozzle: NozzleConfig | None = None, booth: Mapping | None = None,
                   patches_per_m2: float = 400.0, reference: str | None = None,
                   camera: str = DEFAULT_CAMERA, height: int = 620) -> go.Figure:
    """자세 여러 개를 나란히 (예: 기준 자세 B0 vs 추천 자세).

    `results[name]`이 있으면 패치를 제거율 색으로 칠하고(모든 칸 같은 색 범위), 제목에 점수와
    `reference`(기본: 첫 자세) 대비 **상대 개선율**을 붙인다. 절대 제거율은 주장하지 않는다.
    """
    if not poses:
        raise ValueError("poses 가 비었다")
    booth = _resolve_booth(booth)
    nozzle = nozzle if nozzle is not None else load_nozzles()
    names = list(poses)
    reference = reference or names[0]
    results = dict(results or {})

    removal = [results[n].extra.get("removal") for n in names if n in results
               and not results[n].extra.get("infeasible")]
    removal = [r for r in removal if r is not None]
    cmin = 0.0
    cmax = float(max(np.max(r) for r in removal)) if removal else None

    titles = []
    ref = results.get(reference)
    for n in names:
        r = results.get(n)
        if r is None:
            titles.append(n)
        elif r.extra.get("infeasible"):
            titles.append(f"{n}<br><sub>부스 밖 (불가)</sub>")
        elif ref is not None and n != reference and ref.score > 0:
            titles.append(f"{n}<br><sub>점수 {r.score:.3f} · {reference} 대비 {r.score / ref.score - 1:+.0%}</sub>")
        else:
            titles.append(f"{n}<br><sub>점수 {r.score:.3f}</sub>")

    fig = make_subplots(rows=1, cols=len(names), specs=[[{"type": "scene"}] * len(names)],
                        subplot_titles=titles, horizontal_spacing=0.01)
    scene_names = []
    for col, n in enumerate(names, start=1):
        state = build_body(body, poses[n], scenario, patches_per_m2=patches_per_m2)
        r = results.get(n)
        values = None if r is None or r.extra.get("infeasible") else r.extra.get("removal")
        if values is not None and len(values) != state.patch_pos.shape[0]:
            values = None                      # 밀도가 다른 결과면 부위 색으로
        for tr in body_traces(state, values, cmin=cmin, cmax=cmax, showscale=(col == len(names))):
            tr.showlegend = False if col > 1 else tr.showlegend
            fig.add_trace(tr, row=1, col=col)
        for tr in booth_traces(booth) + nozzle_traces(nozzle):
            tr.showlegend = False if col > 1 else tr.showlegend
            fig.add_trace(tr, row=1, col=col)
        scene = "scene" if col == 1 else f"scene{col}"
        scene_names.append(scene)
        fig.update_layout({scene: _scene_layout(booth, camera)})
    fig.update_layout(
        updatemenus=_camera_buttons(scene_names),
        margin=dict(l=0, r=0, t=90, b=0),
        legend=dict(orientation="h", y=0.0, x=0.5, xanchor="center", yanchor="top"),
        height=height,
    )
    return fig
