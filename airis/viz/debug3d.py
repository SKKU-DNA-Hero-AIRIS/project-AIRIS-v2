"""디버그 3D 뷰. 소유자: B. (`docs/tracks/B_body_jet.md` 단계 7)

매 단계 이 뷰로 확인한다. 마네킹이 누워 있거나 팔이 몸을 뚫으면 여기서 보인다.

    from airis.sim.body import build_body
    from airis.viz.debug3d import plot_body, save
    ax = plot_body(build_body(...), nozzle=load_nozzles())
    save(ax.figure, "body_default")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # 헤드리스 저장 전용
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.colors import LogNorm    # noqa: E402

from ..sim.types import PART_NAMES, BodyState, NozzleConfig   # noqa: E402
from ..sim import jet                                          # noqa: E402
from ..sim.scenario import load_physics                        # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_OUT = _ROOT / "outputs" / "debug"

# PART_NAMES 순서와 맞춘 부위별 색
PART_COLORS = {
    "head": "#d95f02",
    "torso_front": "#1b9e77",
    "torso_back": "#7570b3",
    "arms": "#e7298a",
    "legs": "#66a61e",
}


def _equal_aspect(ax, pts: np.ndarray) -> None:
    """3D 축 비율을 동일하게. set_box_aspect + 공통 반경으로 맞춘다."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2.0
    radius = float((hi - lo).max()) / 2.0 or 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def plot_body(state: BodyState, values: np.ndarray | None = None,
              nozzle: NozzleConfig | None = None, ax=None, title: str = ""):
    """패치를 3D 산점도로. values (N,)가 있으면 색으로, 없으면 부위별 색."""
    if ax is None:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")

    pos = np.asarray(state.patch_pos)
    if values is not None:
        sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=np.asarray(values),
                        s=3, cmap="viridis", depthshade=False)
        ax.figure.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    else:
        part = np.asarray(state.patch_part)
        for idx, name in enumerate(PART_NAMES):
            m = part == idx
            if m.any():
                ax.scatter(pos[m, 0], pos[m, 1], pos[m, 2], s=3,
                           c=PART_COLORS[name], label=name, depthshade=False)
        ax.legend(loc="upper left", fontsize=8, markerscale=3)

    extent = [pos]
    if nozzle is not None:
        n, d = np.asarray(nozzle.positions), np.asarray(nozzle.directions)
        # 화살표 길이는 부스 폭에 비해 짧게. 방향만 보이면 된다.
        ax.quiver(n[:, 0], n[:, 1], n[:, 2], d[:, 0], d[:, 1], d[:, 2],
                  length=0.25, color="#377eb8", linewidth=1.0, arrow_length_ratio=0.3)
        ax.scatter(n[:, 0], n[:, 1], n[:, 2], s=14, c="#377eb8", marker="s")
        extent.append(n)
        # 슬롯(4.1b)은 길이 방향 선분으로 그린다
        for a, b in _slot_segments(nozzle):
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#377eb8", linewidth=3.0)
            extent.append(np.stack([a, b]))

    # 가림 전용 캡슐(휠체어 프레임)은 패치가 없으므로 반지름을 살린 관으로 표시한다
    if state.capsule_part is not None:
        caps = np.asarray(state.capsules)
        for cap, part in zip(caps, np.asarray(state.capsule_part)):
            if part == -1:
                extent.append(_plot_occluder(ax, cap[:3], cap[3:6], float(cap[6])))

    _equal_aspect(ax, np.concatenate(extent))
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y lateral (m)")
    ax.set_zlabel("z up (m)")
    ax.set_title(title)
    return ax


def _slot_segments(nozzle: NozzleConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    """슬롯 노즐의 양 끝점 (n ± e·L/2). 원형 노즐은 없다."""
    mask = jet.slot_mask(nozzle)
    if not mask.any():
        return []
    n = np.asarray(nozzle.positions, dtype=np.float64)[mask]
    e = np.asarray(nozzle.slot_axis, dtype=np.float64)[mask]
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    half = 0.5 * np.asarray(nozzle.slot_length, dtype=np.float64)[mask][:, None]
    return list(zip(n - e * half, n + e * half))


def _plot_occluder(ax, p0: np.ndarray, p1: np.ndarray, radius: float) -> np.ndarray:
    """가림 캡슐을 반투명 관으로. 반환값은 축 범위 계산용 코너 점."""
    from ..sim.body import _perp_basis

    axis_vec = np.asarray(p1) - np.asarray(p0)
    length = float(np.linalg.norm(axis_vec))
    axis = axis_vec / length if length > 1e-9 else np.array([0.0, 0.0, 1.0])
    e1, e2 = _perp_basis(axis)

    th = np.linspace(0.0, 2.0 * np.pi, 24)
    ring = radius * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
    t = np.linspace(0.0, 1.0, 2)
    centers = np.asarray(p0) + axis * (t[:, None] * length)
    surf = centers[:, None, :] + ring[None, :, :]                     # (2, 24, 3)
    ax.plot_surface(surf[..., 0], surf[..., 1], surf[..., 2],
                    color="#666666", alpha=0.35, linewidth=0, shade=False)
    for k in (0, 1):                                                  # 양 끝 테두리
        ax.plot(surf[k, :, 0], surf[k, :, 1], surf[k, :, 2], color="#444444", linewidth=0.8)
    return surf.reshape(-1, 3)


def plot_jet_slice(nozzle: NozzleConfig, cfg: dict | None = None, plane: str = "xz",
                   y: float = 0.0, ax=None, n: int = 160, extent: tuple | None = None,
                   title: str = "", body: BodyState | None = None, slab: float = 0.05):
    """평면 격자에서 |u| 등고선. 제트가 어디에 닿는지 확인용.

    plane="xz" 이면 고정 좌표는 y, "xy" 이면 z, "yz" 이면 x (인자 `y`를 고정값으로 그대로 씀).
    자유 제트는 거리에 따라 1/s 로 떨어져 동적 범위가 두 자릿수를 넘으므로 로그 눈금을 쓴다.
    `body` 를 주면 평면에서 ±slab 안의 패치를 겹쳐 그려 제트가 몸에 닿는지 바로 보인다.
    """
    cfg = cfg or load_physics()
    if ax is None:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)

    lo, hi = _slice_extent(nozzle, plane, extent)
    u = np.linspace(lo[0], hi[0], n)
    v = np.linspace(lo[1], hi[1], n)
    uu, vv = np.meshgrid(u, v)
    fixed = np.full(uu.size, y)
    if plane == "xz":
        pts = np.stack([uu.ravel(), fixed, vv.ravel()], axis=1)
        xlabel, ylabel = "x forward (m)", "z up (m)"
    elif plane == "xy":
        pts = np.stack([uu.ravel(), vv.ravel(), fixed], axis=1)
        xlabel, ylabel = "x forward (m)", "y lateral (m)"
    elif plane == "yz":
        pts = np.stack([fixed, uu.ravel(), vv.ravel()], axis=1)
        xlabel, ylabel = "y lateral (m)", "z up (m)"
    else:
        raise ValueError(f"plane 은 'xz', 'xy', 'yz': {plane!r}")

    speed = np.linalg.norm(jet.velocity_field(pts.astype(np.float32), nozzle, 0.0, cfg), axis=1)
    vmax = float(speed.max()) if speed.max() > 0 else 1.0
    vmin = vmax * 1e-3
    levels = np.geomspace(vmin, vmax, 25)
    cs = ax.contourf(uu, vv, np.clip(speed, vmin, vmax).reshape(uu.shape), levels=levels,
                     cmap="magma", norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.figure.colorbar(cs, ax=ax, label="|u| (m/s, log)")

    iu, iv, ifix = {"xz": (0, 2, 1), "xy": (0, 1, 2), "yz": (1, 2, 0)}[plane]
    if body is not None:
        bp = np.asarray(body.patch_pos)
        m = np.abs(bp[:, ifix] - y) < slab
        ax.scatter(bp[m, iu], bp[m, iv], s=2, c="#39ff14", label=f"body |{'xyz'[ifix]}-{y}|<{slab}")
        ax.legend(loc="upper right", fontsize=8, markerscale=4)

    npos = np.asarray(nozzle.positions)
    ndir = np.asarray(nozzle.directions)
    ax.quiver(npos[:, iu], npos[:, iv], ndir[:, iu], ndir[:, iv],
              color="#00e5ff", scale=12, width=0.004)
    for a, b in _slot_segments(nozzle):
        ax.plot([a[iu], b[iu]], [a[iv], b[iv]], color="#00e5ff", linewidth=2.5)

    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"jet |u|, plane={plane}, fixed={y}")
    return ax


def _slice_extent(nozzle: NozzleConfig, plane: str, extent: tuple | None):
    if extent is not None:
        return extent
    from ..sim.body import _booth
    booth = _booth()
    if plane == "xz":
        return (0.0, 0.0), (booth["length_m"], booth["height_m"])
    if plane == "yz":
        return (-booth["width_m"] / 2.0, 0.0), (booth["width_m"] / 2.0, booth["height_m"])
    return (0.0, -booth["width_m"] / 2.0), (booth["length_m"], booth["width_m"] / 2.0)


def save(fig, name: str) -> Path:
    """outputs/debug/<name>.png 저장."""
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
