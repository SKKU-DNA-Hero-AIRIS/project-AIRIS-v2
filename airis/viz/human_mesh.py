"""사람 메시 시각화 (비교 렌더). 소유자: E.

계산(로드·체형·자세·패치)은 B 소유 `airis/sim/human_mesh.py`로 옮겼다 (단계 10 이관). 이 파일은
그 모듈을 그대로 재수출하고 plotly 그림 함수만 둔다. E 가 나중에 정리한다.

비교 렌더: `python -m airis.viz.human_mesh` → `outputs/mesh_preview/makehuman_*.png`
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..sim.human_mesh import *  # noqa: F401,F403  (공개 API 재수출)
from ..sim.human_mesh import (  # noqa: F401  (테스트·렌더가 쓰는 내부 이름)
    MAKEHUMAN_DIR, ROOT, HumanMesh, _axis_angle, _default_booth, _rot, load_makehuman, posed_in_booth,
)
from ..sim.types import BodyParams, PoseParams


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
