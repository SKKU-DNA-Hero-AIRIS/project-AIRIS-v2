"""시나리오 / 물리 상수 로더. 소유자: B."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .types import Scenario, NozzleConfig

_ROOT = Path(__file__).resolve().parents[2]


def load_scenarios(path: Path | None = None) -> dict[str, Scenario]:
    path = path or _ROOT / "configs" / "scenarios.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {name: _resolve(name, raw) for name in raw}


def _resolve(name: str, raw: dict) -> Scenario:
    node = dict(raw[name])
    parent = node.pop("inherits", None)
    merged: dict = {}
    if parent:
        base = raw[parent]
        merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in node.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k].update(v)
        else:
            merged[k] = v
    return Scenario(
        name=name,
        pose_bounds={k: tuple(v) for k, v in merged["pose_bounds"].items()},
        discomfort_weights=merged.get("discomfort_weights", {}),
        fixed_pose=merged.get("fixed_pose", {}),
        nozzle_height_range_m=tuple(merged.get("nozzle_height_range_m", (0.3, 2.0))),
        nozzle_strength_cap=merged.get("nozzle_strength_cap", {}),
        seat_height_m=merged.get("seat_height_m"),
    )


def load_nozzle_layout(path: Path | None = None) -> dict:
    """고정 노즐 배치 원본. NozzleConfig로 전개하는 것은 B의 1주차 구현 (jet.py 또는 여기)."""
    path = path or _ROOT / "configs" / "nozzles.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_nozzles(path: Path | None = None, layout: str | None = None) -> NozzleConfig:
    """configs/nozzles.yaml 의 배치를 전개해 NozzleConfig 를 만든다. B 트랙 단계 5.

    `layout` 이 None 이면 파일의 `active` 배치를 쓴다.
      - "slot_bars": 기준 장비(퓨리움) 슬롯 바. 전 행이 슬롯 제트(00_common.md 4.1b)라
        slot_axis (M,3), slot_length (M,) 를 채운다.
      - "layout": 원형 노즐 비교 배치. slot_axis = slot_length = None (전부 원형, 4.1).
    """
    raw = load_nozzle_layout(path)
    name = layout or raw.get("active", "layout")
    if name == "slot_bars":
        return _expand_slot_bars(raw["slot_bars"])
    if name == "layout":
        return _expand_round_layout(raw["layout"])
    raise ValueError(f"알 수 없는 노즐 배치: {name!r} (slot_bars | layout)")


def _wall_jet_axes(wall_y: float, yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """벽면 토출구의 분사 방향 d 와 수평 접선 e (d ⊥ e, 둘 다 단위 벡터).

    d: 벽면 안쪽 법선 (0, -sign(wall_y), 0) 을 z 축으로 yaw 만큼 진행 방향(+x)으로 돌리고,
       다시 e 축을 중심으로 pitch 만큼 아래로 내린다 (양수 = 아래).
    e: 벽면을 따라가는 수평 방향. yaw = 0 이면 +x. pitch 회전축이므로 pitch 와 무관하게 d 에 수직이다.
    """
    side = np.sign(wall_y)                   # +1 = 왼쪽 벽(+y), -1 = 오른쪽 벽(-y)
    # wall_y<0: d0=(0,+1,0) 을 R_z(-yaw) → (sin yaw,  cos yaw, 0)
    # wall_y>0: d0=(0,-1,0) 을 R_z(+yaw) → (sin yaw, -cos yaw, 0)
    d_h = np.array([np.sin(yaw), -side * np.cos(yaw), 0.0])
    e = np.array([np.cos(yaw), side * np.sin(yaw), 0.0])
    d = d_h * np.cos(pitch) + np.array([0.0, 0.0, -1.0]) * np.sin(pitch)
    return d, e


def _expand_round_layout(lay: dict) -> NozzleConfig:
    """원형 노즐: (x, wall_y, z) for x in x_positions, wall_y in wall_y, z in z_levels."""
    yaw = np.deg2rad(float(lay["yaw_deg"]))
    pitch = np.deg2rad(float(lay["pitch_deg"]))
    positions, directions = [], []
    for x in lay["x_positions"]:
        for wall_y in lay["wall_y"]:
            d, _ = _wall_jet_axes(float(wall_y), yaw, pitch)
            for z in lay["z_levels"]:
                positions.append([float(x), float(wall_y), float(z)])
                directions.append(d)
    return NozzleConfig(
        positions=np.asarray(positions, dtype=np.float32),
        directions=_unit_rows(directions),
        strengths=np.full(len(positions), float(lay["strength"]), dtype=np.float32),
        pulse_phase=None,
    )


def _expand_slot_bars(bars: dict) -> NozzleConfig:
    """슬롯 바: 측면 (wall_y × z_levels) 다음 상단 (x_positions) 순서.

    측면 바: 중심 (x_center, wall_y, z), 분사 = 벽 안쪽 법선에 yaw·pitch, 슬롯 축 = 벽면 수평 접선.
    상단 바: 중심 (x, y_center, z), 분사 = 아래(−z)에서 +x 로 tilt, 슬롯 축 = +y (부스 폭 방향).
    """
    side, top = bars["side"], bars["top"]
    positions, directions, axes, lengths, strengths = [], [], [], [], []

    yaw = np.deg2rad(float(side["yaw_deg"]))
    pitch = np.deg2rad(float(side["pitch_deg"]))
    for wall_y in side["wall_y"]:
        d, e = _wall_jet_axes(float(wall_y), yaw, pitch)
        for z in side["z_levels"]:
            positions.append([float(side["x_center"]), float(wall_y), float(z)])
            directions.append(d)
            axes.append(e)
            lengths.append(float(side["length_m"]))
            strengths.append(float(side["strength"]))

    tilt = np.deg2rad(float(top["tilt_deg"]))
    d_top = np.array([np.sin(tilt), 0.0, -np.cos(tilt)])
    e_top = np.array([0.0, 1.0, 0.0])          # tilt 회전축이라 d_top 에 항상 수직
    for x in top["x_positions"]:
        positions.append([float(x), float(top["y_center"]), float(top["z"])])
        directions.append(d_top)
        axes.append(e_top)
        lengths.append(float(top["length_m"]))
        strengths.append(float(top["strength"]))

    return NozzleConfig(
        positions=np.asarray(positions, dtype=np.float32),
        directions=_unit_rows(directions),
        strengths=np.asarray(strengths, dtype=np.float32),
        pulse_phase=None,
        slot_axis=_unit_rows(axes),
        slot_length=np.asarray(lengths, dtype=np.float32),
    )


def _unit_rows(vectors) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float64)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def load_physics(path: Path | None = None) -> dict:
    path = path or _ROOT / "configs" / "physics.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))
