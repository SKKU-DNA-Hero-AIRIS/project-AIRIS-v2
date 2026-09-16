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


def load_nozzles(path: Path | None = None) -> NozzleConfig:
    """configs/nozzles.yaml의 layout을 전개해 NozzleConfig를 만든다. B 트랙 단계 5.

    위치: (x, wall_y, z) for x in x_positions, wall_y in wall_y, z in z_levels.
    방향: 벽면 안쪽 법선 (0, -sign(wall_y), 0)을
          z 축으로 yaw_deg 만큼 진행 방향(+x)으로 돌리고,
          다시 수평축으로 pitch_deg 만큼 아래로 내린다 (양수 = 아래).
    """
    layout = load_nozzle_layout(path)["layout"]
    wall_ys = layout["wall_y"]
    x_positions = layout["x_positions"]
    z_levels = layout["z_levels"]
    yaw = np.deg2rad(float(layout["yaw_deg"]))
    pitch = np.deg2rad(float(layout["pitch_deg"]))

    positions, directions = [], []
    for x in x_positions:
        for wall_y in wall_ys:
            side = np.sign(wall_y)               # +1 = 왼쪽 벽(+y), -1 = 오른쪽 벽(-y)
            # 안쪽 법선을 +x 쪽으로 yaw 만큼. 회전 부호는 벽면에 따라 반대다.
            # wall_y<0: d0=(0,+1,0) 을 R_z(-yaw) → (sin yaw,  cos yaw, 0)
            # wall_y>0: d0=(0,-1,0) 을 R_z(+yaw) → (sin yaw, -cos yaw, 0)
            d = np.array([np.sin(yaw), -side * np.cos(yaw), 0.0])
            # 수평 방향을 유지한 채 아래로 pitch. d 는 수평이므로 결과는 이미 단위 벡터다.
            d = d * np.cos(pitch) + np.array([0.0, 0.0, -1.0]) * np.sin(pitch)
            for z in z_levels:
                positions.append([x, float(wall_y), float(z)])
                directions.append(d)

    directions = np.asarray(directions, dtype=np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return NozzleConfig(
        positions=np.asarray(positions, dtype=np.float32),
        directions=directions,
        strengths=np.full(len(positions), float(layout["strength"]), dtype=np.float32),
        pulse_phase=None,
    )


def load_physics(path: Path | None = None) -> dict:
    path = path or _ROOT / "configs" / "physics.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))
