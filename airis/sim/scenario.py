"""시나리오 / 물리 상수 로더. 소유자: B."""
from __future__ import annotations

from pathlib import Path

import yaml

from .types import Scenario

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


def load_physics(path: Path | None = None) -> dict:
    path = path or _ROOT / "configs" / "physics.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))
