"""시나리오 / 물리 상수 로더. 소유자: B."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

from .types import ZONE_NAMES, Scenario, NozzleConfig

_ROOT = Path(__file__).resolve().parents[2]


def load_scenarios(path: Path | None = None) -> dict[str, Scenario]:
    path = path or _ROOT / "configs" / "scenarios.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {name: _resolve(name, raw) for name in raw}


def scenario_mesh_targets(name: str, path: Path | None = None) -> dict[str, float]:
    """시나리오의 `mesh_targets` {MakeHuman 타깃 이름: 가중치} (상속 포함). Scenario 타입 밖의 메시 전용 설정."""
    path = path or _ROOT / "configs" / "scenarios.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    node, out = raw.get(name, {}), {}
    parent = node.get("inherits")
    if parent:
        out.update(raw.get(parent, {}).get("mesh_targets", {}) or {})
    out.update(node.get("mesh_targets", {}) or {})
    return {str(k): float(v) for k, v in out.items()}


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
    _check_strength_cap(name, merged.get("nozzle_strength_cap", {}) or {})
    return Scenario(
        name=name,
        pose_bounds={k: tuple(v) for k, v in merged["pose_bounds"].items()},
        discomfort_weights=merged.get("discomfort_weights", {}),
        fixed_pose=merged.get("fixed_pose", {}),
        nozzle_height_range_m=tuple(merged.get("nozzle_height_range_m", (0.3, 2.0))),
        nozzle_strength_cap=merged.get("nozzle_strength_cap", {}),
        seat_height_m=merged.get("seat_height_m"),
    )


#: 쾌적 상한의 옛 부위 키 → 그 부위를 향한 구역 (전환 기간 호환. 새 설정은 구역 이름 키를 쓴다).
CAP_PART_ZONES: dict[str, tuple[str, ...]] = {"torso_front": ("chest_low", "chest_high"),
                                              "torso_back": ("back_low", "back_high")}


def _check_strength_cap(name: str, cap: Mapping) -> None:
    for key, value in cap.items():
        if key not in ZONE_NAMES and key not in CAP_PART_ZONES:
            raise ValueError(f"{name}: nozzle_strength_cap 키 {key!r} 는 구역 이름 {ZONE_NAMES} "
                             f"(또는 옛 부위 키 {sorted(CAP_PART_ZONES)}) 이어야 한다")
        if not float(value) >= 0.0:
            raise ValueError(f"{name}: nozzle_strength_cap[{key!r}] = {value} 는 0 이상이어야 한다")


def zone_strength_caps(scenario: Scenario) -> np.ndarray:
    """(len(ZONE_NAMES),) 구역별 쾌적 상한 (00_common.md 4.7). 상한이 없는 구역은 inf.

    `scenario.nozzle_strength_cap` 의 구역 이름 키를 쓴다 (임산부: chest_low, chest_high ≤ 0.6).
    옛 부위 키(torso_front → chest_*, torso_back → back_*)도 읽고, 겹치면 작은 값을 쓴다.
    구역이 몸 기준이라 상한도 torso_yaw 와 무관하다.
    """
    caps = np.full(len(ZONE_NAMES), np.inf)
    for key, value in (scenario.nozzle_strength_cap or {}).items():
        for z in ((key,) if key in ZONE_NAMES else CAP_PART_ZONES.get(key, ())):
            i = ZONE_NAMES.index(z)
            caps[i] = min(caps[i], float(value))
    return caps


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
    """슬롯 바: 측면 (wall_y × z_levels) 다음 상단 (x_positions × y_positions) 순서.

    측면 바: 중심 (x_center, wall_y, z), 분사 = 벽 안쪽 법선에 yaw·pitch, 슬롯 축 = 벽면 수평 접선.
    상단 바: 중심 (x, y, z), 분사 = 아래(−z)에서 +x 로 tilt, 슬롯 축 = +y (부스 폭 방향).
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
        for y in top["y_positions"]:
            positions.append([float(x), float(y), float(top["z"])])
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


# ---------------------------------------------------------------------------
# 세기 구역 (docs/plan_extension.md 2절, 00_common.md 4.7). 구역 정의는 configs/nozzles.yaml 의 zones.
# ---------------------------------------------------------------------------
#: 가슴 쪽 벽 판정에서 정면·후면으로 보는 |sin(torso_yaw)| 한계 (C 의 PlanEncoder FRONT_EPS 와 같다).
CHEST_SIN_EPS = 1e-6
#: 아래를 향한 노즐(상단 바) 판정: 방향 z 성분이 이보다 작으면 top.
_TOP_DIR_Z = -0.5


@lru_cache(maxsize=8)
def _zone_config_cached(path: str, layout: str | None) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    name = layout or raw.get("active", "layout")
    zones = raw.get("zones") or {}
    if name not in zones:
        raise ValueError(f"nozzles.yaml zones 에 배치 {name!r} 의 구역 정의가 없다")
    z = zones[name]
    return (tuple(float(v) for v in z["side_low_z"]), tuple(float(v) for v in z["side_high_z"]),
            float(zones.get("z_tolerance_m", 0.02)))


def zone_config(layout: str | None = None, path: Path | None = None) -> dict:
    """배치 하나의 구역 정의 {side_low_z, side_high_z, z_tolerance_m}. `layout` 이 None 이면 파일의 `active`."""
    low, high, tol = _zone_config_cached(str(path or _ROOT / "configs" / "nozzles.yaml"), layout)
    return {"side_low_z": list(low), "side_high_z": list(high), "z_tolerance_m": tol}


def chest_wall_sign(torso_yaw: float) -> int:
    """가슴 쪽 벽의 y 부호 (+1 = +y 벽, −1 = −y 벽). `torso_yaw` 는 degree.

    몸 전방 벡터 (cos yaw, sin yaw, 0) 이 향하는 쪽 벽이다. sin(yaw) ≥ 0 이면 +y, 정면·후면
    (|sin| < CHEST_SIN_EPS, ±180 포함)은 +y 로 고정한다. 부동소수점에서 sin(−180°) 가 −1e−16 이 되어
    같은 자세인 180° 와 −180° 의 가슴 쪽 벽이 갈리는 것을 막는다.
    """
    s = float(np.sin(np.deg2rad(float(torso_yaw))))
    return 1 if (s >= 0.0 or abs(s) < CHEST_SIN_EPS) else -1


def _nozzle_zone_kind(nozzle: NozzleConfig, zones: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """(M,) 구역 종류 (0 = 측면 low, 1 = 측면 high, 2 = top) 와 (M,) 벽 부호 (측면 ±1, top 0). 자세와 무관."""
    pos = np.asarray(nozzle.positions, dtype=np.float64)
    dirs = np.asarray(nozzle.directions, dtype=np.float64)
    low = np.asarray(zones["side_low_z"], dtype=np.float64)
    high = np.asarray(zones["side_high_z"], dtype=np.float64)
    tol = float(zones.get("z_tolerance_m", 0.02))
    top = dirs[:, 2] < _TOP_DIR_Z
    side = ~top
    z = pos[:, 2][:, None]
    is_low = side & (np.abs(z - low[None, :]) <= tol).any(axis=1)
    is_high = side & (np.abs(z - high[None, :]) <= tol).any(axis=1)
    kind = np.select([top, is_low & ~is_high, is_high & ~is_low], [2, 0, 1], default=-1)
    wall = np.where(side, np.sign(pos[:, 1]), 0).astype(np.int64)
    bad = np.flatnonzero((kind < 0) | (side & (wall == 0)))
    if bad.size:
        m = int(bad[0])
        raise ValueError(f"노즐 {m} (위치 {pos[m].round(3).tolist()}, 방향 {dirs[m].round(3).tolist()}) 이 "
                         f"한 구역에 들지 않는다 (low z {low.tolist()}, high z {high.tolist()}, 허용 {tol} m)")
    return kind.astype(np.int64), wall


def nozzle_zone_index(nozzle: NozzleConfig, torso_yaw: float, zones: Mapping | None = None) -> np.ndarray:
    """(M,) 노즐별 구역 번호 (ZONE_NAMES 인덱스). 측면 노즐은 가슴 쪽 벽(`chest_wall_sign`)이면 chest_*.

    `zones` 가 None 이면 configs/nozzles.yaml 의 `active` 배치 구역 정의 (`zone_config()`).
    """
    kind, wall = _nozzle_zone_kind(nozzle, zones if zones is not None else zone_config())
    chest = wall == chest_wall_sign(torso_yaw)
    table = np.array([[ZONE_NAMES.index("back_low"), ZONE_NAMES.index("back_high"), ZONE_NAMES.index("top")],
                      [ZONE_NAMES.index("chest_low"), ZONE_NAMES.index("chest_high"), ZONE_NAMES.index("top")]])
    return table[chest.astype(np.int64), kind]


def apply_zone_strengths(nozzle: NozzleConfig, zone_strengths: Sequence[float] | np.ndarray,
                         torso_yaw: float, zones: Mapping | None = None) -> NozzleConfig:
    """구역 세기를 노즐별 `strengths` 에 곱한 새 NozzleConfig (interfaces.md, 00_common.md 4.7).

    s_m = zone_strengths[구역(m)] × nozzle.strengths[m]. 구역은 `nozzle_zone_index` (몸 기준: `torso_yaw`
    로 가슴 쪽 벽을 정한다). 입력 `nozzle` 은 바꾸지 않는다. 한도(s_max, 풍량 cap, 쾌적 상한) 보수는 하지
    않는다 (C 의 인코더 몫). 세기가 ZONE_NAMES 순서 5개가 아니거나 음수·비유한값이면 오류.
    """
    s = np.asarray(zone_strengths, dtype=np.float64).reshape(-1)
    if s.shape != (len(ZONE_NAMES),):
        raise ValueError(f"zone_strengths 는 ZONE_NAMES 순서 {len(ZONE_NAMES)}개여야 한다 (받은 모양 {s.shape})")
    if not np.all(np.isfinite(s)) or (s < 0.0).any():
        raise ValueError(f"zone_strengths 는 0 이상의 유한값이어야 한다: {s.tolist()}")
    idx = nozzle_zone_index(nozzle, torso_yaw, zones)
    strengths = (np.asarray(nozzle.strengths, dtype=np.float64) * s[idx]).astype(np.float32)

    def copy(a):
        return None if a is None else np.array(a, copy=True)

    return NozzleConfig(positions=copy(nozzle.positions), directions=copy(nozzle.directions),
                        strengths=strengths, pulse_phase=copy(nozzle.pulse_phase),
                        slot_axis=copy(nozzle.slot_axis), slot_length=copy(nozzle.slot_length))


def zone_nozzle_counts(nozzle: NozzleConfig | None = None, zones: Mapping | None = None) -> np.ndarray:
    """(len(ZONE_NAMES),) 구역별 노즐 수. 기준 배치 slot_bars 는 [2, 2, 2, 2, 4].

    C 의 풍량 한도 보수(Σ_m s_m ≤ cap_ratio·M)에 쓴다. `nozzle` 이 None 이면 `load_nozzles()`.
    좌우 벽의 구역별 노즐 수가 달라 결과가 yaw 에 따라 바뀌면 오류다 (몸 기준 구역의 전제).
    """
    nozzle = nozzle if nozzle is not None else load_nozzles()
    n = len(ZONE_NAMES)
    plus = np.bincount(nozzle_zone_index(nozzle, 90.0, zones), minlength=n)
    minus = np.bincount(nozzle_zone_index(nozzle, -90.0, zones), minlength=n)
    if not np.array_equal(plus, minus):
        raise ValueError(f"좌우 벽의 구역별 노즐 수가 다르다: +y 가슴 {plus.tolist()}, −y 가슴 {minus.tolist()}")
    return plus.astype(np.float64)


def _unit_rows(vectors) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float64)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def load_physics(path: Path | None = None) -> dict:
    path = path or _ROOT / "configs" / "physics.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))
