"""제트 속도장과 노즐 배치 테스트. 소유자: B. (`docs/tracks/B_body_jet.md` 단계 9)"""
from __future__ import annotations

import copy
import time

import numpy as np
import pytest

from airis.sim.body import build_body
from airis.sim.jet import (
    HALFWIDTH_TO_SIGMA, jet_params, velocity_field, velocity_field_per_nozzle,
)
from airis.sim.scenario import load_nozzles, load_physics, load_scenarios
from airis.sim.types import BodyParams, NozzleConfig, PoseParams

CFG = load_physics()
P = jet_params(CFG)
U0 = P.exit_velocity_mps
LC = P.potential_core_length_m


def _single(strength: float = 1.0, phase=None) -> NozzleConfig:
    """원점에서 +x 로 쏘는 노즐 1개."""
    return NozzleConfig(
        positions=np.zeros((1, 3), dtype=np.float32),
        directions=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        strengths=np.array([strength], dtype=np.float32),
        pulse_phase=phase,
    )


def _speed(points, nozzle=None, t=0.0, cfg=CFG) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, dtype=np.float32))
    return np.linalg.norm(velocity_field(pts, nozzle or _single(), t, cfg), axis=1)


# ---------------------------------------------------------------------------
# 문서 단계 9 명시 항목
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("frac", [0.1, 0.5, 1.0])
def test_on_axis_inside_potential_core_is_exit_velocity(frac):
    """노즐 축 위 s ≤ L_c 에서 속도 = U0."""
    assert _speed([frac * LC, 0, 0])[0] == pytest.approx(U0, rel=1e-5)


def test_on_axis_at_twice_core_length_is_half():
    """s = 2 L_c 에서 U0/2."""
    assert _speed([2 * LC, 0, 0])[0] == pytest.approx(U0 / 2, rel=1e-5)


def test_on_axis_far_field_decays_as_one_over_s():
    for k in (3.0, 10.0, 50.0):
        assert _speed([k * LC, 0, 0])[0] == pytest.approx(U0 / k, rel=1e-4)


@pytest.mark.parametrize("s", [-1.0, -LC, -1e-4, 0.0])
def test_behind_nozzle_is_zero(s):
    """노즐 뒤(s ≤ 0)는 0."""
    assert _speed([s, 0.0, 0.0])[0] == 0.0
    assert _speed([s, 0.05, 0.02])[0] == 0.0


@pytest.mark.parametrize("s", [0.5 * LC, 2 * LC, 0.3, 1.0])
def test_radius_sigma_is_exp_minus_half_of_centre(s):
    """반경 σ 에서 중심의 exp(-0.5)배."""
    sigma = (0.5 * P.nozzle_diameter_m + P.halfwidth_spread_rate * s) / HALFWIDTH_TO_SIGMA
    centre = _speed([s, 0, 0])[0]
    for off in ([s, sigma, 0], [s, 0, sigma], [s, sigma / np.sqrt(2), -sigma / np.sqrt(2)]):
        assert _speed(off)[0] == pytest.approx(centre * np.exp(-0.5), rel=1e-4)


def test_load_nozzles_sixteen_unit_inward():
    """load_nozzles(): 16개, 방향 단위 길이, 모두 부스 안쪽 (d·(0,-sign(y),0) > 0)."""
    nz = load_nozzles()
    assert nz.count == 16
    assert nz.positions.shape == (16, 3) and nz.directions.shape == (16, 3)
    assert nz.strengths.shape == (16,)
    assert nz.pulse_phase is None
    np.testing.assert_allclose(np.linalg.norm(nz.directions, axis=1), 1.0, atol=1e-6)

    inward = np.zeros_like(nz.directions)
    inward[:, 1] = -np.sign(nz.positions[:, 1])
    assert ((nz.directions * inward).sum(1) > 0).all()


# ---------------------------------------------------------------------------
# 노즐 배치 세부
# ---------------------------------------------------------------------------
def test_nozzle_layout_matches_yaml():
    from airis.sim.scenario import load_nozzle_layout
    lay = load_nozzle_layout()["layout"]
    nz = load_nozzles()

    expected = {(x, y, z) for x in lay["x_positions"] for y in lay["wall_y"] for z in lay["z_levels"]}
    got = {tuple(round(float(v), 6) for v in row) for row in nz.positions}
    assert got == {tuple(round(float(v), 6) for v in e) for e in expected}
    np.testing.assert_allclose(nz.strengths, lay["strength"])

    yaw, pitch = np.deg2rad(lay["yaw_deg"]), np.deg2rad(lay["pitch_deg"])
    # yaw 는 진행 방향(+x) 쪽으로, pitch 는 아래(양수)로
    np.testing.assert_allclose(nz.directions[:, 0], np.sin(yaw) * np.cos(pitch), atol=1e-6)
    np.testing.assert_allclose(nz.directions[:, 2], -np.sin(pitch), atol=1e-6)


def test_nozzle_pitch_tilts_down(tmp_path):
    import yaml
    from airis.sim.scenario import load_nozzle_layout
    raw = load_nozzle_layout()
    raw["layout"]["pitch_deg"] = 20
    path = tmp_path / "nozzles.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    nz = load_nozzles(path)
    assert (nz.directions[:, 2] < 0).all()
    np.testing.assert_allclose(np.linalg.norm(nz.directions, axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 속도장 계약
# ---------------------------------------------------------------------------
def test_velocity_is_sum_of_per_nozzle():
    nz = load_nozzles()
    rng = np.random.default_rng(0)
    pts = (rng.random((500, 3)) * [2.0, 1.2, 2.3] - [0, 0.6, 0]).astype(np.float32)
    per = velocity_field_per_nozzle(pts, nz, 0.0, CFG)
    assert per.shape == (nz.count, 500, 3)
    np.testing.assert_allclose(velocity_field(pts, nz, 0.0, CFG), per.sum(0), atol=1e-4)


def test_velocity_is_parallel_to_jet_direction():
    nz = load_nozzles()
    pts = np.array([[1.0, 0.0, 1.3], [0.9, -0.2, 0.5]], dtype=np.float32)
    per = velocity_field_per_nozzle(pts, nz, 0.0, CFG)
    cross = np.cross(per, nz.directions[:, None, :])
    np.testing.assert_allclose(cross, 0.0, atol=1e-5)


def test_strength_scales_exit_velocity():
    assert _speed([LC, 0, 0], _single(0.4))[0] == pytest.approx(0.4 * U0, rel=1e-5)


def test_pulse_off_by_default_ignores_time():
    assert CFG["jet"]["pulse"]["enabled"] is False
    assert _speed([LC, 0, 0], t=0.37)[0] == pytest.approx(U0, rel=1e-5)


def test_pulse_gate():
    """gate(t) = 1 if ((t/period + phase) mod 1) < duty else 0."""
    cfg = copy.deepcopy(CFG)
    cfg["jet"]["pulse"].update(enabled=True, period_s=0.5, duty=0.5)
    on = [LC, 0, 0]
    assert _speed(on, t=0.0, cfg=cfg)[0] == pytest.approx(U0)
    assert _speed(on, t=0.2, cfg=cfg)[0] == pytest.approx(U0)     # 0.4 < 0.5
    assert _speed(on, t=0.3, cfg=cfg)[0] == 0.0                   # 0.6 ≥ 0.5
    assert _speed(on, t=0.5, cfg=cfg)[0] == pytest.approx(U0)     # 다음 주기
    # 위상 0.5 면 반대로
    assert _speed(on, _single(phase=np.array([0.5])), t=0.0, cfg=cfg)[0] == 0.0
    assert _speed(on, _single(phase=np.array([0.5])), t=0.3, cfg=cfg)[0] == pytest.approx(U0)


def test_impingement_not_silently_ignored():
    """단계 8 미구현: 설정은 꺼져 있고, 켜진 채 법선을 넘기면 조용히 자유 제트를 돌려주지 않는다."""
    pts = np.array([[LC, 0, 0]], dtype=np.float32)
    normals = np.array([[-1.0, 0, 0]], dtype=np.float32)
    assert CFG["jet"]["impingement"]["enabled"] is False
    velocity_field(pts, _single(), 0.0, CFG, surface_normals=normals)   # 꺼져 있으면 무시

    cfg = copy.deepcopy(CFG)
    cfg["jet"]["impingement"]["enabled"] = True
    with pytest.raises(NotImplementedError):
        velocity_field(pts, _single(), 0.0, cfg, surface_normals=normals)


@pytest.mark.parametrize("scenario", ["default", "pregnant"])
def test_jets_reach_mannequin(scenario):
    """완료 기준: 노즐 16개의 제트가 마네킹 위치에 닿는다 (서 있는 마네킹).

    노즐마다 몸 패치 중 의미 있는 속도를 받는 패치가 있어야 하고, 제트 축이 마네킹이 선 부스 중앙 근처를 지나야 한다.
    """
    nz = load_nozzles()
    state = build_body(BodyParams(), PoseParams(), load_scenarios()[scenario])
    per = np.linalg.norm(velocity_field_per_nozzle(state.patch_pos, nz, 0.0, CFG), axis=-1)
    assert (per.max(axis=1) > 0.5).all(), per.max(axis=1)

    # 각 제트 축이 몸통 축(부스 중앙 수직선)에 가장 가까이 오는 거리
    centre = state.capsules[1, :3]
    for n, d in zip(nz.positions, nz.directions):
        s = float((centre - n)[:2] @ d[:2]) / float(d[:2] @ d[:2])
        closest = n[:2] + s * d[:2]
        assert s > 0
        assert np.linalg.norm(closest - centre[:2]) < 0.35


def test_wheelchair_top_row_passes_over_seated_head():
    """휠체어: 고정 배치의 최상단(z=1.7) 노즐은 앉은 사람 머리 위로 지나가고, 나머지는 닿는다.

    결함이 아니라 물리적으로 맞는 결과다. scenarios.yaml 이 휠체어 nozzle_height_range_m 을 [0.2, 1.5] 로
    두는 이유와 같다 (README 12절 노즐 확장). 고정 배치는 시나리오별로 바꾸지 않는다.
    """
    nz = load_nozzles()
    sc = load_scenarios()["wheelchair"]
    state = build_body(BodyParams(), PoseParams(), sc)
    per = np.linalg.norm(velocity_field_per_nozzle(state.patch_pos, nz, 0.0, CFG), axis=-1).max(axis=1)

    above_head = nz.positions[:, 2] > state.patch_pos[:, 2].max() + 0.1
    assert above_head.sum() == 4
    assert (per[above_head] < 1e-3).all()
    assert (per[~above_head] > 0.5).all(), per[~above_head]
    assert nz.positions[above_head, 2].min() > sc.nozzle_height_range_m[1]


# ---------------------------------------------------------------------------
# 완료 기준: 설정 키 전부 사용, 성능
# ---------------------------------------------------------------------------
class _Tracking(dict):
    """읽힌 키를 기록하는 dict."""

    def __init__(self, data, prefix, seen):
        super().__init__(data)
        self._prefix, self._seen = prefix, seen

    def __getitem__(self, key):
        path = f"{self._prefix}.{key}"
        self._seen.add(path)
        value = super().__getitem__(key)
        return _Tracking(value, path, self._seen) if isinstance(value, dict) else value


def _leaf_paths(d: dict, prefix: str) -> set[str]:
    out = set()
    for k, v in d.items():
        path = f"{prefix}.{k}"
        out |= _leaf_paths(v, path) if isinstance(v, dict) else {path}
    return out


def test_every_jet_config_key_is_read():
    """완료 기준: configs/physics.yaml 의 jet 섹션 키가 코드에서 전부 읽힌다."""
    seen: set[str] = set()
    cfg = dict(CFG)
    cfg["jet"] = _Tracking(CFG["jet"], "jet", seen)
    velocity_field(np.array([[LC, 0, 0]], dtype=np.float32), _single(), 0.0, cfg)
    missing = _leaf_paths(CFG["jet"], "jet") - seen
    assert not missing, f"읽히지 않는 jet 키: {sorted(missing)}"


def test_velocity_field_under_5ms():
    """완료 기준: 패치 3,600개 × 노즐 16개 5 ms 이하. 타이밍 잡음을 피하려고 최솟값을 쓴다."""
    nz = load_nozzles()
    rng = np.random.default_rng(1)
    pts = (rng.random((3600, 3)) * [2.0, 1.2, 2.3] - [0, 0.6, 0]).astype(np.float32)
    velocity_field(pts, nz, 0.0, CFG)
    best = float("inf")
    for _ in range(15):
        t0 = time.perf_counter()
        velocity_field(pts, nz, 0.0, CFG)
        best = min(best, time.perf_counter() - t0)
    assert best < 0.005, f"velocity_field {best * 1e3:.2f} ms"
