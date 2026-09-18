"""제트 속도장과 노즐 배치 테스트. 소유자: B. (`docs/tracks/B_body_jet.md` 단계 9)

원형 제트는 00_common.md 4.1, 슬롯 제트는 4.1b. 기준 배치는 configs/nozzles.yaml 의 slot_bars,
원형 비교 배치는 layout.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pytest

from airis.sim.body import build_body
from airis.sim.jet import (
    HALFWIDTH_TO_SIGMA, jet_params, slot_mask, velocity_field, velocity_field_per_nozzle,
)
from airis.sim.scenario import load_nozzle_layout, load_nozzles, load_physics, load_scenarios
from airis.sim.types import BodyParams, NozzleConfig, PoseParams

CFG = load_physics()
P = jet_params(CFG)
U0 = P.exit_velocity_mps
LC = P.potential_core_length_m
U0S = P.slot_exit_velocity_mps
LCS = P.slot_core_length_m
H = P.slot_height_m
SLOT_LEN = 0.6
RAW = load_nozzle_layout()


def _single(strength: float = 1.0, phase=None) -> NozzleConfig:
    """원점에서 +x 로 쏘는 노즐 1개."""
    return NozzleConfig(
        positions=np.zeros((1, 3), dtype=np.float32),
        directions=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        strengths=np.array([strength], dtype=np.float32),
        pulse_phase=phase,
    )


def _single_slot(strength: float = 1.0, length: float = SLOT_LEN, axis=(0.0, 1.0, 0.0)) -> NozzleConfig:
    """원점에서 +x 로 쏘는 슬롯 1개. 슬롯 축 기본 +y."""
    return NozzleConfig(
        positions=np.zeros((1, 3), dtype=np.float32),
        directions=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        strengths=np.array([strength], dtype=np.float32),
        slot_axis=np.array([axis], dtype=np.float32),
        slot_length=np.array([length], dtype=np.float32),
    )


def _slot_sigma(s: float) -> float:
    return (0.5 * H + P.slot_spread_rate * s) / HALFWIDTH_TO_SIGMA


def _speed(points, nozzle=None, t=0.0, cfg=CFG) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, dtype=np.float32))
    return np.linalg.norm(velocity_field(pts, nozzle or _single(), t, cfg), axis=1)


def _booth_points(rng, n: int) -> np.ndarray:
    b = RAW["booth"]
    lo = np.array([0.0, -b["width_m"] / 2, 0.0])
    hi = np.array([b["length_m"], b["width_m"] / 2, b["height_m"]])
    return (lo + rng.random((n, 3)) * (hi - lo)).astype(np.float32)


# ---------------------------------------------------------------------------
# 4.1 원형 제트 (문서 단계 9 명시 항목)
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


# ---------------------------------------------------------------------------
# 4.1b 슬롯 제트
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("frac", [0.1, 0.5, 1.0])
def test_slot_on_axis_inside_core_is_exit_velocity(frac):
    """슬롯 중심면 s ≤ L_c = K_p·h 에서 속도 = U0 (jet.slot.exit_velocity_mps)."""
    assert _speed([frac * LCS, 0, 0], _single_slot())[0] == pytest.approx(U0S, rel=1e-5)


def test_slot_far_field_decays_as_one_over_sqrt_s():
    """코어 밖 U_c = U0·sqrt(K_p·h/s). s = 4 L_c 에서 U0/2."""
    assert _speed([4 * LCS, 0, 0], _single_slot())[0] == pytest.approx(U0S / 2, rel=1e-5)
    for k in (2.0, 9.0, 50.0):
        assert _speed([k * LCS, 0, 0], _single_slot())[0] == pytest.approx(U0S / np.sqrt(k), rel=1e-4)


def test_slot_centre_speed_is_continuous_at_core_end():
    eps = 1e-6
    below = _speed([LCS - eps, 0, 0], _single_slot())[0]
    above = _speed([LCS + eps, 0, 0], _single_slot())[0]
    assert below == pytest.approx(above, rel=1e-4)


@pytest.mark.parametrize("s", [-0.3, -1e-4, 0.0])
def test_slot_behind_is_zero(s):
    assert _speed([s, 0.0, 0.0], _single_slot())[0] == 0.0
    assert _speed([s, 0.1, 0.001], _single_slot())[0] == 0.0


def test_slot_uniform_along_length():
    """슬롯 길이 안(|ρ_e| ≤ L/2)에서는 슬롯 축 방향으로 균일하다."""
    s = 0.3
    centre = _speed([s, 0, 0], _single_slot())[0]
    for y in (0.1, -0.2, 0.5 * SLOT_LEN - 1e-4):
        assert _speed([s, y, 0], _single_slot())[0] == pytest.approx(centre, rel=1e-5)


@pytest.mark.parametrize("s", [0.5 * LCS, 2 * LCS, 0.3, 1.0])
def test_slot_end_decays_as_gaussian(s):
    """슬롯 끝에서 σ 만큼 더 나가면 exp(-0.5), 2σ 면 exp(-2)."""
    sig = _slot_sigma(s)
    centre = _speed([s, 0, 0], _single_slot())[0]
    end = 0.5 * SLOT_LEN
    assert _speed([s, end + sig, 0], _single_slot())[0] == pytest.approx(centre * np.exp(-0.5), rel=1e-4)
    assert _speed([s, -(end + 2 * sig), 0], _single_slot())[0] == pytest.approx(centre * np.exp(-2.0), rel=1e-4)


@pytest.mark.parametrize("s", [0.5 * LCS, 2 * LCS, 0.3, 1.0])
def test_slot_thickness_direction_is_gaussian(s):
    """슬롯 두께 방향 ρ_n = σ 에서 exp(-0.5). 끝 밖과 두께 방향 감쇠는 곱해진다."""
    sig = _slot_sigma(s)
    centre = _speed([s, 0, 0], _single_slot())[0]
    assert _speed([s, 0.1, sig], _single_slot())[0] == pytest.approx(centre * np.exp(-0.5), rel=1e-4)
    corner = [s, 0.5 * SLOT_LEN + sig, -sig]
    assert _speed(corner, _single_slot())[0] == pytest.approx(centre * np.exp(-1.0), rel=1e-4)


def test_slot_strength_scales_slot_exit_velocity():
    assert _speed([LCS, 0, 0], _single_slot(2.0))[0] == pytest.approx(2.0 * U0S, rel=1e-5)


def test_slot_mask_per_row_rule():
    """00_common.md 4.1b: slot_axis None → 전부 원형. 배열이면 slot_length>0 이고 slot_axis≠0 인 행만 슬롯."""
    assert not slot_mask(_single()).any()
    nz = NozzleConfig(
        positions=np.zeros((4, 3), np.float32),
        directions=np.tile([1.0, 0.0, 0.0], (4, 1)).astype(np.float32),
        strengths=np.ones(4, np.float32),
        slot_axis=np.array([[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0]], np.float32),
        slot_length=np.array([0.6, 0.6, 0.0, 0.4], np.float32),
    )
    assert slot_mask(nz).tolist() == [True, False, False, True]


def test_mixed_rows_use_their_own_model():
    """같은 배치 안에서 슬롯 행은 4.1b, 원형 행(축 0 또는 길이 0)은 4.1 과 같은 값."""
    nz = NozzleConfig(
        positions=np.zeros((3, 3), np.float32),
        directions=np.tile([1.0, 0.0, 0.0], (3, 1)).astype(np.float32),
        strengths=np.ones(3, np.float32),
        slot_axis=np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]], np.float32),
        slot_length=np.array([SLOT_LEN, SLOT_LEN, 0.0], np.float32),
    )
    pts = np.array([[0.3, 0.05, 0.0], [LC, 0, 0], [2 * LCS, 0.2, 0.003]], np.float32)
    per = np.linalg.norm(velocity_field_per_nozzle(pts, nz, 0.0, CFG), axis=-1)
    np.testing.assert_allclose(per[0], _speed(pts, _single_slot()), rtol=1e-6)
    np.testing.assert_allclose(per[1], _speed(pts, _single()), rtol=1e-6)
    np.testing.assert_allclose(per[2], _speed(pts, _single()), rtol=1e-6)


def test_slot_axis_must_be_perpendicular_to_direction():
    with pytest.raises(ValueError):
        _speed([0.3, 0, 0], _single_slot(axis=(0.2, 1.0, 0.0)))


def test_slot_axis_without_length_is_rejected():
    nz = _single_slot()
    bad = NozzleConfig(nz.positions, nz.directions, nz.strengths, slot_axis=nz.slot_axis)
    with pytest.raises(ValueError):
        _speed([0.3, 0, 0], bad)


# ---------------------------------------------------------------------------
# 노즐 배치 (configs/nozzles.yaml)
# ---------------------------------------------------------------------------
def test_load_nozzles_default_is_slot_bars():
    """기본 배치는 기준 장비 슬롯 바: 측면 2벽 × 4단 + 상단 2 × 2 = 12개 (크로스팬 12개), 전 행 슬롯."""
    assert RAW["active"] == "slot_bars"
    nz = load_nozzles()
    side, top = RAW["slot_bars"]["side"], RAW["slot_bars"]["top"]
    m = len(side["wall_y"]) * len(side["z_levels"]) + len(top["x_positions"]) * len(top["y_positions"])
    assert nz.count == m == 12
    assert nz.positions.shape == (m, 3) and nz.directions.shape == (m, 3) and nz.strengths.shape == (m,)
    assert nz.slot_axis.shape == (m, 3) and nz.slot_length.shape == (m,)
    for arr in (nz.positions, nz.directions, nz.strengths, nz.slot_axis, nz.slot_length):
        assert arr.dtype == np.float32
    assert nz.pulse_phase is None
    assert slot_mask(nz).all()
    np.testing.assert_allclose(np.linalg.norm(nz.directions, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(nz.slot_axis, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose((nz.directions * nz.slot_axis).sum(1), 0.0, atol=1e-6)


def test_slot_bars_match_yaml():
    side, top = RAW["slot_bars"]["side"], RAW["slot_bars"]["top"]
    nz = load_nozzles()
    ns = len(side["wall_y"]) * len(side["z_levels"])
    s_pos, t_pos = nz.positions[:ns], nz.positions[ns:]

    expected = {(side["x_center"], y, z) for y in side["wall_y"] for z in side["z_levels"]}
    assert {tuple(np.round(r.astype(float), 6)) for r in s_pos} == \
           {tuple(np.round(np.array(e, float), 6)) for e in expected}
    np.testing.assert_allclose(nz.strengths[:ns], side["strength"])
    np.testing.assert_allclose(nz.slot_length[:ns], side["length_m"])
    # 측면: 안쪽 수평 분사(yaw 0, pitch 0), 슬롯 축은 진행 방향 x
    inward = -np.sign(s_pos[:, 1])
    zeros = np.zeros_like(inward)
    np.testing.assert_allclose(nz.directions[:ns], np.stack([zeros, inward, zeros], 1), atol=1e-6)
    np.testing.assert_allclose(np.abs(nz.slot_axis[:ns]), [[1, 0, 0]] * ns, atol=1e-6)

    np.testing.assert_allclose(
        t_pos, [[x, y, top["z"]] for x in top["x_positions"] for y in top["y_positions"]], atol=1e-6)
    np.testing.assert_allclose(nz.strengths[ns:], top["strength"])
    np.testing.assert_allclose(nz.slot_length[ns:], top["length_m"])
    # 상단: 수직 아래로 분사, 슬롯 축은 부스 폭 y
    np.testing.assert_allclose(nz.directions[ns:], [[0, 0, -1]] * len(t_pos), atol=1e-6)
    np.testing.assert_allclose(np.abs(nz.slot_axis[ns:]), [[0, 1, 0]] * len(t_pos), atol=1e-6)


def test_slot_bars_inside_booth():
    booth = RAW["booth"]
    pos = load_nozzles().positions
    assert (np.abs(pos[:, 1]) <= booth["width_m"] / 2 + 1e-6).all()
    assert ((0 <= pos[:, 0]) & (pos[:, 0] <= booth["length_m"])).all()
    assert ((0 < pos[:, 2]) & (pos[:, 2] <= booth["height_m"] + 1e-6)).all()


def test_slot_bar_angles_keep_axis_perpendicular(tmp_path):
    """측면 yaw·pitch, 상단 tilt 를 바꿔도 슬롯 축은 분사 방향에 수직이고 방향은 의도대로 기운다."""
    import yaml
    raw = load_nozzle_layout()
    raw["slot_bars"]["side"].update(yaw_deg=20, pitch_deg=15)
    raw["slot_bars"]["top"]["tilt_deg"] = 10
    path = tmp_path / "nozzles.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    nz = load_nozzles(path)
    ns = len(raw["slot_bars"]["side"]["wall_y"]) * len(raw["slot_bars"]["side"]["z_levels"])
    np.testing.assert_allclose((nz.directions * nz.slot_axis).sum(1), 0.0, atol=1e-6)
    assert (nz.directions[:ns, 0] > 0).all() and (nz.directions[:ns, 2] < 0).all()
    assert (nz.directions[ns:, 0] > 0).all() and (nz.directions[ns:, 2] < 0).all()
    velocity_field(np.array([[0.4, 0.0, 1.0]], np.float32), nz, 0.0, CFG)     # 수직 검사 통과


def test_round_layout_sixteen_unit_inward():
    """원형 비교 배치(layout): 16개, 슬롯 필드 None, 방향 단위 길이, 모두 부스 안쪽."""
    nz = load_nozzles(layout="layout")
    assert nz.count == 16
    assert nz.positions.shape == (16, 3) and nz.directions.shape == (16, 3)
    assert nz.strengths.shape == (16,)
    assert nz.pulse_phase is None and nz.slot_axis is None and nz.slot_length is None
    np.testing.assert_allclose(np.linalg.norm(nz.directions, axis=1), 1.0, atol=1e-6)

    inward = np.zeros_like(nz.directions)
    inward[:, 1] = -np.sign(nz.positions[:, 1])
    assert ((nz.directions * inward).sum(1) > 0).all()


def test_round_layout_matches_yaml():
    lay = RAW["layout"]
    nz = load_nozzles(layout="layout")

    expected = {(x, y, z) for x in lay["x_positions"] for y in lay["wall_y"] for z in lay["z_levels"]}
    got = {tuple(round(float(v), 6) for v in row) for row in nz.positions}
    assert got == {tuple(round(float(v), 6) for v in e) for e in expected}
    np.testing.assert_allclose(nz.strengths, lay["strength"])

    yaw, pitch = np.deg2rad(lay["yaw_deg"]), np.deg2rad(lay["pitch_deg"])
    # yaw 는 진행 방향(+x) 쪽으로, pitch 는 아래(양수)로
    np.testing.assert_allclose(nz.directions[:, 0], np.sin(yaw) * np.cos(pitch), atol=1e-6)
    np.testing.assert_allclose(nz.directions[:, 2], -np.sin(pitch), atol=1e-6)


def test_round_layout_pitch_tilts_down(tmp_path):
    import yaml
    raw = load_nozzle_layout()
    raw["layout"]["pitch_deg"] = 20
    path = tmp_path / "nozzles.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    nz = load_nozzles(path, layout="layout")
    assert (nz.directions[:, 2] < 0).all()
    np.testing.assert_allclose(np.linalg.norm(nz.directions, axis=1), 1.0, atol=1e-6)


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError):
        load_nozzles(layout="nope")


# ---------------------------------------------------------------------------
# 속도장 계약
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("layout", ["slot_bars", "layout"])
def test_velocity_is_sum_of_per_nozzle(layout):
    nz = load_nozzles(layout=layout)
    pts = _booth_points(np.random.default_rng(0), 500)
    per = velocity_field_per_nozzle(pts, nz, 0.0, CFG)
    assert per.shape == (nz.count, 500, 3)
    np.testing.assert_allclose(velocity_field(pts, nz, 0.0, CFG), per.sum(0), atol=1e-4)


def _reference_velocity_per_nozzle(points, nozzle, t, cfg):
    """00_common.md 4.1 을 글자 그대로 옮긴 독립 참고 구현 (float64). (P,3) -> (M,P,3).

    tests/fakes.py 의 fake_velocity_field_per_nozzle 을 대신한다 (D 의 PR #11 이 그 함수를 삭제).
    jet.py 와 상수·보조 함수를 공유하지 않도록 1.177 은 4.1 식에서 직접 가져온다.
    """
    jet = cfg["jet"]
    D = float(jet["nozzle_diameter_m"])
    K = float(jet["decay_constant"])
    k = float(jet["halfwidth_spread_rate"])
    p = np.asarray(points, dtype=np.float64)[None, :, :]                   # (1,P,3)
    n = np.asarray(nozzle.positions, dtype=np.float64)[:, None, :]         # (M,1,3)
    d = np.asarray(nozzle.directions, dtype=np.float64)
    d = (d / np.linalg.norm(d, axis=1, keepdims=True))[:, None, :]
    U0 = float(jet["exit_velocity_mps"]) * np.asarray(nozzle.strengths, dtype=np.float64)[:, None]
    r = p - n
    s = (r * d).sum(axis=-1)                                               # (M,P)
    rho = np.linalg.norm(r - s[..., None] * d, axis=-1)
    Lc = K * D
    s_pos = np.where(s > 0.0, s, 1.0)
    Uc = np.where(s_pos <= Lc, U0, U0 * K * D / s_pos)
    sigma = 0.5 * D / 1.177 + k * s_pos / 1.177
    speed = np.where(s > 0.0, Uc * np.exp(-rho**2 / (2.0 * sigma**2)), 0.0)
    pulse = jet["pulse"]
    if pulse["enabled"]:
        phase = np.zeros(len(U0)) if nozzle.pulse_phase is None else np.asarray(nozzle.pulse_phase, float)
        speed = speed * (((t / float(pulse["period_s"]) + phase) % 1.0) < float(pulse["duty"]))[:, None]
    return speed[..., None] * d


def _reference_slot_velocity_per_nozzle(points, nozzle, t, cfg):
    """00_common.md 4.1b 를 글자 그대로 옮긴 독립 참고 구현 (float64, (M,P,3) 벡터로 직접). 전 행 슬롯 가정."""
    jet = cfg["jet"]
    sl = jet["slot"]
    h, Kp, kp = float(sl["height_m"]), float(sl["decay_constant"]), float(sl["spread_rate"])
    p = np.asarray(points, dtype=np.float64)[None, :, :]
    n = np.asarray(nozzle.positions, dtype=np.float64)[:, None, :]
    d = np.asarray(nozzle.directions, dtype=np.float64)
    d = (d / np.linalg.norm(d, axis=1, keepdims=True))[:, None, :]
    e = np.asarray(nozzle.slot_axis, dtype=np.float64)
    e = (e / np.linalg.norm(e, axis=1, keepdims=True))[:, None, :]
    L = np.asarray(nozzle.slot_length, dtype=np.float64)[:, None]
    U0 = float(sl["exit_velocity_mps"]) * np.asarray(nozzle.strengths, dtype=np.float64)[:, None]
    r = p - n
    s = (r * d).sum(axis=-1)
    rho_e = (r * e).sum(axis=-1)
    rho_n = np.linalg.norm(r - s[..., None] * d - rho_e[..., None] * e, axis=-1)
    Lc = Kp * h
    s_pos = np.where(s > 0.0, s, 1.0)
    Uc = np.where(s_pos <= Lc, U0, U0 * np.sqrt(Kp * h / s_pos))
    sigma = 0.5 * h / 1.177 + kp * s_pos / 1.177
    rho_e_out = np.maximum(np.abs(rho_e) - L / 2.0, 0.0)
    speed = np.where(s > 0.0,
                     Uc * np.exp(-rho_n**2 / (2.0 * sigma**2)) * np.exp(-rho_e_out**2 / (2.0 * sigma**2)),
                     0.0)
    pulse = jet["pulse"]
    if pulse["enabled"]:
        phase = np.zeros(len(U0)) if nozzle.pulse_phase is None else np.asarray(nozzle.pulse_phase, float)
        speed = speed * (((t / float(pulse["period_s"]) + phase) % 1.0) < float(pulse["duty"]))[:, None]
    return speed[..., None] * d


def _probe_points(nz, rng) -> np.ndarray:
    """부스 전체 무작위 점 + 첫 노즐 축 근처 점 (출구 근처 σ 가 작은 영역)."""
    s = np.linspace(1e-3, 1.0, 200)[:, None]
    near_axis = nz.positions[0] + s * nz.directions[0] + rng.normal(0.0, 2e-3, (200, 3))
    return np.vstack([_booth_points(rng, 2000), near_axis]).astype(np.float32)


def test_matches_reference_fake_implementation():
    """원형: 4.1 독립 참고 구현(_reference_velocity_per_nozzle)과 같은 점에서 같은 값."""
    from tests.fakes import fake_nozzles

    rng = np.random.default_rng(3)
    for nz in (load_nozzles(layout="layout"), fake_nozzles()):
        pts = _probe_points(nz, rng)
        ours = velocity_field_per_nozzle(pts, nz, 0.0, CFG)
        ref = _reference_velocity_per_nozzle(pts, nz, 0.0, CFG)
        np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("pulse", [False, True])
def test_slot_matches_reference_implementation(pulse):
    """슬롯: 4.1b 독립 참고 구현과 같은 점에서 같은 값 (기준 배치, 무작위 세기·위상)."""
    rng = np.random.default_rng(5)
    base = load_nozzles()
    nz = NozzleConfig(base.positions, base.directions,
                      (0.5 + 1.5 * rng.random(base.count)).astype(np.float32),
                      pulse_phase=rng.random(base.count).astype(np.float32),
                      slot_axis=base.slot_axis, slot_length=base.slot_length)
    cfg = copy.deepcopy(CFG)
    cfg["jet"]["pulse"]["enabled"] = pulse
    pts = _probe_points(nz, rng)
    for t in (0.0, 0.13, 0.37):
        ours = velocity_field_per_nozzle(pts, nz, t, cfg)
        ref = _reference_slot_velocity_per_nozzle(pts, nz, t, cfg)
        assert np.linalg.norm(ref, axis=-1).max() > 0.5 * U0S          # 공허한 비교가 아니다
        np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("layout", ["slot_bars", "layout"])
def test_velocity_is_parallel_to_jet_direction(layout):
    nz = load_nozzles(layout=layout)
    pts = np.array([[0.44, 0.0, 1.3], [0.3, -0.2, 0.5], [0.1, 0.05, 1.0]], dtype=np.float32)
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
    # 슬롯에도 같은 게이트
    assert _speed([LCS, 0, 0], _single_slot(), t=0.3, cfg=cfg)[0] == 0.0


# ---------------------------------------------------------------------------
# 4.2b 충돌 제트 → 벽면 제트 보정
# ---------------------------------------------------------------------------
K_WALL = P.wall_jet_gain
S_WALL = 0.4                                   # 노즐 → 벽 거리 (원거리, 코어 밖)


def _wall_points(offsets_y, offsets_z=None, s=S_WALL):
    """x = s 인 벽(노즐을 마주 봄, 법선 −x) 위의 점과 법선."""
    offsets_z = np.zeros(len(offsets_y)) if offsets_z is None else offsets_z
    pts = np.stack([np.full(len(offsets_y), s), offsets_y, offsets_z], 1).astype(np.float32)
    normals = np.tile([-1.0, 0.0, 0.0], (len(pts), 1)).astype(np.float32)
    return pts, normals


def _correction_only(pts, normals, nozzle, cfg=CFG, t=0.0):
    """(M,P,3) 보정 성분만 = 보정 켬 − 보정 없음."""
    on = velocity_field_per_nozzle(pts, nozzle, t, cfg, surface_normals=normals).astype(np.float64)
    off = velocity_field_per_nozzle(pts, nozzle, t, cfg).astype(np.float64)
    return on - off


def _tangential(u, normals):
    n = np.asarray(normals, np.float64)
    return u - (u * n).sum(-1, keepdims=True) * n


def _round_centre_sigma(s):
    uc = U0 * min(1.0, LC / s)
    return uc, (0.5 * P.nozzle_diameter_m + P.halfwidth_spread_rate * s) / HALFWIDTH_TO_SIGMA


def _slot_centre_sigma(s):
    return U0S * min(1.0, float(np.sqrt(LCS / s))), _slot_sigma(s)


def test_impingement_enabled_in_config():
    assert CFG["jet"]["impingement"]["enabled"] is True
    assert "stagnation_radius_factor" not in CFG["jet"]["impingement"]
    assert "wall_jet_start_factor" not in CFG["jet"]["impingement"]


def test_impingement_off_or_without_normals_is_free_jet():
    """enabled=false 이거나 법선이 없으면 4.1 / 4.1b 그대로."""
    pts, normals = _wall_points(np.linspace(-0.2, 0.2, 21))
    for nz in (_single(), _single_slot(axis=(0.0, 0.0, 1.0))):
        free = velocity_field_per_nozzle(pts, nz, 0.0, CFG)
        np.testing.assert_array_equal(velocity_field_per_nozzle(pts, nz, 0.0, CFG, surface_normals=None), free)
        cfg = copy.deepcopy(CFG)
        cfg["jet"]["impingement"]["enabled"] = False
        np.testing.assert_array_equal(velocity_field_per_nozzle(pts, nz, 0.0, cfg, surface_normals=normals), free)
        np.testing.assert_array_equal(velocity_field(pts, nz, 0.0, cfg, surface_normals=normals),
                                      velocity_field(pts, nz, 0.0, CFG))


def test_stagnation_point_has_no_tangential_velocity():
    """정체점(ρ_w = 0)에서 보정 0, 접선 속도 0 → τ = 0."""
    pts, normals = _wall_points([0.0])
    u = velocity_field_per_nozzle(pts, _single(), 0.0, CFG, surface_normals=normals).astype(np.float64)
    assert np.abs(_correction_only(pts, normals, _single())).max() == 0.0
    assert np.linalg.norm(_tangential(u, normals)) < 1e-6


@pytest.mark.parametrize("xi", [0.3, 1.0, 1.585, 3.0, 10.0])
def test_round_wall_jet_matches_formula(xi):
    """정면 충돌(cosθ = 1): 보정 = k·U_c(H)·(1 − e^{−ξ²/2})/ξ, 방사 방향."""
    uc, sig = _round_centre_sigma(S_WALL)
    pts, normals = _wall_points([xi * sig])
    corr = _correction_only(pts, normals, _single())[0, 0]
    expected = K_WALL * uc * (1 - np.exp(-xi**2 / 2)) / xi
    assert corr[1] == pytest.approx(expected, rel=1e-4)                  # +y 방향 (방사)
    assert abs(corr[0]) < 1e-6 and abs(corr[2]) < 1e-6


def test_round_wall_jet_ring_peak_and_outer_decay():
    """접선 속도가 정체점 0 → ξ ≈ 1.585 고리에서 최대 → 바깥에서 1/ξ 로 감쇠."""
    _, sig = _round_centre_sigma(S_WALL)
    xis = np.linspace(0.05, 12.0, 2400)
    pts, normals = _wall_points(xis * sig)
    ut = np.linalg.norm(_tangential(
        velocity_field(pts, _single(), 0.0, CFG, surface_normals=normals).astype(np.float64), normals), axis=1)
    peak = xis[int(np.argmax(ut))]
    assert 1.4 < peak < 1.8
    assert ut[0] < 0.1 * ut.max()
    far = xis > 8
    np.testing.assert_allclose(ut[far] * xis[far], (ut[far] * xis[far]).mean(), rtol=0.02)   # ∝ 1/ξ


def test_slot_line_impingement():
    """슬롯(축 +z)이 x = s 벽에 선으로 부딪힌다: 슬롯 방향(z)으로는 보정이 없고 두께 방향(y)으로 퍼진다.

    슬롯 길이 안에서는 z 에 따라 같고, 끝 밖은 exp(−ρ_e'²/2σ²) 로 줄며, 두께 방향 모양은
    (1 − e^{−ξ²/2})/√ξ (봉우리 ξ ≈ 2.16).
    """
    nz = _single_slot(axis=(0.0, 0.0, 1.0))
    uc, sig = _slot_centre_sigma(S_WALL)
    # 슬롯 선 위(y = 0)는 z 에 상관없이 보정 0
    pts, normals = _wall_points(np.zeros(5), np.linspace(-0.25, 0.25, 5))
    assert np.abs(_correction_only(pts, normals, nz)).max() == 0.0
    # 두께 방향 모양과 값
    for xi in (0.5, 2.162, 6.0):
        pts, normals = _wall_points([xi * sig, xi * sig], [0.0, 0.2])
        corr = _correction_only(pts, normals, nz)[0]
        expected = K_WALL * uc * (1 - np.exp(-xi**2 / 2)) / np.sqrt(xi)
        np.testing.assert_allclose(corr[:, 1], expected, rtol=1e-4)
        assert np.abs(corr[:, 2]).max() < 1e-6                              # 슬롯 방향 성분 없음
    # 끝 밖 감쇠
    xi = 2.0
    inside, _ = _wall_points([xi * sig], [0.0])
    outside, normals = _wall_points([xi * sig], [0.5 * SLOT_LEN + sig])
    c_in = _correction_only(inside, normals, nz)[0, 0, 1]
    c_out = _correction_only(outside, normals, nz)[0, 0, 1]
    assert c_out == pytest.approx(c_in * np.exp(-0.5), rel=1e-4)


def test_oblique_scales_with_cos_and_back_face_is_zero():
    """비스듬한 면은 cosθ 배. 제트를 등진 면(cosθ ≤ 0)과 노즐 뒤 평면(H ≤ 0)은 보정 0."""
    uc, sig = _round_centre_sigma(S_WALL)
    theta = np.deg2rad(40.0)
    n_tilt = np.array([[-np.cos(theta), np.sin(theta), 0.0]], np.float32)
    # 제트 축 위 x = S_WALL 을 지나는 기울어진 평면. 충돌점은 (S_WALL, 0, 0), 면내 z 방향으로 ξσ 떨어진 점.
    xi = 1.2
    pt = np.array([[S_WALL, 0.0, xi * sig]], np.float32)
    corr = _correction_only(pt, n_tilt, _single())[0, 0]
    expected = K_WALL * np.cos(theta) * uc * (1 - np.exp(-xi**2 / 2)) / xi
    assert corr[2] == pytest.approx(expected, rel=1e-4)
    # 등진 면
    back = np.array([[1.0, 0.0, 0.0]], np.float32)
    assert np.abs(_correction_only(np.array([[S_WALL, 0.05, 0.0]], np.float32), back, _single())).max() == 0.0
    # 노즐 뒤 평면 (법선 −x, 점이 x < 0)
    behind = np.array([[-0.2, 0.05, 0.0]], np.float32)
    assert np.abs(_correction_only(behind, np.array([[-1.0, 0, 0]], np.float32), _single())).max() == 0.0


def test_wall_jet_gain_and_strength_scale_linearly():
    pts, normals = _wall_points([0.02, 0.05])
    base = _correction_only(pts, normals, _single())
    cfg = copy.deepcopy(CFG)
    cfg["jet"]["impingement"]["wall_jet_gain"] = 2.5 * K_WALL
    np.testing.assert_allclose(_correction_only(pts, normals, _single(), cfg), 2.5 * base, rtol=1e-5)
    np.testing.assert_allclose(_correction_only(pts, normals, _single(0.4)), 0.4 * base, rtol=1e-5, atol=1e-7)


def test_impingement_respects_pulse_gate():
    cfg = copy.deepcopy(CFG)
    cfg["jet"]["pulse"].update(enabled=True, period_s=0.5, duty=0.5)
    pts, normals = _wall_points([0.03])
    assert np.abs(_correction_only(pts, normals, _single(), cfg, t=0.0)).max() > 0.1
    assert np.abs(velocity_field(pts, _single(), 0.3, cfg, surface_normals=normals)).max() == 0.0


def test_normals_shape_must_match_points():
    pts, normals = _wall_points([0.0, 0.1])
    with pytest.raises(ValueError):
        velocity_field(pts, _single(), 0.0, CFG, surface_normals=normals[:1])


def _reference_impingement(points, normals, nozzle, t, cfg):
    """00_common.md 4.2b 를 글자 그대로 옮긴 독립 참고 구현 (float64, 노즐·점 이중 루프). (M,P,3) 보정만."""
    jet, imp = cfg["jet"], cfg["jet"]["impingement"]
    k = float(imp["wall_jet_gain"])
    D, K, kr = float(jet["nozzle_diameter_m"]), float(jet["decay_constant"]), float(jet["halfwidth_spread_rate"])
    sl = jet["slot"]
    h, Kp, kp = float(sl["height_m"]), float(sl["decay_constant"]), float(sl["spread_rate"])
    pulse = jet["pulse"]
    out = np.zeros((nozzle.count, len(points), 3))
    for m in range(nozzle.count):
        nm = np.asarray(nozzle.positions[m], float)
        d = np.asarray(nozzle.directions[m], float)
        d /= np.linalg.norm(d)
        is_slot = (nozzle.slot_axis is not None and float(nozzle.slot_length[m]) > 0
                   and np.linalg.norm(nozzle.slot_axis[m]) > 0)
        gate = 1.0
        if pulse["enabled"]:
            ph = 0.0 if nozzle.pulse_phase is None else float(nozzle.pulse_phase[m])
            gate = 1.0 if ((t / float(pulse["period_s"]) + ph) % 1.0) < float(pulse["duty"]) else 0.0
        for i, (x, n) in enumerate(zip(np.asarray(points, float), np.asarray(normals, float))):
            n = n / np.linalg.norm(n)
            cos = -d @ n
            if cos <= 0:
                continue
            H = ((x - nm) @ n) / (d @ n)
            if H <= 0:
                continue
            c = nm + H * d
            r = x - c
            r = r - (r @ n) * n
            end = 1.0
            if is_slot:
                e = np.asarray(nozzle.slot_axis[m], float)
                e /= np.linalg.norm(e)
                et = e - (e @ n) * n
                if np.linalg.norm(et) > 0:
                    et /= np.linalg.norm(et)
                    rho_e = r @ et
                    r = r - rho_e * et
                else:
                    rho_e = 0.0
                sig = 0.5 * h / 1.177 + kp * H / 1.177
                uh = float(sl["exit_velocity_mps"]) * (1.0 if H <= Kp * h else np.sqrt(Kp * h / H))
                end = np.exp(-max(abs(rho_e) - float(nozzle.slot_length[m]) / 2, 0.0) ** 2 / (2 * sig**2))
            else:
                sig = 0.5 * D / 1.177 + kr * H / 1.177
                uh = float(jet["exit_velocity_mps"]) * (1.0 if H <= K * D else K * D / H)
            uh *= float(nozzle.strengths[m])
            rho = np.linalg.norm(r)
            xi = rho / sig
            if xi < 1e-6:
                continue
            F = (1 - np.exp(-xi**2 / 2)) / (np.sqrt(xi) if is_slot else xi)
            out[m, i] = k * cos * uh * F * end * gate * r / rho
    return out


@pytest.mark.parametrize("layout", ["slot_bars", "layout"])
def test_impingement_matches_reference_on_mannequin(layout):
    """마네킹 패치(가림 없음)에서 4.2b 독립 참고 구현과 일치. 무작위 세기·위상, 펄스 켬."""
    rng = np.random.default_rng(11)
    base = load_nozzles(layout=layout)
    nz = NozzleConfig(base.positions, base.directions,
                      (0.5 + rng.random(base.count)).astype(np.float32),
                      pulse_phase=rng.random(base.count).astype(np.float32),
                      slot_axis=base.slot_axis, slot_length=base.slot_length)
    cfg = copy.deepcopy(CFG)
    cfg["jet"]["pulse"]["enabled"] = True
    state = build_body(BodyParams(), PoseParams(shoulder_abduction=45), load_scenarios()["default"])
    idx = rng.choice(len(state.patch_pos), 400, replace=False)
    normals = state.patch_normal[idx]
    pts = (state.patch_pos[idx] + CFG["air"]["wall_offset_m"] * normals).astype(np.float32)
    for t in (0.0, 0.21):
        ours = _correction_only(pts, normals, nz, cfg, t)
        ref = _reference_impingement(pts, normals, nz, t, cfg)
        assert np.abs(ref).max() > 0.5                                      # 공허한 비교가 아니다
        np.testing.assert_allclose(ours, ref, rtol=1e-4, atol=2e-5)


def test_velocity_field_with_normals_is_sum_of_per_nozzle():
    nz = load_nozzles()
    state = build_body(BodyParams(), PoseParams(), load_scenarios()["default"])
    n = state.patch_normal
    pts = (state.patch_pos + CFG["air"]["wall_offset_m"] * n).astype(np.float32)
    per = velocity_field_per_nozzle(pts, nz, 0.0, CFG, surface_normals=n)
    np.testing.assert_allclose(velocity_field(pts, nz, 0.0, CFG, surface_normals=n), per.sum(0), atol=1e-4)


def test_facing_wall_raises_front_tangential_speed():
    """E1 방향: 기준 배치에서 앞면이 벽을 보면(yaw 90) 앞면 접선 속도가 보정으로 오른다 (가림 없음)."""
    from airis.sim.types import PART_NAMES
    front = PART_NAMES.index("torso_front")
    nz = load_nozzles()

    def front_ut(yaw, normals_on):
        st = build_body(BodyParams(), PoseParams(torso_yaw=yaw), load_scenarios()["default"])
        n = st.patch_normal.astype(np.float64)
        pts = (st.patch_pos + CFG["air"]["wall_offset_m"] * n).astype(np.float32)
        u = velocity_field(pts, nz, 0.0, CFG, surface_normals=n if normals_on else None).astype(np.float64)
        m = st.patch_part == front
        return float(np.linalg.norm(_tangential(u, n), axis=1)[m].mean())

    assert front_ut(90.0, True) > 1.5 * front_ut(90.0, False)
    assert front_ut(90.0, True) > front_ut(0.0, True)


# ---------------------------------------------------------------------------
# 완료 기준: 제트가 마네킹에 닿는다
# ---------------------------------------------------------------------------
def _slot_centre_speed(s: float) -> float:
    """4.1b 중심 속도 U_c(s), strength 1."""
    return U0S * min(1.0, float(np.sqrt(LCS / s)))


@pytest.mark.parametrize("scenario", ["default", "pregnant", "wheelchair"])
def test_slot_bars_reach_mannequin(scenario):
    """완료 기준: 기준 배치(슬롯 바 12개)의 제트가 마네킹에 닿는다. 앉은 자세 포함.

    측면 바는 부스 중앙(마네킹 x)에 걸친 수평 슬롯이라 슬롯 길이가 몸 중심 x 를 덮는다.
    몸 높이 안의 측면 바는 몸이 벽과 중심선 사이에 있으므로 몸 위 최대 속도가 중심선 거리
    |wall_y| 에서의 중심 속도 U_c(|wall_y|) 이상이어야 한다. 휠체어에서 앉은 머리보다 높은 측면 바와
    상단 에어커튼 바(x 0.10, 0.79)는 퍼진 가장자리만 닿으므로 0.5 m/s 초과만 요구한다.
    """
    nz = load_nozzles()
    side = RAW["slot_bars"]["side"]
    ns = len(side["wall_y"]) * len(side["z_levels"])
    state = build_body(BodyParams(), PoseParams(), load_scenarios()[scenario])
    per = np.linalg.norm(velocity_field_per_nozzle(state.patch_pos, nz, 0.0, CFG), axis=-1).max(axis=1)
    assert (per > 0.5).all(), per

    within = nz.positions[:ns, 2] <= state.patch_pos[:, 2].max()
    floor = _slot_centre_speed(abs(float(side["wall_y"][1])))
    assert within.sum() >= 6
    assert (per[:ns][within] >= floor).all(), (per[:ns], floor)
    if scenario != "wheelchair":
        assert within.all()

    centre_x = float(state.capsules[1, 0])
    half = 0.5 * nz.slot_length[:ns]
    assert (np.abs(nz.positions[:ns, 0] - centre_x) < half).all()


@pytest.mark.parametrize("scenario", ["default", "pregnant"])
def test_round_layout_jets_reach_mannequin(scenario):
    """원형 비교 배치 16개도 마네킹에 닿는다. 제트 축이 부스 중앙 수직선 근처를 지난다."""
    nz = load_nozzles(layout="layout")
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


def test_round_layout_wheelchair_top_row_passes_over_seated_head():
    """원형 비교 배치: 최상단(z=1.7) 노즐은 앉은 사람 머리 위로 지나가고, 나머지는 닿는다.

    결함이 아니라 물리적으로 맞는 결과다. scenarios.yaml 이 휠체어 nozzle_height_range_m 을 [0.2, 1.5] 로
    두는 이유와 같다 (README 12절 노즐 확장). 고정 배치는 시나리오별로 바꾸지 않는다.
    """
    nz = load_nozzles(layout="layout")
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
    """완료 기준: configs/physics.yaml 의 jet 섹션 키가 코드에서 전부 읽힌다 (원형·슬롯)."""
    seen: set[str] = set()
    cfg = dict(CFG)
    cfg["jet"] = _Tracking(CFG["jet"], "jet", seen)
    pts = np.array([[LC, 0, 0]], dtype=np.float32)
    normals = np.array([[-1.0, 0.0, 0.0]], dtype=np.float32)
    velocity_field(pts, _single(), 0.0, cfg, surface_normals=normals)
    velocity_field(pts, _single_slot(), 0.0, cfg, surface_normals=normals)
    missing = _leaf_paths(CFG["jet"], "jet") - seen
    assert not missing, f"읽히지 않는 jet 키: {sorted(missing)}"


@pytest.mark.parametrize("layout", ["slot_bars", "layout"])
def test_velocity_field_with_impingement_under_25ms(layout):
    """4.2b 보정 켬: 패치 3,600개 × 노즐(슬롯 12 / 원형 16) 25 ms 이하 (보정 없는 경로의 약 4배 여유)."""
    nz = load_nozzles(layout=layout)
    state = build_body(BodyParams(), PoseParams(), load_scenarios()["default"], patches_per_m2=3600 / 2.21)
    n = state.patch_normal
    pts = (state.patch_pos + CFG["air"]["wall_offset_m"] * n).astype(np.float32)
    velocity_field_per_nozzle(pts, nz, 0.0, CFG, surface_normals=n)
    best = float("inf")
    for _ in range(10):
        t0 = time.perf_counter()
        velocity_field_per_nozzle(pts, nz, 0.0, CFG, surface_normals=n)
        best = min(best, time.perf_counter() - t0)
    assert best < 0.025, f"보정 켠 velocity_field_per_nozzle {best * 1e3:.2f} ms (P = {len(pts)})"


@pytest.mark.parametrize("layout", ["slot_bars", "layout"])
def test_velocity_field_under_5ms(layout):
    """완료 기준: 패치 3,600개 × 노즐(슬롯 10 / 원형 16) 5 ms 이하. 타이밍 잡음을 피하려고 최솟값을 쓴다."""
    nz = load_nozzles(layout=layout)
    pts = _booth_points(np.random.default_rng(1), 3600)
    velocity_field(pts, nz, 0.0, CFG)
    best = float("inf")
    for _ in range(15):
        t0 = time.perf_counter()
        velocity_field(pts, nz, 0.0, CFG)
        best = min(best, time.perf_counter() - t0)
    assert best < 0.005, f"velocity_field {best * 1e3:.2f} ms"
