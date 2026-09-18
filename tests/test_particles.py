"""입자 평가기 테스트. 소유자: A.

`docs/tracks/A_particles.md` 단계 10의 후보 격리 테스트 6개 + 시각 확인용 산점도.
개발 중 설정은 `N=1000, B=2, duration 0.5 s` (00_common.md 7절, GPU 공유).

입력
- 격리 테스트: 기하를 완전히 통제하려고 원통 캡슐 BodyState를 여기서 만든다
  (`cylinder_body`, 후보마다 캡슐 수가 다르게). 노즐은 원통을 스치는 원형 4개.
- 실제 경로: B의 `build_body` + `load_nozzles()`(기준 장비 슬롯 바)로 `evaluate`/`batch_evaluate`를
  C가 부르는 그대로 호출한다. D의 `fake_body`/`fake_nozzles`도 한 번 돌린다.
- 물리 상수는 전부 `configs/physics.yaml` 원본이다 (B PR #10 이후 덮어쓰기 없음). 제거율이
  실제로 0보다 큰지 함께 단언해 격리 테스트가 "전부 0 == 전부 0"으로 공허하게 통과하지 않게 한다.

제트 일치 (단계 4)
- 원형(4.1), 슬롯(4.1b), 둘이 섞인 배치를 각각 정상 상태와 펄스로 돌린다. 기준값은 이 파일의
  numpy float64 독립 구현(`_jet_velocity_numpy`)이고 상대 오차 1e-4. B의 `jet.velocity_field`
  (float64)와도 1e-4로 대조한다.
- 4.2b 충돌 제트 보정: 부착 입자가 이탈 판정에 쓰는 속도(`probe_surface_velocity`)를 마네킹
  패치에서 float64 독립 구현(`_impingement_numpy`)과 B의 `velocity_field(..., surface_normals=)`에
  각각 1e-4로 대조한다.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from airis.sim.jet import slot_mask, velocity_field, velocity_field_per_nozzle  # noqa: E402
from airis.sim.patch_baseline import occlusion  # noqa: E402
from airis.sim import scoring  # noqa: E402
from airis.sim.body import build_body  # noqa: E402
from airis.sim.particles import ParticleEvaluator  # noqa: E402
from airis.sim.scenario import (  # noqa: E402
    load_nozzle_layout, load_nozzles, load_physics, load_scenarios,
)
from airis.sim.types import (  # noqa: E402
    PART_NAMES, BodyParams, BodyState, NozzleConfig, PoseParams,
)

ROOT = Path(__file__).resolve().parents[1]
N_DEV = 1000
B_DEV = 2
DURATION_DEV = 0.5
BOOTH = load_nozzle_layout()["booth"]
CENTER_X = BOOTH["length_m"] / 2
TORSO = PART_NAMES.index("torso_front")
ARMS = PART_NAMES.index("arms")


# ------------------------------------------------------------------ 가짜 입력
def _capsule_patches(p0, p1, radius, part, n_axial=16, n_theta=24):
    """캡슐 원통 옆면을 격자 패치로 (반구 끝은 생략). 난수 없음."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    a = axis / length
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    th = 2 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
    ss = length * (np.arange(n_axial) + 0.5) / n_axial
    S, T = np.meshgrid(ss, th, indexing="ij")
    S, T = S.ravel(), T.ravel()
    normal = np.cos(T)[:, None] * e1 + np.sin(T)[:, None] * e2
    pos = p0 + S[:, None] * a + radius * normal
    area = np.full(S.size, 2 * np.pi * radius * length / S.size)
    return pos, normal, area, np.full(S.size, part)


def cylinder_body(pose: PoseParams | None = None) -> BodyState:
    """원통 캡슐 몸통 1개. 자세에 따라 기하가 달라지게 해 격리 테스트가 공허하지 않게 한다.

    - torso_pitch: 몸통 축을 +x 쪽으로 기울임
    - shoulder_abduction >= 45: 팔 캡슐 1개 추가 (후보마다 캡슐 수 K가 달라지는 경우)
    """
    pose = pose or PoseParams()
    base = np.array([CENTER_X, 0.0, 0.55])
    pitch = math.radians(pose.torso_pitch)
    top = base + 1.0 * np.array([math.sin(pitch), 0.0, math.cos(pitch)])
    r_torso = 0.15
    caps = [(base, top, r_torso, TORSO)]
    if pose.shoulder_abduction >= 45:
        ab = math.radians(pose.shoulder_abduction)
        shoulder = top + np.array([0.0, -(r_torso + 0.05), -0.05])
        # 팔 길이 0.45 m: abduction 90도에서도 손끝 y = -0.65로 부스 옆벽(±0.73) 안에 남는다
        hand = shoulder + 0.45 * np.array([0.0, -math.sin(ab), -math.cos(ab)])
        caps.append((shoulder, hand, 0.05, ARMS))

    pieces = [_capsule_patches(p0, p1, r, part) for p0, p1, r, part in caps]
    pos, normal, area, part = (np.concatenate(x) for x in zip(*pieces))
    patch_capsule = np.concatenate([np.full(len(pc[0]), k) for k, pc in enumerate(pieces)])
    return BodyState(
        patch_pos=pos.astype(np.float32),
        patch_normal=normal.astype(np.float32),
        patch_area=area.astype(np.float32),
        patch_part=part.astype(np.int32),
        capsules=np.array([[*p0, *p1, r] for p0, p1, r, _ in caps], dtype=np.float32),
        capsule_part=np.array([part for *_, part in caps], dtype=np.int32),
        patch_capsule=patch_capsule.astype(np.int32),
    )


def grazing_nozzles(strength: float = 1.0) -> NozzleConfig:
    """몸통 원통(반지름 0.15)의 옆면을 스치는 노즐 4개. 양쪽 벽, 두 높이."""
    x_lo, x_hi = CENTER_X - 0.15, CENTER_X + 0.15
    y_wall = BOOTH["width_m"] / 2
    pos = np.array([[x_lo, -y_wall, 0.9], [x_hi, -y_wall, 1.3],
                    [x_lo, y_wall, 1.3], [x_hi, y_wall, 0.9]], dtype=np.float32)
    dirs = np.array([[0, 1, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0]], dtype=np.float32)
    return NozzleConfig(pos, dirs, np.full(4, strength, dtype=np.float32))


def slot_nozzles() -> NozzleConfig:
    """B의 기준 장비 배치 (퓨리움 슬롯 바 12개, 전부 4.1b)."""
    nz = load_nozzles(layout="slot_bars")
    assert nz.slot_axis is not None and slot_mask(nz).all()
    return nz


def round_nozzles() -> NozzleConfig:
    """B의 원형 비교 배치 (16개, 전부 4.1). 기준 배치(`active`)는 슬롯이라 명시해서 부른다."""
    nz = load_nozzles(layout="layout")
    assert nz.slot_axis is None
    return nz


def layout_nozzles(rng: np.random.Generator, kind: str = "round") -> NozzleConfig:
    """제트 테스트용 배치에 무작위 세기와 펄스 위상을 입힌 것.

    - "round": 원형 비교 배치 16개 (slot_axis = None)
    - "slot":  기준 장비 슬롯 바 12개 (slot_axis/slot_length 그대로)
    - "mixed": 둘을 이어 붙인 28개. 원형 행은 slot_axis = 0벡터, slot_length = 0 (행 단위 규약)
    """
    rnd, slt = round_nozzles(), slot_nozzles()
    if kind == "round":
        pos, dirs, axis, length = rnd.positions, rnd.directions, None, None
    elif kind == "slot":
        pos, dirs = slt.positions, slt.directions
        axis, length = slt.slot_axis, slt.slot_length
    elif kind == "mixed":
        pos = np.concatenate([rnd.positions, slt.positions])
        dirs = np.concatenate([rnd.directions, slt.directions])
        axis = np.concatenate([np.zeros((rnd.count, 3), np.float32), slt.slot_axis])
        length = np.concatenate([np.zeros(rnd.count, np.float32), slt.slot_length])
    else:
        raise ValueError(kind)
    m = len(pos)
    return NozzleConfig(np.asarray(pos, np.float32), np.asarray(dirs, np.float32),
                        rng.uniform(0.3, 1.0, m).astype(np.float32),
                        pulse_phase=rng.random(m).astype(np.float32),
                        slot_axis=None if axis is None else np.asarray(axis, np.float32),
                        slot_length=None if length is None else np.asarray(length, np.float32))


# ----------------------------------------------------------------- 픽스처
@pytest.fixture(scope="module")
def scenario():
    return load_scenarios()["default"]


@pytest.fixture(scope="module")
def ev_batch():
    ev = ParticleEvaluator(load_physics(), max_candidates=B_DEV,
                           particles_per_candidate=N_DEV, duration_s=DURATION_DEV)
    yield ev
    ev.destroy()


@pytest.fixture(scope="module")
def ev_single():
    """B=1로 따로 할당한 평가기. 필드 크기와 커널 인스턴스가 배치 평가기와 다르다."""
    ev = ParticleEvaluator(load_physics(), max_candidates=1,
                           particles_per_candidate=N_DEV, duration_s=DURATION_DEV)
    yield ev
    ev.destroy()


POSE_A = PoseParams()                                                  # 캡슐 1개
# 기울임 + 팔, 캡슐 2개. 기울기를 크게 하면 몸통이 스치는 제트에서 벗어나 제거가 0이 되어
# (tau_med 0.125 Pa 기준 15도에서 0) 격리 테스트가 공허해진다. 3도에서 A와 제거 개수가 다르다.
POSE_B = PoseParams(shoulder_abduction=90.0, torso_pitch=3.0)


def _assert_same(r1, r2):
    assert r1.score == r2.score
    assert r1.total_removal == r2.total_removal
    np.testing.assert_array_equal(r1.removal_by_part, r2.removal_by_part)
    np.testing.assert_array_equal(r1.extra["count_removed"], r2.extra["count_removed"])
    np.testing.assert_array_equal(r1.extra["count_init"], r2.extra["count_init"])


# ------------------------------------------------------------ 단계 10: 격리 6개
def test_identical_inputs_are_isolated(ev_batch, scenario):
    """동일 입력 격리: 후보 2개에 같은 자세 -> 두 점수 완전 일치."""
    nz = grazing_nozzles()
    r = ev_batch.batch_evaluate_states(
        [cylinder_body(POSE_A), cylinder_body(POSE_A)], [POSE_A, POSE_A], nz, scenario)
    assert r[0].total_removal > 0.0, "제거가 0이면 이 테스트는 공허하다"
    _assert_same(r[0], r[1])


def test_single_vs_batch_match(ev_batch, ev_single, scenario):
    """단독 vs 배치 일치: A, B를 각각 B=1로 평가한 결과 == B=2로 함께 평가한 결과.

    A와 B는 캡슐 수(1 vs 2)와 기하가 다르다. 슬롯 1의 입자가 슬롯 0의 캡슐을 읽거나
    `i // N` 인덱스가 섞이면 여기서 깨진다. 순서를 뒤집은 배치도 확인한다.
    """
    nz = grazing_nozzles()
    body_a, body_b = cylinder_body(POSE_A), cylinder_body(POSE_B)
    assert body_a.capsules.shape[0] != body_b.capsules.shape[0]

    solo_a = ev_single.batch_evaluate_states([body_a], [POSE_A], nz, scenario)[0]
    solo_b = ev_single.batch_evaluate_states([body_b], [POSE_B], nz, scenario)[0]
    batch = ev_batch.batch_evaluate_states([body_a, body_b], [POSE_A, POSE_B], nz, scenario)
    swapped = ev_batch.batch_evaluate_states([body_b, body_a], [POSE_B, POSE_A], nz, scenario)

    # 두 후보가 실제로 다른 결과를 내야 인덱스 혼선을 검출할 수 있다.
    assert solo_a.total_removal > 0.0 and solo_b.total_removal > 0.0
    assert not np.array_equal(solo_a.extra["count_removed"], solo_b.extra["count_removed"])

    _assert_same(solo_a, batch[0])
    _assert_same(solo_b, batch[1])
    _assert_same(solo_b, swapped[0])
    _assert_same(solo_a, swapped[1])

    # B=2 평가기에서 후보 1개만 넣어도(뒤 슬롯 비활성) 같아야 한다.
    _assert_same(solo_b, ev_batch.batch_evaluate_states([body_b], [POSE_B], nz, scenario)[0])


def test_deterministic_same_seed(ev_batch, scenario):
    """결정론: 같은 시드 두 번 -> 일치. 사이에 다른 입력을 끼워 잔여 상태도 확인."""
    nz = grazing_nozzles()
    bodies, poses = [cylinder_body(POSE_A), cylinder_body(POSE_B)], [POSE_A, POSE_B]
    first = ev_batch.batch_evaluate_states(bodies, poses, nz, scenario)
    ev_batch.batch_evaluate_states([cylinder_body(POSE_B)] * 2, [POSE_B] * 2,
                                   grazing_nozzles(0.5), scenario)
    second = ev_batch.batch_evaluate_states(bodies, poses, nz, scenario)
    for a, b in zip(first, second):
        _assert_same(a, b)


def test_zero_strength_removes_nothing(ev_batch, scenario):
    """세기 0: 모든 노즐 strength=0 -> 이탈도 제거도 0."""
    counts = []
    r = ev_batch.batch_evaluate_states(
        [cylinder_body(POSE_A), cylinder_body(POSE_B)], [POSE_A, POSE_B],
        grazing_nozzles(strength=0.0), scenario,
        step_callback=lambda step, ev, n: counts.append(ev.state_counts(n)),
    )
    for res in r:
        assert res.total_removal == 0.0
        np.testing.assert_array_equal(res.removal_by_part, 0.0)
    # 공기가 0이면 아무도 떨어지지 않는다 (중력만으로는 이탈하지 않음)
    assert all(c[0] == B_DEV * N_DEV for c in counts)


def _is_slot_row(nozzle: NozzleConfig) -> np.ndarray:
    """00_common.md 4.1b 행 단위 규약을 이 파일에서 따로 구현 (B의 slot_mask를 쓰지 않는다)."""
    if nozzle.slot_axis is None:
        return np.zeros(nozzle.count, dtype=bool)
    axis = np.asarray(nozzle.slot_axis, np.float64)
    return (np.asarray(nozzle.slot_length, np.float64) > 0) & (np.abs(axis).sum(axis=1) > 0)


def _jet_velocity_numpy(points, nozzle, cfg, t=0.0):
    """00_common.md 4.1(원형) / 4.1b(슬롯)을 numpy float64로 독립 재구현.

    Taichi 코드와 B의 jet.py를 참조하지 않는다. 노즐마다 식을 고르고 벡터 합한다.
    """
    jet = cfg["jet"]
    pulse = jet.get("pulse") or {}
    p = np.asarray(points, np.float64)
    u = np.zeros_like(p)
    slot_rows = _is_slot_row(nozzle)
    phase = (np.zeros(nozzle.count) if nozzle.pulse_phase is None
             else np.asarray(nozzle.pulse_phase, np.float64))
    for m in range(nozzle.count):
        d = np.asarray(nozzle.directions[m], np.float64)
        d = d / np.linalg.norm(d)
        r = p - np.asarray(nozzle.positions[m], np.float64)
        s = r @ d
        u0_strength = float(nozzle.strengths[m])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if slot_rows[m]:
                sl = jet["slot"]
                h, kp, spread = sl["height_m"], sl["decay_constant"], sl["spread_rate"]
                e = np.asarray(nozzle.slot_axis[m], np.float64)
                e = e / np.linalg.norm(e)
                rho_e = r @ e
                rho_n = np.linalg.norm(r - s[:, None] * d - rho_e[:, None] * e, axis=1)
                u0 = sl["exit_velocity_mps"] * u0_strength
                l_core = kp * h
                u_c = np.where(s <= l_core, u0, u0 * np.sqrt(kp * h / s))
                sigma = 0.5 * h / 1.177 + spread * s / 1.177
                rho_e_out = np.maximum(np.abs(rho_e) - float(nozzle.slot_length[m]) / 2, 0.0)
                mag = (u_c * np.exp(-rho_n ** 2 / (2 * sigma ** 2))
                       * np.exp(-rho_e_out ** 2 / (2 * sigma ** 2)))
            else:
                big_d, k = jet["nozzle_diameter_m"], jet["decay_constant"]
                rho = np.linalg.norm(r - s[:, None] * d, axis=1)
                u0 = jet["exit_velocity_mps"] * u0_strength
                l_core = k * big_d
                u_c = np.where(s <= l_core, u0, u0 * k * big_d / s)
                sigma = 0.5 * big_d / 1.177 + jet["halfwidth_spread_rate"] * s / 1.177
                mag = u_c * np.exp(-rho ** 2 / (2 * sigma ** 2))
        mag = np.where(s > 0, mag, 0.0)
        if pulse.get("enabled", False):
            gate = float(np.mod(t / pulse["period_s"] + phase[m], 1.0) < pulse["duty"])
            mag = mag * gate
        u += mag[:, None] * d
    return u


def _jet_points(nozzle, cfg, rng, n=1000):
    """무작위 점 1000개: 절반은 부스 안 균일, 절반은 제트 근처 (값이 큰 곳).

    제트 근처 점은 그 거리의 제트 폭 sigma(s)에 비례해 흩뿌려 코어/감쇠 구간,
    가우시안 중심/꼬리, 노즐 뒤(s <= 0)를 고루 지나게 한다. 슬롯 노즐은 슬롯 길이
    방향으로 끝 바깥까지 (+-(L/2 + 0.15 m)) 퍼뜨려 끝단 감쇠 구간도 지나게 한다.
    """
    jet = cfg["jet"]
    half = n // 2
    box = rng.uniform([0.0, -BOOTH["width_m"] / 2, 0.0],
                      [BOOTH["length_m"], BOOTH["width_m"] / 2, BOOTH["height_m"]], (half, 3))
    slot_rows = _is_slot_row(nozzle)
    m = rng.integers(0, nozzle.count, n - half)
    s = rng.uniform(-0.05, 1.0, n - half)            # 음수 = 노즐 뒤 (u = 0 분기)
    near = np.empty((n - half, 3))
    for i, (mi, si) in enumerate(zip(m, s)):
        d = nozzle.directions[mi] / np.linalg.norm(nozzle.directions[mi])
        base = nozzle.positions[mi] + si * d
        if slot_rows[mi]:
            sl = jet["slot"]
            sigma = (0.5 * sl["height_m"] + sl["spread_rate"] * abs(si)) / 1.177
            e = nozzle.slot_axis[mi] / np.linalg.norm(nozzle.slot_axis[mi])
            reach = nozzle.slot_length[mi] / 2 + 0.15
            near[i] = base + rng.uniform(-reach, reach) * e + rng.normal(0, 1.5 * sigma, 3)
        else:
            sigma = (0.5 * jet["nozzle_diameter_m"]
                     + jet["halfwidth_spread_rate"] * abs(si)) / 1.177
            near[i] = base + rng.normal(0, 1.5 * sigma, 3)
    return np.concatenate([box, near]).astype(np.float32)


def _relative_error(u_ti, u_np):
    """점별 |Δu| / |u|. |u|가 사실상 0인 점은 필드 최대값 1e-6배를 바닥으로 둔다
    (f32 exp 꼬리의 언더플로가 상대 오차를 부풀리지 않게)."""
    mag = np.linalg.norm(u_np, axis=1)
    floor = 1e-6 * max(mag.max(), 1e-12)
    return np.linalg.norm(u_ti - u_np, axis=1) / np.maximum(mag, floor)


@pytest.mark.parametrize("kind", ["round", "slot", "mixed"])
@pytest.mark.parametrize("pulse_t", [None, 0.37], ids=["steady", "pulse"])
def test_jet_matches_numpy_reference(kind, pulse_t):
    """제트 일치: Taichi ti.func vs numpy 재구현, 무작위 점 1000개, 상대 오차 1e-4 이내.

    D의 패치판과 점수를 맞추는 핵심. physics.yaml 원본 상수를 쓴다. 원형(4.1),
    슬롯(4.1b), 둘이 섞인 배치(행 단위 규약)를 각각 돌린다.
    """
    cfg = copy.deepcopy(load_physics())
    t = 0.0
    if pulse_t is not None:
        cfg["jet"]["pulse"]["enabled"] = True
        t = pulse_t
    rng = np.random.default_rng(1234)
    nozzle = layout_nozzles(rng, kind)
    if kind != "round":
        assert _is_slot_row(nozzle).sum() == slot_nozzles().count
    points = _jet_points(nozzle, cfg, rng)

    ev = ParticleEvaluator(cfg, max_candidates=1, particles_per_candidate=1, duration_s=0.0)
    try:
        u_ti = ev.probe_velocity(points, nozzle, t).astype(np.float64)
    finally:
        ev.destroy()
    u_np = _jet_velocity_numpy(points, nozzle, cfg, t)

    speed = np.linalg.norm(u_np, axis=1)
    assert (speed > 0.1).sum() >= 150, f"의미 있는 속도의 점 {(speed > 0.1).sum()}개: 공허한 비교"
    rel = _relative_error(u_ti, u_np)
    assert rel.max() < 1e-4, f"최대 상대 오차 {rel.max():.3e} (점 {int(rel.argmax())})"

    if pulse_t is not None:
        _assert_pulse_gate(points, nozzle, cfg, t, u_ti)


def _subset(nozzle: NozzleConfig, rows: np.ndarray) -> NozzleConfig:
    return NozzleConfig(
        nozzle.positions[rows], nozzle.directions[rows], nozzle.strengths[rows],
        pulse_phase=nozzle.pulse_phase[rows],
        slot_axis=None if nozzle.slot_axis is None else nozzle.slot_axis[rows],
        slot_length=None if nozzle.slot_length is None else nozzle.slot_length[rows])


def _assert_pulse_gate(points, nozzle, cfg, t, u_ti_all):
    """게이트 확인: 꺼진 노즐의 기여는 정확히 0이고, 전체 결과는 켜진 노즐만의 합과 1e-4.

    "속도가 정확히 0인 점이 생긴다"로 확인하면 다른 노즐이 부스를 덮는 배치에서 공허하게
    깨진다. 게이트를 노즐 단위로 직접 본다.
    """
    pulse = cfg["jet"]["pulse"]
    on = np.mod(t / pulse["period_s"] + nozzle.pulse_phase.astype(np.float64), 1.0) < pulse["duty"]
    assert on.any() and (~on).any(), "켜진 노즐과 꺼진 노즐이 둘 다 있어야 게이트를 검증한다"

    ev = ParticleEvaluator(cfg, max_candidates=1, particles_per_candidate=1, duration_s=0.0)
    try:
        u_off = ev.probe_velocity(points, _subset(nozzle, ~on), t)
    finally:
        ev.destroy()
    np.testing.assert_array_equal(u_off, 0.0)

    steady = copy.deepcopy(cfg)
    steady["jet"]["pulse"]["enabled"] = False
    u_on_ref = _jet_velocity_numpy(points, _subset(nozzle, on), steady)
    off_ref = _jet_velocity_numpy(points, _subset(nozzle, ~on), steady)
    assert (np.linalg.norm(off_ref, axis=1) > 0.1).sum() > 0, "꺼진 노즐이 켜져 있었다면 기여가 있었어야 한다"
    rel = _relative_error(u_ti_all, u_on_ref)
    assert rel.max() < 1e-4, f"켜진 노즐 합 대비 최대 상대 오차 {rel.max():.3e}"


@pytest.mark.parametrize("kind", ["round", "slot", "mixed"])
def test_jet_matches_b_velocity_field(kind):
    """제트 일치 (B 기준값): Taichi vs `airis.sim.jet.velocity_field`, 몸 주변 점 1000개, 1e-4.

    점은 부스 중앙 몸 주변이며 모든 노즐 중심에서 0.1 m 이상 떨어져 있다. B의 jet.py가
    float64로 바뀌어 (B PR #21) 이전의 1e-3 완화를 없앴다. float64 독립 구현과도 1e-4.
    """
    cfg = load_physics()
    rng = np.random.default_rng(99)
    nozzle = layout_nozzles(rng, kind)
    cand = rng.uniform([0.1, -0.5, 0.2], [BOOTH["length_m"] - 0.1, 0.5, 1.95], (20000, 3))
    d_min = np.linalg.norm(cand[:, None] - nozzle.positions[None], axis=2).min(axis=1)
    points = cand[d_min >= 0.1][:1000].astype(np.float32)
    assert len(points) == 1000

    ev = ParticleEvaluator(cfg, max_candidates=1, particles_per_candidate=1, duration_s=0.0)
    try:
        u_ti = ev.probe_velocity(points, nozzle).astype(np.float64)
    finally:
        ev.destroy()
    u_b = velocity_field(points, nozzle, 0.0, cfg).astype(np.float64)
    u_exact = _jet_velocity_numpy(points, nozzle, cfg)

    assert (np.linalg.norm(u_exact, axis=1) > 0.1).sum() >= 150
    assert _relative_error(u_ti, u_exact).max() < 1e-4
    rel_b = _relative_error(u_ti, u_b)
    assert rel_b.max() < 1e-4, f"Taichi vs B 최대 상대 오차 {rel_b.max():.3e}"


def test_slot_nozzle_validation():
    """슬롯 노즐 입력 검사: 축이 분사 방향에 수직이 아니면, 또는 jet.slot 상수가 없으면 막는다.
    slot_axis 영벡터 / slot_length 0인 행은 원형(4.1)으로 계산한다 (행 단위 규약)."""
    cfg = load_physics()
    ev = ParticleEvaluator(cfg, max_candidates=1, particles_per_candidate=1, duration_s=0.0)
    try:
        slt = slot_nozzles()
        bent = NozzleConfig(slt.positions, slt.directions, slt.strengths,
                            slot_axis=slt.slot_axis + 0.2 * slt.directions,
                            slot_length=slt.slot_length)
        with pytest.raises(ValueError, match="수직"):
            ev.probe_velocity(np.zeros((1, 3), np.float32), bent)

        rnd = round_nozzles()
        as_round = NozzleConfig(rnd.positions, rnd.directions, rnd.strengths,
                                slot_axis=np.zeros((rnd.count, 3), np.float32),
                                slot_length=np.ones(rnd.count, np.float32))
        pts = (rnd.positions + rnd.directions * 0.3).astype(np.float32)
        np.testing.assert_array_equal(ev.probe_velocity(pts, as_round),
                                      ev.probe_velocity(pts, rnd))
    finally:
        ev.destroy()

    no_slot = copy.deepcopy(cfg)
    del no_slot["jet"]["slot"]
    ev = ParticleEvaluator(no_slot, max_candidates=1, particles_per_candidate=1, duration_s=0.0)
    try:
        with pytest.raises(ValueError, match="jet.slot"):
            ev.probe_velocity(np.zeros((1, 3), np.float32), slot_nozzles())
    finally:
        ev.destroy()


def test_torso_redeposition_splits_front_back():
    """재부착 부위: 몸통 캡슐(capsule_part = torso_front)에 부딪히면 충돌 법선과 몸 전방
    (cos yaw, sin yaw, 0)의 부호로 torso_front / torso_back을 가른다."""
    cfg = copy.deepcopy(load_physics())
    cfg["adhesion"]["redeposition_prob"] = 1.0               # 충돌하면 반드시 재부착
    ev = ParticleEvaluator(cfg, max_candidates=2, particles_per_candidate=4, duration_s=0.0)
    scenario = load_scenarios()["default"]
    poses = [PoseParams(torso_yaw=0.0), PoseParams(torso_yaw=180.0)]
    try:
        ev.batch_evaluate_states([cylinder_body(p) for p in poses], poses,
                                 grazing_nozzles(0.0), scenario)
        # 각 후보의 입자 4개를 몸통 안쪽 +x, -x, 앞쪽 비스듬히, 뒤쪽 비스듬히 두고 부유 상태로.
        # 법선이 정확히 좌우(·전방 = 0)인 점은 yaw 180에서 sin(pi) 반올림으로 부호가 흔들려 피한다.
        offsets = np.array([[0.1, 0, 0], [-0.1, 0, 0], [0.08, -0.05, 0], [-0.08, 0.05, 0]],
                           np.float32)
        pos = np.tile(np.array([CENTER_X, 0.0, 1.0], np.float32) + offsets, (2, 1))
        ev.f.pos.from_numpy(pos)
        ev.f.state.from_numpy(np.ones(8, np.int32))
        ev.f.rand_redep.from_numpy(np.zeros(8, np.float32))
        ev.f.k_collide(2)
        state, part = ev.f.state.to_numpy(), ev.f.part.to_numpy()
    finally:
        ev.destroy()
    front, back = TORSO, PART_NAMES.index("torso_back")
    assert np.all(state == 0)
    # yaw 0: 전방 +x
    np.testing.assert_array_equal(part[:4], [front, back, front, back])
    # yaw 180: 전방 -x -> 앞뒤가 뒤집힌다
    np.testing.assert_array_equal(part[4:], [back, front, back, front])


def test_mass_conservation_every_step(ev_batch, scenario):
    """질량 보존: state 0/1/2 개수 합 == B·N, 매 스텝. 제거는 되돌아가지 않는다."""
    history = []
    ev_batch.batch_evaluate_states(
        [cylinder_body(POSE_A), cylinder_body(POSE_B)], [POSE_A, POSE_B],
        grazing_nozzles(), scenario,
        step_callback=lambda step, ev, n: history.append(ev.state_counts(n)),
    )
    history = np.array(history)
    assert len(history) == round(DURATION_DEV / ev_batch.dt)
    np.testing.assert_array_equal(history.sum(axis=1), B_DEV * N_DEV)
    assert np.all(np.diff(history[:, 2]) >= 0)
    assert history[-1, 2] > 0


# ------------------------------------------------------ 시각 확인 (완료 기준 2)
def test_particles_detach_fly_and_leave_booth(scenario):
    """원통 1개 + 노즐 1개: 입자가 이탈해 날아가 부스 밖으로 나간다. 산점도를 저장한다.

    그림: outputs/debug/particles_escape.png (위에서 본 x-y, 정면 y-z).
    """
    ev = ParticleEvaluator(load_physics(), max_candidates=1,
                           particles_per_candidate=N_DEV, duration_s=DURATION_DEV)
    x_graze = CENTER_X - 0.15
    nozzle = NozzleConfig(np.array([[x_graze, -BOOTH["width_m"] / 2, 1.05]], np.float32),
                          np.array([[0.0, 1.0, 0.0]], np.float32),
                          np.array([1.0], np.float32))
    body = cylinder_body(POSE_A)
    assert body.capsules.shape[0] == 1

    frames = []
    try:
        r = ev.batch_evaluate_states(
            [body], [POSE_A], nozzle, scenario,
            step_callback=lambda step, e, n: frames.append((step, e.snapshot(0)))
            if step % 5 == 0 else None,
        )[0]
        final = ev.snapshot(0)
    finally:
        ev.destroy()

    st, pos = final["state"], final["pos"]
    removed = st == 2
    assert removed.sum() > 0, "부스 밖으로 나간 입자가 없다"
    assert r.total_removal == removed.sum() / N_DEV
    half_w = BOOTH["width_m"] / 2
    outside = ((pos[:, 0] < 0) | (pos[:, 0] > BOOTH["length_m"]) | (np.abs(pos[:, 1]) > half_w)
               | (pos[:, 2] < 0) | (pos[:, 2] > BOOTH["height_m"]))
    np.testing.assert_array_equal(outside, removed)
    # 제트가 +y로 불므로 나간 입자는 반대편 벽(+y)으로 나가야 한다
    assert np.all(pos[removed, 1] > half_w)

    _save_escape_plot(frames, final, body, nozzle, r)


def _save_escape_plot(frames, final, body, nozzle, result):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    installed = {f.name for f in mpl.font_manager.fontManager.ttflist}
    for font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):     # 한글 라벨
        if font in installed:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

    out = ROOT / "outputs" / "debug"
    out.mkdir(parents=True, exist_ok=True)
    moved = final["state"] != 0
    ever = np.zeros_like(moved)
    for _, snap in frames:
        ever |= snap["state"] != 0

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    views = [(0, 1, "x (m, 진행 방향)", "y (m)", "위에서 본 모습 (x-y)"),
             (1, 2, "y (m)", "z (m)", "정면 (y-z)")]
    cmap = plt.get_cmap("viridis")
    n_frames = len(frames)
    for ax, (i, j, xl, yl, title) in zip(axes, views):
        attached = final["state"] == 0
        ax.scatter(final["pos"][attached, i], final["pos"][attached, j],
                   s=3, c="#b0b0b0", label="부착 (state 0)")
        for k, (step, snap) in enumerate(frames):
            sel = ever & (snap["state"] != 0)
            ax.scatter(snap["pos"][sel, i], snap["pos"][sel, j], s=4,
                       color=cmap(k / max(n_frames - 1, 1)),
                       label="이탈 입자 궤적 (색 = 시간)" if k == n_frames // 2 else None)
        rem = final["state"] == 2
        ax.scatter(final["pos"][rem, i], final["pos"][rem, j], s=18, marker="x",
                   c="#d62728", label="제거 (부스 밖, state 2)")
        lims = {0: (0, BOOTH["length_m"]), 1: (-BOOTH["width_m"] / 2, BOOTH["width_m"] / 2),
                2: (0, BOOTH["height_m"])}
        (x0, x1), (y0, y1) = lims[i], lims[j]
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k--", lw=1, label="부스 경계")
        npos = nozzle.positions[0]
        ax.annotate("", xy=(npos[i] + 0.25 * nozzle.directions[0][i],
                            npos[j] + 0.25 * nozzle.directions[0][j]),
                    xytext=(npos[i], npos[j]),
                    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2))
        ax.plot(npos[i], npos[j], "s", c="#1f77b4", ms=8, label="노즐")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.margins(0.08)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"원통 1개 + 노즐 1개, N={len(moved)}, {DURATION_DEV} s  |  "
                 f"제거 {result.total_removal:.1%}  |  "
                 f"이탈 경험 {ever.mean():.1%}")
    fig.savefig(out / "particles_escape.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------- 실제 입력 (B build_body, D fakes)
def test_batch_evaluate_with_build_body_matches_evaluate(ev_batch, scenario):
    """C가 부르는 그대로: batch_evaluate([(PoseParams, NozzleConfig)], body, scenario)
    -> (B,) float32, 입력 순서 유지, 각 원소 == evaluate(...).score. 기준 장비 슬롯 배치."""
    body = BodyParams()
    nozzle = load_nozzles()
    assert slot_mask(nozzle).all()
    poses = [PoseParams(shoulder_abduction=120.0, torso_yaw=30.0), PoseParams()]
    scores = ev_batch.batch_evaluate([(p, nozzle) for p in poses], body, scenario)
    assert scores.dtype == np.float32 and scores.shape == (2,)
    singles = [ev_batch.evaluate(p, nozzle, body, scenario) for p in poses]
    np.testing.assert_array_equal(scores, np.array([r.score for r in singles], np.float32))
    reversed_scores = ev_batch.batch_evaluate([(p, nozzle) for p in poses[::-1]], body, scenario)
    np.testing.assert_array_equal(reversed_scores, scores[::-1])
    for r in singles:
        assert r.extra["infeasible"] is False
        assert r.total_removal > 0.0
        assert r.extra["count_init"].sum() == N_DEV


def test_fake_body_batch_isolation(ev_batch, scenario):
    """D의 fake_body(캡슐 4개) + fake_nozzles로 배치가 돌고 격리가 유지되는지."""
    fakes = pytest.importorskip("tests.fakes", reason="tests/fakes.py 미병합 (D)")
    down, up = fakes.fake_body(arms_up=False), fakes.fake_body(arms_up=True)
    pose_down, pose_up = PoseParams(), PoseParams(shoulder_abduction=120.0)
    nz = fakes.fake_nozzles()
    batch = ev_batch.batch_evaluate_states([down, up], [pose_down, pose_up], nz, scenario)
    solo = ev_batch.batch_evaluate_states([up], [pose_up], nz, scenario)[0]
    _assert_same(solo, batch[1])
    for res in batch:
        assert 0.0 <= res.total_removal <= 1.0


# ------------------------------------------------------ 부스 밖 자세 불가 (00_common 5절)
def _moved_patch(state: BodyState, index: int, xyz) -> BodyState:
    """패치 하나를 옮긴 사본. 나머지는 그대로."""
    pos = state.patch_pos.copy()
    pos[index] = xyz
    return BodyState(pos, state.patch_normal, state.patch_area, state.patch_part,
                     state.capsules, state.capsule_part, state.patch_capsule)


def test_outside_booth_is_infeasible_per_slot(ev_batch, ev_single, scenario):
    """패치가 하나라도 옆벽(|y| > W/2)이나 천장(z > H) 밖이면 그 후보만 불가.

    - 불가 후보: score = -1 - 10·d_out, removal 0, total 0, extra["infeasible"] = True,
      시뮬레이션 안 함 (d_out = 벽/천장 초과 거리 m)
    - 가능 후보: 불가 후보가 섞여도 단독 평가와 결과가 같다 (슬롯이 밀려도 격리 유지)
    - x 방향은 열린 문이라 패치가 x < 0이어도 가능
    """
    nz = grazing_nozzles()
    half_w, height = BOOTH["width_m"] / 2, BOOTH["height_m"]
    body_a, body_b = cylinder_body(POSE_A), cylinder_body(POSE_B)
    out_y = _moved_patch(body_a, 0, [CENTER_X, -(half_w + 0.02), 1.0])      # d_out = 0.02
    out_z = _moved_patch(body_a, 0, [CENTER_X, 0.0, height + 0.05])         # d_out = 0.05
    out_x = _moved_patch(body_a, 0, [-0.05, 0.0, 1.0])
    pose_y, pose_z = PoseParams(shoulder_abduction=30.0), PoseParams(torso_pitch=10.0)

    solo_a = ev_single.batch_evaluate_states([body_a], [POSE_A], nz, scenario)[0]
    solo_b = ev_single.batch_evaluate_states([body_b], [POSE_B], nz, scenario)[0]

    calls = []
    mixed = ev_batch.batch_evaluate_states(
        [out_y, body_a, out_z, body_b], [pose_y, POSE_A, pose_z, POSE_B], nz, scenario,
        step_callback=lambda step, ev, n: calls.append(n))
    assert set(calls) == {2}, "가능 후보 2개만 시뮬레이션해야 한다"
    _assert_same(solo_a, mixed[1])
    _assert_same(solo_b, mixed[3])
    for res, pose, d_out in ((mixed[0], pose_y, 0.02), (mixed[2], pose_z, 0.05)):
        assert res.score == pytest.approx(-1.0 - 10.0 * d_out, abs=1e-5)
        assert res.extra["d_out_m"] == pytest.approx(d_out, abs=1e-6)
        assert res.total_removal == 0.0
        np.testing.assert_array_equal(res.removal_by_part, 0.0)
        assert res.extra["infeasible"] is True
        assert res.discomfort == scoring.discomfort(pose, scenario)
    assert mixed[1].extra["infeasible"] is False and solo_a.total_removal > 0.0

    x_ok = ev_batch.batch_evaluate_states([out_x], [POSE_A], nz, scenario)[0]
    assert x_ok.extra["infeasible"] is False

    calls.clear()
    all_out = ev_batch.batch_evaluate_states([out_y, out_z], [pose_y, pose_z], nz, scenario,
                                             step_callback=lambda step, ev, n: calls.append(n))
    assert calls == [] and all(r.score <= -1.0 for r in all_out)

    # 벽 바로 안쪽은 가능. (patch_pos가 float32라 벽과 "같은" 값은 반올림으로 밖이 될 수 있어
    # 0.1 mm 안쪽으로 둔다. D의 outside_booth도 같은 float32 패치를 받는다.)
    near_wall = _moved_patch(body_a, 0, [CENTER_X, half_w - 1e-4, height - 1e-4])
    assert ev_batch.batch_evaluate_states([near_wall], [POSE_A], nz, scenario)[0].extra[
        "infeasible"] is False


def test_outside_booth_matches_patch_evaluator(ev_batch, scenario):
    """실제 몸: 불가 여부와 등급제 벌점 값이 D의 패치판과 같다 (00_common.md 5절).

    어깨 벌림 90/120도는 팔 끝이 옆벽을 넘고, 180도(만세)는 체형에 따라 천장에 닿는다.
    """
    from airis.sim.patch_baseline import PatchEvaluator

    body, nozzle = BodyParams(), load_nozzles()
    poses = [PoseParams(shoulder_abduction=90.0), PoseParams(shoulder_abduction=120.0),
             PoseParams(shoulder_abduction=180.0, elbow_flexion=0.0), PoseParams()]
    scores = ev_batch.batch_evaluate([(p, nozzle) for p in poses], body, scenario)
    assert scores[0] < -1.0 and scores[1] < -1.0 and scores[3] > -1.0

    patch = PatchEvaluator(load_physics())
    for pose, score in zip(poses, scores):
        ref = patch.evaluate(pose, nozzle, body, scenario)
        assert ref.extra["infeasible"] == (score <= -1.0)
        if ref.extra["infeasible"]:
            # batch_evaluate는 float32로 돌려준다
            assert score == pytest.approx(ref.score, abs=1e-6)
            single = ev_batch.evaluate(pose, nozzle, body, scenario)
            assert single.score == pytest.approx(ref.score, abs=1e-12)
            assert single.discomfort == pytest.approx(ref.discomfort, abs=1e-12)


# ------------------------------------------------ 4.2b 충돌 제트 -> 벽면 제트 보정
def _impingement_numpy(points, normals, nozzle, cfg, t=0.0):
    """00_common.md 4.2b를 numpy float64로 독립 재구현 (B의 jet.py, Taichi 코드 참조 없음).

    조회점 x, 바깥 법선 n마다 노즐별 w·e_r을 합한 (P,3) 보정을 돌려준다.
    """
    jet = cfg["jet"]
    k = jet["impingement"]["wall_jet_gain"]
    pulse = jet.get("pulse") or {}
    slot_rows = _is_slot_row(nozzle)
    phase = (np.zeros(nozzle.count) if nozzle.pulse_phase is None
             else np.asarray(nozzle.pulse_phase, np.float64))
    x_all = np.asarray(points, np.float64)
    n_all = np.asarray(normals, np.float64)
    n_all = n_all / np.linalg.norm(n_all, axis=1, keepdims=True)
    out = np.zeros_like(x_all)
    for m in range(nozzle.count):
        n_m = np.asarray(nozzle.positions[m], np.float64)
        d = np.asarray(nozzle.directions[m], np.float64)
        d = d / np.linalg.norm(d)
        gate = 1.0
        if pulse.get("enabled", False):
            gate = float(np.mod(t / pulse["period_s"] + phase[m], 1.0) < pulse["duty"])
        strength = float(nozzle.strengths[m])
        for i in range(len(x_all)):
            x, n = x_all[i], n_all[i]
            cos_t = max(-(d @ n), 0.0)
            if cos_t == 0.0:
                continue
            h_ax = ((x - n_m) @ n) / (d @ n)
            if h_ax <= 0.0:
                continue
            c = n_m + h_ax * d
            r = (x - c) - ((x - c) @ n) * n
            end = 1.0
            if slot_rows[m]:
                sl = jet["slot"]
                h, kp = sl["height_m"], sl["decay_constant"]
                sigma = 0.5 * h / 1.177 + sl["spread_rate"] * h_ax / 1.177
                u_h = sl["exit_velocity_mps"] * strength * (
                    1.0 if h_ax <= kp * h else np.sqrt(kp * h / h_ax))
                e = np.asarray(nozzle.slot_axis[m], np.float64)
                e = e / np.linalg.norm(e)
                e_t = e - (e @ n) * n
                rho_e = 0.0
                if np.linalg.norm(e_t) > 0.0:
                    e_t = e_t / np.linalg.norm(e_t)
                    rho_e = r @ e_t
                    r = r - rho_e * e_t
                end = np.exp(-max(abs(rho_e) - float(nozzle.slot_length[m]) / 2, 0.0) ** 2
                             / (2 * sigma ** 2))
            else:
                big_d, kd = jet["nozzle_diameter_m"], jet["decay_constant"]
                sigma = 0.5 * big_d / 1.177 + jet["halfwidth_spread_rate"] * h_ax / 1.177
                u_h = jet["exit_velocity_mps"] * strength * (
                    1.0 if h_ax <= kd * big_d else kd * big_d / h_ax)
            rho = np.linalg.norm(r)
            xi = rho / sigma
            if xi < 1e-6:
                continue
            core = 1.0 - np.exp(-xi ** 2 / 2)
            shape = core / np.sqrt(xi) if slot_rows[m] else core / xi
            out[i] += k * cos_t * u_h * shape * end * gate * r / rho
    return out


def _mannequin_probe(pose: PoseParams, scenario, cfg):
    """실제 마네킹 패치의 조회점 (x = 패치 + δ·n)과 법선."""
    st = build_body(BodyParams(), pose, scenario)
    n = np.asarray(st.patch_normal, np.float64)
    x = st.patch_pos + cfg["air"]["wall_offset_m"] * n
    return x.astype(np.float32), n.astype(np.float32)


@pytest.mark.parametrize("kind", ["round", "slot", "mixed"])
@pytest.mark.parametrize("pulse_t", [None, 0.37], ids=["steady", "pulse"])
def test_impingement_matches_references(kind, pulse_t, scenario):
    """4.2b: 부착 입자의 이탈 판정 속도 (자유 제트 + 보정)가 float64 독립 구현 및 B와 1e-4.

    마네킹 두 자세(정면, 옆으로 90도)의 패치 전체에서 본다. 무작위 세기와 펄스 위상.
    """
    cfg = copy.deepcopy(load_physics())
    assert cfg["jet"]["impingement"]["enabled"]
    t = 0.0
    if pulse_t is not None:
        cfg["jet"]["pulse"]["enabled"] = True
        t = pulse_t
    nozzle = layout_nozzles(np.random.default_rng(7), kind)
    ev = ParticleEvaluator(cfg, max_candidates=1, particles_per_candidate=1, duration_s=0.0)
    try:
        for yaw in (0.0, 90.0):
            x, n = _mannequin_probe(PoseParams(torso_yaw=yaw), scenario, cfg)
            u_ti = ev.probe_surface_velocity(x, n, nozzle, t).astype(np.float64)
            corr = _impingement_numpy(x, n, nozzle, cfg, t)
            u_ref = _jet_velocity_numpy(x, nozzle, cfg, t) + corr
            u_b = velocity_field(x, nozzle, t, cfg, surface_normals=n).astype(np.float64)

            assert np.linalg.norm(corr, axis=1).max() > 0.5, "보정이 거의 0이면 공허한 비교"
            rel = _relative_error(u_ti, u_ref)
            assert rel.max() < 1e-4, f"yaw {yaw}: float64 독립 구현 대비 {rel.max():.3e}"
            rel_b = _relative_error(u_ti, u_b)
            assert rel_b.max() < 1e-4, f"yaw {yaw}: B 대비 {rel_b.max():.3e}"
    finally:
        ev.destroy()


def test_impingement_only_for_attached_and_switchable(scenario):
    """보정은 이탈 판정(부착 입자)에만: 자유 제트 조회(`probe_velocity`, 부유 입자가 쓰는 값)는
    보정과 무관하고, `enabled = false`면 이탈 판정 속도도 자유 제트와 같다."""
    cfg = load_physics()
    off = copy.deepcopy(cfg)
    off["jet"]["impingement"]["enabled"] = False
    nozzle = slot_nozzles()
    x, n = _mannequin_probe(PoseParams(), scenario, cfg)
    ev_on = ParticleEvaluator(cfg, max_candidates=1, particles_per_candidate=1, duration_s=0.0)
    ev_off = ParticleEvaluator(off, max_candidates=1, particles_per_candidate=1, duration_s=0.0)
    try:
        free_on = ev_on.probe_velocity(x, nozzle)
        np.testing.assert_array_equal(free_on, ev_off.probe_velocity(x, nozzle))
        # 커널이 달라 f32 연산 순서(fast math)가 달라지므로 비트 일치 대신 제트 비교 기준 1e-4
        # (측정 최대 3e-5)
        rel = _relative_error(ev_off.probe_surface_velocity(x, n, nozzle).astype(np.float64),
                              free_on.astype(np.float64))
        assert rel.max() < 1e-4
        assert np.abs(ev_on.probe_surface_velocity(x, n, nozzle) - free_on).max() > 0.5
    finally:
        ev_on.destroy()
        ev_off.destroy()


def test_head_on_jet_detaches_only_with_impingement(scenario):
    """정면으로 제트를 받는 원통: 보정이 없으면 접선 성분이 거의 0이라 이탈이 적고,
    4.2b를 켜면 정체점 둘레의 벽면 제트가 입자를 더 많이 뗀다 (4.2b의 동기).

    이탈한 입자는 자유 제트(몸 쪽을 향함)에 밀려 부스 밖까지 못 나가므로 제거율이 아니라
    한 번이라도 이탈한 입자 수(state != 0)로 본다.
    """
    cfg = load_physics()
    off = copy.deepcopy(cfg)
    off["jet"]["impingement"]["enabled"] = False
    head_on = NozzleConfig(np.array([[CENTER_X, -BOOTH["width_m"] / 2, 1.05]], np.float32),
                           np.array([[0.0, 1.0, 0.0]], np.float32),
                           np.array([1.0], np.float32))
    body = cylinder_body(POSE_A)
    detached = {}
    for name, c in (("on", cfg), ("off", off)):
        ev = ParticleEvaluator(c, max_candidates=1, particles_per_candidate=N_DEV,
                               duration_s=DURATION_DEV)
        try:
            ev.batch_evaluate_states([body], [POSE_A], head_on, scenario)
            detached[name] = int((ev.snapshot(0)["state"] != 0).sum())
        finally:
            ev.destroy()
    assert detached["on"] > detached["off"], detached
    assert detached["on"] >= 5, detached


# ------------------------------------------------------------ 단계 12. 프레임 덤프
def test_frame_dump_format_and_candidate_mapping(tmp_path, scenario):
    """`dump_every` 스텝마다 frames/<step>.npz. interfaces.md 필드(pos, attached, part, candidate)와
    내용이 그 스텝의 실제 입자 상태와 같고, candidate는 입력 순서 번호다 (불가 후보는 빠짐)."""
    every, duration = 5, 0.1
    ev = ParticleEvaluator(load_physics(), max_candidates=B_DEV, particles_per_candidate=N_DEV,
                           duration_s=duration, dump_every=every, dump_dir=tmp_path)
    body_a = cylinder_body(POSE_A)
    out_y = _moved_patch(body_a, 0, [CENTER_X, BOOTH["width_m"] / 2 + 0.02, 1.0])
    seen = {}

    def grab(step, e, n):
        if step % every == 0:
            seen[step] = {"pos": e.f.pos.to_numpy()[:n * N_DEV],
                          "state": e.f.state.to_numpy()[:n * N_DEV]}
    try:
        res = ev.batch_evaluate_states(
            [out_y, body_a, cylinder_body(POSE_B)], [PoseParams(shoulder_abduction=30.0),
                                                     POSE_A, POSE_B],
            grazing_nozzles(), scenario, step_callback=grab)
    finally:
        ev.destroy()
    assert res[0].extra["infeasible"] and not res[1].extra["infeasible"]

    n_steps = round(duration / ev.dt)
    files = sorted((tmp_path / "frames").glob("*.npz"))
    assert [f.stem for f in files] == [f"{k:05d}" for k in range(0, n_steps, every)]
    for f in files:
        with np.load(f) as fr:
            step = int(fr["step"])
            assert fr["pos"].shape == (2 * N_DEV, 3) and fr["pos"].dtype == np.float32
            np.testing.assert_array_equal(fr["pos"], seen[step]["pos"])
            np.testing.assert_array_equal(fr["state"], seen[step]["state"])
            np.testing.assert_array_equal(fr["attached"], seen[step]["state"] == 0)
            np.testing.assert_array_equal(fr["candidate"], np.repeat([1, 2], N_DEV))
            assert fr["part"].shape == fr["part_init"].shape == (2 * N_DEV,)
            assert float(fr["t_s"]) == pytest.approx(step * ev.dt)
    with np.load(files[-1]) as last:
        assert (~last["attached"]).any(), "마지막 프레임까지 이탈이 하나도 없으면 공허한 덤프"


def test_frame_dump_off_and_missing_dir(tmp_path, scenario):
    """dump_every = 0이면 아무것도 쓰지 않고, > 0인데 디렉터리가 없으면 막는다."""
    body = cylinder_body(POSE_A)
    ev = ParticleEvaluator(load_physics(), max_candidates=1, particles_per_candidate=10,
                           duration_s=0.02, dump_every=0, dump_dir=tmp_path)
    try:
        ev.batch_evaluate_states([body], [POSE_A], grazing_nozzles(), scenario)
    finally:
        ev.destroy()
    assert not (tmp_path / "frames").exists()

    ev = ParticleEvaluator(load_physics(), max_candidates=1, particles_per_candidate=10,
                           duration_s=0.02, dump_every=2)
    try:
        with pytest.raises(ValueError, match="dump_dir"):
            ev.batch_evaluate_states([body], [POSE_A], grazing_nozzles(), scenario)
        ev.batch_evaluate_states([body], [POSE_A], grazing_nozzles(), scenario,
                                 dump_dir=tmp_path / "call")
    finally:
        ev.destroy()
    assert len(list((tmp_path / "call" / "frames").glob("*.npz"))) == 5


# ------------------------------------------------------------- 단계 11. 성능 경로
@pytest.mark.parametrize("variant", ["default", "redeposit", "pulse"])
def test_fused_run_matches_stepwise(variant, scenario):
    """여러 스텝을 한 커널에서 도는 빠른 경로(k_run, 조기 종료 포함)가 매 스텝 경로(k_step)와
    입자 하나까지 같은 결과를 낸다. steps_per_launch를 바꿔도 같다.

    - redeposit: 재부착 확률 0.5로 "한 스텝 안에서 이탈 후 재부착" 경우를 많이 만든다
      (조기 종료가 이 입자를 멈추면 안 된다)
    - pulse: 펄스가 켜지면 조기 종료를 쓰지 않는다
    """
    cfg = copy.deepcopy(load_physics())
    if variant == "redeposit":
        cfg["adhesion"]["redeposition_prob"] = 0.5
    if variant == "pulse":
        cfg["jet"]["pulse"]["enabled"] = True
    nozzle = load_nozzles()
    poses = [PoseParams(), PoseParams(torso_yaw=90.0, shoulder_abduction=40.0)]
    states = [build_body(BodyParams(), p, scenario) for p in poses]

    ev = ParticleEvaluator(cfg, max_candidates=2, particles_per_candidate=N_DEV,
                           duration_s=DURATION_DEV)

    def run(steps_per_launch, stepwise):
        ev.steps_per_launch = steps_per_launch
        cb = (lambda *a: None) if stepwise else None
        res = ev.batch_evaluate_states(states, poses, nozzle, scenario, step_callback=cb)
        return res, ev.f.state.to_numpy()[:2 * N_DEV], ev.f.pos.to_numpy()[:2 * N_DEV]

    try:
        ref, ref_state, ref_pos = run(100, stepwise=True)
        assert sum(r.total_removal for r in ref) > 0.0
        for spl in (1, 7, 100):
            res, state, pos = run(spl, stepwise=False)
            for a, b in zip(res, ref):
                _assert_same(a, b)
            np.testing.assert_array_equal(state, ref_state)
            np.testing.assert_array_equal(pos, ref_pos)
    finally:
        ev.destroy()


# ------------------------------------------------------------ 가림 (occlusion)
def _occluded_reference(ev, states, pidx, nozzle, cfg):
    """D 패치판과 같은 식: u = Σ_m occlusion[m, 패치] · u_m (B의 노즐별 속도, 4.2b 포함).

    반환: (n·N, 3) 기준 속도와 합산 크기 척도 Σ_m w·|u_m| (상쇄 판정용).
    """
    n_all = ev.N * len(states)
    pos = ev._h["pos"][:n_all].astype(np.float64)
    nrm = ev._h["normal"][:n_all].astype(np.float64)
    x = pos + cfg["air"]["wall_offset_m"] * nrm
    ref = np.zeros((n_all, 3))
    scale = np.zeros(n_all)
    for b, st in enumerate(states):
        sl = slice(b * ev.N, (b + 1) * ev.N)
        u_m = velocity_field_per_nozzle(x[sl], nozzle, 0.0, cfg,
                                        surface_normals=nrm[sl]).astype(np.float64)   # (M,P,3)
        w = occlusion(st, nozzle, cfg)[:, pidx[sl]]                                   # (M,P)
        ref[sl] = np.einsum("mpk,mp->pk", u_m, w)
        scale[sl] = np.einsum("mp,mp->p", np.linalg.norm(u_m, axis=2), w)
    return ref, scale


def test_occlusion_matches_patch_evaluator_visibility(scenario):
    """부착 입자의 이탈 판정 속도 = Σ_m occlusion[m, 소속 패치] · (자유 제트 + 4.2b)_m.

    D의 `patch_baseline.occlusion` 배열을 입자별 소속 패치로 그대로 읽는지 본다. 가림을 끄면
    B의 `velocity_field(..., surface_normals=)`와 같다. 좌우 벽의 대칭 노즐이 서로 상쇄해
    합이 거의 0인 점이 있어, 상대 오차의 분모는 max(|u|, 1e-3·Σ_m w·|u_m|)로 둔다.
    """
    cfg = load_physics()
    nozzle = load_nozzles()
    poses = [PoseParams(), PoseParams(torso_yaw=90.0, shoulder_abduction=40.0)]
    states = [build_body(BodyParams(), p, scenario) for p in poses]
    ev = ParticleEvaluator(cfg, max_candidates=2, particles_per_candidate=2000,
                           duration_s=DURATION_DEV)
    ev_off = ParticleEvaluator(cfg, max_candidates=2, particles_per_candidate=2000,
                               duration_s=DURATION_DEV, occlusion=False)
    try:
        vel, pidx = ev.probe_attached_velocity(states, poses, nozzle)
        vel_off, pidx_off = ev_off.probe_attached_velocity(states, poses, nozzle)
        ref, scale = _occluded_reference(ev, states, pidx, nozzle, cfg)
        pos = ev_off._h["pos"][:4000].astype(np.float64)
        nrm = ev_off._h["normal"][:4000].astype(np.float64)
    finally:
        ev.destroy()
        ev_off.destroy()

    np.testing.assert_array_equal(pidx, pidx_off)
    err = np.linalg.norm(vel.astype(np.float64) - ref, axis=1)
    rel = err / np.maximum(np.linalg.norm(ref, axis=1), 1e-3 * scale + 1e-12)
    assert rel.max() < 1e-4, f"가림 적용 속도 최대 상대 오차 {rel.max():.3e}"

    free = velocity_field(pos + cfg["air"]["wall_offset_m"] * nrm, nozzle, 0.0, cfg,
                          surface_normals=nrm).astype(np.float64)
    assert _relative_error(vel_off.astype(np.float64), free).max() < 1e-4
    # 가림이 실제로 무언가를 가린다 (공허하지 않다)
    assert np.linalg.norm(vel_off - vel, axis=1).max() > 1.0


def test_occlusion_shadowed_patches_never_detach(scenario):
    """가림을 켜면 모든 노즐에서 완전히 가린 패치(occlusion 전부 0)의 입자는 이탈하지 않는다.
    가림을 끄면 그 입자 일부가 이탈하고, 전체 이탈도 가림을 켤 때 더 적다.

    ("몸통 뒤 이탈이 준다"는 반드시 성립하지 않는다: 가림 없이는 양쪽 벽의 마주 보는 제트가
    몸통 뒤에서 서로 상쇄되는데, 가림이 한쪽을 지우면 남은 쪽 때문에 접선 속도가 커질 수 있다.
    기본 자세에서 torso_back 이탈이 가림 끔 14 -> 켬 23으로 늘었다. PR 본문 참고.)
    """
    cfg = load_physics()
    nozzle = load_nozzles()
    poses = [PoseParams(), PoseParams(torso_yaw=90.0)]
    states = [build_body(BodyParams(), p, scenario) for p in poses]
    detached, hidden_detached, n_hidden = {}, {}, None
    for flag in (True, False):
        ev = ParticleEvaluator(cfg, max_candidates=2, particles_per_candidate=N_DEV,
                               duration_s=DURATION_DEV, occlusion=flag)
        try:
            ev.batch_evaluate_states(states, poses, nozzle, scenario)
            moved = ev.f.state.to_numpy()[:2 * N_DEV] != 0
            pidx = ev._h["patch_idx"][:2 * N_DEV]
        finally:
            ev.destroy()
        hidden = np.concatenate([
            occlusion(st, nozzle, cfg)[:, pidx[b * N_DEV:(b + 1) * N_DEV]].max(axis=0) == 0.0
            for b, st in enumerate(states)])
        n_hidden = int(hidden.sum())
        detached[flag] = int(moved.sum())
        hidden_detached[flag] = int((moved & hidden).sum())
    assert n_hidden > 100, "완전히 가린 패치의 입자가 거의 없으면 공허한 검사"
    assert hidden_detached[True] == 0, hidden_detached
    assert hidden_detached[False] > 0, hidden_detached
    assert detached[True] < detached[False], detached
