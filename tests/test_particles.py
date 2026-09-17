"""입자 평가기 테스트. 소유자: A.

`docs/tracks/A_particles.md` 단계 10의 후보 격리 테스트 6개 + 시각 확인용 산점도.
개발 중 설정은 `N=1000, B=2, duration 0.5 s` (00_common.md 7절, GPU 공유).

입력
- 격리 테스트: 기하를 완전히 통제하려고 원통 캡슐 BodyState를 여기서 만든다
  (`cylinder_body`, 후보마다 캡슐 수가 다르게). 노즐은 원통을 스치는 4개.
- 실제 경로: B의 `build_body` + `load_nozzles(layout="layout")`(원형 16개)로
  `evaluate`/`batch_evaluate`를 C가 부르는 그대로 호출한다. D의 `fake_body`/`fake_nozzles`도
  한 번 돌린다.
- 슬롯(4.1b) 노즐: A의 Taichi 커널(⑤b)이 아직 없어 제트 테스트는 전부 4.1 원형 검증이다.
  기준 배치(`load_nozzles()` = 퓨리움 슬롯 바)는 입자판이 NotImplementedError로 막는지만 본다.

물리 상수 오버라이드 (`_sim_physics`)
- configs/physics.yaml 그대로(노즐 지름 4 mm)면 몸 위치(노즐에서 0.45 m)의 중심 속도가
  약 1.5 m/s, 전단 약 0.007 Pa로 임계 전단 중앙값 0.5 Pa보다 두 자릿수 작다. 제거율이
  사실상 0이 되어 격리 테스트가 "전부 0 == 전부 0"으로 공허하게 통과한다.
- 그래서 시뮬레이션 테스트는 노즐 지름과 임계 전단만 바꾼 사본을 쓰고, 제거율이 실제로
  0보다 큰지 함께 단언한다. configs/는 수정하지 않는다 (PR에 B 앞 제안으로 남김).
- 제트 일치 테스트는 오버라이드 없이 physics.yaml 원본을 쓴다 (D와 맞춰야 하는 값).
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from airis.sim.jet import velocity_field  # noqa: E402
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
        hand = shoulder + 0.6 * np.array([0.0, -math.sin(ab), -math.cos(ab)])
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


def round_nozzles() -> NozzleConfig:
    """B의 원형 비교 배치 (16개, 전부 4.1). 기준 배치(`active`)는 슬롯이라 명시해서 부른다."""
    nz = load_nozzles(layout="layout")
    assert nz.slot_axis is None
    return nz


def layout_nozzles(rng: np.random.Generator) -> NozzleConfig:
    """원형 비교 배치에 무작위 세기와 펄스 위상을 입힌 것.

    TODO(A, ⑤b): 4.1b 커널이 들어오면 슬롯 배치(slot_axis/slot_length 포함)로도 돌린다.
    지금은 slot_axis=None을 명시해 "4.1 원형 검증"임을 드러낸다.
    """
    nz = round_nozzles()
    return NozzleConfig(nz.positions, nz.directions,
                        rng.uniform(0.3, 1.0, nz.count).astype(np.float32),
                        pulse_phase=rng.random(nz.count).astype(np.float32),
                        slot_axis=None, slot_length=None)


def _sim_physics() -> dict:
    cfg = copy.deepcopy(load_physics())
    cfg["jet"]["nozzle_diameter_m"] = 0.05
    cfg["adhesion"]["critical_shear_pa_median"] = 0.1
    return cfg


# ----------------------------------------------------------------- 픽스처
@pytest.fixture(scope="module")
def scenario():
    return load_scenarios()["default"]


@pytest.fixture(scope="module")
def ev_batch():
    ev = ParticleEvaluator(_sim_physics(), max_candidates=B_DEV,
                           particles_per_candidate=N_DEV, duration_s=DURATION_DEV)
    yield ev
    ev.destroy()


@pytest.fixture(scope="module")
def ev_single():
    """B=1로 따로 할당한 평가기. 필드 크기와 커널 인스턴스가 배치 평가기와 다르다."""
    ev = ParticleEvaluator(_sim_physics(), max_candidates=1,
                           particles_per_candidate=N_DEV, duration_s=DURATION_DEV)
    yield ev
    ev.destroy()


POSE_A = PoseParams()                                                  # 캡슐 1개
POSE_B = PoseParams(shoulder_abduction=90.0, torso_pitch=15.0)        # 기울임 + 팔, 캡슐 2개


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


def _jet_velocity_numpy(points, nozzle, cfg, t=0.0):
    """00_common.md 4.1을 numpy float64로 독립 재구현. Taichi 코드를 참조하지 않는다."""
    jet = cfg["jet"]
    big_d = jet["nozzle_diameter_m"]
    k = jet["decay_constant"]
    u_exit = jet["exit_velocity_mps"]
    spread = jet["halfwidth_spread_rate"]
    pulse = jet.get("pulse") or {}

    p = np.asarray(points, np.float64)[:, None, :]                    # (P,1,3)
    n = np.asarray(nozzle.positions, np.float64)[None]                # (1,M,3)
    d = np.asarray(nozzle.directions, np.float64)
    d = (d / np.linalg.norm(d, axis=1, keepdims=True))[None]
    r = p - n
    s = np.sum(r * d, axis=-1)                                        # (P,M)
    rho = np.linalg.norm(r - s[..., None] * d, axis=-1)
    u0 = u_exit * np.asarray(nozzle.strengths, np.float64)[None]
    l_core = k * big_d
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        u_c = np.where(s <= l_core, u0, u0 * k * big_d / s)
        sigma = 0.5 * big_d / 1.177 + spread * s / 1.177
        mag = u_c * np.exp(-rho ** 2 / (2 * sigma ** 2))
    mag = np.where(s > 0, mag, 0.0)
    if pulse.get("enabled", False):
        phase = np.zeros(nozzle.count) if nozzle.pulse_phase is None else nozzle.pulse_phase
        gate = (np.mod(t / pulse["period_s"] + np.asarray(phase, np.float64), 1.0)
                < pulse["duty"]).astype(np.float64)
        mag = mag * gate[None]
    return np.sum(mag[..., None] * d, axis=1)                         # (P,3)


def _jet_points(nozzle, cfg, rng, n=1000):
    """무작위 점 1000개: 절반은 부스 안 균일, 절반은 제트 축 근처 (값이 큰 곳).

    축 근처 점은 그 거리의 제트 폭 sigma(s)에 비례해 흩뿌려 코어/감쇠 구간,
    가우시안 중심/꼬리, 노즐 뒤(s <= 0)를 고루 지나게 한다.
    """
    jet = cfg["jet"]
    half = n // 2
    box = rng.uniform([0.0, -BOOTH["width_m"] / 2, 0.0],
                      [BOOTH["length_m"], BOOTH["width_m"] / 2, BOOTH["height_m"]], (half, 3))
    m = rng.integers(0, nozzle.count, n - half)
    s = rng.uniform(-0.05, 0.8, n - half)            # 음수 = 노즐 뒤 (u = 0 분기)
    sigma = (0.5 * jet["nozzle_diameter_m"] + jet["halfwidth_spread_rate"] * np.abs(s)) / 1.177
    off = rng.normal(0.0, 1.0, (n - half, 3)) * 1.5 * sigma[:, None]
    near = nozzle.positions[m] + s[:, None] * nozzle.directions[m] + off
    return np.concatenate([box, near]).astype(np.float32)


def _relative_error(u_ti, u_np):
    """점별 |Δu| / |u|. |u|가 사실상 0인 점은 필드 최대값 1e-6배를 바닥으로 둔다
    (f32 exp 꼬리의 언더플로가 상대 오차를 부풀리지 않게)."""
    mag = np.linalg.norm(u_np, axis=1)
    floor = 1e-6 * max(mag.max(), 1e-12)
    return np.linalg.norm(u_ti - u_np, axis=1) / np.maximum(mag, floor)


@pytest.mark.parametrize("pulse_t", [None, 0.37], ids=["steady", "pulse"])
def test_jet_matches_numpy_reference(pulse_t):
    """제트 일치: Taichi ti.func vs numpy 재구현, 무작위 점 1000개, 상대 오차 1e-4 이내.

    D의 패치판과 점수를 맞추는 핵심. physics.yaml 원본 상수를 쓴다.
    """
    cfg = copy.deepcopy(load_physics())
    t = 0.0
    if pulse_t is not None:
        cfg["jet"]["pulse"]["enabled"] = True
        t = pulse_t
    rng = np.random.default_rng(1234)
    nozzle = layout_nozzles(rng)
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
    return NozzleConfig(nozzle.positions[rows], nozzle.directions[rows], nozzle.strengths[rows],
                        pulse_phase=nozzle.pulse_phase[rows])


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


def test_jet_matches_b_velocity_field():
    """제트 일치 (B 기준값): Taichi vs `airis.sim.jet.velocity_field`, 몸 주변 점 1000개.

    점은 부스 중앙 몸 주변이며 모든 노즐에서 0.1 m 이상 떨어져 있다 (통합 관리자 안내).
    B 구현은 float32로 rho^2 = |r|^2 - s^2를 계산해, 좌표 크기(~1 m)의 반올림 오차가 가는
    제트의 지수에서 증폭된다. 정확한 float64 수식 대비 최대 약 4e-4 어긋난다 (측정값).
    Taichi는 rho = |r - s*d|를 직접 계산해 float64 대비 1e-4 안에 든다. 그래서
    - Taichi vs float64 독립 구현: 1e-4 (엄격)
    - Taichi vs B: 1e-3. B가 정밀도를 고치면 1e-4로 조인다 (PR 본문에 제안).
    """
    cfg = load_physics()
    rng = np.random.default_rng(99)
    nozzle = layout_nozzles(rng)
    cand = rng.uniform([CENTER_X - 0.3, -0.4, 0.2], [CENTER_X + 0.3, 0.4, 1.9], (20000, 3))
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
    assert rel_b.max() < 1e-3, f"Taichi vs B 최대 상대 오차 {rel_b.max():.3e}"


def test_torso_redeposition_splits_front_back():
    """재부착 부위: 몸통 캡슐(capsule_part = torso_front)에 부딪히면 충돌 법선과 몸 전방
    (cos yaw, sin yaw, 0)의 부호로 torso_front / torso_back을 가른다."""
    cfg = _sim_physics()
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
    ev = ParticleEvaluator(_sim_physics(), max_candidates=1,
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
    -> (B,) float32, 입력 순서 유지, 각 원소 == evaluate(...).score."""
    body = BodyParams()
    nozzle = round_nozzles()
    poses = [PoseParams(shoulder_abduction=120.0, torso_yaw=30.0), PoseParams()]
    scores = ev_batch.batch_evaluate([(p, nozzle) for p in poses], body, scenario)
    assert scores.dtype == np.float32 and scores.shape == (2,)
    singles = [ev_batch.evaluate(p, nozzle, body, scenario) for p in poses]
    np.testing.assert_array_equal(scores, np.array([r.score for r in singles], np.float32))
    reversed_scores = ev_batch.batch_evaluate([(p, nozzle) for p in poses[::-1]], body, scenario)
    np.testing.assert_array_equal(reversed_scores, scores[::-1])
    assert singles[0].total_removal > 0.0
    assert singles[0].extra["count_init"].sum() == N_DEV


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


# ---------------------------------------------- 슬롯(4.1b) 노즐 방어 (⑤b 전까지)
def test_slot_nozzles_raise_until_kernel_exists(ev_batch, scenario):
    """기준 배치(퓨리움 슬롯 바)를 받으면 원형으로 조용히 계산하지 않고 막는다.

    TODO(A, ⑤b): 4.1b 커널이 들어오면 이 테스트를 슬롯 제트 일치 테스트로 바꾼다.
    """
    slot = load_nozzles()
    assert slot.slot_axis is not None and slot.slot_length is not None
    with pytest.raises(NotImplementedError, match="4.1b"):
        ev_batch.batch_evaluate([(PoseParams(), slot)], BodyParams(), scenario)
    with pytest.raises(NotImplementedError, match="4.1b"):
        ev_batch.probe_velocity(np.zeros((1, 3), np.float32), slot)

    # 행 단위 규약: slot_length = 0 또는 slot_axis = 0벡터인 행은 원형이라 통과해야 한다.
    rnd = round_nozzles()
    mixed = NozzleConfig(rnd.positions, rnd.directions, rnd.strengths,
                         slot_axis=np.zeros((rnd.count, 3), np.float32),
                         slot_length=np.ones(rnd.count, np.float32))
    np.testing.assert_array_equal(
        ev_batch.probe_velocity(rnd.positions + rnd.directions * 0.3, mixed),
        ev_batch.probe_velocity(rnd.positions + rnd.directions * 0.3, rnd))
