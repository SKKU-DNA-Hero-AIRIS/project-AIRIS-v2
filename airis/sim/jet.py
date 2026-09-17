"""제트 속도장. 소유자: B. (`docs/tracks/B_body_jet.md` 단계 6)

수식은 `docs/tracks/00_common.md` 4.1 (원형) / 4.1b (슬롯) 를 글자 그대로 구현한다. 상수는 전부
`configs/physics.yaml` 의 `jet` 섹션에서 읽는다 (코드에 숫자를 박지 않는다).

노즐별 판정 (4.1b): nozzle.slot_axis 가 None 이면 전부 원형. 배열이면 행 단위로
slot_length[m] > 0 이고 slot_axis[m] ≠ 0 인 노즐만 슬롯, 나머지는 원형.

4.1 원형 제트:

    노즐 위치 n, 단위 방향 d, D = jet.nozzle_diameter_m, U0 = jet.exit_velocity_mps × strength
    r = p - n,  s = r·d,  ρ = |r - s·d|

    s ≤ 0   : u = 0
    L_c = K·D,  K = jet.decay_constant
    U_c(s) = U0                (s ≤ L_c)
    U_c(s) = U0 · K·D / s      (s > L_c)
    σ(s)   = 0.5·D/1.177 + jet.halfwidth_spread_rate · s / 1.177
    u(p)   = U_c(s) · exp(-ρ² / (2σ²)) · d

4.1b 슬롯 제트: 슬롯 축 e (e ⊥ d), 길이 L = slot_length, h = jet.slot.height_m,
U0 = jet.slot.exit_velocity_mps × strength

    r = p - n,  s = r·d,  ρ_e = r·e,  ρ_n = |r - s·d - ρ_e·e|
    s ≤ 0   : u = 0
    L_c = K_p·h,  K_p = jet.slot.decay_constant
    U_c(s) = U0                     (s ≤ L_c)
    U_c(s) = U0 · sqrt(K_p·h / s)   (s > L_c)
    σ(s)   = 0.5·h/1.177 + jet.slot.spread_rate · s / 1.177
    ρ_e'   = max(|ρ_e| - L/2, 0)
    u(p)   = U_c(s) · exp(-ρ_n² / (2σ²)) · exp(-ρ_e'² / (2σ²)) · d

펄스가 켜져 있으면 gate(t) = 1 if ((t/period + phase) mod 1) < duty else 0 을 곱한다 (두 모델 공통).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import NozzleConfig

# 4.1 의 σ 식에 나오는 상수. 반속도 반경 r_1/2 를 가우시안 표준편차로 바꾸는 계수
# (r_1/2 = sqrt(2 ln2) σ ≈ 1.177 σ). 물리 상수가 아니라 수식 자체의 일부라 여기 둔다.
HALFWIDTH_TO_SIGMA = 1.177

# 슬롯 축 e 와 분사 방향 d 의 수직 판정 허용치 |e·d| (단위 벡터 기준, 약 0.06도). 물리 상수가 아니라 입력 검증용.
SLOT_AXIS_PERP_TOL = 1e-3


@dataclass(frozen=True)
class JetParams:
    """configs/physics.yaml 의 jet 섹션 전체. 안 쓰는 키가 없는지 여기서 확인한다."""
    nozzle_diameter_m: float
    exit_velocity_mps: float
    decay_constant: float
    halfwidth_spread_rate: float
    slot_height_m: float
    slot_exit_velocity_mps: float
    slot_decay_constant: float
    slot_spread_rate: float
    impingement_enabled: bool
    stagnation_radius_factor: float
    wall_jet_start_factor: float
    pulse_enabled: bool
    pulse_period_s: float
    pulse_duty: float

    @property
    def potential_core_length_m(self) -> float:
        """L_c = K·D. 00_common.md 4.1 이 포텐셜 코어 길이를 감쇠 상수로 정의한다.

        U_c 가 s = L_c 에서 연속이려면 (U0 = U0·K·D/L_c) L_c = K·D 여야 한다.
        """
        return self.decay_constant * self.nozzle_diameter_m

    @property
    def slot_core_length_m(self) -> float:
        """슬롯 제트 L_c = K_p·h (4.1b). s = L_c 에서 sqrt(K_p·h/L_c) = 1 이라 U_c 가 연속이다."""
        return self.slot_decay_constant * self.slot_height_m


def jet_params(cfg: dict) -> JetParams:
    """physics.yaml 의 jet 섹션을 파싱한다. 모든 키를 읽는다."""
    j = cfg["jet"]
    imp = j["impingement"]
    slot = j["slot"]
    pulse = j["pulse"]
    return JetParams(
        nozzle_diameter_m=float(j["nozzle_diameter_m"]),
        exit_velocity_mps=float(j["exit_velocity_mps"]),
        decay_constant=float(j["decay_constant"]),
        halfwidth_spread_rate=float(j["halfwidth_spread_rate"]),
        slot_height_m=float(slot["height_m"]),
        slot_exit_velocity_mps=float(slot["exit_velocity_mps"]),
        slot_decay_constant=float(slot["decay_constant"]),
        slot_spread_rate=float(slot["spread_rate"]),
        impingement_enabled=bool(imp["enabled"]),
        stagnation_radius_factor=float(imp["stagnation_radius_factor"]),
        wall_jet_start_factor=float(imp["wall_jet_start_factor"]),
        pulse_enabled=bool(pulse["enabled"]),
        pulse_period_s=float(pulse["period_s"]),
        pulse_duty=float(pulse["duty"]),
    )


def pulse_gate(t: float, nozzle: NozzleConfig, p: JetParams) -> np.ndarray:
    """(M,) 0 또는 1. gate(t) = 1 if ((t/period + phase) mod 1) < duty else 0."""
    m = nozzle.count
    if not p.pulse_enabled:
        return np.ones(m, dtype=np.float32)
    phase = (np.zeros(m) if nozzle.pulse_phase is None
             else np.asarray(nozzle.pulse_phase, dtype=np.float64))
    frac = np.mod(t / p.pulse_period_s + phase, 1.0)
    return (frac < p.pulse_duty).astype(np.float32)


def slot_mask(nozzle: NozzleConfig) -> np.ndarray:
    """(M,) bool. 슬롯(4.1b) 노즐이면 True. 00_common.md 4.1b 의 행 단위 규약.

    slot_axis 가 None 이면 전부 False (원형). 배열이면 slot_length[m] > 0 이고 slot_axis[m] 이
    0벡터가 아닌 행만 True.
    """
    m = nozzle.count
    if nozzle.slot_axis is None:
        return np.zeros(m, dtype=bool)
    if nozzle.slot_length is None:
        raise ValueError("NozzleConfig.slot_axis 가 있으면 slot_length (M,) 도 있어야 한다.")
    axis = np.asarray(nozzle.slot_axis, dtype=np.float64).reshape(m, 3)
    length = np.asarray(nozzle.slot_length, dtype=np.float64).reshape(m)
    return (length > 0.0) & (np.linalg.norm(axis, axis=1) > 0.0)


def _speed_per_nozzle(points: np.ndarray, nozzle: NozzleConfig, t: float, cfg: dict,
                      surface_normals: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """(M,P) 속도 크기와 (M,3) 방향. 4.1 / 4.1b 의 U_c·exp(...) 부분.

    s 와 거리 제곱을 (M,P,3) 중간 배열 없이 행렬곱으로 구한다 (패치 3,600 × 노즐 16 에서 ~3 ms).
      s    = d·p - d·n
      |r|² = |p|² - 2 n·p + |n|²,   원형 ρ² = |r|² - s²
      슬롯 ρ_e = e·p - e·n,         ρ_n² = |r|² - s² - ρ_e²   (e ⊥ d)
    """
    p = jet_params(cfg)
    # float64 로 계산한다. ρ² = |r|² - s² 는 제트 축 근처에서 뺄셈 상쇄가 커서 float32 로는
    # 출구 근처(σ ≈ 수 mm)에서 0.6% 오차가 났다. 반환 배열만 float32 로 내린다.
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pos = np.asarray(nozzle.positions, dtype=np.float64)
    dirs = np.asarray(nozzle.directions, dtype=np.float64)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    if surface_normals is not None and p.impingement_enabled:
        # 단계 8 (2주차 옵션). 아직 미구현이므로 조용히 자유 제트를 돌려주지 않고 명시적으로 막는다.
        raise NotImplementedError(
            "충돌 제트 보정은 B 단계 8 (2주차 옵션)이라 아직 없다. "
            "configs/physics.yaml 의 jet.impingement.enabled 를 false 로 두거나 "
            "surface_normals 를 넘기지 마라."
        )

    strengths = np.asarray(nozzle.strengths, dtype=np.float64)
    s = dirs @ pts.T - np.einsum("mk,mk->m", pos, dirs)[:, None]                # (M,P)
    r2 = ((pts * pts).sum(1)[None, :]
          - 2.0 * (pos @ pts.T)
          + (pos * pos).sum(1)[:, None])                                        # (M,P)
    live = s > 0.0                                       # s ≤ 0 이면 u = 0
    s_safe = np.where(live, s, 1.0)                      # 0 나눗셈 방지용 더미

    mag = np.empty_like(s)
    slot = slot_mask(nozzle)
    rnd = ~slot

    if rnd.any():
        # 4.1 원형: 포텐셜 코어 안에서는 U0, 밖에서는 1/s 감쇠
        D, KD = p.nozzle_diameter_m, p.potential_core_length_m                  # L_c = K·D
        sr = s_safe[rnd]
        rho2 = np.maximum(r2[rnd] - s[rnd] ** 2, 0.0)   # 반올림 오차로 음수가 될 수 있다
        u0 = p.exit_velocity_mps * strengths[rnd]
        u_c = np.where(sr <= KD, 1.0, KD / sr) * u0[:, None]
        sigma = (0.5 * D + p.halfwidth_spread_rate * sr) / HALFWIDTH_TO_SIGMA
        mag[rnd] = u_c * np.exp(-rho2 / (2.0 * sigma * sigma))

    if slot.any():
        # 4.1b 슬롯: 코어 밖 1/sqrt(s) 감쇠, 슬롯 길이 안은 균일, 끝에서 가우시안
        h, Lc = p.slot_height_m, p.slot_core_length_m                           # L_c = K_p·h
        e = np.asarray(nozzle.slot_axis, dtype=np.float64)[slot]
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        dot = np.abs((e * dirs[slot]).sum(1))
        if (dot > SLOT_AXIS_PERP_TOL).any():
            bad = np.flatnonzero(slot)[dot > SLOT_AXIS_PERP_TOL]
            raise ValueError(f"슬롯 축이 분사 방향에 수직이 아니다 (4.1b e ⊥ d): 노즐 {bad.tolist()}")
        half_len = 0.5 * np.asarray(nozzle.slot_length, dtype=np.float64)[slot]

        sr = s_safe[slot]
        rho_e = e @ pts.T - np.einsum("mk,mk->m", pos[slot], e)[:, None]       # (Ms,P)
        rho_n2 = np.maximum(r2[slot] - s[slot] ** 2 - rho_e ** 2, 0.0)
        rho_e_out = np.maximum(np.abs(rho_e) - half_len[:, None], 0.0)          # ρ_e'
        u0 = p.slot_exit_velocity_mps * strengths[slot]
        u_c = np.where(sr <= Lc, 1.0, np.sqrt(Lc / sr)) * u0[:, None]
        sigma = (0.5 * h + p.slot_spread_rate * sr) / HALFWIDTH_TO_SIGMA
        mag[slot] = u_c * np.exp(-(rho_n2 + rho_e_out ** 2) / (2.0 * sigma * sigma))

    mag = np.where(live, mag, 0.0) * pulse_gate(t, nozzle, p)[:, None]
    return mag, dirs


def velocity_field_per_nozzle(points: np.ndarray, nozzle: NozzleConfig, t: float,
                              cfg: dict, surface_normals: np.ndarray | None = None) -> np.ndarray:
    """(P,3) → (M,P,3). 노즐별 기여. D가 가림 판정 후 합산한다. 수식: docs/tracks/00_common.md 4.1 / 4.1b"""
    mag, dirs = _speed_per_nozzle(points, nozzle, t, cfg, surface_normals)
    return (mag[..., None] * dirs[:, None, :]).astype(np.float32)


def velocity_field(points: np.ndarray, nozzle: NozzleConfig, t: float,
                   cfg: dict, surface_normals: np.ndarray | None = None) -> np.ndarray:
    """(P,3) → (P,3). velocity_field_per_nozzle의 합.

    (M,P,3) 을 만들지 않고 바로 합산한다 (결과는 동일).
    """
    mag, dirs = _speed_per_nozzle(points, nozzle, t, cfg, surface_normals)
    return (mag.T @ dirs).astype(np.float32)
