"""제트 속도장. 소유자: B. (`docs/tracks/B_body_jet.md` 단계 6)

수식은 `docs/tracks/00_common.md` 4.1 을 글자 그대로 구현한다. 상수는 전부
`configs/physics.yaml` 의 `jet` 섹션에서 읽는다 (코드에 숫자를 박지 않는다).

    노즐 위치 n, 단위 방향 d, D = jet.nozzle_diameter_m, U0 = jet.exit_velocity_mps × strength
    r = p - n,  s = r·d,  ρ = |r - s·d|

    s ≤ 0   : u = 0
    L_c = K·D,  K = jet.decay_constant
    U_c(s) = U0                (s ≤ L_c)
    U_c(s) = U0 · K·D / s      (s > L_c)
    σ(s)   = 0.5·D/1.177 + jet.halfwidth_spread_rate · s / 1.177
    u(p)   = U_c(s) · exp(-ρ² / (2σ²)) · d

펄스가 켜져 있으면 gate(t) = 1 if ((t/period + phase) mod 1) < duty else 0 을 곱한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import NozzleConfig

# 4.1 의 σ 식에 나오는 상수. 반속도 반경 r_1/2 를 가우시안 표준편차로 바꾸는 계수
# (r_1/2 = sqrt(2 ln2) σ ≈ 1.177 σ). 물리 상수가 아니라 수식 자체의 일부라 여기 둔다.
HALFWIDTH_TO_SIGMA = 1.177


@dataclass(frozen=True)
class JetParams:
    """configs/physics.yaml 의 jet 섹션 전체. 안 쓰는 키가 없는지 여기서 확인한다."""
    nozzle_diameter_m: float
    exit_velocity_mps: float
    potential_core_diameters: float
    decay_constant: float
    halfwidth_spread_rate: float
    impingement_enabled: bool
    stagnation_radius_factor: float
    wall_jet_start_factor: float
    pulse_enabled: bool
    pulse_period_s: float
    pulse_duty: float

    @property
    def potential_core_length_m(self) -> float:
        """L_c = K·D. 00_common.md 4.1 이 포텐셜 코어 길이를 감쇠 상수로 정의한다.

        `potential_core_diameters`(=6.0)와는 다른 값이다. 4.1 의 U_c 가 s = L_c 에서
        연속이려면 (U0 = U0·K·D/L_c) 반드시 L_c = K·D 여야 하므로 4.1 을 따른다.
        두 키의 정합은 PR 에 공용 설정 변경 제안으로 남긴다.
        """
        return self.decay_constant * self.nozzle_diameter_m


def jet_params(cfg: dict) -> JetParams:
    """physics.yaml 의 jet 섹션을 파싱한다. 모든 키를 읽는다."""
    j = cfg["jet"]
    imp = j["impingement"]
    pulse = j["pulse"]
    return JetParams(
        nozzle_diameter_m=float(j["nozzle_diameter_m"]),
        exit_velocity_mps=float(j["exit_velocity_mps"]),
        potential_core_diameters=float(j["potential_core_diameters"]),
        decay_constant=float(j["decay_constant"]),
        halfwidth_spread_rate=float(j["halfwidth_spread_rate"]),
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


def _speed_per_nozzle(points: np.ndarray, nozzle: NozzleConfig, t: float, cfg: dict,
                      surface_normals: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """(M,P) 속도 크기와 (M,3) 방향. 4.1 의 U_c·exp(-ρ²/2σ²) 부분.

    s 와 ρ² 를 (M,P,3) 중간 배열 없이 행렬곱으로 구한다 (패치 3,600 × 노즐 16 에서 ~1 ms).
      s   = d·p - d·n
      |r|² = |p|² - 2 n·p + |n|²,   ρ² = |r|² - s²
    """
    p = jet_params(cfg)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    pos = np.asarray(nozzle.positions, dtype=np.float32)
    dirs = np.asarray(nozzle.directions, dtype=np.float32)

    if surface_normals is not None and p.impingement_enabled:
        # 단계 8 (2주차 옵션). 아직 미구현이므로 조용히 자유 제트를 돌려주지 않고 명시적으로 막는다.
        raise NotImplementedError(
            "충돌 제트 보정은 B 단계 8 (2주차 옵션)이라 아직 없다. "
            "configs/physics.yaml 의 jet.impingement.enabled 를 false 로 두거나 "
            "surface_normals 를 넘기지 마라."
        )

    D = p.nozzle_diameter_m
    KD = p.potential_core_length_m                       # L_c = K·D
    u0 = p.exit_velocity_mps * np.asarray(nozzle.strengths, dtype=np.float32)   # (M,)

    s = dirs @ pts.T - np.einsum("mk,mk->m", pos, dirs)[:, None]                # (M,P)
    r2 = ((pts * pts).sum(1)[None, :]
          - 2.0 * (pos @ pts.T)
          + (pos * pos).sum(1)[:, None])                                        # (M,P)
    rho2 = np.maximum(r2 - s * s, 0.0)                   # 반올림 오차로 음수가 될 수 있다

    live = s > 0.0                                       # s ≤ 0 이면 u = 0
    s_safe = np.where(live, s, 1.0)                      # 0 나눗셈 방지용 더미

    # 중심 속도: 포텐셜 코어 안에서는 U0, 밖에서는 1/s 감쇠
    u_c = np.where(s_safe <= KD, 1.0, KD / s_safe) * u0[:, None]                # (M,P)
    sigma = (0.5 * D + p.halfwidth_spread_rate * s_safe) / HALFWIDTH_TO_SIGMA
    mag = u_c * np.exp(-rho2 / (2.0 * sigma * sigma))
    mag = np.where(live, mag, 0.0) * pulse_gate(t, nozzle, p)[:, None]
    return mag.astype(np.float32), dirs


def velocity_field_per_nozzle(points: np.ndarray, nozzle: NozzleConfig, t: float,
                              cfg: dict, surface_normals: np.ndarray | None = None) -> np.ndarray:
    """(P,3) → (M,P,3). 노즐별 기여. D가 가림 판정 후 합산한다. 수식: docs/tracks/00_common.md 4.1"""
    mag, dirs = _speed_per_nozzle(points, nozzle, t, cfg, surface_normals)
    return mag[..., None] * dirs[:, None, :]


def velocity_field(points: np.ndarray, nozzle: NozzleConfig, t: float,
                   cfg: dict, surface_normals: np.ndarray | None = None) -> np.ndarray:
    """(P,3) → (P,3). velocity_field_per_nozzle의 합.

    (M,P,3) 을 만들지 않고 바로 합산한다 (결과는 동일).
    """
    mag, dirs = _speed_per_nozzle(points, nozzle, t, cfg, surface_normals)
    return (mag.T @ dirs).astype(np.float32)
