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

4.2b 충돌 보정 (surface_normals 가 주어지고 jet.impingement.enabled 일 때, 노즐별):
점 p 와 바깥 법선 n. 정면으로 받는 면에서 법선 성분만 남던 제트를 방사상 벽면 제트로 바꾼다.

    cosθ = −d·n                           (cosθ ≤ 0 이면 보정 0: 제트를 등진 면)
    H    = ((p − n_m)·n) / (d·n)          제트 축이 p 의 접평면과 만나는 축 거리 (H ≤ 0 이면 보정 0)
    c    = n_m + H·d                      충돌점,  r = p − c (접평면 안)
    슬롯: 슬롯 축의 면내 성분 e_t 방향을 뺀다 (선 충돌).  ρ_e = r·ê_t,  r ← r − ρ_e·ê_t
    ξ    = |r| / σ(H),  U_H = U_c(H)     (4.1 / 4.1b 의 σ, U_c 를 충돌 거리 H 에서, strength 포함)
    F    = (1 − exp(−ξ²/2)) / ξ           원형 (방사상 벽면 제트, u ∝ 1/ρ)
    F    = (1 − exp(−ξ²/2)) / sqrt(ξ)     슬롯 (평면 벽면 제트, u ∝ 1/√ρ)
    슬롯 끝 밖은 exp(−ρ_e'²/(2σ²)),  ρ_e' = max(|ρ_e| − L/2, 0)
    w    = k · cosθ · U_H · F · gate(t),  k = jet.impingement.wall_jet_gain   (ξ < 1e-6 이면 0)
    u_corr = u + w · e_r,  e_r = r/|r|

법선 성분을 지우지 않고 면내 방사 성분만 더하므로 4.2 의 접선 투영이 u_t + w·e_r 를 만든다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import NozzleConfig

# 4.1 의 σ 식에 나오는 상수. 반속도 반경 r_1/2 를 가우시안 표준편차로 바꾸는 계수
# (r_1/2 = sqrt(2 ln2) σ ≈ 1.177 σ). 물리 상수가 아니라 수식 자체의 일부라 여기 둔다.
HALFWIDTH_TO_SIGMA = 1.177

# 4.2b 에서 ξ 가 이보다 작으면 방사 방향 e_r 이 정의되지 않으므로 보정을 0 으로 둔다 (정체점).
IMPINGEMENT_XI_MIN = 1e-6

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
    wall_jet_gain: float
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
        wall_jet_gain=float(imp["wall_jet_gain"]),
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


def _speed_per_nozzle(points: np.ndarray, nozzle: NozzleConfig, t: float,
                      cfg: dict) -> tuple[np.ndarray, np.ndarray]:
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


def _impingement_per_nozzle(points: np.ndarray, normals: np.ndarray, nozzle: NozzleConfig,
                            t: float, p: JetParams) -> np.ndarray:
    """(M,P,3) 4.2b 벽면 제트 보정 w·e_r. 모듈 docstring 의 식 그대로.

    제트를 마주 보고(cosθ > 0) 충돌점이 하류(H > 0)인 (노즐, 점) 쌍만 모아 1차원으로 계산한다.
    몸 표면에서는 대략 절반만 해당하고, (M,P,3) 중간 배열을 여러 번 만들지 않아 빠르다.
    """
    x = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    if n.shape != x.shape:
        raise ValueError(f"surface_normals {n.shape} 가 points {x.shape} 와 맞지 않는다.")
    n = n / np.sqrt(np.einsum("ij,ij->i", n, n))[:, None]
    pos = np.asarray(nozzle.positions, dtype=np.float64)
    d = np.asarray(nozzle.directions, dtype=np.float64)
    d = d / np.sqrt(np.einsum("ij,ij->i", d, d))[:, None]
    out = np.zeros((nozzle.count, x.shape[0], 3))

    cos = -(d @ n.T)                                                   # (M,P) cosθ = −d·n
    xn = np.einsum("ij,ij->i", x, n)[None, :] - pos @ n.T                          # (M,P) (x − n_m)·n
    # H = ((x − n_m)·n) / (d·n) = −xn / cosθ.  cosθ > 0 이고 H > 0 ⇔ xn < 0
    mi, pi = np.nonzero((cos > 0.0) & (xn < 0.0))
    if mi.size == 0:
        return out
    c = cos[mi, pi]
    H = -xn[mi, pi] / c
    # r = x − c_m = (x − n_m) − H·d : 접평면 안 (해석적으로 r·n = 0)
    r = x[pi] - pos[mi] - H[:, None] * d[mi]                          # (L,3)

    slot_rows = slot_mask(nozzle)
    sl = slot_rows[mi]
    u_h = np.empty_like(H)
    sigma = np.empty_like(H)
    strengths = np.asarray(nozzle.strengths, dtype=np.float64)[mi]
    if (~sl).any():
        KD, h = p.potential_core_length_m, H[~sl]
        u_h[~sl] = np.where(h <= KD, 1.0, KD / h) * p.exit_velocity_mps
        sigma[~sl] = (0.5 * p.nozzle_diameter_m + p.halfwidth_spread_rate * h) / HALFWIDTH_TO_SIGMA
    end = np.ones_like(H)
    if sl.any():
        Lc, h = p.slot_core_length_m, H[sl]
        u_h[sl] = np.where(h <= Lc, 1.0, np.sqrt(Lc / h)) * p.slot_exit_velocity_mps
        sigma[sl] = (0.5 * p.slot_height_m + p.slot_spread_rate * h) / HALFWIDTH_TO_SIGMA
        e = np.asarray(nozzle.slot_axis, dtype=np.float64)[mi[sl]]
        e = e / np.sqrt(np.einsum("ij,ij->i", e, e))[:, None]
        ns = n[pi[sl]]
        et = e - np.einsum("ij,ij->i", e, ns)[:, None] * ns                   # 슬롯 축의 면내 성분
        et_len = np.sqrt(np.einsum("ij,ij->i", et, et))[:, None]
        et = np.where(et_len > 0.0, et / np.where(et_len > 0.0, et_len, 1.0), 0.0)
        rs = r[sl]
        rho_e = np.einsum("ij,ij->i", rs, et)
        r[sl] = rs - rho_e[:, None] * et                               # 선 충돌: 슬롯 방향 성분 제거
        half = 0.5 * np.asarray(nozzle.slot_length, dtype=np.float64)[mi[sl]]
        rho_e_out = np.maximum(np.abs(rho_e) - half, 0.0)
        end[sl] = np.exp(-rho_e_out ** 2 / (2.0 * sigma[sl] ** 2))

    rho = np.sqrt(np.einsum("ij,ij->i", r, r))
    xi = rho / sigma
    ok = xi >= IMPINGEMENT_XI_MIN
    xi_safe = np.where(ok, xi, 1.0)
    core = 1.0 - np.exp(-0.5 * xi_safe * xi_safe)
    F = np.where(sl, core / np.sqrt(xi_safe), core / xi_safe)
    gate = pulse_gate(t, nozzle, p).astype(np.float64)[mi]
    w = np.where(ok, p.wall_jet_gain * c * u_h * strengths * F * end * gate, 0.0)
    out[mi, pi] = (w / np.where(ok, rho, 1.0))[:, None] * r           # w · e_r
    return out


def _correction(points, nozzle, t, cfg, surface_normals) -> np.ndarray | None:
    """보정이 켜져 있고 법선이 오면 (M,P,3) 보정, 아니면 None."""
    if surface_normals is None:
        return None
    p = jet_params(cfg)
    if not p.impingement_enabled:
        return None
    return _impingement_per_nozzle(points, surface_normals, nozzle, t, p)


def velocity_field_per_nozzle(points: np.ndarray, nozzle: NozzleConfig, t: float,
                              cfg: dict, surface_normals: np.ndarray | None = None) -> np.ndarray:
    """(P,3) → (M,P,3). 노즐별 기여. D가 가림 판정 후 합산한다. 수식: docs/tracks/00_common.md 4.1 / 4.1b.

    `surface_normals` (P,3) 가 주어지고 `jet.impingement.enabled` 이면 4.2b 벽면 제트 보정을 노즐별로
    더한다. `points` 는 표면 위(또는 wall_offset_m 만큼 띄운) 조회점이고 법선은 그 점의 바깥 법선이다.
    """
    mag, dirs = _speed_per_nozzle(points, nozzle, t, cfg)
    u = mag[..., None] * dirs[:, None, :]
    corr = _correction(points, nozzle, t, cfg, surface_normals)
    if corr is not None:
        u = u + corr
    return u.astype(np.float32)


def velocity_field(points: np.ndarray, nozzle: NozzleConfig, t: float,
                   cfg: dict, surface_normals: np.ndarray | None = None) -> np.ndarray:
    """(P,3) → (P,3). velocity_field_per_nozzle의 합.

    보정이 없으면 (M,P,3) 을 만들지 않고 바로 합산한다 (결과는 동일). 4.2b 보정이 켜지면
    노즐별 방사 방향이 달라 (M,P,3) 을 만들어 합한다.
    """
    mag, dirs = _speed_per_nozzle(points, nozzle, t, cfg)
    u = mag.T @ dirs
    corr = _correction(points, nozzle, t, cfg, surface_normals)
    if corr is not None:
        u = u + corr.sum(0)
    return u.astype(np.float32)
