"""벽면 전단 / 제거율 / 불편도 / 점수. 소유자: D. A와 C도 import한다.

수식의 단일 기준은 `docs/tracks/00_common.md` 4.2~4.4(+ 계획 확장 4.6·4.7)다. 여기가 어긋나면 D와 A의
점수가 달라져 E2(순위 상관)가 실패하므로, 네 함수 모두 문서 수식을 그대로 옮긴다.

물리 상수는 전부 `configs/physics.yaml`(= `physics_cfg` dict)에서 읽는다.
숫자를 코드에 박지 않는다. 상수를 바꿔 보려면 dict를 복사해 덮어쓴다.

`removal_fraction`은 `scipy.stats.norm.cdf` 대신 `scipy.special.erf`로 쓴다.
A가 Taichi로 이식할 때 그대로 옮길 수 있어야 하기 때문이다.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import erf

from .types import PART_NAMES, Phase, PoseParams, Scenario

_SQRT2 = math.sqrt(2.0)


def wall_shear(u: np.ndarray, normals: np.ndarray, physics_cfg: dict) -> np.ndarray:
    """(P,3), (P,3) -> (P,). `00_common.md` 4.2.

    u_t = u - (u·n)·n  (접선 성분)
    tau = 0.5 · air.density · air.friction_coeff · |u_t|^2
    """
    air = physics_cfg["air"]
    u = np.asarray(u, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)

    u_t = u - (u * n).sum(axis=-1, keepdims=True) * n
    return 0.5 * float(air["density"]) * float(air["friction_coeff"]) * (u_t * u_t).sum(axis=-1)


def removal_fraction(tau: np.ndarray, physics_cfg: dict) -> np.ndarray:
    """(P,) -> (P,). `00_common.md` 4.3의 D판 (닫힌 식).

    입자의 임계 전단이 로그 정규 분포이므로, 어떤 패치의 제거율은
    그 패치에서 `tau_c < tau`인 입자 비율 = 로그 정규 CDF다.

        R = Phi((ln tau - ln tau_med) / sigma_log),  tau <= 0 이면 R = 0
        tau_med = adhesion.critical_shear_pa_median × adhesion.fabric_roughness_factor
    """
    adhesion = physics_cfg["adhesion"]
    tau_med = (float(adhesion["critical_shear_pa_median"])
               * float(adhesion["fabric_roughness_factor"]))
    sigma_log = float(adhesion["critical_shear_sigma_log"])

    tau = np.asarray(tau, dtype=np.float64)
    positive = tau > 0.0
    z = np.zeros_like(tau)
    np.divide(np.log(np.where(positive, tau, 1.0)) - math.log(tau_med), sigma_log,
              out=z, where=positive)

    # Phi(z) = 0.5 · (1 + erf(z / sqrt(2)))
    return np.where(positive, 0.5 * (1.0 + erf(z / _SQRT2)), 0.0)


def discomfort(pose: PoseParams, scenario: Scenario) -> float:
    """`00_common.md` 4.4. 기본 자세면 0.

        불편도 = sum_k  c_k · |theta_k - theta_k,기본| / (hi_k - lo_k)
        c_k = scenario.discomfort_weights.get(k, 0),  (lo_k, hi_k) = scenario.pose_bounds[k]
    """
    default = PoseParams()
    total = 0.0
    for key, weight in scenario.discomfort_weights.items():
        if not weight:
            continue
        if key not in scenario.pose_bounds:
            raise KeyError(
                f"시나리오 '{scenario.name}'의 discomfort_weights['{key}']에 대응하는 "
                f"pose_bounds가 없어 정규화할 수 없다"
            )
        lo, hi = scenario.pose_bounds[key]
        span = float(hi) - float(lo)
        if span <= 0.0:
            raise ValueError(f"시나리오 '{scenario.name}'의 pose_bounds['{key}'] 범위가 0 이하다")
        total += float(weight) * abs(getattr(pose, key) - getattr(default, key)) / span
    return total


def score(removal_by_part: np.ndarray, pose: PoseParams, scenario: Scenario,
          physics_cfg: dict) -> tuple[float, float]:
    """`00_common.md` 4.4 -> (score, discomfort).

        score = sum_부위 scoring.part_weights[부위] · R_부위
                - scoring.discomfort_weight · 불편도

    `removal_by_part`는 `PART_NAMES` 순서의 (5,) 배열이다.
    `part_weights`에 없는 부위의 가중치는 0으로 본다.
    """
    scoring_cfg = physics_cfg["scoring"]
    part_weights = scoring_cfg["part_weights"]

    removal_by_part = np.asarray(removal_by_part, dtype=np.float64)
    if removal_by_part.shape != (len(PART_NAMES),):
        raise ValueError(
            f"removal_by_part는 {(len(PART_NAMES),)} 여야 한다: {removal_by_part.shape}")

    weights = np.array([float(part_weights.get(name, 0.0)) for name in PART_NAMES])
    disc = discomfort(pose, scenario)
    total = float(weights @ removal_by_part) - float(scoring_cfg["discomfort_weight"]) * disc
    return total, disc


# --- 계획 확장 (docs/plan_extension.md, 00_common.md 4.4·4.6·4.7) ----------------

def removal_fraction_plan(tau: np.ndarray, durations: np.ndarray,
                          physics_cfg: dict) -> np.ndarray:
    """(K,P) 전단과 (K,) 단계 시간 -> (P,) 제거율. `00_common.md` 4.6.

    임계 전단 `tau_c` 인 입자는 `tau_ik >= tau_c` 인 단계에서만 떨어질 수 있다. 단계를 패치마다
    전단 오름차순으로 정렬해 `tau_(1) <= ... <= tau_(K)`, `F` = 로그 정규 CDF(4.3), `F(tau_(0)) = 0`,
    `T_r = adhesion.kinetics.time_constant_s` 라 하면

        R_i = sum_j [F(tau_(j)) - F(tau_(j-1))] · (1 - exp(-(sum_{k: tau_ik >= tau_(j)} t_k) / T_r))

    `adhesion.kinetics.enabled` 가 거짓이면 시간 항이 없는 점근값 `R_i = F(max_k tau_ik)` 다
    (위 식에서 t -> 무한대인 극한이고, 단계가 1개면 `removal_fraction` 과 같다).
    """
    tau = np.atleast_2d(np.asarray(tau, dtype=np.float64))
    durations = np.asarray(durations, dtype=np.float64).reshape(-1)
    if durations.shape[0] != tau.shape[0]:
        raise ValueError(f"durations {durations.shape} 가 tau {tau.shape} 의 단계 수와 다르다")
    if (durations < 0.0).any() or not np.all(np.isfinite(durations)):
        raise ValueError("durations 는 0 이상의 유한값이어야 한다")

    kinetics = physics_cfg["adhesion"].get("kinetics", {})
    if not kinetics.get("enabled", False):
        return removal_fraction(tau.max(axis=0), physics_cfg)

    t_r = float(kinetics["time_constant_s"])
    if t_r <= 0.0:
        raise ValueError(f"adhesion.kinetics.time_constant_s 는 양수여야 한다: {t_r}")

    order = np.argsort(tau, axis=0, kind="stable")                  # (K,P) 패치마다 전단 오름차순
    tau_sorted = np.take_along_axis(tau, order, axis=0)
    t_sorted = durations[order]                                     # (K,P)
    # j 번째 밴드가 노출되는 시간 = 전단이 tau_(j) 이상인 단계들의 시간 합 (정렬 뒤 뒤쪽 누적).
    exposed = np.cumsum(t_sorted[::-1], axis=0)[::-1]
    f = removal_fraction(tau_sorted, physics_cfg)                   # (K,P)
    band = np.diff(f, axis=0, prepend=0.0)                          # F(tau_(j)) - F(tau_(j-1))
    return (band * (1.0 - np.exp(-exposed / t_r))).sum(axis=0)


def energy(strengths: np.ndarray, duration_s: float, physics_cfg: dict) -> float:
    """노즐별 세기 (M,) 와 시간 -> 무차원 에너지 e. `00_common.md` 4.7.

        e = (sum_m s_m^p · T) / (M · T_ref),  p = fan.power_exponent, T_ref = scoring.reference_duration_s

    전 노즐 s = 1, T = T_ref 인 현행 운전이면 e = 1 이다.
    """
    s = np.asarray(strengths, dtype=np.float64).reshape(-1)
    if s.size == 0:
        raise ValueError("strengths 가 비어 있다")
    if not np.all(np.isfinite(s)) or (s < 0.0).any():
        raise ValueError("strengths 는 0 이상의 유한값이어야 한다")
    p = float(physics_cfg["fan"]["power_exponent"])
    t_ref = float(physics_cfg["scoring"]["reference_duration_s"])
    return float((s ** p).sum() * float(duration_s) / (s.size * t_ref))


def score_plan(removal_by_part: np.ndarray, phases: list[Phase], scenario: Scenario,
               physics_cfg: dict, energy_value: float) -> tuple[float, float]:
    """`00_common.md` 4.4 의 계획 점수 -> (score, 시간 가중 불편도).

        score = sum_부위 w_부위 · R_부위
                - scoring.discomfort_weight · sum_k 불편도_k · t_k / T_ref
                - scoring.energy_weight · e
                - scoring.time_weight · T / T_ref

    두 번째 항의 `sum_k 불편도_k · t_k / T_ref` 를 "시간 가중 불편도"로 함께 돌려준다
    (`EvalResult.discomfort`). 단일 자세를 T_ref 만큼 유지하면 `discomfort(pose, scenario)` 와 같다.
    """
    scoring_cfg = physics_cfg["scoring"]
    removal_by_part = np.asarray(removal_by_part, dtype=np.float64)
    if removal_by_part.shape != (len(PART_NAMES),):
        raise ValueError(
            f"removal_by_part는 {(len(PART_NAMES),)} 여야 한다: {removal_by_part.shape}")
    if not phases:
        raise ValueError("phases 가 비어 있다")

    t_ref = float(scoring_cfg["reference_duration_s"])
    total_time = float(sum(ph.duration_s for ph in phases))
    disc = sum(discomfort(ph.pose, scenario) * float(ph.duration_s) for ph in phases) / t_ref

    weights = np.array([float(scoring_cfg["part_weights"].get(name, 0.0)) for name in PART_NAMES])
    total = (float(weights @ removal_by_part)
             - float(scoring_cfg["discomfort_weight"]) * disc
             - float(scoring_cfg["energy_weight"]) * float(energy_value)
             - float(scoring_cfg["time_weight"]) * total_time / t_ref)
    return total, disc
