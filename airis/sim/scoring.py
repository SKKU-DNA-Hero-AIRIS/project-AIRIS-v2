"""벽면 전단 / 제거율 / 불편도 / 점수. 소유자: D. A와 C도 import한다.

수식의 단일 기준은 `docs/tracks/00_common.md` 4.2~4.4다. 여기가 어긋나면 D와 A의
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

from .types import PART_NAMES, PoseParams, Scenario

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
