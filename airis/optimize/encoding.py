"""자세 벡터 인코딩/디코딩. 소유자: C.

PoseParams 중 시나리오가 고정하지 않은 변수만 골라 pose_bounds 로 [-1, 1] 정규화한다.
규약: docs/tracks/C_optimize.md 단계 1, docs/interfaces.md "최적화 벡터 인코딩".

    x = 2·(θ − lo)/(hi − lo) − 1,   (lo, hi) = scenario.pose_bounds[key]
"""
from __future__ import annotations

from dataclasses import fields

import numpy as np

from airis.sim import PoseParams, Scenario

# PoseParams 필드 순서가 곧 벡터 순서다. types.py 가 단일 소스.
POSE_FIELDS: list[str] = [f.name for f in fields(PoseParams)]

# configs/scenarios.yaml 의 default 에 pose_bounds 항목이 없는 자세 변수의 대체 범위.
# 현재 knee_flexion 하나가 해당한다. configs/ 는 B 소유라 C 가 고칠 수 없어 코드에 둔다.
# TODO(B): configs/scenarios.yaml 의 default.pose_bounds 에 knee_flexion 을 추가하면 이 표를 지운다.
FALLBACK_BOUNDS: dict[str, tuple[float, float]] = {
    "knee_flexion": (0.0, 30.0),
}


class PoseEncoder:
    """시나리오 하나에 대한 자세 ↔ 정규화 벡터 변환기."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.free_keys: list[str] = [k for k in POSE_FIELDS if k not in scenario.fixed_pose]
        self._bounds = np.array(
            [self.bounds_for(k) for k in self.free_keys], dtype=np.float64
        ).reshape(-1, 2)
        span = self._bounds[:, 1] - self._bounds[:, 0]
        if np.any(span <= 0):
            bad = [k for k, s in zip(self.free_keys, span) if s <= 0]
            raise ValueError(f"pose_bounds 의 hi 가 lo 보다 크지 않다: {bad}")
        self._lo = self._bounds[:, 0]
        self._span = span

    @property
    def dim(self) -> int:
        return len(self.free_keys)

    def bounds_for(self, key: str) -> tuple[float, float]:
        """자세 변수 하나의 (lo, hi). 시나리오에 없으면 FALLBACK_BOUNDS."""
        if key in self.scenario.pose_bounds:
            lo, hi = self.scenario.pose_bounds[key]
            return float(lo), float(hi)
        if key in FALLBACK_BOUNDS:
            return FALLBACK_BOUNDS[key]
        raise KeyError(f"{key} 의 pose_bounds 가 시나리오에도 대체 표에도 없다")

    def encode(self, pose: PoseParams) -> np.ndarray:
        """PoseParams → [-1, 1]^dim. 범위 밖 자세는 경계로 clip 된다."""
        theta = np.array([getattr(pose, k) for k in self.free_keys], dtype=np.float64)
        x = 2.0 * (theta - self._lo) / self._span - 1.0
        return np.clip(x, -1.0, 1.0)

    def decode(self, x: np.ndarray) -> PoseParams:
        """[-1, 1]^dim → PoseParams. x 를 먼저 clip 하므로 경계 밖 제안도 안전하다."""
        xc = np.clip(np.asarray(x, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if xc.size != self.dim:
            raise ValueError(f"벡터 길이가 {xc.size}, 기대값은 {self.dim}")
        theta = self._lo + (xc + 1.0) * self._span / 2.0
        values = {f.name: f.default for f in fields(PoseParams)}
        values.update(dict(zip(self.free_keys, (float(t) for t in theta))))
        for k, v in self.scenario.fixed_pose.items():
            if k in values:
                values[k] = float(v)
        return PoseParams(**values)

    def default_x(self) -> np.ndarray:
        """PoseParams() 기본값의 인코딩. CMA-ES 시작점."""
        return self.encode(PoseParams())

    def clip_pose(self, pose: PoseParams) -> PoseParams:
        """자세를 시나리오 제약(범위 + fixed_pose) 안으로 투영한다."""
        return self.decode(self.encode(pose))
