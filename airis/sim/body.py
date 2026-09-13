"""마네킹 기하. 소유자: B.

TODO(B, 1주차):
- BodyParams + PoseParams → 관절 위치 (forward kinematics)
- 관절 위치 → 캡슐 목록
- 캡슐 표면 → 패치 샘플링 (위치, 법선, 면적, 부위)
- wheelchair 시나리오: seat_height와 fixed_pose 반영
"""
from __future__ import annotations

from .types import BodyParams, PoseParams, Scenario, BodyState


def build_body(body: BodyParams, pose: PoseParams, scenario: Scenario,
               patches_per_m2: float = 2000.0) -> BodyState:
    raise NotImplementedError("B: 1주차 구현 대상")
