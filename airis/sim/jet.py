"""제트 속도장. 소유자: B.

TODO(B, 1~2주차):
- 자유 제트: 포텐셜 코어 + 1/x 감쇠 + 가우시안 단면
- 충돌 보정: 표면 근처에서 벽면 전단 분포로 치환 (정체점 문제 해결)
- 펄스: 시간에 따른 on/off 또는 사인 변조
- 입력: 점 (P, 3), 노즐 설정, 시각 t → 속도 (P, 3)
"""
from __future__ import annotations

import numpy as np

from .types import NozzleConfig


def velocity_field(points: np.ndarray, nozzle: NozzleConfig, t: float,
                   cfg: dict) -> np.ndarray:
    raise NotImplementedError("B: 1주차 구현 대상")
