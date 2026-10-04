"""가까운 학습 체형의 최적 자세를 후보로 돌려주는 표. 소유자: F.

`predict.py` 의 혼합 추천(총괄 2026-09-30: flow 샘플 + kNN 후보 + 고정 후보 → 재채점)에서 kNN 쪽을 맡는다.
학습 데이터셋에서 (체형, 시나리오, 최적 자세) 표를 뽑아 parquet 한 장으로 저장하고, 예측할 때 같은 시나리오
행 중 **표준화한 체형 5개 값의 유클리드 거리**가 가까운 k 개의 자세를 돌려준다.

    표 열: body_idx, scenario, body_*(5), pose_*(7, yaw 는 0~90° 로 접힌 값), score, arm_class
           + nozzle_layout_hash, physics_hash, body_model, patches_per_m2, commit (설정 도장)

표준화(평균·표준편차)는 표 전체에서 한 번 계산한다. 거리 동률은 body_idx 오름차순으로 끊어 같은 입력이면
같은 후보 순서가 나오게 한다. 자세는 항상 `PoseEncoder(scenario).clip_pose()` 를 거쳐 돌려준다
(interfaces.md 회귀 계약: 범위 안 + fixed_pose 적용).

산출물은 `data/models/pose_knn.parquet` (git 밖). `scripts/build_pose_knn.py` 로 만든다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from airis.sim import BodyParams, PoseParams, Scenario

from .flow import BODY_KEYS, POSE_KEYS

#: 학습 데이터셋의 도장 열 (predict.py 가 현재 설정과 비교한다).
STAMP_COLS = ("nozzle_layout_hash", "physics_hash", "body_model", "patches_per_m2", "commit")
#: 도장 중 출처 기록일 뿐 설정이 아닌 열. 여러 값이어도 된다 (데이터셋 생성이 중단·재개되면 HEAD 가 바뀐다).
#: 나머지 도장 열(설정)은 데이터셋에서 한 값이어야 한다.
PROVENANCE_COLS = ("commit",)


def stamp_value(series) -> str:
    """도장 열 → 기록할 값. 한 값이면 그 값, 여러 값이면 많은 순으로 '+' 로 잇는다 (예: '55681cc+8af76a9')."""
    counts = series.astype(str).value_counts()
    return "+".join(counts.index.tolist())


TABLE_COLS = (["body_idx", "scenario"] + [f"body_{k}" for k in BODY_KEYS]
              + [f"pose_{k}" for k in POSE_KEYS] + ["score", "arm_class"] + list(STAMP_COLS))


class PoseKNN:
    """체형 → 가까운 학습 체형들의 최적 자세."""

    def __init__(self, table) -> None:
        missing = [c for c in TABLE_COLS if c not in table.columns and c not in STAMP_COLS]
        if missing:
            raise ValueError(f"kNN 표에 열이 없다: {missing}")
        self.table = table.reset_index(drop=True)
        self.meta = {c: (self.table[c].iloc[0] if c in self.table.columns and len(self.table) else None)
                     for c in STAMP_COLS}
        for c in PROVENANCE_COLS:                # 출처 열은 여러 값일 수 있다 → 전부 적는다
            if c in self.table.columns and len(self.table):
                self.meta[c] = stamp_value(self.table[c])
        cols = [f"body_{k}" for k in BODY_KEYS]
        bodies = self.table[cols].to_numpy(dtype=np.float64)
        self.mean = bodies.mean(axis=0)
        self.std = np.maximum(bodies.std(axis=0), 1e-9)
        self._by_scenario: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for name, g in self.table.groupby("scenario"):
            g = g.sort_values("body_idx", kind="mergesort")
            self._by_scenario[str(name)] = (
                (g[cols].to_numpy(dtype=np.float64) - self.mean) / self.std,
                g[[f"pose_{k}" for k in POSE_KEYS]].to_numpy(dtype=np.float64),
                g["body_idx"].to_numpy(dtype=np.int64),
            )

    @property
    def scenario_names(self) -> list[str]:
        return sorted(self._by_scenario)

    # ---------- 만들기 / 읽기 ----------

    @classmethod
    def from_dataset(cls, df) -> "PoseKNN":
        """데이터셋 DataFrame → 표 (필요한 열만 남긴다)."""
        cols = [c for c in TABLE_COLS if c in df.columns]
        need = [c for c in TABLE_COLS if c not in STAMP_COLS and c not in df.columns]
        if need:
            raise ValueError(f"데이터셋에 열이 없다: {need}")
        return cls(df[cols].copy())

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        self.table.to_parquet(tmp, index=False)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "PoseKNN":
        import pandas as pd

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"kNN 표가 없다: {path} (scripts/build_pose_knn.py 로 만든다)")
        return cls(pd.read_parquet(path))

    # ---------- 후보 ----------

    def candidates(self, body: BodyParams, scenario: Scenario, k: int) -> list[PoseParams]:
        """가까운 학습 체형 k 개의 최적 자세. 학습에 없던 시나리오면 KeyError."""
        from airis.optimize.encoding import PoseEncoder

        if scenario.name not in self._by_scenario:
            raise KeyError(f"kNN 표에 없는 시나리오: {scenario.name} (표: {self.scenario_names})")
        if k <= 0:
            return []
        B, Y, idx = self._by_scenario[scenario.name]
        b = (np.array([getattr(body, key) for key in BODY_KEYS], dtype=np.float64) - self.mean) / self.std
        dist = ((B - b) ** 2).sum(axis=1)
        # 거리 오름차순, 동률은 body_idx 오름차순 (결정론).
        order = np.lexsort((idx, dist))[:int(k)]
        enc = PoseEncoder(scenario)
        return [enc.clip_pose(PoseParams(**dict(zip(POSE_KEYS, map(float, Y[i]))))) for i in order]
