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


# ---------- 계획 (docs/plan_extension.md, interfaces.md "계획 모델") ----------

#: 계획 데이터셋의 도장 열. 자세 도장에 계획 채점 조건을 더한다 (C 와 합의 2026-10-06).
#: 계획 점수는 에너지 가중(energy_weight)과 단계 수(n_phases)에 따라 달라 이 둘이 다르면 행끼리 비교할 수 없다.
#: 데이터셋 생성 설정 (C, PR #159). 후보 선정(개수·점수 허용 폭·최소 거리)과 따뜻한 시작·자세 최적화 예산이 다르면
#: 후보 열과 기준선(pose_*, score_p1opt_10)의 뜻이 달라진다. 지금 설정과 비교할 값은 없고, 섞였는지만 본다.
PLAN_GENERATION_COLS = ("candidate_k", "candidate_tol", "candidate_min_dist", "warm_start_sigma0", "pose_max_evals")
PLAN_STAMP_COLS = STAMP_COLS + ("kinetics_enabled", "time_constant_s", "zone_nozzle_counts", "n_phases",
                                "energy_weight") + PLAN_GENERATION_COLS
#: 계획 한도(C 의 PlanLimits). 데이터셋을 만든 한도로 모델의 출력 범위와 투영을 맞춘다. 있으면 읽는다.
PLAN_LIMIT_COLS = ("duration_lo_s", "duration_hi_s", "min_phase_s", "transition_s", "s_max", "cap_ratio")


class PlanKNN:
    """체형 → 가까운 학습 체형들의 최적 계획. PoseKNN 의 계획 판이다.

        표 열: body_idx, scenario, body_*(5), plan_<키>(원래 단위, PlanSpace.plan_keys 순서), score
               + 도장(PLAN_STAMP_COLS) · 계획 한도(PLAN_LIMIT_COLS) 중 데이터셋에 있는 것

    단계 수 N 은 `plan_p<k>_torso_yaw` 열의 수에서 읽는다 (C 의 설계 결정값, 코드에 고정하지 않는다).
    단계 최소 시간은 표의 `min_phase_s` 열, 없으면 생성자 인자, 그것도 없으면 설정 파일의 값이다.
    거리·동률 규칙은 PoseKNN 과 같다 (표준화한 체형 5개 값의 유클리드 거리, 동률은 body_idx 오름차순).
    돌려주는 계획의 자세는 시나리오 제약으로 투영한다. 시간·세기 범위와 풍량 한도는 predict_plan 이
    PlanEncoder.clip_plan 으로 투영한다.
    """

    def __init__(self, table, *, min_phase_s: float | None = None) -> None:
        from .flow import PlanSpace, plan_phase_count

        self.n_phases = plan_phase_count(table.columns)
        self.keys = PlanSpace.plan_keys(self.n_phases)
        need = (["body_idx", "scenario", "score"] + [f"body_{k}" for k in BODY_KEYS]
                + [f"plan_{k}" for k in self.keys])
        missing = [c for c in need if c not in table.columns]
        if missing:
            raise ValueError(f"계획 kNN 표에 열이 없다: {missing}")
        self.table = table.reset_index(drop=True)
        if "min_phase_s" in self.table.columns and len(self.table):
            self.min_phase_s = float(self.table["min_phase_s"].iloc[0])
        elif min_phase_s is not None:
            self.min_phase_s = float(min_phase_s)
        else:
            from airis.optimize.plan_encoding import PlanLimits
            from airis.sim.scenario import load_physics

            self.min_phase_s = float(PlanLimits.from_config(load_physics()).min_phase_s)
        stamp_cols = [c for c in PLAN_STAMP_COLS + PLAN_LIMIT_COLS if c in self.table.columns]
        self.meta = {c: (stamp_value(self.table[c]) if c in PROVENANCE_COLS else self.table[c].iloc[0])
                     for c in stamp_cols if len(self.table)}
        self.meta.setdefault("n_phases", self.n_phases)
        self.meta.setdefault("min_phase_s", self.min_phase_s)
        cols = [f"body_{k}" for k in BODY_KEYS]
        bodies = self.table[cols].to_numpy(dtype=np.float64)
        self.mean = bodies.mean(axis=0)
        self.std = np.maximum(bodies.std(axis=0), 1e-9)
        self._by_scenario: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for name, g in self.table.groupby("scenario"):
            g = g.sort_values("body_idx", kind="mergesort")
            self._by_scenario[str(name)] = (
                (g[cols].to_numpy(dtype=np.float64) - self.mean) / self.std,
                g[[f"plan_{k}" for k in self.keys]].to_numpy(dtype=np.float64),
                g["body_idx"].to_numpy(dtype=np.int64),
            )

    @property
    def scenario_names(self) -> list[str]:
        return sorted(self._by_scenario)

    @classmethod
    def from_dataset(cls, df, *, min_phase_s: float | None = None) -> "PlanKNN":
        """계획 데이터셋 DataFrame → 표 (필요한 열만 남긴다). 후보 열(cand_*)은 넣지 않는다."""
        from .flow import PlanSpace, plan_phase_count

        keys = PlanSpace.plan_keys(plan_phase_count(df.columns))
        cols = (["body_idx", "scenario"] + [f"body_{k}" for k in BODY_KEYS] + [f"plan_{k}" for k in keys] + ["score"]
                + [c for c in PLAN_STAMP_COLS + PLAN_LIMIT_COLS if c in df.columns])
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"데이터셋에 열이 없다: {missing}")
        return cls(df[cols].copy(), min_phase_s=min_phase_s)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        table = self.table
        if "min_phase_s" not in table.columns:          # 다시 읽을 때 같은 단계 시간이 되게 표에 남긴다
            table = table.assign(min_phase_s=self.min_phase_s)
        table.to_parquet(tmp, index=False)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "PlanKNN":
        import pandas as pd

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"계획 kNN 표가 없다: {path} (scripts/train_plan_models.py 로 만든다)")
        return cls(pd.read_parquet(path))

    def candidates(self, body: BodyParams, scenario: Scenario, k: int) -> list:
        """가까운 학습 체형 k 개의 최적 계획 (Plan 목록). 학습에 없던 시나리오면 KeyError."""
        from .flow import plan_from_raw

        if scenario.name not in self._by_scenario:
            raise KeyError(f"계획 kNN 표에 없는 시나리오: {scenario.name} (표: {self.scenario_names})")
        if k <= 0:
            return []
        B, Y, idx = self._by_scenario[scenario.name]
        b = (np.array([getattr(body, key) for key in BODY_KEYS], dtype=np.float64) - self.mean) / self.std
        dist = ((B - b) ** 2).sum(axis=1)
        order = np.lexsort((idx, dist))[:int(k)]
        return [plan_from_raw(Y[i], scenario, self.n_phases, self.min_phase_s) for i in order]
