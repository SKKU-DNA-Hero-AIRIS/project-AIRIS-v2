"""조건부 flow matching 모델. 소유자: F. 제안: docs/proposals/flow_matching.md, 확장: docs/plan_extension.md.

체형 5개 + 시나리오를 조건으로 p(출력 | 체형, 시나리오)를 학습한다. 출력은 두 가지다.

    PoseSpace   자세 7개 (단일 자세, interfaces.md "회귀 모델")
    PlanSpace   계획 = 자세 순서 K단계 + 총 시간 + 단계 분할 + 구역 세기 5개 (K = 2 면 21차원, "계획 모델")

모델은 출력 길이에 묶이지 않는다. 출력 공간(VectorSpace)이 키 순서·범위·시나리오별 마스크를 정하고, 같은
학습·샘플링 코드가 두 공간을 모두 다룬다. 최적 출력이 봉우리 여러 개(팔 만세 / 팔 내림, yaw 대칭)로 갈리므로
평균을 내는 회귀 대신 분포를 학습하고, 샘플 여러 개를 뽑아 시뮬레이터로 고른다(predict.py).

학습 (conditional flow matching, 직선 경로)

    x_0 ~ 시작 분포,  x_1 = 데이터 출력,  t ~ U(0, 1)
    x_t = (1 − t)·x_0 + t·x_1
    loss = Σ_d m_d·(v_θ(x_t, t, c) − (x_1 − x_0))_d² / Σ_d m_d

    m 은 시나리오가 고정하지 않은 변수 마스크(휠체어 hip·knee = 0), c 는 표준화한 체형 + 시나리오 원핫.
    배치는 (체형, 시나리오) 묶음마다 합이 1 이 되는 가중치로 뽑는다. 후보를 여러 개 저장한 묶음이
    학습을 독차지하지 않게 하기 위해서다.

샘플: x_0 에서 t = 0 → 1 로 v_θ 를 중점법으로 적분하고 [-1, 1] 로 자른다.

자세 공간 (PoseSpace): 시나리오 공통 7차원. 변수마다 학습 시나리오들의 pose_bounds 합집합으로 [-1, 1] 에
정규화한다. torso_yaw 는 데이터셋이 0~90° 로 접어 두므로(e4.fold_yaw, interfaces.md 회귀 계약) 접은 범위
[0, min(90, max|bound|)] 로 정규화한다.

계획 공간 (PlanSpace): 키 순서는 p1_<자세 7개>, …, pK_<자세 7개>, duration_s, share_1 … share_{K−1},
zone_<ZONE_NAMES>. 대칭(좌우 거울, 앞뒤 등가)은 계획 전체에 함께 적용되므로 1단계 yaw 만 0~90° 로 접고
나머지 단계의 yaw 는 시나리오 범위 그대로 둔다. 풍량 한도 보수·쾌적 상한은 C 의 PlanEncoder.clip_plan 몫이고,
여기 to_plan 은 범위 자르기와 단계 시간 분할만 한다 (predict_plan 이 채점 전에 clip_plan 을 거친다).

torch 는 이 모듈의 함수 안에서 import 한다 (패키지 전체가 torch 에 묶이지 않게).
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from airis.sim import ZONE_NAMES, BodyParams, Phase, Plan, PoseParams, Scenario

POSE_KEYS: list[str] = [f.name for f in fields(PoseParams)]
BODY_KEYS: list[str] = [f.name for f in fields(BodyParams)]
YAW_KEY = "torso_yaw"
ARTIFACT_VERSION = 1


def folded_yaw_bounds(lo: float, hi: float) -> tuple[float, float]:
    """yaw 범위 [lo, hi] 가 fold_yaw(90° − |90° − |θ||) 뒤에 차지하는 범위."""
    return 0.0, min(90.0, max(abs(float(lo)), abs(float(hi))))


# ---------- 출력 공간 ----------

class VectorSpace:
    """임의 길이 출력 벡터 ↔ [-1, 1] 변환. keys 순서가 벡터 순서다.

    하위 클래스가 정하는 것: 데이터셋 열 접두사(column_prefix), 시나리오가 고정하는 키(fixed_keys),
    벡터 → 객체 변환(to_object).
    """

    kind = "vector"
    column_prefix = "x_"

    def __init__(self, bounds: dict[str, tuple[float, float]], keys: Sequence[str] | None = None) -> None:
        self.keys = list(keys) if keys is not None else list(bounds)
        missing = [k for k in self.keys if k not in bounds]
        if missing:
            raise KeyError(f"출력 공간 범위가 없는 키: {missing}")
        arr = np.array([bounds[k] for k in self.keys], dtype=np.float64).reshape(-1, 2)
        span = arr[:, 1] - arr[:, 0]
        if np.any(span <= 0):
            bad = [k for k, s in zip(self.keys, span) if s <= 0]
            raise ValueError(f"출력 공간 범위의 hi 가 lo 보다 크지 않다: {bad}")
        self._lo, self._span = arr[:, 0], span

    @property
    def dim(self) -> int:
        return len(self.keys)

    def to_dict(self) -> dict[str, list[float]]:
        return {k: [float(lo), float(lo + s)] for k, lo, s in zip(self.keys, self._lo, self._span)}

    def extra(self) -> dict:
        """산출물에 함께 저장할 공간 설정 (기본 타입만)."""
        return {}

    @classmethod
    def from_dict(cls, d: dict, extra: dict | None = None) -> "VectorSpace":
        return cls({k: (float(v[0]), float(v[1])) for k, v in d.items()}, keys=list(d))

    def fixed_keys(self, scenario: Scenario) -> set[str]:
        """시나리오가 값을 고정해 학습·샘플링에서 빼는 키."""
        return set()

    def mask(self, scenario: Scenario) -> np.ndarray:
        """시나리오가 고정하지 않은 변수 1, 고정한 변수 0. (dim,) float32."""
        fixed = self.fixed_keys(scenario)
        return np.array([0.0 if k in fixed else 1.0 for k in self.keys], dtype=np.float32)

    def encode(self, raw: np.ndarray) -> np.ndarray:
        """(N, dim) 원래 단위 → (N, dim) [-1, 1]. 범위 밖은 자른다."""
        raw = np.atleast_2d(np.asarray(raw, dtype=np.float64))
        return np.clip(2.0 * (raw - self._lo) / self._span - 1.0, -1.0, 1.0)

    def decode(self, x: np.ndarray) -> np.ndarray:
        """(N, dim) [-1, 1] → (N, dim) 원래 단위."""
        x = np.clip(np.atleast_2d(np.asarray(x, dtype=np.float64)), -1.0, 1.0)
        return self._lo + (x + 1.0) * self._span / 2.0

    def to_object(self, x: np.ndarray, scenario: Scenario):
        """정규화 벡터 하나 → 평가기에 넣을 객체. 기본은 원래 단위 벡터."""
        return self.decode(x)[0]


def _pose_bounds_union(scenarios: Sequence[Scenario], key: str, *, fold_yaw: bool) -> tuple[float, float]:
    los, his = [], []
    for s in scenarios:
        if key in s.fixed_pose:
            continue
        lo, hi = s.pose_bounds[key]
        if key == YAW_KEY and fold_yaw:
            lo, hi = folded_yaw_bounds(lo, hi)
        los.append(float(lo))
        his.append(float(hi))
    if not los:        # 모든 시나리오가 고정한 변수: 마스크로 빠지므로 폭은 아무래도 된다
        v = float(scenarios[0].fixed_pose[key])
        los, his = [v - 1.0], [v + 1.0]
    return min(los), max(his)


class PoseSpace(VectorSpace):
    """시나리오 공통 7차원 자세 ↔ [-1, 1] 변환. 순서는 PoseParams 필드 순서."""

    kind = "pose"
    column_prefix = "pose_"

    def __init__(self, bounds: dict[str, tuple[float, float]], keys: Sequence[str] | None = None) -> None:
        super().__init__(bounds, POSE_KEYS)

    @classmethod
    def from_scenarios(cls, scenarios: Sequence[Scenario]) -> "PoseSpace":
        return cls({k: _pose_bounds_union(scenarios, k, fold_yaw=True) for k in POSE_KEYS})

    @classmethod
    def from_dict(cls, d: dict, extra: dict | None = None) -> "PoseSpace":
        return cls({k: (float(v[0]), float(v[1])) for k, v in d.items()})

    def fixed_keys(self, scenario: Scenario) -> set[str]:
        return set(scenario.fixed_pose)

    def to_pose(self, x: np.ndarray, scenario: Scenario) -> PoseParams:
        """정규화 벡터 하나 → 시나리오 제약(범위 + fixed_pose) 안의 PoseParams (interfaces.md 회귀 계약)."""
        from airis.optimize.encoding import PoseEncoder

        deg = self.decode(x)[0]
        return PoseEncoder(scenario).clip_pose(PoseParams(**dict(zip(self.keys, map(float, deg)))))

    def to_object(self, x: np.ndarray, scenario: Scenario) -> PoseParams:
        return self.to_pose(x, scenario)


class PlanSpace(VectorSpace):
    """계획(자세 순서 + 시간 + 구역 세기) ↔ [-1, 1] 변환. docs/plan_extension.md 2절.

    share_k 는 1단계부터 K−1단계까지의 시간 몫(0~1)이고 마지막 단계가 나머지를 갖는다. 단계 시간은
    min_phase_s + (T − K·min_phase_s)·몫 이라 어떤 벡터든 단계 최소 시간을 지킨다.
    """

    kind = "plan"
    column_prefix = "plan_"

    def __init__(self, bounds: dict[str, tuple[float, float]], *, n_phases: int = 2,
                 min_phase_s: float = 2.0, keys: Sequence[str] | None = None) -> None:
        if n_phases < 1:
            raise ValueError("n_phases 는 1 이상")
        self.n_phases = int(n_phases)
        self.min_phase_s = float(min_phase_s)
        super().__init__(bounds, self.plan_keys(self.n_phases))
        lo_t = float(self._lo[self.keys.index("duration_s")])
        if lo_t < self.n_phases * self.min_phase_s:
            raise ValueError(f"총 시간 하한 {lo_t} s 가 단계 {self.n_phases}개 × 최소 {self.min_phase_s} s 보다 짧다")

    @staticmethod
    def plan_keys(n_phases: int) -> list[str]:
        return ([f"p{k}_{p}" for k in range(1, n_phases + 1) for p in POSE_KEYS]
                + ["duration_s"]
                + [f"share_{k}" for k in range(1, n_phases)]
                + [f"zone_{z}" for z in ZONE_NAMES])

    @classmethod
    def from_scenarios(cls, scenarios: Sequence[Scenario], *, n_phases: int = 2,
                       duration_bounds_s: tuple[float, float] = (5.0, 20.0), s_max: float = 1.0,
                       min_phase_s: float = 2.0) -> "PlanSpace":
        bounds: dict[str, tuple[float, float]] = {}
        for k in range(1, n_phases + 1):
            for p in POSE_KEYS:
                bounds[f"p{k}_{p}"] = _pose_bounds_union(scenarios, p, fold_yaw=(k == 1))
        bounds["duration_s"] = (float(duration_bounds_s[0]), float(duration_bounds_s[1]))
        for k in range(1, n_phases):
            bounds[f"share_{k}"] = (0.0, 1.0)
        for z in ZONE_NAMES:
            bounds[f"zone_{z}"] = (0.0, float(s_max))
        return cls(bounds, n_phases=n_phases, min_phase_s=min_phase_s)

    @classmethod
    def from_config(cls, scenarios: Sequence[Scenario], physics_cfg: dict) -> "PlanSpace":
        """configs/physics.yaml 의 plan.* · fan.s_max (docs/plan_extension.md 4절). 없는 키는 문서의 작업값."""
        plan = physics_cfg.get("plan", {}) or {}
        fan = physics_cfg.get("fan", {}) or {}
        return cls.from_scenarios(
            scenarios,
            n_phases=int(plan.get("n_phases", 2)),
            duration_bounds_s=tuple(plan.get("duration_bounds_s", (5.0, 20.0))),
            s_max=float(fan.get("s_max", 1.0)),
            min_phase_s=float(plan.get("min_phase_s", 2.0)),
        )

    def extra(self) -> dict:
        return {"n_phases": self.n_phases, "min_phase_s": self.min_phase_s}

    @classmethod
    def from_dict(cls, d: dict, extra: dict | None = None) -> "PlanSpace":
        extra = extra or {}
        return cls({k: (float(v[0]), float(v[1])) for k, v in d.items()},
                   n_phases=int(extra.get("n_phases", 2)), min_phase_s=float(extra.get("min_phase_s", 2.0)))

    def fixed_keys(self, scenario: Scenario) -> set[str]:
        return {f"p{k}_{p}" for k in range(1, self.n_phases + 1) for p in scenario.fixed_pose}

    def durations(self, total_s: float, shares: Sequence[float]) -> list[float]:
        """총 시간과 몫 K−1 개 → 단계 시간 K 개 (각각 min_phase_s 이상, 합 = total_s)."""
        return plan_durations(total_s, shares, self.n_phases, self.min_phase_s)

    def to_plan(self, x: np.ndarray, scenario: Scenario) -> Plan:
        """정규화 벡터 하나 → 범위 안의 Plan. 자세는 시나리오 제약(범위 + fixed_pose)으로 투영한다."""
        return plan_from_raw(self.decode(x)[0], scenario, self.n_phases, self.min_phase_s)

    def from_plan(self, plan: Plan) -> np.ndarray:
        """Plan → 원래 단위 벡터 (dim,). 데이터셋 plan_* 열과 테스트용. to_plan 의 역."""
        if len(plan.phases) != self.n_phases:
            raise ValueError(f"단계 수 {len(plan.phases)} (공간은 {self.n_phases})")
        total = plan.duration_s
        free = total - self.n_phases * self.min_phase_s
        raw: dict[str, float] = {}
        for k, ph in enumerate(plan.phases, start=1):
            for p in POSE_KEYS:
                raw[f"p{k}_{p}"] = float(getattr(ph.pose, p))
        raw["duration_s"] = total
        for k in range(1, self.n_phases):
            t = plan.phases[k - 1].duration_s
            raw[f"share_{k}"] = (t - self.min_phase_s) / free if free > 0 else 1.0 / self.n_phases
        for z, s in zip(ZONE_NAMES, np.asarray(plan.zone_strengths, dtype=np.float64)):
            raw[f"zone_{z}"] = float(s)
        return np.array([raw[k] for k in self.keys], dtype=np.float64)

    def to_object(self, x: np.ndarray, scenario: Scenario) -> Plan:
        return self.to_plan(x, scenario)


SPACE_KINDS: dict[str, type[VectorSpace]] = {"vector": VectorSpace, "pose": PoseSpace, "plan": PlanSpace}


def plan_phase_count(columns, prefix: str = "plan_") -> int:
    """열 이름에서 계획의 단계 수 N 을 읽는다 (`<prefix>p<k>_torso_yaw` 가 있는 k 의 개수).

    계획 데이터셋의 단계 수는 C 의 설계 결정값이고 열 수로 고정된다. 모델은 이 값을 데이터에서 읽는다.
    열이 p1 … pN 로 이어지지 않으면 ValueError.
    """
    ks = sorted(int(c[len(prefix) + 1:].split("_", 1)[0]) for c in columns
                if c.startswith(prefix + "p") and c.endswith("_" + YAW_KEY)
                and c[len(prefix) + 1:].split("_", 1)[0].isdigit())
    if not ks or ks != list(range(1, len(ks) + 1)):
        raise ValueError(f"계획 열({prefix}p<k>_…)에서 단계 수를 읽을 수 없다: 찾은 단계 {ks}")
    return len(ks)


def plan_durations(total_s: float, shares: Sequence[float], n_phases: int, min_phase_s: float) -> list[float]:
    """총 시간과 몫 N−1 개 → 단계 시간 N 개 (각각 min_phase_s 이상, 합 = total_s). PlanEncoder.durations 와 같은 식."""
    r = [min(1.0, max(0.0, float(s))) for s in shares]
    r.append(max(0.0, 1.0 - sum(r)))
    tot = sum(r)
    r = [v / tot for v in r] if tot > 0 else [1.0 / n_phases] * n_phases
    free = max(0.0, float(total_s) - n_phases * min_phase_s)
    return [min_phase_s + free * v for v in r]


def plan_from_raw(raw, scenario: Scenario, n_phases: int, min_phase_s: float) -> Plan:
    """원래 단위 계획 벡터(PlanSpace.plan_keys 순서) → Plan. 자세는 시나리오 제약(범위 + fixed_pose)으로 투영한다.

    데이터셋의 plan_* 행과 kNN 표의 행을 계획으로 되돌릴 때 쓴다. 시간·세기 범위와 풍량 한도는 여기서 다루지
    않는다 (predict_plan 이 PlanEncoder.clip_plan 으로 투영한다).
    """
    from airis.optimize.encoding import PoseEncoder

    keys = PlanSpace.plan_keys(n_phases)
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.size != len(keys):
        raise ValueError(f"계획 벡터 길이 {values.size} (단계 {n_phases}개면 {len(keys)})")
    v = dict(zip(keys, map(float, values)))
    enc = PoseEncoder(scenario)
    times = plan_durations(v["duration_s"], [v[f"share_{k}"] for k in range(1, n_phases)], n_phases, min_phase_s)
    phases = [Phase(enc.clip_pose(PoseParams(**{p: v[f"p{k}_{p}"] for p in POSE_KEYS})), t)
              for k, t in zip(range(1, n_phases + 1), times)]
    return Plan(phases, np.array([v[f"zone_{z}"] for z in ZONE_NAMES], dtype=np.float64))


# ---------- 학습 데이터 ----------

def _cand_cols() -> list[str]:
    return [f"cand_pose_{k}" for k in POSE_KEYS]


def _has_candidates(row, col: str = "cand_pose_shoulder_abduction") -> bool:
    v = row.get(col) if hasattr(row, "get") else None
    return v is not None and not (isinstance(v, float) and math.isnan(v)) and len(v) > 0


def body_from_row(row) -> BodyParams:
    """데이터셋 행의 body_* 열 → BodyParams."""
    return BodyParams(**{k: float(row[f"body_{k}"]) for k in BODY_KEYS})


def candidate_matrix(value, dim: int) -> np.ndarray:
    """후보 벡터 목록 열의 한 칸 → (후보 수, dim) 배열.

    parquet 에서 읽은 2차원 목록(배열의 배열)과 행 우선으로 편 1차원 목록을 모두 받는다.
    길이가 dim 의 배수가 아니면 ValueError (단계 수가 다른 데이터가 섞였다는 뜻이다).
    """
    rows = [np.asarray(v, dtype=np.float64).reshape(-1) for v in value] if len(value) and np.ndim(value[0]) else None
    flat = np.concatenate(rows) if rows is not None else np.asarray(value, dtype=np.float64).reshape(-1)
    if flat.size == 0 or flat.size % dim:
        raise ValueError(f"후보 벡터 길이 {flat.size} 가 출력 차원 {dim} 의 배수가 아니다")
    if rows is not None and any(r.size != dim for r in rows):
        raise ValueError(f"후보 벡터 길이가 출력 차원 {dim} 과 다르다: {sorted({r.size for r in rows})}")
    return flat.reshape(-1, dim)


def training_arrays(df, space: VectorSpace, scenarios: dict[str, Scenario]) -> dict[str, np.ndarray]:
    """데이터셋 DataFrame → 학습 배열.

    열 이름은 공간이 정한다: `<column_prefix><key>` (pose_* / plan_*). 후보 목록은 두 형태 중 하나다.
    - `cand_<column_prefix>raw`: 후보마다 원래 단위 벡터(space.keys 순서)를 모은 목록 열. (후보 수 × dim) 2차원,
      또는 행 우선으로 편 1차원. 계획 데이터셋이 쓴다 (C 와 합의 2026-10-06: N = 9 면 키가 77개라 키마다 열을
      두지 않는다. 대칭 정규화를 거친 PlanEncoder.raw_vector).
    - `cand_<column_prefix><key>`: 키마다 목록 열. 자세 데이터셋이 쓴다.
    행에 후보 목록(candidate_k > 0)이 있으면 그 후보들을, 없으면 최적 출력 하나를 쓴다.
    반환: body (N,5), scenario (N,) str, x (N,dim), mask (N,dim), weight (N,) (묶음마다 합 1), group (N,) int
    """
    cols = [f"{space.column_prefix}{k}" for k in space.keys]
    cand_cols = [f"cand_{c}" for c in cols]
    raw_col = f"cand_{space.column_prefix}raw"
    body, scen, raw, group = [], [], [], []
    for g, (_, row) in enumerate(df.iterrows()):
        b = [float(row[f"body_{k}"]) for k in BODY_KEYS]
        if _has_candidates(row, raw_col):
            cand = candidate_matrix(row[raw_col], space.dim)
        elif _has_candidates(row, cand_cols[0]):
            cand = np.stack([np.asarray(row[c], dtype=np.float64) for c in cand_cols], axis=1)
        else:
            cand = np.array([[float(row[c]) for c in cols]])
        for d in cand:
            body.append(b)
            scen.append(str(row["scenario"]))
            raw.append(d)
            group.append(g)
    group_arr = np.array(group, dtype=np.int64)
    counts = np.bincount(group_arr)
    missing = sorted(set(scen) - set(scenarios))
    if missing:
        raise KeyError(f"시나리오 설정에 없는 이름: {missing}")
    return {
        "body": np.array(body, dtype=np.float32).reshape(-1, len(BODY_KEYS)),
        "scenario": np.array(scen),
        "x": space.encode(np.array(raw)).astype(np.float32),
        "mask": np.stack([space.mask(scenarios[s]) for s in scen]).astype(np.float32),
        "weight": (1.0 / counts[group_arr]).astype(np.float64),
        "group": group_arr,
    }


# ---------- 모델 ----------

@dataclass
class FlowConfig:
    hidden: int = 128
    layers: int = 3
    time_freqs: int = 8            # 시각 t 의 사인 임베딩 주파수 수
    steps: int = 4000              # 학습 스텝 (미니배치 수)
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    source: str = "gaussian"       # "gaussian": N(0, I) | "default_pose": 기본 자세(PoseParams()) + 노이즈
    source_sigma: float = 1.0
    ode_steps: int = 16            # 샘플링 적분 스텝 (중점법)
    seed: int = 0
    device: str = "cpu"            # 학습 장치 "cpu" | "cuda" | "auto". 샘플링은 항상 CPU


def _make_net(in_dim: int, out_dim: int, hidden: int, layers: int):
    import torch.nn as nn

    mods: list = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


@dataclass
class PoseFlow:
    """조건부 flow matching 모델. 이름은 처음 쓰임(자세)에서 왔고, 출력 길이는 space 가 정한다."""
    space: VectorSpace
    scenario_names: list[str]
    cond_mean: np.ndarray            # (5,) 체형 표준화
    cond_std: np.ndarray             # (5,)
    source_mean: dict[str, list[float]]  # 시나리오별 시작 분포 중심 (gaussian 이면 0)
    cfg: FlowConfig
    net: object = None
    meta: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    @property
    def out_dim(self) -> int:
        return self.space.dim

    @property
    def in_dim(self) -> int:
        return self.out_dim + 1 + 2 * self.cfg.time_freqs + len(BODY_KEYS) + len(self.scenario_names)

    def build_net(self):
        self.net = _make_net(self.in_dim, self.out_dim, self.cfg.hidden, self.cfg.layers)
        return self.net

    # --- 텐서 도우미 ---
    def _cond(self, body: np.ndarray, scen: Sequence[str]):
        import torch

        unknown = sorted(set(scen) - set(self.scenario_names))
        if unknown:
            raise KeyError(f"학습에 없던 시나리오: {unknown} (학습: {self.scenario_names})")
        b = (np.asarray(body, dtype=np.float32) - self.cond_mean) / self.cond_std
        onehot = np.zeros((len(scen), len(self.scenario_names)), dtype=np.float32)
        onehot[np.arange(len(scen)), [self.scenario_names.index(s) for s in scen]] = 1.0
        return torch.from_numpy(np.concatenate([b, onehot], axis=1).astype(np.float32))

    def _time_embed(self, t):
        import torch

        k = torch.arange(self.cfg.time_freqs, dtype=t.dtype, device=t.device)
        ang = t * (2.0 ** k) * math.pi
        return torch.cat([t, torch.sin(ang), torch.cos(ang)], dim=1)

    def velocity(self, x, t, c):
        import torch

        return self.net(torch.cat([x, self._time_embed(t), c], dim=1))

    def _source_center(self, scen: Sequence[str]):
        import torch

        return torch.tensor(np.array([self.source_mean[s] for s in scen], dtype=np.float32))

    # --- 샘플링 ---
    def sample(self, body: BodyParams, scenario_name: str, n: int, *, seed: int = 0) -> np.ndarray:
        """체형 하나 × 시나리오 하나에서 정규화 출력 n 개. (n, dim), 같은 seed 면 같은 결과."""
        import torch

        bvec = np.array([[getattr(body, k) for k in BODY_KEYS]] * n, dtype=np.float32)
        scen = [scenario_name] * n
        c = self._cond(bvec, scen)
        g = torch.Generator().manual_seed(int(seed))
        x = self._source_center(scen) + self.cfg.source_sigma * torch.randn(n, self.out_dim, generator=g)
        steps = max(1, int(self.cfg.ode_steps))
        dt = 1.0 / steps
        self.net.eval()
        with torch.no_grad():
            for i in range(steps):
                t = torch.full((n, 1), i * dt)
                v1 = self.velocity(x, t, c)
                v2 = self.velocity(x + 0.5 * dt * v1, t + 0.5 * dt, c)
                x = x + dt * v2
        return x.clamp(-1.0, 1.0).numpy().astype(np.float64)

    def sample_objects(self, body: BodyParams, scenario: Scenario, n: int, *, seed: int = 0) -> list:
        """sample → 공간이 정한 객체 목록 (PoseSpace 면 PoseParams, PlanSpace 면 Plan)."""
        return [self.space.to_object(x, scenario) for x in self.sample(body, scenario.name, n, seed=seed)]

    def sample_poses(self, body: BodyParams, scenario: Scenario, n: int, *, seed: int = 0) -> list[PoseParams]:
        """sample → 시나리오 제약 안의 PoseParams 목록. 자세 모델 전용."""
        if not isinstance(self.space, PoseSpace):
            raise TypeError(f"자세 모델이 아니다 (space.kind = {self.space.kind})")
        return self.sample_objects(body, scenario, n, seed=seed)

    def sample_plans(self, body: BodyParams, scenario: Scenario, n: int, *, seed: int = 0) -> list[Plan]:
        """sample → 범위 안의 Plan 목록. 계획 모델 전용."""
        if not isinstance(self.space, PlanSpace):
            raise TypeError(f"계획 모델이 아니다 (space.kind = {self.space.kind})")
        return self.sample_objects(body, scenario, n, seed=seed)

    # --- 저장 ---
    def save(self, path: Path | str) -> Path:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "version": ARTIFACT_VERSION,
            "config": asdict(self.cfg),
            "space_kind": self.space.kind,
            "space": self.space.to_dict(),
            "space_extra": json.loads(json.dumps(self.space.extra())),
            "scenario_names": list(self.scenario_names),
            "cond_mean": [float(v) for v in self.cond_mean],
            "cond_std": [float(v) for v in self.cond_std],
            "source_mean": {k: [float(v) for v in vs] for k, vs in self.source_mean.items()},
            "meta": json.loads(json.dumps(self.meta, default=str)),   # weights_only 로드가 되게 기본 타입만
            "history": json.loads(json.dumps(self.history, default=float)),
            "state_dict": self.net.state_dict(),
        }, path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "PoseFlow":
        import torch

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"모델 산출물이 없다: {path}")
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck.get("version") != ARTIFACT_VERSION:
            raise ValueError(f"산출물 버전 {ck.get('version')} (코드 {ARTIFACT_VERSION})")
        kind = ck.get("space_kind", "pose")          # space_kind 이전 산출물은 자세 모델
        if kind not in SPACE_KINDS:
            raise ValueError(f"알 수 없는 출력 공간 {kind!r} (가능: {sorted(SPACE_KINDS)})")
        model = cls(
            space=SPACE_KINDS[kind].from_dict(ck["space"], ck.get("space_extra") or {}),
            scenario_names=list(ck["scenario_names"]),
            cond_mean=np.array(ck["cond_mean"], dtype=np.float32),
            cond_std=np.array(ck["cond_std"], dtype=np.float32),
            source_mean={k: list(v) for k, v in ck["source_mean"].items()},
            cfg=FlowConfig(**ck["config"]),
            meta=dict(ck.get("meta", {})),
            history=list(ck.get("history", [])),
        )
        model.build_net().load_state_dict(ck["state_dict"])
        model.net.eval()
        return model


def _resolve_device(name: str) -> str:
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def train_flow(df, scenarios: dict[str, Scenario], cfg: FlowConfig = FlowConfig(), *,
               space: VectorSpace | None = None, meta: dict | None = None, log=None) -> PoseFlow:
    """데이터셋 DataFrame 으로 모델을 학습한다. 시나리오 목록은 df 의 scenario 열에서 정한다.

    space 를 주지 않으면 자세 공간(PoseSpace). 계획 모델은 PlanSpace 를 넘기고 데이터셋에 plan_* 열이 있어야 한다.
    """
    import torch

    if cfg.source not in ("gaussian", "default_pose"):
        raise ValueError(f"source 는 gaussian | default_pose: {cfg.source!r}")
    names = sorted(set(df["scenario"].astype(str)))
    if space is None:
        space = PoseSpace.from_scenarios([scenarios[n] for n in names])
    arr = training_arrays(df, space, scenarios)
    if cfg.source == "default_pose":
        if not isinstance(space, PoseSpace):
            raise ValueError("source = default_pose 는 자세 공간에서만 쓴다")
        from airis.optimize.e4 import fold_pose
        from airis.optimize.encoding import PoseEncoder

        source_mean = {}
        for n in names:
            d = fold_pose(PoseEncoder(scenarios[n]).clip_pose(PoseParams()))
            source_mean[n] = space.encode(np.array([[d[k] for k in POSE_KEYS]]))[0].tolist()
    else:
        source_mean = {n: [0.0] * space.dim for n in names}

    model = PoseFlow(
        space=space, scenario_names=names,
        cond_mean=arr["body"].mean(axis=0),
        cond_std=np.maximum(arr["body"].std(axis=0), 1e-6).astype(np.float32),
        source_mean=source_mean, cfg=cfg, meta=dict(meta or {}),
    )
    torch.manual_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    net = model.build_net().to(device)

    x1_all = torch.from_numpy(arr["x"]).to(device)
    m_all = torch.from_numpy(arr["mask"]).to(device)
    c_all = model._cond(arr["body"], arr["scenario"]).to(device)
    x0c_all = model._source_center(arr["scenario"]).to(device)
    probs = torch.from_numpy(arr["weight"] / arr["weight"].sum()).float().to(device)
    g = torch.Generator(device=device).manual_seed(cfg.seed)

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    net.train()
    running = 0.0
    for step in range(1, cfg.steps + 1):
        idx = torch.multinomial(probs, cfg.batch_size, replacement=True, generator=g)
        x1, m, c = x1_all[idx], m_all[idx], c_all[idx]
        x0 = x0c_all[idx] + cfg.source_sigma * torch.randn(x1.shape, generator=g, device=device)
        t = torch.rand((x1.shape[0], 1), generator=g, device=device)
        xt = (1.0 - t) * x0 + t * x1
        err = (model.velocity(xt, t, c) - (x1 - x0)) ** 2
        loss = ((err * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        running += loss.item()
        if step % 100 == 0 or step == cfg.steps:
            n = 100 if step % 100 == 0 else step % 100
            model.history.append({"step": step, "loss": running / n})
            if log is not None:
                log(f"[flow] step {step}/{cfg.steps}  loss {running / n:.4f}")
            running = 0.0

    model.net = net.to("cpu").eval()
    model.meta.setdefault("space_kind", space.kind)
    model.meta.setdefault("n_rows", int(len(df)))
    model.meta.setdefault("n_train_points", int(len(arr["x"])))
    model.meta["device"] = str(device)          # 실제로 학습한 장치 (auto 가 무엇으로 풀렸는지, 재현성 기록)
    return model


def train_pose_flow(df, scenarios: dict[str, Scenario], cfg: FlowConfig = FlowConfig(), *,
                    meta: dict | None = None, log=None) -> PoseFlow:
    """자세 모델 학습 (train_flow 의 자세 공간 버전)."""
    return train_flow(df, scenarios, cfg, meta=meta, log=log)


def flow_loss(model: PoseFlow, df, scenarios: dict[str, Scenario], *, n_t: int = 8, seed: int = 0) -> float:
    """검증용 flow matching 손실 (t 와 x_0 를 n_t 번 뽑아 평균). 학습 손실과 같은 식."""
    import torch

    arr = training_arrays(df, model.space, scenarios)
    g = torch.Generator().manual_seed(seed)
    x1 = torch.from_numpy(arr["x"])
    m = torch.from_numpy(arr["mask"])
    c = model._cond(arr["body"], arr["scenario"])
    x0c = model._source_center(arr["scenario"])
    w = torch.from_numpy(arr["weight"]).float()
    total = 0.0
    with torch.no_grad():
        for _ in range(n_t):
            x0 = x0c + model.cfg.source_sigma * torch.randn(x1.shape, generator=g)
            t = torch.rand((x1.shape[0], 1), generator=g)
            err = (model.velocity((1 - t) * x0 + t * x1, t, c) - (x1 - x0)) ** 2
            per = (err * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            total += float((per * w).sum() / w.sum())
    return total / n_t
