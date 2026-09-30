"""계획(자세 순서 + 시간 + 구역 세기) ↔ 최적화 벡터. 소유자: C.

규약: `docs/plan_extension.md` 2·4절, `docs/tracks/00_common.md` 4.7, `docs/interfaces.md`(키 순서).
K = 2 면 21차원이다.

    키 순서  p1_<자세 7개>, p2_<자세 7개>, duration_s, share_1, zone_<ZONE_NAMES 5개>
    정규화   x = 2·(v − lo)/(hi − lo) − 1,  decode 는 x 를 [-1, 1] 로 clip 한 뒤 되돌린다
    단계 시간 t_k = min_phase_s + (T − K·min_phase_s)·몫_k,  마지막 단계가 나머지 몫
             (몫은 0~1 로 자른 뒤 합이 1 이 되게 정규화한다. 어떤 벡터든 단계 최소 시간을 지킨다.)

`clip_plan` 이 계획을 실행 가능한 범위로 투영한다 (평가 전에 반드시 거친다).

1. 자세: 시나리오 `pose_bounds` + `fixed_pose` (`PoseEncoder.clip_pose`)
2. 시간: `plan.duration_bounds_s`, 단계마다 `plan.min_phase_s` 이상
3. 구역 세기: 0 ≤ s_z ≤ `fan.s_max`
4. 쾌적 상한: `airis.sim.scenario.zone_strength_caps(scenario)` (임산부 chest_low·chest_high ≤ 0.6)
5. 풍량 한도 보수: Σ_노즐 s ≤ `fan.cap_ratio` · M 을 넘으면 **전 구역을 같은 비율로** 줄인다.
   구역별 노즐 수는 B 의 `zone_nozzle_counts()` (기준 배치 [2, 2, 2, 2, 4], top 은 상단 바 4개)

`normalize` 는 같은 점수인 계획을 하나로 모은다 (00_common 4.7).

- 좌우 거울(전 단계 yaw → −yaw)과 앞뒤 등가(전 단계 yaw → 180° − yaw)를 **계획 전체에 함께** 적용해
  1단계 yaw 가 0~90° 가 되게 고른다. 단계마다 따로 접으면 "1단계 왼쪽 벽, 2단계 오른쪽 벽" 계획이 사라진다.
- 변환 뒤 yaw 는 (−180°, 180°] 로 감고, **전 단계 yaw 가 시나리오 범위 안일 때만** 그 변환을 쓴다
  (휠체어 ±45° 는 앞뒤 등가를 못 쓴다).
- 구역 세기는 몸 기준이라 두 변환에서 바뀌지 않는다. 다만 1단계가 정면·후면(|sin yaw| < 1e−6)이면
  가슴 쪽 벽이 +y 로 고정돼 좌우를 맞바꾼 계획이 같은 점수이므로, chest 합 ≥ back 합이 되게 맞바꾼다.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np

from airis.sim import ZONE_NAMES, Phase, Plan, PoseParams, Scenario
from airis.sim.scenario import zone_nozzle_counts as _load_zone_counts
from airis.sim.scenario import zone_strength_caps

from .e4 import fold_yaw
from .encoding import PoseEncoder

#: 계획 평가에서 켜야 하는 설정 (docs/plan_extension.md 4절: "false → 확장 실험에서 true").
KINETICS_KEY = "kinetics"

POSE_KEYS: list[str] = [f.name for f in fields(PoseParams)]
YAW_KEY = "torso_yaw"
FRONT_EPS = 1e-6




@dataclass(frozen=True)
class PlanLimits:
    """계획 범위. 값은 `configs/physics.yaml` 의 plan.* · fan.* (없으면 docs/plan_extension.md 4절 작업값)."""
    n_phases: int = 2
    duration_bounds_s: tuple[float, float] = (5.0, 20.0)
    min_phase_s: float = 2.0
    transition_s: float = 1.5
    s_max: float = 1.0
    cap_ratio: float = 1.0

    @classmethod
    def from_config(cls, physics_cfg: dict | None = None) -> "PlanLimits":
        cfg = physics_cfg if physics_cfg is not None else {}
        plan = cfg.get("plan", {}) or {}
        fan = cfg.get("fan", {}) or {}
        bounds = tuple(plan.get("duration_bounds_s", (5.0, 20.0)))
        return cls(
            n_phases=int(plan.get("n_phases", 2)),
            duration_bounds_s=(float(bounds[0]), float(bounds[1])),
            min_phase_s=float(plan.get("min_phase_s", 2.0)),
            transition_s=float(plan.get("transition_s", 1.5)),
            s_max=float(fan.get("s_max", 1.0)),
            cap_ratio=float(fan.get("cap_ratio", 1.0)),
        )


def _default_zone_counts() -> np.ndarray:
    """기준 노즐 배치의 구역별 노즐 수. 배치 파일이 없으면 구역마다 1개로 본다.

    설정 오류(좌우 노즐 수 불일치 등)는 조용히 넘기지 않고 경고를 낸 뒤 폴백한다.
    """
    try:
        return np.asarray(_load_zone_counts(), dtype=np.float64)
    except FileNotFoundError:               # 노즐 설정이 없는 단위 테스트 등
        return np.ones(len(ZONE_NAMES), dtype=np.float64)
    except Exception as exc:                # 구역 정의·배치가 어긋난 설정
        warnings.warn(f"구역별 노즐 수를 읽지 못해 1 로 본다: {exc}", RuntimeWarning)
        return np.ones(len(ZONE_NAMES), dtype=np.float64)


def plan_physics_cfg(physics_cfg: dict | None = None) -> dict:
    """계획 평가용 physics 설정 (복사본). `adhesion.kinetics.enabled` 를 켠다.

    설정 파일 기본값은 false 라 그대로 두면 시간이 제거율에 영향을 주지 않아 총 시간이 하한으로 몰린다
    (docs/plan_extension.md 4절, 통합·D 검토 2026-09-30). 계획을 다루는 모든 경로 — run_e7,
    run_cmaes_plan 을 부르는 스크립트, 기준선 P0~P2, 계획 데이터셋, F 의 predict_plan 재채점,
    E 의 계획 화면 — 가 이 함수를 거쳐 같은 설정을 쓴다. 입력 dict 는 바꾸지 않는다.
    단일 자세 경로(evaluate)는 시간 항이 없는 점근값을 그대로 쓴다 (건드리지 않는다).
    """
    import copy

    from airis.sim.scenario import load_physics

    cfg = copy.deepcopy(physics_cfg if physics_cfg is not None else load_physics())
    cfg.setdefault("adhesion", {}).setdefault(KINETICS_KEY, {})["enabled"] = True
    return cfg


def plan_kinetics_stamp(physics_cfg: dict) -> dict:
    """stamp·결과 요약에 남길 시간 상수 정보."""
    kin = (physics_cfg.get("adhesion", {}) or {}).get(KINETICS_KEY, {}) or {}
    return {"kinetics_enabled": bool(kin.get("enabled", False)),
            "time_constant_s": float(kin.get("time_constant_s", float("nan")))}


def wrap_deg(deg: float) -> float:
    """(−180°, 180°] 로 감는다."""
    v = (float(deg) + 180.0) % 360.0 - 180.0
    return 180.0 if v == -180.0 else v


def plan_keys(n_phases: int) -> list[str]:
    """interfaces.md 키 순서 (airis.model.flow.PlanSpace 와 같다)."""
    return ([f"p{k}_{p}" for k in range(1, n_phases + 1) for p in POSE_KEYS]
            + ["duration_s"]
            + [f"share_{k}" for k in range(1, n_phases)]
            + [f"zone_{z}" for z in ZONE_NAMES])


class PlanEncoder:
    """시나리오 하나에 대한 계획 ↔ [-1, 1]^dim 변환기."""

    def __init__(self, scenario: Scenario, limits: PlanLimits | None = None,
                 zone_nozzle_counts: Sequence[float] | None = None,
                 fixed_poses: Sequence[PoseParams] | None = None) -> None:
        self.scenario = scenario
        self.limits = limits or PlanLimits()
        if self.limits.n_phases < 1:
            raise ValueError("n_phases 는 1 이상")
        lo_t, hi_t = self.limits.duration_bounds_s
        if lo_t < self.limits.n_phases * self.limits.min_phase_s:
            raise ValueError(
                f"총 시간 하한 {lo_t} s 가 단계 {self.limits.n_phases}개 × 최소 {self.limits.min_phase_s} s 보다 짧다")
        counts = (np.asarray(zone_nozzle_counts, dtype=np.float64) if zone_nozzle_counts is not None
                  else _default_zone_counts())
        if counts.shape != (len(ZONE_NAMES),) or counts.sum() <= 0:
            raise ValueError(f"구역별 노즐 수는 {len(ZONE_NAMES)}개 양수여야 한다: {zone_nozzle_counts}")
        self.zone_counts = counts
        self.pose_encoder = PoseEncoder(scenario)
        # 단계 자세를 고정하면(P3·P4 처럼 세기·시간만 탐색) decode 가 그 자세로 덮어쓴다.
        self.fixed_poses = ([self.pose_encoder.clip_pose(p) for p in fixed_poses]
                            if fixed_poses is not None else None)
        if self.fixed_poses is not None and len(self.fixed_poses) != self.limits.n_phases:
            raise ValueError(f"fixed_poses 는 단계 수({self.limits.n_phases})만큼 필요하다")
        self.keys = plan_keys(self.limits.n_phases)
        self._bounds = np.array([self._bounds_for(k) for k in self.keys], dtype=np.float64)
        self._lo = self._bounds[:, 0]
        self._span = self._bounds[:, 1] - self._bounds[:, 0]
        if np.any(self._span <= 0):
            bad = [k for k, s in zip(self.keys, self._span) if s <= 0]
            raise ValueError(f"범위의 hi 가 lo 보다 크지 않다: {bad}")

    # ---------- 범위 ----------

    def _bounds_for(self, key: str) -> tuple[float, float]:
        if key == "duration_s":
            return self.limits.duration_bounds_s
        if key.startswith("share_"):
            return 0.0, 1.0
        if key.startswith("zone_"):
            return 0.0, self.limits.s_max
        phase, _, pose_key = key.partition("_")
        lo, hi = self.pose_encoder.bounds_for(pose_key)
        if pose_key == YAW_KEY and phase == "p1":
            # 1단계 yaw 만 대칭 정규화로 0~90° 에 든다 (00_common 4.7).
            return 0.0, min(90.0, max(abs(lo), abs(hi)))
        return lo, hi

    @property
    def dim(self) -> int:
        return len(self.keys)

    @property
    def free_keys(self) -> list[str]:
        """탐색 변수 이름 (고정 자세 변수도 벡터에는 있지만 clip 이 덮어쓴다). 로그용."""
        if self.fixed_poses is not None:
            return [k for k in self.keys if not k.startswith("p")]
        fixed = {f"p{k}_{p}" for k in range(1, self.limits.n_phases + 1) for p in self.scenario.fixed_pose}
        return [k for k in self.keys if k not in fixed]

    def durations(self, total_s: float, shares: Sequence[float]) -> list[float]:
        """총 시간과 몫 K−1 개 → 단계 시간 K 개 (각각 min_phase_s 이상, 합 = total_s)."""
        n, min_s = self.limits.n_phases, self.limits.min_phase_s
        r = [min(1.0, max(0.0, float(s))) for s in shares]
        r.append(max(0.0, 1.0 - sum(r)))
        tot = sum(r)
        r = [v / tot for v in r] if tot > 0 else [1.0 / n] * n
        free = max(0.0, float(total_s) - n * min_s)
        return [min_s + free * v for v in r]

    # ---------- 투영 ----------

    def clip_zone_strengths(self, zones: Sequence[float]) -> np.ndarray:
        """구역 세기를 [0, s_max] → 쾌적 상한 → 풍량 한도(전 구역 같은 비율 축소) 순으로 투영한다."""
        s = np.clip(np.asarray(zones, dtype=np.float64).reshape(-1), 0.0, self.limits.s_max)
        if s.size != len(ZONE_NAMES):
            raise ValueError(f"구역 세기는 {len(ZONE_NAMES)}개: {s.size}")
        s = np.minimum(s, zone_strength_caps(self.scenario))     # 쾌적 상한 (임산부 가슴 0.6 등)
        total = float(self.zone_counts @ s)                      # 풍량 한도 Σ_노즐 s ≤ cap_ratio·M
        budget = self.limits.cap_ratio * float(self.zone_counts.sum())
        if total > budget > 0:
            s *= budget / total
        return s

    def clip_plan(self, plan: Plan) -> Plan:
        """계획을 실행 가능한 범위로 투영한다 (자세·시간·세기). 단계 수가 다르면 ValueError."""
        if len(plan.phases) != self.limits.n_phases:
            raise ValueError(f"단계 수 {len(plan.phases)} (인코더는 {self.limits.n_phases})")
        lo_t, hi_t = self.limits.duration_bounds_s
        total = float(np.clip(plan.duration_s, lo_t, hi_t))
        free = max(0.0, total - self.limits.n_phases * self.limits.min_phase_s)
        raw = [max(0.0, ph.duration_s - self.limits.min_phase_s) for ph in plan.phases]
        shares = [v / sum(raw) for v in raw] if sum(raw) > 0 else [1.0 / self.limits.n_phases] * self.limits.n_phases
        times = [self.limits.min_phase_s + free * v for v in shares]
        phases = [Phase(self.pose_encoder.clip_pose(ph.pose), t) for ph, t in zip(plan.phases, times)]
        return Plan(phases, self.clip_zone_strengths(plan.zone_strengths))

    # ---------- 대칭 정규화 ----------

    def _in_bounds(self, yaws: Sequence[float]) -> bool:
        lo, hi = self.pose_encoder.bounds_for(YAW_KEY)
        return all(lo - 1e-9 <= y <= hi + 1e-9 for y in yaws)

    def normalize(self, plan: Plan) -> Plan:
        """같은 점수인 계획을 하나로 모은다 (좌우 거울·앞뒤 등가·정면 동률). 00_common 4.7."""
        yaws = [float(ph.pose.torso_yaw) for ph in plan.phases]
        best = yaws
        for mirror, flip in ((False, False), (True, False), (False, True), (True, True)):
            cand = [-y for y in yaws] if mirror else list(yaws)
            if flip:
                cand = [180.0 - y for y in cand]
            cand = [wrap_deg(y) for y in cand]
            if self._in_bounds(cand) and 0.0 <= cand[0] <= 90.0:
                best = cand
                break
        phases = [Phase(_with_yaw(ph.pose, y), ph.duration_s) for ph, y in zip(plan.phases, best)]
        zones = np.asarray(plan.zone_strengths, dtype=np.float64).copy()
        if abs(np.sin(np.deg2rad(best[0]))) < FRONT_EPS:
            # 정면·후면이면 가슴 쪽 벽이 +y 로 고정된다. chest 합 ≥ back 합이 되게 맞바꾼다.
            c0, c1, b0, b1 = (ZONE_NAMES.index(z) for z in ("chest_low", "chest_high", "back_low", "back_high"))
            if zones[c0] + zones[c1] < zones[b0] + zones[b1]:
                zones[[c0, c1, b0, b1]] = zones[[b0, b1, c0, c1]]
        return Plan(phases, zones)

    # ---------- 벡터 ↔ 계획 ----------

    def encode(self, plan: Plan) -> np.ndarray:
        """Plan → [-1, 1]^dim. 대칭 정규화와 범위 투영을 먼저 거친다."""
        p = self.clip_plan(self.normalize(plan))
        raw = self.raw_vector(p)
        return np.clip(2.0 * (raw - self._lo) / self._span - 1.0, -1.0, 1.0)

    def raw_vector(self, plan: Plan) -> np.ndarray:
        """Plan → 원래 단위 벡터 (dim,). 데이터셋 plan_* 열과 같은 순서다."""
        values: dict[str, float] = {}
        for k, ph in enumerate(plan.phases, start=1):
            for key in POSE_KEYS:
                values[f"p{k}_{key}"] = float(getattr(ph.pose, key))
        total = plan.duration_s
        values["duration_s"] = total
        free = total - self.limits.n_phases * self.limits.min_phase_s
        for k in range(1, self.limits.n_phases):
            t = plan.phases[k - 1].duration_s
            values[f"share_{k}"] = ((t - self.limits.min_phase_s) / free if free > 0
                                    else 1.0 / self.limits.n_phases)
        for z, s in zip(ZONE_NAMES, np.asarray(plan.zone_strengths, dtype=np.float64)):
            values[f"zone_{z}"] = float(s)
        return np.array([values[k] for k in self.keys], dtype=np.float64)

    def decode(self, x: np.ndarray) -> Plan:
        """[-1, 1]^dim → 실행 가능한 Plan. x 를 먼저 clip 하므로 경계 밖 제안도 안전하다."""
        xc = np.clip(np.asarray(x, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if xc.size != self.dim:
            raise ValueError(f"벡터 길이가 {xc.size}, 기대값은 {self.dim}")
        raw = dict(zip(self.keys, self._lo + (xc + 1.0) * self._span / 2.0))
        times = self.durations(raw["duration_s"],
                               [raw[f"share_{k}"] for k in range(1, self.limits.n_phases)])
        poses = (self.fixed_poses if self.fixed_poses is not None
                 else [PoseParams(**{p: raw[f"p{k}_{p}"] for p in POSE_KEYS})
                       for k in range(1, self.limits.n_phases + 1)])
        phases = [Phase(pose, t) for pose, t in zip(poses, times)]
        return self.clip_plan(Plan(phases, [raw[f"zone_{z}"] for z in ZONE_NAMES]))

    def default_x(self) -> np.ndarray:
        """기본 계획(기본 자세 K 단계, 총 시간 상한, 균등 분할, 전 구역 최대 세기)의 인코딩."""
        return self.encode(self.default_plan())

    def default_plan(self) -> Plan:
        total = self.limits.duration_bounds_s[1]
        t = total / self.limits.n_phases
        phases = [Phase(self.pose_encoder.clip_pose(PoseParams()), t) for _ in range(self.limits.n_phases)]
        return Plan(phases, np.full(len(ZONE_NAMES), self.limits.s_max, dtype=np.float64))

    def plan_from_pose(self, pose: PoseParams, *, duration_s: float | None = None,
                       zone_strengths: Sequence[float] | None = None) -> Plan:
        """자세 하나를 K 단계 모두에 쓰는 계획 (단일 자세 최적 결과를 계획 기준선·시작점으로 쓸 때)."""
        total = self.limits.duration_bounds_s[1] if duration_s is None else float(duration_s)
        t = total / self.limits.n_phases
        zones = (np.full(len(ZONE_NAMES), self.limits.s_max, dtype=np.float64)
                 if zone_strengths is None else zone_strengths)
        return self.clip_plan(Plan([Phase(pose, t) for _ in range(self.limits.n_phases)], zones))


def _with_yaw(pose: PoseParams, yaw_deg: float) -> PoseParams:
    from dataclasses import replace

    return replace(pose, torso_yaw=float(yaw_deg))
