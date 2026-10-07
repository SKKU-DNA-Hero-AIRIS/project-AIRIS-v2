"""회전·계획 안내. 소유자: E. (`docs/e_plan_screen_draft.md`, 확장 5단계)

화면 ④-2 가 쓰는 것들이다. **물리 계산을 하지 않는다** — 자세는 ④ 의 추천을 그대로 받고, 수치는 C 의
실험 결과 파일(`docs/e7_sweep_reference.json`)에서 읽는다.

    모드 A (기본)  제안 자세를 **유지한 채** 제자리에서 N 방향으로 돌아선다 (`rotation_plan`).
                  C 의 단계 수 스윕에서 모든 시나리오 1위였다 (`P1opt_*`). 자세를 바꾸지 않으므로
                  단계 전환 시간(`plan.transition_s`)이 들지 않는다 — 그래서 20초를 10칸으로 쪼갤 수 있다.
    모드 B        단계마다 **다른** 자세로 가는 계획 (`Plan`/`Phase`, `P5`). F 의 `predict_plan` 이
                  낸다 (`plan_model`). 산출물(`plan_flow.pt`·`plan_knn.parquet`)이 없으면 화면은
                  "준비 중" 으로 두고 모드 A 만 쓴다.

**단계 수를 고정으로 가정하지 않는다** (`interfaces.md` "계획 모델"). 모델 계획은 산출물의 N, 고정 회전
계획은 10 이다. 화면은 `len(plan.phases)` 로만 그린다.

장비 제어 JSON (`control_json`) 의 노즐 세기는 E 가 계산하지 않고 B 의 `apply_zone_strengths` 결과를
그대로 담는다. 단계마다 가슴 쪽 벽이 바뀌면 같은 구역 세기라도 노즐 배정이 달라지므로 **단계 안에** 둔다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..sim.scenario import apply_zone_strengths, load_nozzles
from ..sim.types import ZONE_NAMES, BodyParams, EvalResult, Phase, Plan, PoseParams, Scenario
from .recommend import pose_instructions

ROOT = Path(__file__).resolve().parents[2]
#: C 의 E7 단계 수 스윕 결과 (회전 기준선 포함). 없으면 수치 없이 안내만 한다.
SWEEP_REFERENCE = ROOT / "docs" / "e7_sweep_reference.json"
#: E7 본 실행 결과 (P0 "안내 없이 통과" 가 여기 있다. 스윕 파일에는 없다)
E7_REFERENCE = ROOT / "docs" / "e7_reference.json"
#: 회전 칸 수와 칸당 시간 (총괄 확정 2026-10-04: 표준 기준선 P1_10, 10방향 × 2초 = 20초).
ROTATION_STEPS = 10
STEP_SECONDS = 2.0
#: 화면에 견줄 조건. 내부 식별자 → 표시 이름 (문구 규약: 약어를 화면에 쓰지 않는다)
ROTATION_CONDITIONS = {
    "P0": "안내 없이 통과",
    f"P1_{ROTATION_STEPS}": f"제조사 안내 ({ROTATION_STEPS}방향)",
    f"P1opt_{ROTATION_STEPS}": f"제안 자세로 {ROTATION_STEPS}방향",
}


def rotation_plan(pose: PoseParams, *, steps: int = ROTATION_STEPS,
                  step_s: float = STEP_SECONDS) -> Plan:
    """제안 자세를 유지한 채 `steps` 방향으로 도는 계획. 구역 세기는 전부 1.0 (기본 운전)."""
    if steps < 1 or step_s <= 0:
        raise ValueError(f"칸 수·칸 시간이 이상하다: steps={steps}, step_s={step_s}")
    turn = 360.0 / steps
    phases = []
    for i in range(steps):
        yaw = pose.torso_yaw + i * turn
        yaw = (yaw + 180.0) % 360.0 - 180.0                 # -180~180 으로 접는다
        phases.append(Phase(PoseParams(**{**vars(pose), "torso_yaw": yaw}), float(step_s)))
    return Plan(phases, np.ones(len(ZONE_NAMES), dtype=np.float64))


def rotation_instructions(pose: PoseParams, scenario: Scenario, *, steps: int = ROTATION_STEPS,
                          step_s: float = STEP_SECONDS) -> list[str]:
    """사람에게 읽어 줄 안내. 자세는 한 번만 말하고, 방향은 "한 바퀴를 N칸으로"로 말한다.

    단계마다 자세를 다시 말하지 않는다 — 바뀌는 것은 몸 방향뿐이기 때문이다. 좌우 어느 쪽으로 도는지는
    좌우 대칭이라 점수가 같으므로 사용자가 고른다.
    """
    lines = pose_instructions(pose, scenario, include_rotation=False)
    turn = 360.0 / steps
    verb = "돌아앉으세요" if scenario.name == "wheelchair" else "돌아서세요"
    lines.append(f"이 자세 그대로 제자리에서 한 바퀴 {verb}. "
                 f"{steps}칸으로 나눠 한 칸에 약 {turn:.0f}°씩, 칸마다 {step_s:.0f}초 멈춥니다 "
                 f"(모두 {steps * step_s:.0f}초).")
    lines.append("도는 방향은 왼쪽·오른쪽 어느 쪽이든 같습니다.")
    return lines


@dataclass(frozen=True)
class ConditionResult:
    """실험 결과 한 줄 (C 의 파일에서 읽은 값. E 가 다시 계산하지 않는다)."""
    key: str
    label: str
    total_removal: float
    duration_s: float
    n_phases: int


@lru_cache(maxsize=8)
def _rows() -> tuple:
    """두 참조 파일의 행을 합친다 (P0 은 본 실행, 회전 조건은 스윕에 있다)."""
    out: list = []
    for path in (E7_REFERENCE, SWEEP_REFERENCE):
        if path.exists():
            out += json.loads(path.read_text(encoding="utf-8"))["rows"]
    return tuple(out)


def rotation_reference(scenario_name: str, *, energy_weight: float = 0.1) -> list[ConditionResult]:
    """시나리오의 회전 기준선 수치 (제거율). 파일이 없거나 조건이 없으면 그 줄은 빠진다.

    `score` 는 에너지 가중마다 목적함수가 달라 가로로 비교하면 안 되므로 쓰지 않는다
    (파일의 `score_note`). 화면에는 **제거율**과 시간만 쓴다.
    """
    rows = _rows()
    out: list[ConditionResult] = []
    for key, label in ROTATION_CONDITIONS.items():
        hit = [r for r in rows if r["condition"] == key and r["scenario"] == scenario_name
               and r["energy_weight"] == energy_weight]
        if not hit:
            continue
        r = hit[0]
        out.append(ConditionResult(key, label, float(r["total_removal"]),
                                   float(r["duration_s"]), int(r["n_phases"])))
    return out


def control_json(plan: Plan, body: BodyParams | None, scenario: Scenario, *, source: str,
                 reference: dict | None = None, transition_s: float = 0.0) -> dict:
    """장비 제어용 JSON (`docs/e_plan_screen_draft.md` 4절).

    `fans` 는 노즐 순서대로의 출구 속도 배수다. E 가 지어내지 않고 B 의 `apply_zone_strengths` 결과를
    그대로 담는다. 단계마다 가슴 쪽 벽이 바뀌면 같은 구역 세기라도 배정이 달라지므로 단계 안에 둔다.
    """
    nozzle = load_nozzles()
    zones = {z: float(v) for z, v in zip(ZONE_NAMES, np.asarray(plan.zone_strengths).reshape(-1))}
    phases = []
    for i, ph in enumerate(plan.phases, start=1):
        fans = apply_zone_strengths(nozzle, plan.zone_strengths, ph.pose.torso_yaw).strengths
        phases.append({
            "index": i,
            "duration_s": round(float(ph.duration_s), 2),
            "pose": {k: round(float(v), 2) for k, v in vars(ph.pose).items()},
            "fans": [round(float(x), 4) for x in np.asarray(fans).reshape(-1)],
        })
    return {
        "schema": "airis.plan.v1",
        "scenario": scenario.name,
        "body": None if body is None else {k: round(float(v), 4) for k, v in vars(body).items()},
        "duration_s": round(float(plan.duration_s), 2),
        "transition_s": round(float(transition_s), 2),
        "phases": phases,
        "zones": zones,
        "source": source,
        "reference": reference or {},
    }


# ---------------------------------------------------------------------------
# 모드 B: 자세를 바꿔 가는 계획
# ---------------------------------------------------------------------------
#: 구역 표시 이름 (몸 기준. 화면 문구 규약대로 약어를 쓰지 않는다)
ZONE_LABELS = {"chest_low": "가슴 쪽 아래", "chest_high": "가슴 쪽 위",
               "back_low": "등 쪽 아래", "back_high": "등 쪽 위", "top": "천장"}


@dataclass
class PlanGuide:
    """모드 B 화면에 필요한 것 묶음. 계획이 없으면 `plan is None` 이고 `note` 에 이유가 있다."""
    plan: Plan | None
    source: str = ""                      # 고른 후보의 출처 ("flow" | "knn" | "fixed" 등)
    note: str = ""                        # 못 쓸 때의 이유 (화면에 그대로 띄운다)
    elapsed_s: float = 0.0
    n_candidates: int = 0
    #: 실제로 채점한 후보 수. F 의 `rescore_top`(소수 재채점)을 쓰면 후보 수보다 적다 (#171).
    #: 화면이 "후보 N개를 재채점했다" 고 잘못 말하지 않도록 따로 받는다.
    n_rescored: int = 0
    margin_fallback: bool = False

    @property
    def n_phases(self) -> int:
        return 0 if self.plan is None else len(self.plan.phases)

    @property
    def scoring_text(self) -> str:
        """채점한 후보를 사람이 읽을 한 줄. 전부 채점했으면 수를 하나만 말한다."""
        if not self.n_candidates:
            return ""
        if self.n_rescored and self.n_rescored < self.n_candidates:
            return f"후보 {self.n_candidates}개 중 {self.n_rescored}개를 자세히 채점했습니다."
        return f"후보 {self.n_candidates}개를 모두 채점했습니다."


def plan_model(body: BodyParams | None, scenario: Scenario, rotation_pose: PoseParams, *,
               evaluator=None, nozzle=None) -> PlanGuide:
    """F 의 계획 추천을 부른다. 산출물이 없거나 못 쓰면 `plan=None` + 이유.

    `rotation_pose` 에 ④ 의 추천 자세를 넘기면 "제안 자세 그대로 회전"(= 모드 A)이 후보에 들어간다.
    두 모드가 **같은 후보군에서** 겨루므로, 모드 B 가 뽑혔다는 것은 모드 A 보다 높았다는 뜻이다.

    인자는 `interfaces.md` "계획 모델" 에 있는 것만 넘긴다 — 모르는 인자는 `TypeError` 다
    (중복 제거·스레드는 계획 쪽에 아직 없다, F 확인 2026-10-06).
    """
    import time

    t0 = time.perf_counter()
    try:
        from ..model.predict import predict_plan_candidates
    except ImportError as exc:
        return PlanGuide(None, note=f"계획 추천을 쓸 수 없습니다 ({exc})")
    kw = {}
    if evaluator is not None:
        kw["evaluator"] = evaluator
    if nozzle is not None:
        kw["nozzle"] = nozzle
    try:
        pred = predict_plan_candidates(body, scenario, rotation_pose=rotation_pose, **kw)
    except FileNotFoundError:
        return PlanGuide(None, note="계획 추천 모델이 아직 준비 중입니다 (학습 산출물 없음).",
                         elapsed_s=time.perf_counter() - t0)
    except ImportError as exc:
        return PlanGuide(None, note=f"계획 추천에 필요한 것이 없습니다 ({exc}).",
                         elapsed_s=time.perf_counter() - t0)
    except KeyError:
        return PlanGuide(None, note=f"계획 추천 모델이 이 유형({scenario.name})을 아직 배우지 않았습니다.",
                         elapsed_s=time.perf_counter() - t0)
    n_cand = len(getattr(pred, "candidates", []) or [])
    return PlanGuide(pred.plan, source=getattr(pred, "source", ""),
                     elapsed_s=time.perf_counter() - t0,
                     n_candidates=n_cand,
                     n_rescored=int(getattr(pred, "n_rescored", 0) or n_cand),
                     margin_fallback=bool(getattr(pred, "margin_fallback", False)))


def plan_steps(plan: Plan, scenario: Scenario) -> list[dict]:
    """단계 표. 단계마다 자세 안내 문장·시간·그 단계의 벽별 바람 세기.

    구역 세기는 계획 전체에 하나지만 **가슴 쪽 벽이 단계마다 바뀌므로** 노즐에 실제로 실리는 값은
    단계마다 다르다. 그래서 `apply_zone_strengths` 를 단계마다 다시 불러 그 단계의 값을 보여 준다.
    """
    nozzle = load_nozzles()
    zones = np.asarray(plan.zone_strengths, dtype=np.float64).reshape(-1)
    rows = []
    for i, ph in enumerate(plan.phases, start=1):
        fans = np.asarray(apply_zone_strengths(nozzle, zones, ph.pose.torso_yaw).strengths,
                          dtype=np.float64).reshape(-1)
        rows.append({
            "단계": i,
            "시간": f"{ph.duration_s:.1f}초",
            "자세": " / ".join(pose_instructions(ph.pose, scenario)),
            "바람 세기": " · ".join(f"{ZONE_LABELS[z]} {v:.0%}" for z, v in zip(ZONE_NAMES, zones)),
            "분사구 평균": f"{fans.mean():.0%}",
        })
    return rows


def plan_effect(result: EvalResult) -> dict:
    """`evaluate_plan` 결과 → 화면에 쓸 값. 없는 키는 None 으로 둔다 (평가기가 바뀌어도 안 깨지게)."""
    extra = getattr(result, "extra", {}) or {}
    return {
        "먼지 제거 효과": float(result.total_removal),
        "바람 에너지": float(extra["energy"]) if "energy" in extra else None,
        "자세 불편도": float(result.discomfort),
        "총 시간": float(extra["duration_s"]) if "duration_s" in extra else None,
        "불가": bool(extra.get("infeasible", False)),
    }
