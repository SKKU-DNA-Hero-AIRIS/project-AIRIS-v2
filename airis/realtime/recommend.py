"""추천 자세와 기준 자세 점수 비교. 소유자: E. (`docs/tracks/E_realtime.md` 단계 4)

`recommend_pose(body, scenario) -> PoseParams`

1. C의 회귀 모델 `airis.model.predict.predict_pose` (4주차, `docs/interfaces.md` "회귀 모델")와
   학습 산출물 `data/models/pose_regressor.joblib`이 있으면 그것을 쓴다.
2. 없으면 **스텁**: E4 정식 결과의 시나리오별 최적 자세 표(`STUB_TABLE`)를 돌려준다.
   표는 봉우리 후보다(만세 + 옆으로 회전, 팔 내림 + 옆으로 회전). 키가 커서 만세가 천장에
   닿는 체형이면 D의 `outside_booth`로 거르고, 남은 후보는 이 체형으로 패치판 점수를 재서 고른다
   (표는 기본 체형 결과라 체형이 다르면 순위가 바뀔 수 있다).

출력은 항상 `PoseEncoder(scenario).clip_pose()`로 시나리오 제약 안에 넣는다 (interfaces.md 규약).

점수 비교(`compare_with_baselines`)는 D의 `PatchEvaluator`(400/m²)를 부른다. E는 물리 계산을 하지
않는다. 결과는 절대 제거율이 아니라 기준 자세(B0·B1·B2) 대비 **상대 개선율**로 보여 준다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..optimize.encoding import PoseEncoder
from ..sim.body import build_body
from ..sim.patch_baseline import PatchEvaluator, outside_booth
from ..sim.scenario import load_nozzle_layout, load_nozzles, load_physics
from ..sim.types import PART_NAMES, BodyParams, EvalResult, PoseParams, Scenario

#: 부스 안 판정·점수 계산에 쓰는 패치 밀도 (최적화 루프와 같음)
PATCHES_PER_M2 = 400.0


@dataclass(frozen=True)
class StubEntry:
    label: str
    pose: PoseParams
    source: str


# 첫 후보 = E4 정식 결과 (패치판, 대칭 격자 main 5f353be, 시드 5개 평균, 체형 BodyParams() 기본값).
#   default·wheelchair 는 만세 봉우리, pregnant 는 팔 내림 봉우리다 (벌림 불편도 가중이 커서).
# 둘째 후보 = 다른 봉우리. 체형 때문에 첫 후보가 부스 밖이거나 점수가 낮을 때 쓴다. #42(패치 격자
#   대칭화) 이전 docs/experiments.md 기록이라 값은 참고용이고, recommend() 가 이 체형으로 다시 잰다.
# yaw 는 좌우 대칭이라 |yaw| 로 적는다 (interfaces.md 거울 정규화).
E4_SOURCE = "E4 e4_20260918_125531_5c81a2 시드 평균"

STUB_TABLE: dict[str, list[StubEntry]] = {
    "default": [
        StubEntry("만세 + 옆으로 회전",
                  PoseParams(179.8, 0.0, 1.4, -0.8, 94.2, 0.6, 1.5), E4_SOURCE),
        StubEntry("팔 내림 + 옆으로 회전",
                  PoseParams(4.8, -3.2, 3.1, 0.5, 98.1, 0.1, 0.2),
                  "remeasure_20260918_103149_00b235 (#42 이전, 대체 후보)"),
    ],
    "pregnant": [
        StubEntry("팔 내림 + 옆으로 회전",
                  PoseParams(12.5, -3.0, 7.4, -0.1, 84.6, 2.4, 5.3), E4_SOURCE),
        StubEntry("만세 + 옆으로 회전",
                  PoseParams(180.0, 6.7, 2.6, -0.1, 97.2, 0.9, 2.1),
                  "starts_20260918_110639_46f4dd (#42 이전, 대체 후보)"),
    ],
    "wheelchair": [
        StubEntry("만세 + 몸 45° 회전",
                  PoseParams(179.9, 7.9, 9.6, 4.4, 44.9, 90.0, 90.0), E4_SOURCE),
        StubEntry("팔 내림 + 몸 45° 회전",
                  PoseParams(0.0, -29.9, 14.5, 0.3, 45.0, 90.0, 90.0),
                  "remeasure_20260918_103307_2bd141 (#42 이전, 대체 후보)"),
    ],
}


@dataclass
class Recommendation:
    pose: PoseParams
    label: str
    source: str                      # "model" 또는 "stub: <exp_id>"
    notes: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _booth() -> dict:
    return load_nozzle_layout()["booth"]


def is_inside_booth(body: BodyParams, pose: PoseParams, scenario: Scenario) -> bool:
    state = build_body(body, pose, scenario, patches_per_m2=PATCHES_PER_M2)
    return not outside_booth(state.patch_pos, _booth())


def _model_predict(body: BodyParams, scenario: Scenario) -> PoseParams | None:
    """C의 회귀 모델이 있으면 예측, 없으면 None (모듈·산출물·시나리오 없음)."""
    try:
        from ..model.predict import predict_pose          # C 4주차. 아직 없다
    except ImportError:
        return None
    try:
        return predict_pose(body, scenario)
    except (FileNotFoundError, KeyError):
        return None


def recommend(body: BodyParams, scenario: Scenario, *, use_model: bool = True,
              rank_by_score: bool = True) -> Recommendation:
    """추천 자세 + 출처·메모. 항상 시나리오 제약 안이고, 가능하면 부스 안이다.

    스텁은 표의 봉우리 중 부스 안인 것을 고르고, `rank_by_score`면 그 후보들(시나리오당 2개)을
    이 체형으로 D의 패치판에서 평가해 가장 높은 것을 고른다 (약 30 ms × 2).
    """
    enc = PoseEncoder(scenario)
    notes: list[str] = []
    if use_model:
        pred = _model_predict(body, scenario)
        if pred is not None:
            pose = enc.clip_pose(pred)
            if is_inside_booth(body, pose, scenario):
                return Recommendation(pose, "회귀 모델 예측", "model")
            notes.append("회귀 모델 예측 자세가 부스 밖이라 표의 자세로 대체")

    entries = STUB_TABLE.get(scenario.name)
    if entries is None:
        raise KeyError(f"추천 표에 없는 시나리오: {scenario.name}")
    inside = []
    for i, e in enumerate(entries):
        pose = enc.clip_pose(e.pose)
        if is_inside_booth(body, pose, scenario):
            inside.append((i, e, pose))
        else:
            notes.append(f"'{e.label}' 자세는 이 체형에서 부스(천장·벽) 밖이라 제외")
    if inside:
        if rank_by_score and len(inside) > 1:
            # 표는 기본 체형의 봉우리다. 체형이 다르면 순위가 바뀔 수 있어 이 체형으로 다시 잰다.
            ev, nz = patch_evaluator(), _nozzles()
            scores = [ev.evaluate(pose, nz, body, scenario).score for _, _, pose in inside]
            best = int(np.argmax(scores))
            if best != 0:
                notes.append(f"이 체형에서는 '{inside[best][1].label}'가 "
                             f"'{inside[0][1].label}'보다 점수가 높아 바꿈 "
                             f"({scores[best]:.3f} > {scores[0]:.3f})")
            inside = [inside[best]]
        _, e, pose = inside[0]
        return Recommendation(pose, e.label, f"stub: {e.source}", notes)
    notes.append("표의 자세가 모두 부스 밖이라 기본 자세(B0)로 안내")
    return Recommendation(enc.clip_pose(PoseParams()), "기본 자세", "fallback: B0", notes)


def recommend_pose(body: BodyParams, scenario: Scenario) -> PoseParams:
    """`E_realtime.md` 단계 4 시그니처. 회귀 모델 → 없으면 스텁."""
    return recommend(body, scenario).pose


def pose_instructions(pose: PoseParams, scenario: Scenario) -> list[str]:
    """자세 7개 → 사람에게 읽어 줄 안내 문장. 5° 수준 차이는 모델 오차라 대략값으로 말한다."""
    lines: list[str] = []
    if scenario.name == "wheelchair":
        lines.append("휠체어에 앉은 채로 멈추세요.")
    yaw = abs(pose.torso_yaw)
    if yaw < 15:
        lines.append("진행 방향(정면)을 보고 서세요." if scenario.name != "wheelchair"
                     else "진행 방향(정면)을 보세요.")
    elif yaw < 60:
        lines.append(f"몸을 한쪽으로 약 {round(yaw / 5) * 5:.0f}° 돌리세요 (좌우 어느 쪽이든 같습니다).")
    elif yaw <= 120:
        lines.append("몸을 옆으로 돌려 한쪽 벽을 보고 서세요 (약 90°, 좌우 어느 쪽이든 같습니다).")
    else:
        lines.append("뒤로 돌아 들어온 문을 보고 서세요.")

    a, f = pose.shoulder_abduction, pose.shoulder_flexion
    if a >= 150:
        lines.append("두 팔을 머리 위로 곧게 드세요 (만세).")
    elif a >= 60:
        lines.append(f"두 팔을 옆으로 약 {round(a / 10) * 10:.0f}° 벌리세요.")
    elif f >= 45:
        lines.append(f"두 팔을 앞으로 약 {round(f / 10) * 10:.0f}° 드세요.")
    else:
        lines.append("팔은 자연스럽게 내리세요.")
    if pose.elbow_flexion >= 45:
        lines.append(f"팔꿈치를 약 {round(pose.elbow_flexion / 10) * 10:.0f}° 굽히세요.")
    if pose.torso_pitch >= 10:
        lines.append(f"상체를 약 {round(pose.torso_pitch / 5) * 5:.0f}° 앞으로 숙이세요.")
    elif pose.torso_pitch <= -10:
        lines.append(f"상체를 약 {round(-pose.torso_pitch / 5) * 5:.0f}° 뒤로 젖히세요.")
    if scenario.name != "wheelchair" and pose.knee_flexion >= 15:
        lines.append("무릎을 살짝 굽히세요.")
    return lines


# ---------------------------------------------------------------------------
# 기준 자세와 점수 비교
# ---------------------------------------------------------------------------
#: E4 기준선 (C의 `scripts/run_baselines.py` BASELINES 와 같은 정의).
B1_YAWS_DEG = [float(y) for y in range(0, 360, 30)]


def baseline_poses() -> dict[str, list[PoseParams]]:
    return {
        "B0 기본": [PoseParams()],
        "B1 몸 회전": [PoseParams(torso_yaw=y if y <= 180 else y - 360) for y in B1_YAWS_DEG],
        "B2 만세": [PoseParams(shoulder_abduction=180.0, elbow_flexion=0.0)],
    }


@dataclass
class ScoreRow:
    name: str
    score: float
    total_removal: float
    discomfort: float
    removal_by_part: np.ndarray
    infeasible: bool
    n_feasible: int = 1
    n_poses: int = 1
    result: EvalResult | None = None     # 단일 자세면 원래 결과 (패치 색칠용)


@lru_cache(maxsize=1)
def patch_evaluator() -> PatchEvaluator:
    return PatchEvaluator(load_physics(), patches_per_m2=PATCHES_PER_M2)


@lru_cache(maxsize=1)
def _nozzles():
    return load_nozzles()


def _in_bounds(pose: PoseParams, scenario: Scenario) -> bool:
    for key, (lo, hi) in scenario.pose_bounds.items():
        if key not in scenario.fixed_pose and not lo <= getattr(pose, key) <= hi:
            return False
    return True


def evaluate_condition(name: str, poses: list[PoseParams], body: BodyParams,
                       scenario: Scenario) -> ScoreRow:
    """자세가 여러 개면(B1) 범위 안·부스 안인 것의 평균 (run_baselines.py 집계와 같음)."""
    ev, nz, enc = patch_evaluator(), _nozzles(), PoseEncoder(scenario)
    usable = [enc.clip_pose(p) for p in poses if len(poses) == 1 or _in_bounds(p, scenario)]
    results = [ev.evaluate(p, nz, body, scenario) for p in usable]
    feasible = [r for r in results if not r.extra.get("infeasible")]
    pool = feasible or results
    return ScoreRow(
        name=name,
        score=float(np.mean([r.score for r in pool])),
        total_removal=float(np.mean([r.total_removal for r in feasible])) if feasible else 0.0,
        discomfort=float(results[0].discomfort),
        removal_by_part=(np.mean([r.removal_by_part for r in feasible], axis=0) if feasible
                         else np.zeros(len(PART_NAMES))),
        infeasible=not feasible,
        n_feasible=len(feasible),
        n_poses=len(usable),
        result=results[0] if len(results) == 1 else None,
    )


def compare_with_baselines(body: BodyParams, scenario: Scenario,
                           recommended: PoseParams) -> list[ScoreRow]:
    """[추천, B0, B1, B2] 점수. 패치판 1회 약 30 ms × 15회."""
    rows = [evaluate_condition("추천", [recommended], body, scenario)]
    rows += [evaluate_condition(n, ps, body, scenario) for n, ps in baseline_poses().items()]
    return rows


def improvement(rec: ScoreRow, base: ScoreRow) -> float | None:
    """상대 개선율 rec/base − 1. 기준이 불가이거나 점수가 0 이하면 None."""
    if base.infeasible or rec.infeasible or base.score <= 0:
        return None
    return rec.score / base.score - 1.0
