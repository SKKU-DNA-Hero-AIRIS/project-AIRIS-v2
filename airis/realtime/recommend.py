"""추천 자세와 기준 자세 점수 비교. 소유자: E. (`docs/tracks/E_realtime.md` 단계 4)

`recommend_pose(body, scenario) -> PoseParams`

1. C의 혼합 추천 `airis.model.predict.predict` (`docs/interfaces.md` "회귀 모델", 총괄 2026-09-30 확정)를
   부른다. 후보 = flow 샘플 8개 + 가까운 학습 체형 8개 + **표(`STUB_TABLE`)의 고정 후보**를
   `extra_candidates`로 함께 넘긴 것이고, 전부 이 체형·몸 모델의 패치판으로 다시 채점해 **풀 전체에서**
   가장 높은 후보를 쓴다. 표 후보도 같은 풀에 있으니 결과가 표보다 나빠지지 않는다.
2. 산출물(`data/models/pose_flow.pt`, `pose_knn.parquet`)이 없거나 torch 가 없으면 **스텁**:
   표(`STUB_TABLE`)만 쓴다. 표는 봉우리 후보다(만세 + 옆으로 회전, 팔 내림 + 옆으로 회전). 키가 커서
   만세가 천장에 닿는 체형이면 D의 `outside_booth`로 거르고, 남은 후보는 이 체형으로 패치판 점수를
   재서 고른다 (표는 기본 체형 결과라 체형이 다르면 순위가 바뀔 수 있다).

`Recommendation.source` 는 `"model: hybrid (flow|knn|extra)"`(고른 후보의 출처) 또는 `"stub: …"`,
`"fallback: B0"` 이다. `stats` 에 출처별 후보 수·가능 수·최고 점수, `elapsed_s` 에 응답 시간(목표 1.5 s)이
들어간다. 산출물 경로·학습 커밋·설정 해시 일치는 `model_artifacts()`(= C의 `predict.artifact_status()`)로 읽기만 한다.

출력은 항상 `PoseEncoder(scenario).clip_pose()`로 시나리오 제약 안에 넣는다 (interfaces.md 규약).

점수 비교(`compare_with_baselines`)는 D의 `PatchEvaluator`(400/m²)를 부른다. E는 물리 계산을 하지
않는다. 결과는 절대 제거율이 아니라 기준 자세(B0·B1·B2) 대비 **상대 개선율**로 보여 준다.

몸 모델: 함수마다 `model`(`"mesh"` | `"capsule"` | None = `configs/physics.yaml` `body.model`)을 받는다.
`body`가 None이면 그 모델의 기본 체형(메시 `MESH_DEFAULT_BODY`, 캡슐 `BodyParams()`)이다. 추천 표 첫 후보는
메시판 E4 결과이고, 부스 안 판정과 후보 채점은 고른 몸 모델로 다시 한다.

패치 밀도 (`scoring_patches_per_m2`): 메시 2,000/m² (C 메시판 E4 재채점 기준. 400/m² 는 메시에서 점수가
±10~16% 흔들린다, 1회 약 40~60 ms), 캡슐 400/m² (캡슐판 E4·최적화 루프 기준, 1회 약 25 ms).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache, partial

import numpy as np

from ..optimize.encoding import PoseEncoder
from ..sim.body import build_body
from ..sim.patch_baseline import PatchEvaluator, outside_booth
from ..sim.scenario import load_nozzle_layout, load_nozzles, load_physics
from ..sim.types import PART_NAMES, BodyParams, EvalResult, PoseParams, Scenario

#: 캡슐 몸의 부스 안 판정·점수 계산 패치 밀도 (캡슐판 최적화 루프와 같음)
PATCHES_PER_M2 = 400.0
#: 메시 몸의 패치 밀도 (C 메시판 E4 재채점과 같음. 400/m² 는 메시에서 점수가 ±10~16% 흔들린다)
MESH_PATCHES_PER_M2 = 2000.0


def scoring_patches_per_m2(model: str | None = None) -> float:
    """몸 모델별 채점 패치 밀도. 그림을 같은 결과로 칠하려면 같은 밀도로 `build_body` 해야 한다."""
    return MESH_PATCHES_PER_M2 if resolve_model(model) == "mesh" else PATCHES_PER_M2


@dataclass(frozen=True)
class StubEntry:
    label: str
    pose: PoseParams
    source: str


# 첫 후보 = 메시판 E4 정식 결과 (e4m_20260921_193042_bca7ad: 1,500/m² 탐색 + 2,000/m² 재채점,
#   3시나리오 × 5시드, 메시 기본 체형). 2,000/m² 재채점 최고 시드. 세 시나리오 모두 만세 + 옆으로 서기
#   (임산부 포함), 휠체어는 약 36° 회전 (시나리오 yaw 상한 45° 안).
#   pregnant 원 시드는 yaw 102.1 이지만 앞뒤 등가(θ ↔ 180° − θ 점수 동일)라 안내 일관성을 위해 77.9 로 쓴다.
# 둘째 후보 = 팔 내림 봉우리 (캡슐판 E4 값, 참고). 체형 때문에 첫 후보가 부스 밖(키가 커서 만세가 천장에
#   닿음)이거나 이 체형에서 점수가 낮을 때 쓴다. recommend() 가 이 체형·몸 모델로 다시 잰다.
# yaw 는 좌우 대칭이라 |yaw| 로 적는다 (interfaces.md 거울 정규화).
E4_SOURCE = "메시판 E4 e4m_20260921_193042_bca7ad 시드 최고"
CAPSULE_REF_SOURCE = "캡슐판 참고"

STUB_TABLE: dict[str, list[StubEntry]] = {
    "default": [
        StubEntry("만세 + 옆으로 회전",
                  PoseParams(179.9, -4.4, 0.1, 0.2, 71.0, 1.7, 11.2), E4_SOURCE),     # 0.6544
        StubEntry("팔 내림 + 옆으로 회전",
                  PoseParams(4.8, -3.2, 3.1, 0.5, 98.1, 0.1, 0.2),
                  f"{CAPSULE_REF_SOURCE} (remeasure_20260918_103149_00b235, 대체 후보)"),
    ],
    "pregnant": [
        StubEntry("만세 + 옆으로 회전",
                  PoseParams(179.4, -6.0, 0.0, -0.2, 77.9, 0.1, 10.6), E4_SOURCE),    # 0.6342
        StubEntry("팔 내림 + 옆으로 회전",
                  PoseParams(12.5, -3.0, 7.4, -0.1, 84.6, 2.4, 5.3),
                  f"{CAPSULE_REF_SOURCE} (e4_20260918_125531_5c81a2, 대체 후보)"),
    ],
    "wheelchair": [
        StubEntry("만세 + 몸 36° 회전",
                  PoseParams(179.9, 2.5, 8.9, -0.4, 35.8, 90.0, 90.0), E4_SOURCE),    # 0.4319
        StubEntry("팔 내림 + 몸 45° 회전",
                  PoseParams(0.0, -29.9, 14.5, 0.3, 45.0, 90.0, 90.0),
                  f"{CAPSULE_REF_SOURCE} (remeasure_20260918_103307_2bd141, 대체 후보)"),
    ],
}


#: 응답 시간 목표 (총괄 확정. 넘으면 대시보드가 알린다)
RESPONSE_BUDGET_S = 1.5
#: 후보 출처 표시 이름
SOURCE_LABELS = {"flow": "flow 샘플", "knn": "가까운 학습 체형", "extra": "고정 후보표(E4)"}


@dataclass(frozen=True)
class SourceStat:
    """출처별 후보 수와 최고 점수 (화면 ④ 접힌 표)."""
    source: str                      # "flow" | "knn" | "extra"
    n: int
    n_feasible: int
    best: float | None               # 부스 안 후보 중 최고 점수 (전부 밖이면 None)

    @property
    def label(self) -> str:
        return SOURCE_LABELS.get(self.source, self.source)


@dataclass
class Recommendation:
    pose: PoseParams
    label: str
    source: str                      # "model: hybrid (<출처>)" | "stub: <exp_id>" | "fallback: B0"
    notes: list[str] = field(default_factory=list)
    stats: list[SourceStat] = field(default_factory=list)
    elapsed_s: float = 0.0


@lru_cache(maxsize=1)
def _booth() -> dict:
    return load_nozzle_layout()["booth"]


def resolve_model(model: str | None) -> str:
    """None 이면 `configs/physics.yaml` `body.model`."""
    if model is None:
        from ..sim.body import _configured_body_model
        return _configured_body_model()
    return model


def default_body(model: str | None = None) -> BodyParams:
    """몸 모델의 기본 체형 (메시 `MESH_DEFAULT_BODY`, 캡슐 `BodyParams()`)."""
    if resolve_model(model) == "mesh":
        from ..sim.human_mesh import MESH_DEFAULT_BODY
        return MESH_DEFAULT_BODY
    return BodyParams()


def is_inside_booth(body: BodyParams | None, pose: PoseParams, scenario: Scenario,
                    model: str | None = None) -> bool:
    model = resolve_model(model)
    state = build_body(body, pose, scenario, patches_per_m2=scoring_patches_per_m2(model), model=model)
    return not outside_booth(state.patch_pos, _booth())


def _model_predict(body: BodyParams, scenario: Scenario, model: str,
                   extra_candidates: tuple[PoseParams, ...] = (),
                   notes: list[str] | None = None):
    """C의 혼합 추천. 쓸 수 없으면 None 과 폴백 사유(`notes`).

    재채점은 **E의 평가기**(이 몸 모델·밀도)로 한다. 그래야 ④의 점수·개선율과 같은 자로 잰 값이다.
    """
    def note(msg: str) -> None:
        if notes is not None:
            notes.append(msg)

    try:
        from ..model.predict import predict
    except ImportError as exc:                       # torch 없이도 kNN 은 되지만 모듈 자체가 없을 때
        note(f"추천 모델을 쓸 수 없어 표(E4)로 안내합니다 ({exc})")
        return None
    try:
        return predict(body, scenario, evaluator=patch_evaluator(model), nozzle=_nozzles(),
                       extra_candidates=extra_candidates)
    except FileNotFoundError:
        note("학습 산출물(data/models/pose_flow.pt · pose_knn.parquet)이 없어 표(E4)로 안내합니다")
    except ImportError as exc:
        note(f"flow 산출물만 있고 torch 가 없어 표(E4)로 안내합니다 ({exc})")
    except KeyError as exc:      # 학습에 없던 시나리오. 다른 KeyError 도 삼키지 않게 내용을 그대로 보여 준다
        note(f"추천 모델이 이 입력을 처리하지 못해 표(E4)로 안내합니다 (KeyError: {exc})")
    return None


def _source_stats(pred) -> list[SourceStat]:
    """출처별 후보 수·가능 수·최고 점수. 재채점을 안 했으면 점수는 None."""
    stats: list[SourceStat] = []
    for src in ("flow", "knn", "extra"):
        idx = [i for i, s in enumerate(pred.sources) if s == src]
        if not idx:
            continue
        if pred.scores is None:
            stats.append(SourceStat(src, len(idx), len(idx), None))
            continue
        ok = [i for i in idx if not bool(pred.infeasible[i])]
        stats.append(SourceStat(src, len(idx), len(ok),
                                float(max(pred.scores[i] for i in ok)) if ok else None))
    return stats


def _picked(pred) -> int:
    """고른 후보의 자리 (predict 는 candidates[i] 를 그대로 돌려준다)."""
    for i, c in enumerate(pred.candidates):
        if c is pred.pose:
            return i
    return -1


def recommend(body: BodyParams | None, scenario: Scenario, *, use_model: bool = True,
              rank_by_score: bool = True, model: str | None = None) -> Recommendation:
    """추천 자세 + 출처·메모. 항상 시나리오 제약 안이고, 가능하면 부스 안이다.

    `use_model`이면 C의 혼합 추천에 표의 고정 후보를 `extra_candidates`로 함께 넘기고, 재채점한 풀
    전체에서 최고를 고른다. 산출물이 없으면 표만 쓴다.

    스텁은 표의 봉우리 중 부스 안인 것을 고르고, `rank_by_score`면 그 후보들(시나리오당 2개)을
    이 체형·몸 모델로 D의 패치판에서 평가해 가장 높은 것을 고른다 (약 30 ms × 2).
    """
    t0 = time.perf_counter()
    model = resolve_model(model)
    body = body if body is not None else default_body(model)
    enc = PoseEncoder(scenario)
    notes: list[str] = []

    entries = STUB_TABLE.get(scenario.name)
    if entries is None:
        raise KeyError(f"추천 표에 없는 시나리오: {scenario.name}")

    if use_model:
        pred = _model_predict(body, scenario, model, tuple(e.pose for e in entries), notes)
        if pred is not None:
            i, stats = _picked(pred), _source_stats(pred)
            pose = enc.clip_pose(pred.pose)
            if pred.infeasible is not None and bool(pred.infeasible.all()):
                notes.append("모델 후보가 모두 부스(천장·벽) 밖이라 표의 자세로 대체")
            elif not is_inside_booth(body, pose, scenario, model):
                # 평가기의 불가 판정과 별개로 E가 한 번 더 본다 (clip_pose 로 자세가 바뀌었을 수 있다).
                notes.append("모델이 고른 자세가 부스(천장·벽) 밖이라 표의 자세로 대체")
            else:
                src = pred.sources[i] if 0 <= i < len(pred.sources) else pred.source
                label = "모델 추천"
                if src == "extra":                 # 고른 것이 표 후보면 표의 이름을 그대로 쓴다
                    j = [k for k, t in enumerate(pred.sources) if t == "extra"].index(i)
                    label = f"모델 추천 · {entries[j].label}"
                return Recommendation(pose, label, f"model: hybrid ({src})",
                                      notes, stats, time.perf_counter() - t0)
    inside = []
    for i, e in enumerate(entries):
        pose = enc.clip_pose(e.pose)
        if is_inside_booth(body, pose, scenario, model):
            inside.append((i, e, pose))
        else:
            notes.append(f"'{e.label}' 자세는 이 체형에서 부스(천장·벽) 밖이라 제외")
    if inside:
        if rank_by_score and len(inside) > 1:
            # 표는 기본 체형의 봉우리다. 체형이 다르면 순위가 바뀔 수 있어 이 체형으로 다시 잰다.
            ev, nz = patch_evaluator(model), _nozzles()
            scores = [ev.evaluate(pose, nz, body, scenario).score for _, _, pose in inside]
            best = int(np.argmax(scores))
            if best != 0:
                notes.append(f"이 체형에서는 '{inside[best][1].label}'가 "
                             f"'{inside[0][1].label}'보다 점수가 높아 바꿈 "
                             f"({scores[best]:.3f} > {scores[0]:.3f})")
            inside = [inside[best]]
        _, e, pose = inside[0]
        return Recommendation(pose, e.label, f"stub: {e.source}", notes,
                              elapsed_s=time.perf_counter() - t0)
    notes.append("표의 자세가 모두 부스 밖이라 기본 자세(B0)로 안내")
    return Recommendation(enc.clip_pose(PoseParams()), "기본 자세", "fallback: B0", notes,
                          elapsed_s=time.perf_counter() - t0)


@dataclass(frozen=True)
class ArtifactInfo:
    """학습 산출물 상태 (사이드바 읽기 전용). 경로·판정 모두 C의 `predict.artifact_status()` 값이다."""
    name: str                        # "flow" | "knn"
    path: str
    exists: bool
    commit: str | None = None        # 학습 커밋
    config_ok: bool | None = None    # 노즐·물리 해시가 지금 설정과 같은가 (None = 확인 불가·도장 없음)
    message: str = ""                # 못 읽은 이유


def model_artifacts() -> list[ArtifactInfo]:
    """산출물 상태. 판정은 C의 `predict.artifact_status()` 하나만 쓴다 (해시 규약이 바뀌어도 E는 그대로).

    화면에 보여 주기만 하고 아무것도 바꾸지 않는다. 호출마다 산출물 파일을 다시 읽으므로
    화면 쪽에서 캐시한다.
    """
    try:
        from ..model.predict import artifact_status
    except ImportError as exc:
        return [ArtifactInfo("model", "-", False, message=str(exc))]

    status = artifact_status()
    out: list[ArtifactInfo] = []
    for name in ("flow", "knn"):
        e = status[name]
        # match 를 그대로 쓴다: #99 뒤 도장이 없거나 일부만 찍힌 산출물도 None(확인 불가)으로 온다.
        out.append(ArtifactInfo(name, e["path"], bool(e["exists"]), (e.get("stamp") or {}).get("commit"),
                                e["match"], e["error"] or ("" if e["exists"] else "없음")))
    return out


def recommend_pose(body: BodyParams, scenario: Scenario) -> PoseParams:
    """`E_realtime.md` 단계 4 시그니처. 혼합 추천(flow + kNN + 표 후보) → 산출물이 없으면 표."""
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
        lines.append(f"몸을 옆으로 약 {round(yaw / 5) * 5:.0f}° 돌려 한쪽 벽 쪽을 보고 서세요 "
                     "(좌우 어느 쪽이든 같습니다).")
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


def patch_evaluator(model: str | None = None) -> PatchEvaluator:
    """D의 패치판. 몸 모델(None = 설정 파일)에 맞는 밀도(`scoring_patches_per_m2`)로 `build_body` 를 부른다."""
    return _patch_evaluator(resolve_model(model))


@lru_cache(maxsize=4)
def _patch_evaluator(model: str) -> PatchEvaluator:
    return PatchEvaluator(load_physics(), partial(build_body, model=model),
                          patches_per_m2=scoring_patches_per_m2(model))


@lru_cache(maxsize=1)
def _nozzles():
    return load_nozzles()


def _in_bounds(pose: PoseParams, scenario: Scenario) -> bool:
    for key, (lo, hi) in scenario.pose_bounds.items():
        if key not in scenario.fixed_pose and not lo <= getattr(pose, key) <= hi:
            return False
    return True


def evaluate_condition(name: str, poses: list[PoseParams], body: BodyParams | None,
                       scenario: Scenario, model: str | None = None) -> ScoreRow:
    """자세가 여러 개면(B1) 범위 안·부스 안인 것의 평균 (run_baselines.py 집계와 같음)."""
    ev, nz, enc = patch_evaluator(model), _nozzles(), PoseEncoder(scenario)
    body = body if body is not None else default_body(model)
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


def compare_with_baselines(body: BodyParams | None, scenario: Scenario,
                           recommended: PoseParams, model: str | None = None) -> list[ScoreRow]:
    """[추천, B0, B1, B2] 점수. 패치판 15회 (메시 2,000/m² 약 0.7 s, 캡슐 400/m² 약 0.4 s). 같은 몸 모델로 채점한다."""
    rows = [evaluate_condition("추천", [recommended], body, scenario, model)]
    rows += [evaluate_condition(n, ps, body, scenario, model) for n, ps in baseline_poses().items()]
    return rows


def improvement(rec: ScoreRow, base: ScoreRow) -> float | None:
    """상대 개선율 rec/base − 1. 기준이 불가이거나 점수가 0 이하면 None."""
    if base.infeasible or rec.infeasible or base.score <= 0:
        return None
    return rec.score / base.score - 1.0
