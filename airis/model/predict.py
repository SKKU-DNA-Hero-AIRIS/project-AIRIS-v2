"""체형 → 자세·계획 추천. 소유자: F. docs/interfaces.md "회귀 모델"·"계획 모델" 계약,
docs/proposals/flow_matching.md, docs/plan_extension.md.

    predict_pose(body, scenario) -> PoseParams        자세 모델 (혼합: flow + kNN + 고정 후보)
    predict_plan(body, scenario) -> Plan              계획 모델 (data/models/plan_flow.pt)

**혼합 추천** (총괄 2026-09-30 확정, 기본 backend="hybrid")

    후보 = flow matching 샘플 n_flow 개 (data/models/pose_flow.pt)
         + 가까운 학습 체형의 최적 자세 n_knn 개 (data/models/pose_knn.parquet, knn.PoseKNN)
         + extra_candidates (E 가 넘기는 고정 후보표)
    → 전부 시나리오 제약 안으로 투영(clip_pose) → 패치판으로 재채점 → 가장 높은 후보

부스 밖 후보는 평가기의 불가 판정으로 탈락한다. 재채점을 끄면(rescore=False) 첫 후보를 그대로 돌려준다
(E5 비교용). extra_candidates 를 함께 재채점하므로 결과가 그 후보들보다 나빠지지 않는다.
backend="flow" 는 flow 샘플만, "knn" 은 kNN 후보만 쓴다 (E5 비교용).
`Prediction.sources` 는 후보마다 "flow" | "knn" | "extra", `Prediction.source` 는 고른 후보의 출처다.
계획은 평가기의 evaluate_plan 으로 채점한다 (구현 전 평가기는 NotImplementedError 를 그대로 올린다).
계획 후보(flow 샘플 + extra_candidates)는 채점 전에 전부 C 의 `PlanEncoder.clip_plan`(자세·시간·구역 세기,
쾌적 상한, 풍량 한도 보수)을 거친다. 범위는 configs/physics.yaml 의 plan.* · fan.* (`PlanLimits.from_config`).

산출물이 하나만 있으면 있는 쪽으로 돌아간다 (처음 한 번 경고). flow 는 torch 가 필요하고, torch 가 없으면
kNN 쪽만 쓴다.

예외 규약 (E 의 recommend 가 스텁으로 폴백할 때 구분한다):
- flow 도 kNN 도 쓸 수 없으면: 산출물이 아예 없으면 FileNotFoundError, flow 산출물만 있고 torch 가 없으면 ImportError
- 학습에 없던 시나리오면 KeyError
그 외 예외는 삼키지 않는다.

산출물에는 학습 데이터의 nozzle_layout_hash · physics_hash · 학습 커밋이 들어 있고, 로드할 때 현재 설정과 다르면
경고한다. 물리 기준이 바뀌면 모델은 무효다 (docs/experiments.md 규칙).
"""
from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path

import numpy as np

from airis.sim import BodyParams, Evaluator, NozzleConfig, Plan, PoseParams, Scenario

from .flow import PoseFlow          # torch 는 flow.py 안에서만 import 한다 (kNN 만으로도 돌아가게)
from .knn import PoseKNN

ROOT = Path(__file__).resolve().parents[2]
#: 산출물 폴더. 환경변수 AIRIS_MODEL_DIR 이 있으면 그 폴더 (worktree 에서 본 폴더 산출물을 쓸 때 등).
MODEL_DIR = Path(os.environ.get("AIRIS_MODEL_DIR") or ROOT / "data" / "models")
DEFAULT_MODEL_PATH = MODEL_DIR / "pose_flow.pt"
DEFAULT_PLAN_MODEL_PATH = MODEL_DIR / "plan_flow.pt"
DEFAULT_KNN_PATH = MODEL_DIR / "pose_knn.parquet"
N_SAMPLES = 16
N_FLOW = 8
N_KNN = 8
BACKENDS = ("hybrid", "flow", "knn")
POSE_YAW_INDEX = [f.name for f in fields(PoseParams)].index("torso_yaw")


@dataclass
class Prediction:
    pose: PoseParams
    candidates: list[PoseParams]
    scores: np.ndarray | None        # 재채점 점수 (rescore=False 면 None)
    infeasible: np.ndarray | None
    sources: list[str] = field(default_factory=list)   # 후보마다 "flow" | "knn" | "extra"
    source: str = ""                                   # 고른 후보의 출처
    #: 재채점 단축 옵션을 썼을 때의 기록 (기본 동작에서는 n_dropped 0, screen_scores None, n_rescored = 후보 수)
    n_dropped: int = 0                                 # 중복 제거로 뺀 후보 수 (candidates 에는 남은 것만)
    screen_scores: np.ndarray | None = None            # 저밀도 선별 점수 (선별을 안 했으면 None)
    n_rescored: int = 0                                # 최종 평가기로 채점한 후보 수
    #: 부스 안 판정 여유(feasibility_margin > 0)를 썼을 때의 기록 (기본 동작에서는 0, False)
    n_margin_checks: int = 0                           # 키운 체형으로 다시 채점한 횟수
    n_margin_rejected: int = 0                         # 키운 체형에서 불가라 넘긴 후보 수
    margin_fallback: bool = False                      # 전부 걸려서 여유 없이 고른 후보를 돌려줬는가


@dataclass
class PlanPrediction:
    plan: Plan
    candidates: list[Plan]
    scores: np.ndarray | None
    infeasible: np.ndarray | None


#: 산출물이 유효한지 가르는 설정 해시 (학습 데이터의 도장과 지금 설정을 비교한다).
STAMP_KEYS = ("nozzle_layout_hash", "physics_hash")
#: 계획 산출물만 갖는 도장: 계획 채점 설정(C 의 plan_kinetics_stamp). 산출물에 있을 때만 비교한다.
KINETICS_KEYS = ("kinetics_enabled", "time_constant_s")
#: artifact_status 가 함께 보여 주는 도장 (학습 조건 기록용).
INFO_KEYS = ("body_model", "patches_per_m2", "commit")


def current_stamp() -> dict:
    """지금 설정의 도장 {nozzle_layout_hash, physics_hash, kinetics_enabled, time_constant_s}.

    해시 규약은 airis.optimize.explog, 시간 상수는 계획 채점 설정(plan_physics_cfg)을 따른다.
    """
    from airis.optimize import cli, explog
    from airis.optimize.plan_encoding import plan_kinetics_stamp

    return {"nozzle_layout_hash": cli.nozzle_hash(cli.resolve_nozzles()[0]),
            "physics_hash": explog.file_hash(explog.ROOT / "configs" / "physics.yaml"),
            **plan_kinetics_stamp(_physics_cfg(plan=True))}


def _same(key: str, a, b) -> bool:
    """도장 값 비교. 해시는 문자열로 (16진 해시 '1e500000' 을 숫자로 읽지 않게), kinetics 는 값으로
    (시간 상수 2 == 2.0, numpy bool, parquet 에서 읽은 문자열 'True')."""
    if key == "kinetics_enabled":
        def truthy(v) -> bool:
            return str(v).strip().lower() in ("true", "1", "1.0")
        return truthy(a) == truthy(b)
    if key == "time_constant_s":
        try:
            return bool(np.isclose(float(a), float(b)))
        except (TypeError, ValueError):
            return str(a) == str(b)
    return str(a) == str(b)


def stamp_mismatch(meta: dict, now: dict | None = None) -> list[str]:
    """학습 도장과 지금 설정이 다른 키. 산출물에 없는 키(자세 산출물의 kinetics 등)는 비교하지 않는다."""
    now = current_stamp() if now is None else now
    return [k for k in STAMP_KEYS + KINETICS_KEYS
            if meta.get(k) is not None and now.get(k) is not None and not _same(k, meta[k], now[k])]


def _check_stamp(meta: dict, path: Path) -> None:
    """학습 때의 노즐·물리 설정 해시가 지금과 다르면 경고."""
    try:
        now = current_stamp()
    except Exception as exc:          # 설정을 못 읽으면 판정만 건너뛴다
        warnings.warn(f"{path.name}: 현재 설정 해시를 읽지 못해 비교하지 않는다 ({exc})", RuntimeWarning)
        return
    diff = stamp_mismatch(meta, now)
    if diff:
        warnings.warn(
            f"{path.name}: 학습 때와 설정이 다르다 ({', '.join(f'{k} {meta[k]} → {now[k]}' for k in diff)}). "
            "물리 기준이 바뀌었으면 모델을 다시 학습해야 한다.", RuntimeWarning)


def artifact_status(model_path: Path | str | None = None, knn_path: Path | str | None = None) -> dict:
    """자세 추천 산출물 상태 (E 대시보드 표시용). 경고를 내지 않고, 캐시도 쓰지 않는다.

    반환::

        {"current": {nozzle_layout_hash, physics_hash, kinetics_enabled, time_constant_s} | None,  # 설정을 못 읽으면 None
         "current_error": str | None,
         "flow": 항목, "knn": 항목}

        항목 = {"path": str, "exists": bool,
                "stamp": {nozzle_layout_hash, physics_hash, kinetics_enabled, time_constant_s,
                          body_model, patches_per_m2, commit},  # 없는 키는 None (kinetics 는 계획 산출물만)
                "match": bool | None,         # 도장 키가 전부 있고 전부 같으면 True, 다른 키가 있으면 False,
                                              # 도장이 없거나 일부만 있거나 산출물·설정을 못 읽으면 None
                "mismatched": [키 ...],       # 다른 키 (STAMP_KEYS 와, 산출물에 있으면 KINETICS_KEYS 중)
                "error": str | None}          # 읽기 실패 (torch 없음 등)
    """
    try:
        now, now_error = current_stamp(), None
    except Exception as exc:
        now, now_error = None, f"{type(exc).__name__}: {exc}"

    def entry(path: Path, loader) -> dict:
        out = {"path": str(path), "exists": path.exists(), "stamp": {}, "match": None, "mismatched": [],
               "error": None}
        if not out["exists"]:
            return out
        try:
            meta = loader(path).meta
        except Exception as exc:             # torch 없음, 손상된 파일 등
            out["error"] = f"{type(exc).__name__}: {exc}"
            return out
        out["stamp"] = {k: meta.get(k) for k in STAMP_KEYS + KINETICS_KEYS + INFO_KEYS}
        if now is not None:
            # 다른 키가 하나라도 있으면 False. 전부 찍혀 있고 전부 같을 때만 True. 그 밖(도장 없음·일부만)은 확인 불가 None.
            out["mismatched"] = stamp_mismatch(meta, now)
            if out["mismatched"]:
                out["match"] = False
            elif all(meta.get(k) is not None for k in STAMP_KEYS):
                out["match"] = True
        return out

    return {"current": now, "current_error": now_error,
            "flow": entry(Path(model_path or DEFAULT_MODEL_PATH), PoseFlow.load),
            "knn": entry(Path(knn_path or DEFAULT_KNN_PATH), PoseKNN.load)}


@lru_cache(maxsize=4)
def _load_cached(path: str) -> PoseFlow:
    model = PoseFlow.load(path)
    _check_stamp(model.meta, Path(path))
    return model


@lru_cache(maxsize=4)
def _load_knn_cached(path: str) -> PoseKNN:
    table = PoseKNN.load(path)
    _check_stamp(table.meta, Path(path))
    return table


def load_knn(path: Path | str | None = None) -> PoseKNN:
    """kNN 표 로드 (경로별 캐시). 없으면 FileNotFoundError."""
    return _load_knn_cached(str(Path(path or DEFAULT_KNN_PATH).resolve()))


def load_model(path: Path | str | None = None, *, kind: str | None = None) -> PoseFlow:
    """산출물 로드 (경로별 캐시). 없으면 FileNotFoundError. kind 를 주면 출력 공간이 다를 때 ValueError."""
    default = DEFAULT_PLAN_MODEL_PATH if kind == "plan" else DEFAULT_MODEL_PATH
    p = Path(path) if path is not None else default
    if not p.exists():
        raise FileNotFoundError(f"모델 산출물이 없다: {p} (scripts/train_pose_flow.py 로 만든다)")
    model = _load_cached(str(p.resolve()))
    if kind is not None and model.space.kind != kind:
        raise ValueError(f"{p.name} 은 {model.space.kind} 모델이다 (필요: {kind})")
    return model


@lru_cache(maxsize=4)
def _rescore_evaluator(body_model: str, patches_per_m2: float, plan: bool = False) -> Evaluator:
    """패치판 재채점기. plan=True 면 C 의 plan_physics_cfg() (시간 의존 제거 kinetics 켬, 계획 데이터셋과 같은 설정)."""
    from functools import partial

    from airis.sim.body import build_body
    from airis.sim.patch_baseline import PatchEvaluator

    return PatchEvaluator(_physics_cfg(plan), partial(build_body, model=body_model),
                          patches_per_m2=patches_per_m2)


def _physics_cfg(plan: bool) -> dict:
    """자세 경로는 설정 파일 그대로, 계획 경로는 plan_physics_cfg (docs/plan_extension.md 4절)."""
    from airis.sim.scenario import load_physics

    if plan:
        from airis.optimize.plan_encoding import plan_physics_cfg

        return plan_physics_cfg()
    return load_physics()


@lru_cache(maxsize=1)
def _default_nozzles() -> NozzleConfig:
    from airis.optimize import cli

    return cli.resolve_nozzles()[0]


def default_rescorer(model: PoseFlow | PoseKNN | dict, *, plan: bool = False) -> tuple[Evaluator, NozzleConfig]:
    """학습 데이터와 같은 몸 모델·패치 밀도의 패치판 (점수를 데이터셋 score 와 비교할 수 있게).

    plan=True 면 계획 채점용 설정(plan_physics_cfg, kinetics 켬)이다. 계획 데이터셋도 같은 함수로 만든다.
    """
    from airis.optimize.dataset import configured_body_model

    meta = model if isinstance(model, dict) else model.meta
    body_model = str(meta.get("body_model") or configured_body_model())
    density = float(meta.get("patches_per_m2") or 400.0)
    return _rescore_evaluator(body_model, density, plan), _default_nozzles()


def _gather(body: BodyParams, scenario: Scenario, *, backend: str, n_flow: int, n_knn: int, seed: int,
            path: Path | str | None, knn_path: Path | str | None,
            extra_candidates: Sequence[PoseParams]) -> tuple[list[PoseParams], list[str], dict]:
    """후보와 출처를 모은다. 산출물이 없거나 torch 가 없으면 있는 쪽으로 돌아간다 (경고 1회)."""
    from airis.optimize.encoding import PoseEncoder

    want_flow = backend in ("hybrid", "flow") and n_flow > 0
    want_knn = backend in ("hybrid", "knn") and n_knn > 0
    model = knn = None
    missing: list[str] = []

    if want_flow:
        try:
            model = load_model(path, kind="pose")
        except FileNotFoundError as exc:
            missing.append(f"flow 산출물 없음({exc})")
        except ImportError as exc:
            missing.append(f"flow 를 쓸 수 없음(torch: {exc})")
    if want_knn:
        try:
            knn = load_knn(knn_path)
        except FileNotFoundError as exc:
            missing.append(f"kNN 표 없음({exc})")

    if model is None and knn is None:
        if backend == "flow":                       # 한 쪽만 쓰라고 했으면 그 실패를 그대로 올린다
            load_model(path, kind="pose")
        if backend == "knn":
            load_knn(knn_path)
        reason = "; ".join(missing)
        if any("torch" in m for m in missing) and not any("없음" in m for m in missing):
            raise ImportError(f"flow 도 kNN 도 쓸 수 없다: {reason}")
        raise FileNotFoundError(
            f"자세 모델 산출물이 없다: {reason} "
            f"(scripts/train_pose_flow.py, scripts/build_pose_knn.py 로 만든다)")
    if missing and backend == "hybrid":
        warnings.warn(f"혼합 추천에서 일부만 쓴다: {'; '.join(missing)}", RuntimeWarning)

    names = set()
    for m in (model, knn):
        if m is not None:
            names |= set(m.scenario_names)
    if scenario.name not in names:
        raise KeyError(f"학습에 없던 시나리오: {scenario.name} (학습: {sorted(names)})")

    candidates: list[PoseParams] = []
    sources: list[str] = []
    if model is not None and scenario.name in model.scenario_names:
        flow_cands = model.sample_poses(body, scenario, max(1, int(n_flow)), seed=seed)
        candidates += flow_cands
        sources += ["flow"] * len(flow_cands)
    if knn is not None and scenario.name in knn.scenario_names:
        knn_cands = knn.candidates(body, scenario, int(n_knn))
        candidates += knn_cands
        sources += ["knn"] * len(knn_cands)
    if extra_candidates:
        enc = PoseEncoder(scenario)
        extra = [enc.clip_pose(p) for p in extra_candidates]
        candidates += extra
        sources += ["extra"] * len(extra)
    return candidates, sources, (model.meta if model is not None else knn.meta)


def _pose_gap(a: PoseParams, b: PoseParams) -> float:
    """두 자세의 관절 각도 차 최대값(°). yaw 는 접은 값(좌우 거울·앞뒤 등가는 점수가 같다)으로 비교한다."""
    from airis.optimize.e4 import fold_yaw

    va, vb = a.to_vector().astype(np.float64), b.to_vector().astype(np.float64)
    iy = POSE_YAW_INDEX
    va[iy], vb[iy] = fold_yaw(va[iy]), fold_yaw(vb[iy])
    return float(np.max(np.abs(va - vb)))


def dedup_candidates(candidates: Sequence[PoseParams], sources: Sequence[str],
                     tol_deg: float) -> tuple[list[PoseParams], list[str], int]:
    """관절 각도 차 최대값이 tol_deg 이하인 후보를 하나로 묶는다. (남은 후보, 출처, 뺀 수).

    고정 후보("extra")는 전부 남긴다 (E 가 extra 안의 순서로 표 항목과 짝짓는다). flow·kNN 후보는 앞에서부터
    보며, 고정 후보나 앞서 남긴 후보와 겹치면 뺀다. 순서는 바꾸지 않으므로 결과는 결정적이다.
    """
    if tol_deg <= 0:
        return list(candidates), list(sources), 0
    extras = [c for c, s in zip(candidates, sources) if s == "extra"]
    kept_other: list[PoseParams] = []
    out_c: list[PoseParams] = []
    out_s: list[str] = []
    for c, s in zip(candidates, sources):
        if s != "extra":
            if any(_pose_gap(c, k) <= tol_deg for k in extras + kept_other):
                continue
            kept_other.append(c)
        out_c.append(c)
        out_s.append(s)
    return out_c, out_s, len(candidates) - len(out_c)


def score_candidates(evaluator: Evaluator, poses: Sequence[PoseParams], nozzle: NozzleConfig, body: BodyParams,
                     scenario: Scenario, *, n_threads: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """score_batch 와 같은 (점수, 불가). n_threads > 1 이면 후보마다 스레드로 나눠 채점한다 (순서 유지).

    D 확인(2026-09-30): 패치판은 같은 인스턴스를 여러 스레드가 동시에 불러도 결과가 비트 단위로 같고,
    속도는 스레드 2~3개에서 1.6~1.7배로 포화한다 (GIL).
    batch_evaluate 를 재정의한 평가기(입자판)는 스레드 채점을 확인하지 않았으므로 순차로 채점한다 (경고).
    """
    from airis.optimize.cmaes_runner import score_batch

    poses = list(poses)
    if n_threads > 1 and type(evaluator).batch_evaluate is not Evaluator.batch_evaluate:
        warnings.warn(f"{type(evaluator).__name__} 는 batch_evaluate 를 재정의한 평가기라 스레드 채점의 동일성이 "
                      "확인되지 않았다. 순차로 채점한다.", RuntimeWarning)
        n_threads = 1
    if n_threads <= 1 or len(poses) <= 1:
        return score_batch(evaluator, poses, nozzle, body, scenario)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=int(n_threads)) as pool:
        parts = list(pool.map(lambda p: score_batch(evaluator, [p], nozzle, body, scenario), poses))
    return (np.concatenate([sc for sc, _ in parts]).astype(np.float64),
            np.concatenate([bad for _, bad in parts]).astype(bool))


def _pick(scores: np.ndarray, infeasible: np.ndarray) -> int:
    """가능한 후보 중 최고. 전부 불가면 벌점이 가장 작은(벽을 가장 적게 넘는) 후보. 동률은 앞 후보."""
    pick = np.where(infeasible, -np.inf, scores) if not infeasible.all() else scores
    return int(np.argmax(pick))


def enlarged_body(body: BodyParams, margin: float) -> BodyParams:
    """체형 값 전부에 (1 + margin) 을 곱한 체형. 부스 안 판정에 여유를 둘 때 쓴다."""
    return type(body)(**{f.name: float(getattr(body, f.name)) * (1.0 + float(margin)) for f in fields(body)})


def margin_pick(evaluator: Evaluator, candidates: Sequence[PoseParams], scores: np.ndarray, infeasible: np.ndarray,
                considered: Sequence[int], nozzle: NozzleConfig, body: BodyParams, scenario: Scenario,
                margin: float) -> tuple[int, int, int, bool]:
    """부스 안 판정에 여유를 둔 선택. (고른 자리, 다시 채점한 횟수, 넘긴 후보 수, 폴백 여부).

    체형 입력에 오차가 있으면(키를 작게 추정) 추정 체형에서는 부스 안이던 자세가 실제 몸에서는 천장·벽을
    넘을 수 있다 (docs/experiments_model.md 7절). 그래서 점수 순으로 후보를 보며 **키운 체형**
    (체형 값 × (1 + margin))에서도 부스 안인 첫 후보를 고른다. 점수는 추정 체형의 것을 그대로 쓴다.
    considered 는 최종 평가기로 채점한 후보의 자리다. 추정 체형에서 불가인 후보는 보지 않는다.
    전부 걸리면 여유 없이 고른 후보로 돌아간다 (폴백, 불가 추천을 늘리지는 않는다).
    """
    from airis.optimize.cmaes_runner import score_batch

    considered = list(considered)
    base = considered[_pick(scores[considered], infeasible[considered])]
    order = sorted((j for j in considered if not infeasible[j]), key=lambda j: (-float(scores[j]), j))
    big = enlarged_body(body, margin)
    checks = 0
    for rejected, j in enumerate(order):
        _, bad = score_batch(evaluator, [candidates[j]], nozzle, big, scenario)
        checks += 1
        if not bool(bad[0]):
            return j, checks, rejected, False
    return base, checks, len(order), True


def predict(body: BodyParams, scenario: Scenario, *, backend: str = "hybrid",
            n_flow: int = N_FLOW, n_knn: int = N_KNN, n_samples: int | None = None,
            rescore: bool = True, seed: int = 0, path: Path | str | None = None,
            knn_path: Path | str | None = None,
            evaluator: Evaluator | None = None, nozzle: NozzleConfig | None = None,
            extra_candidates: Sequence[PoseParams] = (),
            dedup_deg: float = 0.0, screen_density: float | None = None, screen_top: int = 3,
            screen_evaluator: Evaluator | None = None, screen_keep_extra: bool = False,
            n_threads: int = 1, feasibility_margin: float = 0.0) -> Prediction:
    """후보까지 돌려주는 예측. evaluator·nozzle 을 주지 않으면 default_rescorer.

    backend "hybrid"(기본) = flow n_flow 개 + kNN n_knn 개 + extra_candidates, "flow"·"knn" 은 한 쪽만 쓴다.
    n_samples 를 주면 쓰는 쪽 후보 수를 그 값으로 맞춘다 (backend="flow"·"knn" 의 예전 인자 이름).
    extra_candidates 는 시나리오 제약으로 투영해 함께 재채점한다 (rescore=False 면 첫 후보를 그대로 돌려준다).

    재채점 단축 옵션 (기본값이면 지금까지와 같은 동작. 총괄 2026-09-30, E5 로 비교 중):
    - dedup_deg > 0: 관절 각도 차 최대값이 이 값 이하인 후보를 하나로 묶는다 (dedup_candidates, 고정 후보는 남김).
    - screen_density: 모든 후보를 이 패치 밀도로 먼저 채점하고 상위 screen_top 개만 최종 평가기로 다시 채점한다.
      선별 평가기는 screen_evaluator 로 줄 수 있다. 주지 않으면 **evaluator 를 줬더라도** 산출물 meta 의 몸 모델로
      만든 패치판이다 (E 처럼 자기 평가기를 넘기는 쪽은 screen_evaluator 도 같은 몸 모델로 넘긴다).
      선별했으면 scores 는 최종 채점한 후보만 값이 있고 나머지는 NaN, infeasible 은 그 밖의 후보에서 선별 결과다.
      **선별을 켜면 고정 후보(표)보다 나빠질 수 있다** (저밀도에서 표 후보가 탈락할 수 있다).
      screen_keep_extra=True 면 고정 후보는 선별 결과와 상관없이 최종 채점에 넣어 표보다 나빠지지 않게 한다.
      선별 상위가 최종 밀도에서 전부 불가면 나머지 후보도 최종 평가기로 채점해 가능한 후보를 찾는다.
    - n_threads > 1: 채점을 스레드로 나눈다 (score_candidates). 패치판처럼 batch_evaluate 를 재정의하지 않은
      평가기 전용이고, 재정의한 평가기(입자판)에서는 순차로 채점한다 (경고).
    - 중복 제거도 고정 후보는 남기므로 표보다 나빠지지 않지만, R0(전부 채점)보다는 나빠질 수 있다
      (3° 안에서도 점수가 크게 바뀌는 자세가 있다, docs/experiments_model.md 5절).

    부스 안 판정 여유 (기본 0 이면 지금까지와 같은 동작이고 추가 채점도 없다. 총괄 2026-10-04):
    - feasibility_margin > 0: 점수 순으로 후보를 보며, 체형 값을 (1 + 여유) 배로 키운 몸에서도 부스 안인 첫 후보를
      고른다 (margin_pick). 체형 입력 오차로 실제 몸이 더 클 때 천장·벽을 넘는 추천을 막는다. 추가 비용은 보통
      채점 1회다. 전부 걸리면 여유 없이 고른 후보를 돌려주고 margin_fallback 을 켠다.
    """

    if backend not in BACKENDS:
        raise ValueError(f"backend 는 {BACKENDS} 중 하나: {backend!r}")
    if n_samples is not None:
        if backend == "flow":
            n_flow, n_knn = int(n_samples), 0
        elif backend == "knn":
            n_flow, n_knn = 0, int(n_samples)
        else:
            raise ValueError("n_samples 는 backend='flow'·'knn' 에서만 쓴다 (hybrid 는 n_flow·n_knn)")

    candidates, sources, meta = _gather(
        body, scenario, backend=backend, n_flow=n_flow, n_knn=n_knn, seed=seed, path=path,
        knn_path=knn_path, extra_candidates=extra_candidates if rescore else ())
    if not rescore or len(candidates) == 1:
        return Prediction(candidates[0], candidates, None, None, sources, sources[0])

    candidates, sources, n_dropped = dedup_candidates(candidates, sources, dedup_deg)
    if evaluator is None or nozzle is None:
        ev, nz = default_rescorer(meta)
        evaluator, nozzle = evaluator or ev, nozzle or nz

    screening = screen_density is not None or screen_evaluator is not None
    if not screening or len(candidates) <= max(1, int(screen_top)):
        scores, infeasible = score_candidates(evaluator, candidates, nozzle, body, scenario, n_threads=n_threads)
        i = _pick(scores, infeasible)
        out = Prediction(candidates[i], candidates, scores, infeasible, sources, sources[i],
                         n_dropped=n_dropped, n_rescored=len(candidates))
        return _with_margin(out, range(len(candidates)), evaluator, nozzle, body, scenario, feasibility_margin)

    if screen_evaluator is None:
        from airis.optimize.dataset import configured_body_model

        screen_evaluator = _rescore_evaluator(str(meta.get("body_model") or configured_body_model()),
                                              float(screen_density))
    screen, screen_bad = score_candidates(screen_evaluator, candidates, nozzle, body, scenario,
                                          n_threads=n_threads)
    # 선별 순위: 가능한 후보 먼저, 그 안에서 점수 내림차순, 동률은 앞 후보 (안정 정렬이라 결정적)
    key = np.where(screen_bad, -np.inf, screen) if not screen_bad.all() else screen
    order = np.argsort(-key, kind="stable").tolist()
    top = set(order[:max(1, int(screen_top))])
    if screen_keep_extra:                    # 고정 후보는 선별에서 떨어뜨리지 않는다 (표보다 나빠지지 않게)
        top |= {j for j, src in enumerate(sources) if src == "extra"}
    top = sorted(top)
    scores = np.full(len(candidates), np.nan)
    infeasible = screen_bad.copy()
    final, final_bad = score_candidates(evaluator, [candidates[j] for j in top], nozzle, body, scenario,
                                        n_threads=n_threads)
    scores[top], infeasible[top] = final, final_bad
    rescored = list(top)
    if final_bad.all():
        # 최종 밀도에서 선별 상위가 전부 불가면 나머지 후보도 최종 평가기로 채점한다 (불가 추천은 안전 문제)
        rest = [j for j in order if j not in set(top)]
        if rest:
            r_sc, r_bad = score_candidates(evaluator, [candidates[j] for j in rest], nozzle, body, scenario,
                                           n_threads=n_threads)
            scores[rest], infeasible[rest] = r_sc, r_bad
            rescored += rest
    rescored.sort()
    i = rescored[_pick(scores[rescored], infeasible[rescored])]
    out = Prediction(candidates[i], candidates, scores, infeasible, sources, sources[i],
                     n_dropped=n_dropped, screen_scores=screen, n_rescored=len(rescored))
    return _with_margin(out, rescored, evaluator, nozzle, body, scenario, feasibility_margin)


def _with_margin(p: Prediction, considered: Sequence[int], evaluator: Evaluator, nozzle: NozzleConfig,
                 body: BodyParams, scenario: Scenario, margin: float) -> Prediction:
    """feasibility_margin > 0 이면 margin_pick 으로 선택을 바꾼다. 0 이면 그대로 (추가 채점 없음)."""
    if not margin or margin <= 0:
        return p
    i, checks, rejected, fallback = margin_pick(evaluator, p.candidates, p.scores, p.infeasible, considered,
                                                nozzle, body, scenario, margin)
    p.pose, p.source = p.candidates[i], p.sources[i]
    p.n_margin_checks, p.n_margin_rejected, p.margin_fallback = checks, rejected, fallback
    return p


def predict_pose(body: BodyParams, scenario: Scenario, **kwargs) -> PoseParams:
    """interfaces.md 계약. 항상 PoseEncoder(scenario).clip_pose() 를 거친 자세 (fixed_pose 적용)."""
    return predict(body, scenario, **kwargs).pose


def score_plans(evaluator: Evaluator, plans: Sequence[Plan], nozzle: NozzleConfig, body: BodyParams,
                scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """계획 목록을 evaluate_plan 으로 채점. (점수 (n,), 불가 (n,) bool)."""
    from airis.optimize.cmaes_runner import is_infeasible

    results = [evaluator.evaluate_plan(p, nozzle, body, scenario) for p in plans]
    return (np.array([r.score for r in results], dtype=np.float64),
            np.array([is_infeasible(r) for r in results], dtype=bool))


@lru_cache(maxsize=1)
def _default_plan_limits():
    from airis.optimize.plan_encoding import PlanLimits
    from airis.sim.scenario import load_physics

    return PlanLimits.from_config(load_physics())


def plan_encoder(scenario: Scenario, *, limits=None, zone_nozzle_counts: Sequence[float] | None = None,
                 nozzle: NozzleConfig | None = None):
    """계획 후보를 투영할 C 의 PlanEncoder. limits 를 주지 않으면 configs/physics.yaml 의 plan.* · fan.*.

    zone_nozzle_counts 는 풍량 한도 보수의 구역별 노즐 수다. 주지 않으면 실제 장비 구성
    `airis.sim.scenario.zone_nozzle_counts(nozzle)` (기준 배치 [2, 2, 2, 2, 4]). 계획 데이터셋을 만든 C 의
    PlanEncoder 기본값과 같다 (통합 2026-09-30, 데이터셋 stamp 열 zone_nozzle_counts).
    """
    from airis.optimize.plan_encoding import PlanEncoder
    from airis.sim.scenario import zone_nozzle_counts as counts_of

    if zone_nozzle_counts is None:
        zone_nozzle_counts = counts_of(nozzle if isinstance(nozzle, NozzleConfig) else None)
    return PlanEncoder(scenario, limits if limits is not None else _default_plan_limits(), zone_nozzle_counts)


def predict_plan_candidates(body: BodyParams, scenario: Scenario, *, n_samples: int = N_SAMPLES,
                            rescore: bool = True, seed: int = 0, path: Path | str | None = None,
                            evaluator: Evaluator | None = None, nozzle: NozzleConfig | None = None,
                            extra_candidates: Sequence[Plan] = (), limits=None,
                            zone_nozzle_counts: Sequence[float] | None = None) -> PlanPrediction:
    """계획 후보까지 돌려주는 예측. predict 의 계획 버전.

    후보는 전부 plan_encoder(scenario).clip_plan 을 거친 뒤 채점한다 (rescore=False 여도 투영한다).
    """
    model = load_model(path, kind="plan")
    if scenario.name not in model.scenario_names:
        raise KeyError(f"학습에 없던 시나리오: {scenario.name} (학습: {model.scenario_names})")
    candidates = model.sample_plans(body, scenario, max(1, int(n_samples)), seed=seed)
    if rescore and extra_candidates:
        candidates = candidates + list(extra_candidates)
    enc = plan_encoder(scenario, limits=limits, zone_nozzle_counts=zone_nozzle_counts, nozzle=nozzle)
    candidates = [enc.clip_plan(p) for p in candidates]
    if not rescore or len(candidates) == 1:
        return PlanPrediction(candidates[0], candidates, None, None)

    if evaluator is None or nozzle is None:
        ev, nz = default_rescorer(model, plan=True)
        evaluator, nozzle = evaluator or ev, nozzle or nz
    scores, infeasible = score_plans(evaluator, candidates, nozzle, body, scenario)
    pick = np.where(infeasible, -np.inf, scores) if not infeasible.all() else scores
    return PlanPrediction(candidates[int(np.argmax(pick))], candidates, scores, infeasible)


def predict_plan(body: BodyParams, scenario: Scenario, **kwargs) -> Plan:
    """interfaces.md "계획 모델" 계약. 자세는 시나리오 제약 안, 시간·구역 세기는 범위 안이다."""
    return predict_plan_candidates(body, scenario, **kwargs).plan
