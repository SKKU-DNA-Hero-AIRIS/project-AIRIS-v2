"""체형 → 자세·계획 추천. 소유자: C. docs/interfaces.md "회귀 모델"·"계획 모델" 계약,
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

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from airis.sim import BodyParams, Evaluator, NozzleConfig, Plan, PoseParams, Scenario

from .flow import PoseFlow          # torch 는 flow.py 안에서만 import 한다 (kNN 만으로도 돌아가게)
from .knn import PoseKNN

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "data" / "models" / "pose_flow.pt"
DEFAULT_PLAN_MODEL_PATH = ROOT / "data" / "models" / "plan_flow.pt"
DEFAULT_KNN_PATH = ROOT / "data" / "models" / "pose_knn.parquet"
N_SAMPLES = 16
N_FLOW = 8
N_KNN = 8
BACKENDS = ("hybrid", "flow", "knn")


@dataclass
class Prediction:
    pose: PoseParams
    candidates: list[PoseParams]
    scores: np.ndarray | None        # 재채점 점수 (rescore=False 면 None)
    infeasible: np.ndarray | None
    sources: list[str] = field(default_factory=list)   # 후보마다 "flow" | "knn" | "extra"
    source: str = ""                                   # 고른 후보의 출처


@dataclass
class PlanPrediction:
    plan: Plan
    candidates: list[Plan]
    scores: np.ndarray | None
    infeasible: np.ndarray | None


def _check_stamp(meta: dict, path: Path) -> None:
    """학습 때의 노즐·물리 설정 해시가 지금과 다르면 경고."""
    from airis.optimize import cli, explog

    want = {"nozzle_layout_hash": meta.get("nozzle_layout_hash"),
            "physics_hash": meta.get("physics_hash")}
    try:
        now = {"nozzle_layout_hash": cli.nozzle_hash(cli.resolve_nozzles()[0]),
               "physics_hash": explog.file_hash(explog.ROOT / "configs" / "physics.yaml")}
    except Exception as exc:          # 설정을 못 읽으면 판정만 건너뛴다
        warnings.warn(f"{path.name}: 현재 설정 해시를 읽지 못해 비교하지 않는다 ({exc})", RuntimeWarning)
        return
    diff = [k for k in want if want[k] is not None and str(want[k]) != str(now[k])]
    if diff:
        warnings.warn(
            f"{path.name}: 학습 때와 설정이 다르다 ({', '.join(f'{k} {want[k]} → {now[k]}' for k in diff)}). "
            "물리 기준이 바뀌었으면 모델을 다시 학습해야 한다.", RuntimeWarning)


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
def _rescore_evaluator(body_model: str, patches_per_m2: float) -> Evaluator:
    from functools import partial

    from airis.sim.body import build_body
    from airis.sim.patch_baseline import PatchEvaluator
    from airis.sim.scenario import load_physics

    return PatchEvaluator(load_physics(), partial(build_body, model=body_model),
                          patches_per_m2=patches_per_m2)


@lru_cache(maxsize=1)
def _default_nozzles() -> NozzleConfig:
    from airis.optimize import cli

    return cli.resolve_nozzles()[0]


def default_rescorer(model: PoseFlow | PoseKNN | dict) -> tuple[Evaluator, NozzleConfig]:
    """학습 데이터와 같은 몸 모델·패치 밀도의 패치판 (점수를 데이터셋 score 와 비교할 수 있게)."""
    from airis.optimize.dataset import configured_body_model

    meta = model if isinstance(model, dict) else model.meta
    body_model = str(meta.get("body_model") or configured_body_model())
    density = float(meta.get("patches_per_m2") or 400.0)
    return _rescore_evaluator(body_model, density), _default_nozzles()


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


def predict(body: BodyParams, scenario: Scenario, *, backend: str = "hybrid",
            n_flow: int = N_FLOW, n_knn: int = N_KNN, n_samples: int | None = None,
            rescore: bool = True, seed: int = 0, path: Path | str | None = None,
            knn_path: Path | str | None = None,
            evaluator: Evaluator | None = None, nozzle: NozzleConfig | None = None,
            extra_candidates: Sequence[PoseParams] = ()) -> Prediction:
    """후보까지 돌려주는 예측. evaluator·nozzle 을 주지 않으면 default_rescorer.

    backend "hybrid"(기본) = flow n_flow 개 + kNN n_knn 개 + extra_candidates, "flow"·"knn" 은 한 쪽만 쓴다.
    n_samples 를 주면 쓰는 쪽 후보 수를 그 값으로 맞춘다 (backend="flow"·"knn" 의 예전 인자 이름).
    extra_candidates 는 시나리오 제약으로 투영해 함께 재채점한다 (rescore=False 면 첫 후보를 그대로 돌려준다).
    """
    from airis.optimize.cmaes_runner import score_batch

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

    if evaluator is None or nozzle is None:
        ev, nz = default_rescorer(meta)
        evaluator, nozzle = evaluator or ev, nozzle or nz
    scores, infeasible = score_batch(evaluator, candidates, nozzle, body, scenario)
    # 가능한 후보 중 최고. 전부 불가면 벌점이 가장 작은(벽을 가장 적게 넘는) 후보.
    pick = np.where(infeasible, -np.inf, scores) if not infeasible.all() else scores
    i = int(np.argmax(pick))
    return Prediction(candidates[i], candidates, scores, infeasible, sources, sources[i])


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


def predict_plan_candidates(body: BodyParams, scenario: Scenario, *, n_samples: int = N_SAMPLES,
                            rescore: bool = True, seed: int = 0, path: Path | str | None = None,
                            evaluator: Evaluator | None = None, nozzle: NozzleConfig | None = None,
                            extra_candidates: Sequence[Plan] = ()) -> PlanPrediction:
    """계획 후보까지 돌려주는 예측. predict 의 계획 버전."""
    model = load_model(path, kind="plan")
    if scenario.name not in model.scenario_names:
        raise KeyError(f"학습에 없던 시나리오: {scenario.name} (학습: {model.scenario_names})")
    candidates = model.sample_plans(body, scenario, max(1, int(n_samples)), seed=seed)
    if rescore and extra_candidates:
        candidates = candidates + list(extra_candidates)
    if not rescore or len(candidates) == 1:
        return PlanPrediction(candidates[0], candidates, None, None)

    if evaluator is None or nozzle is None:
        ev, nz = default_rescorer(model)
        evaluator, nozzle = evaluator or ev, nozzle or nz
    scores, infeasible = score_plans(evaluator, candidates, nozzle, body, scenario)
    pick = np.where(infeasible, -np.inf, scores) if not infeasible.all() else scores
    return PlanPrediction(candidates[int(np.argmax(pick))], candidates, scores, infeasible)


def predict_plan(body: BodyParams, scenario: Scenario, **kwargs) -> Plan:
    """interfaces.md "계획 모델" 계약. 자세는 시나리오 제약 안, 시간·구역 세기는 범위 안이다."""
    return predict_plan_candidates(body, scenario, **kwargs).plan
