"""체형 → 자세 추천. 소유자: C. docs/interfaces.md "회귀 모델" 계약, docs/proposals/flow_matching.md.

    predict_pose(body, scenario) -> PoseParams

조건부 flow matching 모델(flow.py)에서 자세 후보를 n_samples 개 뽑고, 시나리오 제약 안으로 투영(clip_pose)한 뒤
패치판으로 다시 채점해 가장 높은 것을 고른다. 부스 밖 후보는 평가기의 불가 판정으로 탈락한다.
재채점을 끄면(rescore=False) 첫 샘플을 그대로 돌려준다 (E5 비교용).

예외 규약 (E 의 recommend 가 스텁으로 폴백할 때 구분한다):
- torch 가 없으면 이 모듈 import 에서 ImportError
- 산출물(DEFAULT_MODEL_PATH)이 없으면 FileNotFoundError
- 학습에 없던 시나리오면 KeyError
그 외 예외는 삼키지 않는다.

산출물에는 학습 데이터의 nozzle_layout_hash · physics_hash · 학습 커밋이 들어 있고, 로드할 때 현재 설정과 다르면
경고한다. 물리 기준이 바뀌면 모델은 무효다 (docs/experiments.md 규칙).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch  # noqa: F401  계약: torch 가 없으면 import 단계에서 ImportError

from airis.sim import BodyParams, Evaluator, NozzleConfig, PoseParams, Scenario

from .flow import PoseFlow

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "data" / "models" / "pose_flow.pt"
N_SAMPLES = 16


@dataclass
class Prediction:
    pose: PoseParams
    candidates: list[PoseParams]
    scores: np.ndarray | None        # 재채점 점수 (rescore=False 면 None)
    infeasible: np.ndarray | None


def _check_stamp(model: PoseFlow, path: Path) -> None:
    """학습 때의 노즐·물리 설정 해시가 지금과 다르면 경고."""
    from airis.optimize import cli, explog

    want = {"nozzle_layout_hash": model.meta.get("nozzle_layout_hash"),
            "physics_hash": model.meta.get("physics_hash")}
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
    _check_stamp(model, Path(path))
    return model


def load_model(path: Path | str | None = None) -> PoseFlow:
    """산출물 로드 (경로별 캐시). 없으면 FileNotFoundError."""
    p = Path(path) if path is not None else DEFAULT_MODEL_PATH
    if not p.exists():
        raise FileNotFoundError(f"자세 모델 산출물이 없다: {p} (scripts/train_pose_flow.py 로 만든다)")
    return _load_cached(str(p.resolve()))


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


def default_rescorer(model: PoseFlow) -> tuple[Evaluator, NozzleConfig]:
    """학습 데이터와 같은 몸 모델·패치 밀도의 패치판 (점수를 데이터셋 score 와 비교할 수 있게)."""
    from airis.optimize.dataset import configured_body_model

    body_model = str(model.meta.get("body_model") or configured_body_model())
    density = float(model.meta.get("patches_per_m2") or 400.0)
    return _rescore_evaluator(body_model, density), _default_nozzles()


def predict(body: BodyParams, scenario: Scenario, *, n_samples: int = N_SAMPLES, rescore: bool = True,
            seed: int = 0, path: Path | str | None = None,
            evaluator: Evaluator | None = None, nozzle: NozzleConfig | None = None) -> Prediction:
    """후보까지 돌려주는 예측. evaluator·nozzle 을 주지 않으면 default_rescorer."""
    from airis.optimize.cmaes_runner import score_batch

    model = load_model(path)
    if scenario.name not in model.scenario_names:
        raise KeyError(f"학습에 없던 시나리오: {scenario.name} (학습: {model.scenario_names})")
    candidates = model.sample_poses(body, scenario, max(1, int(n_samples)), seed=seed)
    if not rescore or len(candidates) == 1:
        return Prediction(candidates[0], candidates, None, None)

    if evaluator is None or nozzle is None:
        ev, nz = default_rescorer(model)
        evaluator, nozzle = evaluator or ev, nozzle or nz
    scores, infeasible = score_batch(evaluator, candidates, nozzle, body, scenario)
    # 가능한 후보 중 최고. 전부 불가면 벌점이 가장 작은(벽을 가장 적게 넘는) 후보.
    pick = np.where(infeasible, -np.inf, scores) if not infeasible.all() else scores
    return Prediction(candidates[int(np.argmax(pick))], candidates, scores, infeasible)


def predict_pose(body: BodyParams, scenario: Scenario, **kwargs) -> PoseParams:
    """interfaces.md 계약. 항상 PoseEncoder(scenario).clip_pose() 를 거친 자세 (fixed_pose 적용)."""
    return predict(body, scenario, **kwargs).pose
