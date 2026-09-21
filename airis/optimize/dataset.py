"""회귀용 데이터셋 생성. 소유자: C. docs/tracks/C_optimize.md 단계 9, docs/interfaces.md 데이터셋 스키마·회귀 계약.

체형을 시드 고정으로 샘플링하고, 체형 × 시나리오마다 CMA-ES(두 시작점)로 최적 자세를 찾아
parquet 한 행으로 남긴다.

체형 분포 (C_optimize.md 단계 9). 키는 두 몸 모델 공통 N(1.68, 0.09), clip [1.45, 1.95].
나머지 4개는 `비율·height + N(0, σ)` 이고 몸 모델(configs/physics.yaml body.model)마다 정의가 다르다.

    capsule (캡슐 마네킹 시절 정의)       비율     σ
      shoulder_width_m                    0.245    0.015
      torso_depth_m                       0.13     0.015
      arm_length_m                        0.365    0.02
      leg_length_m                        0.50     0.025
    mesh (관절 중심 정의, docs/mesh_transition.md 8번; 비율 = human_mesh.MESH_DEFAULT_BODY / 1.70,
          σ = capsule σ 를 같은 변동계수로 옮긴 값)
      shoulder_width_m  어깨 관절 간격     0.2012   0.012
      torso_depth_m     유두선 가슴 두께   0.1141   0.013
      arm_length_m      상완 + 전완        0.2724   0.015
      leg_length_m      고관절 높이        0.5194   0.026
    두 모델은 같은 난수열을 쓰므로 같은 시드의 i 번 체형은 키가 같다.

열 (interfaces.md 스키마 + 회귀 계약 요구):
    body_idx, body_*                      체형 번호와 BodyParams 필드
    scenario                              시나리오 이름
    pose_*                                최적 자세. **pose_torso_yaw 는 0~90° 로 접은 값**
                                          (좌우 |yaw| + 앞뒤 90° − |90° − |yaw||, e4.fold_yaw, interfaces.md 회귀 계약)
    pose_torso_yaw_raw                    접기 전 yaw
    arm_class                             팔 봉우리 hands_up(벌림 ≥ 90°) / arms_down — 회귀의 봉우리 분류 라벨
    start_best_default, start_best_hands_up, peak_gap   시작점별 best 와 차이(hands_up − default)
    hands_up_feasible                     이 체형에서 만세 기준 자세(B2)가 부스 안인가 (큰 체형은 천장 2.15 m 에 걸린다)
    score, total_removal, discomfort      best 자세의 평가값
    n_infeasible, elapsed_s
    nozzle_layout_hash, physics_hash, exp_id, commit, seed, max_evals, popsize, patches_per_m2, body_seed,
    body_model                            체형 샘플링·몸 생성에 쓴 몸 모델 (capsule | mesh)

재개: 출력 parquet 이 있으면 (body_idx, scenario) 키가 있는 작업을 건너뛴다. 설정 해시·체형 시드가 다르면
섞지 않고 멈춘다. flush_every 행마다 파일 전체를 임시 파일에 다시 쓰고 바꿔 넣는다.
병렬: multiprocessing.Pool, 프로세스마다 평가기를 시나리오별로 한 번 만든다.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from airis.sim import BodyParams, PoseParams

POSE_KEYS: list[str] = [f.name for f in fields(PoseParams)]
BODY_KEYS: list[str] = [f.name for f in fields(BodyParams)]
KEY_COLS = ("body_idx", "scenario")

#: 이 값들이 다르면 이어 쓰지 않는다 (다른 물리 기준·다른 탐색 설정의 행이 섞인다).
CONSISTENCY_COLS = ("nozzle_layout_hash", "physics_hash", "body_seed", "max_evals", "popsize",
                    "patches_per_m2", "evaluator", "body_model")

#: 몸 모델별 체형 분포: 필드 → (키 대비 비율, 잡음 표준편차 m). 위 모듈 설명 참고.
BODY_DISTRIBUTIONS: dict[str, dict[str, tuple[float, float]]] = {
    "capsule": {
        "shoulder_width_m": (0.245, 0.015),
        "torso_depth_m": (0.13, 0.015),
        "arm_length_m": (0.365, 0.02),
        "leg_length_m": (0.50, 0.025),
    },
    "mesh": {
        "shoulder_width_m": (0.342 / 1.70, 0.012),
        "torso_depth_m": (0.194 / 1.70, 0.013),
        "arm_length_m": (0.463 / 1.70, 0.015),
        "leg_length_m": (0.883 / 1.70, 0.026),
    },
}


def sample_bodies(n: int, seed: int, model: str = "capsule") -> list[BodyParams]:
    """시드 고정 체형 샘플러. 같은 (n, seed, model) 이면 같은 목록, n 을 늘리면 앞부분은 그대로다."""
    if model not in BODY_DISTRIBUTIONS:
        raise ValueError(f"몸 모델은 {sorted(BODY_DISTRIBUTIONS)} 중 하나: {model!r}")
    dist = BODY_DISTRIBUTIONS[model]
    rng = np.random.default_rng(seed)
    bodies = []
    for _ in range(n):
        # 체형마다 난수 5개를 같은 순서로 뽑아 n 에 무관하게 앞부분이 고정되게 한다.
        z = rng.standard_normal(5)
        height = float(np.clip(1.68 + 0.09 * z[0], 1.45, 1.95))
        values = {key: ratio * height + sigma * float(z[i + 1])
                  for i, (key, (ratio, sigma)) in enumerate(dist.items())}
        bodies.append(BodyParams(height_m=height, **values))
    return bodies


def configured_body_model() -> str:
    """configs/physics.yaml 의 body.model (없으면 capsule). 평가기의 build_body 와 같은 값을 읽는다."""
    from airis.sim.scenario import load_physics

    return str(load_physics().get("body", {}).get("model", "capsule"))


@dataclass(frozen=True)
class DatasetConfig:
    evaluator: str = "patch"
    max_evals: int = 2000
    popsize: int = 50
    starts: str = "default,hands_up"
    patches_per_m2: float = 400.0
    seed: int = 0                 # CMA-ES 시드 기준값. 작업 시드 = seed·100000 + body_idx
    body_seed: int = 0
    dataset_id: str = "ds"
    #: 체형 분포의 몸 모델. None 이면 physics.yaml body.model. 평가기가 만드는 몸과 달라지면 거부한다.
    body_model: str | None = None


# ---------- 작업자 (프로세스마다 평가기 캐시) ----------

_WORKER: dict = {}


def _worker_init(cfg: DatasetConfig) -> None:
    _WORKER.clear()
    _WORKER["cfg"] = cfg


def _evaluator_for(scenario_name: str, body: BodyParams):
    from airis.sim.scenario import load_scenarios

    from . import cli

    cfg: DatasetConfig = _WORKER["cfg"]
    if "scenarios" not in _WORKER:
        _WORKER["scenarios"] = load_scenarios()
        _WORKER["nozzle"], _ = cli.resolve_nozzles()
        _WORKER["evaluators"] = {}
    scenario = _WORKER["scenarios"][scenario_name]
    if scenario_name not in _WORKER["evaluators"]:
        _WORKER["evaluators"][scenario_name] = cli.make_evaluator(
            cfg.evaluator, scenario, body=body, nozzle=_WORKER["nozzle"],
            patches_per_m2=cfg.patches_per_m2)
    return _WORKER["evaluators"][scenario_name], scenario, _WORKER["nozzle"]


def run_task(task: tuple[int, dict, str]) -> dict:
    """체형 하나 × 시나리오 하나를 최적화해 데이터셋 행을 돌려준다 (Pool 작업자에서 호출)."""
    from . import cli, e4, sensitivity
    from .baselines import BASELINES, evaluate_condition
    from .cmaes_runner import run_cmaes
    from .encoding import PoseEncoder

    body_idx, body_dict, scenario_name = task
    cfg: DatasetConfig = _WORKER["cfg"]
    body = BodyParams(**body_dict)
    evaluator, scenario, nozzle = _evaluator_for(scenario_name, body)

    seed = cfg.seed * 100000 + body_idx
    t0 = time.perf_counter()
    result = run_cmaes(
        evaluator, body, scenario, nozzle,
        max_evals=cfg.max_evals, seed=seed, popsize=cfg.popsize,
        starts=cli.parse_starts(cfg.starts),
    )
    b2 = evaluate_condition(evaluator, BASELINES["B2"], PoseEncoder(scenario), nozzle, body, scenario)
    start_best = {ps["start"]: ps["best_score"] for ps in result.per_start}

    row: dict = {"body_idx": body_idx, "scenario": scenario_name}
    row.update({f"body_{k}": float(v) for k, v in body_dict.items()})
    folded = e4.fold_pose(result.best_pose)
    row.update({f"pose_{k}": folded[k] for k in POSE_KEYS})
    row["pose_torso_yaw_raw"] = float(result.best_pose.torso_yaw)
    row["arm_class"] = sensitivity.arm_class(result.best_pose)
    row["start_best_default"] = float(start_best.get("default", np.nan))
    row["start_best_hands_up"] = float(start_best.get("hands_up", np.nan))
    row["peak_gap"] = e4.peak_gap(result.per_start)
    row["hands_up_feasible"] = not b2["infeasible"]
    row["score"] = float(result.best_score)
    row["total_removal"] = float(result.best_result.total_removal)
    row["discomfort"] = float(result.best_result.discomfort)
    row["n_infeasible"] = int(result.n_infeasible)
    row["elapsed_s"] = time.perf_counter() - t0
    row["exp_id"] = f"{cfg.dataset_id}_{body_idx:05d}_{scenario_name}"
    row["seed"] = seed
    return row


# ---------- 저장·재개 ----------

def _read(path: Path):
    import pandas as pd

    return pd.read_parquet(path) if path.exists() else None


def _write_atomic(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def build_dataset(
    out_path: Path | str,
    *,
    n_bodies: int,
    scenarios: list[str],
    cfg: DatasetConfig = DatasetConfig(),
    processes: int = 1,
    flush_every: int = 100,
    log=print,
) -> dict:
    """데이터셋을 만들거나 이어 만든다. {"path", "n_rows", "n_new", "n_skipped"} 를 돌려준다."""
    import multiprocessing as mp

    import pandas as pd

    from . import cli, explog

    out_path = Path(out_path)
    body_model = configured_body_model()
    if cfg.body_model is not None and cfg.body_model != body_model:
        raise ValueError(
            f"체형 분포 몸 모델({cfg.body_model})이 physics.yaml body.model({body_model})과 다르다. "
            f"평가기는 physics.yaml 의 몸으로 평가하므로 섞지 않는다.")
    nozzle, _ = cli.resolve_nozzles()
    stamp = {
        "nozzle_layout_hash": cli.nozzle_hash(nozzle),
        "physics_hash": explog.file_hash(explog.ROOT / "configs" / "physics.yaml"),
        "commit": explog.git_commit(),
        "body_seed": cfg.body_seed,
        "max_evals": cfg.max_evals,
        "popsize": cfg.popsize,
        "patches_per_m2": cfg.patches_per_m2,
        "evaluator": cfg.evaluator,
        "body_model": body_model,
    }

    existing = _read(out_path)
    done: set = set()
    if existing is not None and len(existing):
        if "body_model" not in existing.columns:        # body_model 열 이전 파일은 캡슐판이다
            existing = existing.assign(body_model="capsule")
        for col in CONSISTENCY_COLS:
            vals = set(existing[col].astype(str))
            if vals != {str(stamp[col])}:
                raise ValueError(
                    f"{out_path} 의 {col} 가 {sorted(vals)} 라 이번 실행({stamp[col]})과 섞을 수 없다. "
                    f"다른 파일로 만들거나 기존 파일을 옮겨라.")
        done = set(zip(existing["body_idx"].astype(int), existing["scenario"].astype(str)))

    bodies = sample_bodies(n_bodies, cfg.body_seed, body_model)
    tasks = [(i, asdict(b), s) for i, b in enumerate(bodies) for s in scenarios if (i, s) not in done]
    n_skipped = n_bodies * len(scenarios) - len(tasks)
    log(f"[dataset] {out_path.name}: 몸 {body_model}, 작업 {len(tasks)}개 (건너뜀 {n_skipped}), 프로세스 {processes}")

    frames = [existing] if existing is not None else []
    pending: list[dict] = []
    n_new = 0
    t0 = time.perf_counter()

    def flush():
        nonlocal frames, pending
        if not pending:
            return
        new = pd.DataFrame(pending)
        merged = pd.concat(frames + [new], ignore_index=True) if frames else new
        merged = merged.sort_values(list(KEY_COLS), kind="mergesort").reset_index(drop=True)
        _write_atomic(merged, out_path)
        frames = [merged]
        pending = []

    def consume(row):
        nonlocal n_new
        row.update({k: v for k, v in stamp.items() if k not in row})
        pending.append(row)
        n_new += 1
        if n_new % max(1, flush_every) == 0:
            flush()
        if n_new % 10 == 0 or n_new == len(tasks):
            rate = n_new / max(time.perf_counter() - t0, 1e-9)
            log(f"[dataset] {n_new}/{len(tasks)}  {rate * 60:.1f} 행/분")

    if processes <= 1:
        _worker_init(cfg)
        for task in tasks:
            consume(run_task(task))
    elif tasks:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes, initializer=_worker_init, initargs=(cfg,)) as pool:
            for row in pool.imap_unordered(run_task, tasks, chunksize=1):
                consume(row)
    flush()

    n_rows = len(frames[0]) if frames else 0
    return {"path": out_path, "n_rows": n_rows, "n_new": n_new, "n_skipped": n_skipped}


def height_bin_summary(df, bins=(1.45, 1.60, 1.70, 1.80, 1.85, 1.95)):
    """키 구간 × 시나리오별 만세 봉우리 비율, 만세 가능 비율, 평균 점수 (Q2 체형 의존성)."""
    import pandas as pd

    d = df.copy()
    d["height_bin"] = pd.cut(d["body_height_m"], bins=list(bins), include_lowest=True)
    return (d.groupby(["scenario", "height_bin"], observed=True)
             .agg(n=("score", "size"),
                  hands_up_frac=("arm_class", lambda s: float((s == "hands_up").mean())),
                  hands_up_feasible_frac=("hands_up_feasible", "mean"),
                  peak_gap_mean=("peak_gap", "mean"),
                  score_mean=("score", "mean"),
                  yaw_mean=("pose_torso_yaw", "mean"))
             .reset_index())
