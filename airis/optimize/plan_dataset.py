"""계획 데이터셋 (체형 × 시나리오 → 최적 계획). 소유자: C.

자세 데이터셋(`dataset.py`)의 계획판이다. 한 행 = 체형 하나 × 시나리오 하나이고, 그 조합의
**단계 N개 계획**을 따뜻한 시작으로 최적화해 담는다. F 의 계획 추천 모델이 학습·평가에 쓴다.
열 규약은 `docs/interfaces.md` "계획 데이터셋 스키마" 절(C·F 합의 2026-10-06)이 기준이다.

    plan_<키>            8N + 5 열 (p1_shoulder_abduction … duration_s, zone_*). PlanEncoder 순서
    score · total_removal · energy · duration_s · discomfort
    cand_plan_raw        후보마다 raw_vector — **대칭 정규화(normalize) 뒤 원래 단위**, [-1,1] 아님
    cand_score · n_candidates
    도장 10개            physics_hash, nozzle_layout_hash, body_model, patches_per_m2, commit,
                        kinetics_enabled, time_constant_s, zone_nozzle_counts(문자열), n_phases,
                        energy_weight
    한도 6개             duration_lo_s, duration_hi_s, min_phase_s, transition_s, s_max, cap_ratio
                        (실행이 하한을 N × min_phase_s 로 올리므로 설정 파일 값과 다를 수 있다)
    선택 열              score_p1_10, score_p1opt_10, pose_<키> (단일 자세 최적)

후보 거리는 **encode() 공간에서 재고 차원으로 나눈다**(÷√dim). 원래 단위로 재면 yaw 180° 차이와
시간 2 s 차이가 같은 무게가 된다. 대칭 정규화를 거친 뒤 재므로 거울상이 다른 후보로 잡히지 않는다.

이어 만들기·atomic 쓰기·도장 섞임 거부는 자세 데이터셋과 같은 규칙이다.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from airis.sim import BodyParams, PoseParams

from .dataset import BODY_KEYS, POSE_KEYS, configured_body_model, sample_bodies

#: 행을 가리키는 키 (이어 만들 때 이미 한 조합을 건너뛴다).
KEY_COLS = ("body_idx", "scenario")
#: 이 열들이 기존 파일과 다르면 섞지 않는다 (자세 데이터셋 CONSISTENCY_COLS 의 계획판).
CONSISTENCY_COLS = ("nozzle_layout_hash", "physics_hash", "body_seed", "max_evals", "popsize",
                    "patches_per_m2", "evaluator", "body_model", "n_phases", "energy_weight",
                    "kinetics_enabled", "time_constant_s", "zone_nozzle_counts",
                    "duration_lo_s", "duration_hi_s", "min_phase_s",
                    # 후보 선정·따뜻한 시작 설정이 다르면 행의 뜻이 달라진다 (통합 검토 10-07)
                    "candidate_k", "candidate_tol", "candidate_min_dist",
                    "warm_start_sigma0", "pose_max_evals")
#: 고정 회전 기준선의 단계 수 (총괄 10-04: min_phase_s 를 지키는 표준은 10단계).
ROTATION_STEPS = 10

_WORKER: dict = {}


@dataclass
class PlanDatasetConfig:
    """계획 데이터셋 생성 설정. 기본값은 총괄이 채택한 따뜻한 시작 설정(10-06)이다."""
    n_phases: int = 9
    energy_weight: float = 0.1
    evaluator: str = "patch"
    patches_per_m2: float = 1500.0
    max_evals: int = 12000
    pose_max_evals: int = 3000
    popsize: int = 100
    sigma0: float = 0.5
    warm_start_sigma0: float = 0.2
    tol_stagnation_gens: int = 20
    seed: int = 0
    body_seed: int = 0
    starts: str = "default,hands_up"          # 단일 자세 최적을 찾을 때 쓰는 시작점
    candidate_k: int = 16
    candidate_tol: float = 0.02
    #: encode 공간에서 ÷√dim 한 거리. 자세(7차원 0.1)와 달라 1행 실측으로 정한다.
    candidate_min_dist: float = 0.05
    baselines: bool = True                    # score_p1_10 · score_p1opt_10 · pose_* 를 넣는다
    body_model: str | None = None


def plan_limits(cfg: PlanDatasetConfig):
    """이 설정이 쓰는 계획 한도. 단계를 늘리면 총 시간 하한을 N × min_phase_s 로 올린다.

    `run_e7 --n-phases` 와 같은 규칙이다 — 폭이 0 이 되면(상한과 같아지면) 계획을 정규화할 수
    없으므로 거부한다 (N × min_phase_s < 상한 이어야 한다).
    """
    import dataclasses

    from .plan_encoding import PlanLimits, plan_physics_cfg

    limits = PlanLimits.from_config(plan_physics_cfg())
    limits = dataclasses.replace(limits, n_phases=cfg.n_phases)
    lo, hi = limits.duration_bounds_s
    need = limits.n_phases * limits.min_phase_s
    if need >= hi:
        raise ValueError(
            f"단계 {limits.n_phases}개 × 최소 {limits.min_phase_s:g} s = {need:g} s 가 총 시간 상한 "
            f"{hi:g} s 이상이다. N × min_phase_s < 상한 이어야 한다 "
            f"(지금 값이면 N ≤ {int((hi - 1e-9) // limits.min_phase_s)}).")
    if need > lo:
        limits = dataclasses.replace(limits, duration_bounds_s=(need, hi))
    return limits


def plan_columns(encoder) -> list[str]:
    """`plan_<키>` 열 이름 (PlanEncoder 의 raw 벡터 순서)."""
    from .plan_encoding import plan_keys

    return [f"plan_{k}" for k in plan_keys(encoder.limits.n_phases)]


def candidate_columns(candidates: list[dict], encoder, best_score: float, *,
                      k: int, tol: float, min_dist: float) -> dict:
    """후보 목록 → `cand_plan_raw` · `cand_score` · `n_candidates`.

    점수가 `best − tol·|best|` 이상인 **가능한** 후보를 점수순으로 보고, encode 공간에서
    ÷√dim 한 거리가 `min_dist` 이상 떨어진 것만 최대 k 개 고른다 (자세 데이터셋과 같은 규칙).
    저장 벡터는 **대칭 정규화 뒤 원래 단위**(`raw_vector`)다.
    """
    if k <= 0 or not candidates:
        return {"cand_plan_raw": [], "cand_score": [], "n_candidates": 0}

    floor = best_score - tol * abs(best_score)
    pool = sorted((c for c in candidates
                   if c.get("score", -np.inf) >= floor and not c.get("infeasible", False)),
                  key=lambda c: -c["score"])
    dim = encoder.dim
    chosen: list[dict] = []
    picked: list[np.ndarray] = []
    for cand in pool:
        # run_cmaes 가 남기는 후보는 **이미 encode 공간 벡터**(x)다 — 그대로 거리에 쓴다.
        x = np.asarray(cand["x"], dtype=np.float64)
        if any(float(np.linalg.norm(x - px)) / math.sqrt(dim) < min_dist for px in picked):
            continue
        chosen.append(cand)
        picked.append(x)
        if len(chosen) >= k:
            break
    # 저장은 원래 단위. **decode 는 clip 만 하므로 normalize 를 다시 거친다** — 1단계 yaw 가
    # 정면(0°) 경계면 chest/back 구역을 맞바꾸는 규칙(plan_encoding.normalize)이 decode 에는
    # 없어서, 그냥 decode 하면 같은 계획이 구역 순서만 다른 값으로 저장된다 (통합 검토 10-07).
    return {
        "cand_plan_raw": [[float(v) for v in encoder.raw_vector(encoder.normalize(encoder.decode(x)))]
                          for x in picked],
        "cand_score": [float(c["score"]) for c in chosen],
        "n_candidates": len(chosen),
    }


def _worker_init(cfg: PlanDatasetConfig) -> None:
    _WORKER["cfg"] = cfg


def _evaluator_for(scenario_name: str, body: BodyParams, cfg: PlanDatasetConfig):
    """계획 평가기 (kinetics 켠 설정 + 이 행의 에너지 가중)."""
    from airis.sim.scenario import load_scenarios

    from . import cli
    from .plan_encoding import plan_physics_cfg

    physics = plan_physics_cfg()
    physics.setdefault("scoring", {})["energy_weight"] = float(cfg.energy_weight)
    scenario = load_scenarios()[scenario_name]
    nozzle, _ = cli.resolve_nozzles()
    evaluator = cli.make_evaluator(cfg.evaluator, scenario, body=body, nozzle=nozzle,
                                   patches_per_m2=cfg.patches_per_m2, physics_cfg=physics)
    return evaluator, scenario, nozzle


def run_task(task: tuple[int, dict, str]) -> dict:
    """체형 하나 × 시나리오 하나의 계획을 최적화해 행 하나를 돌려준다 (Pool 작업자)."""
    from . import cli
    from .baselines import evaluate_plan_row, rotation_plan
    from .cmaes_runner import Start, run_cmaes, run_cmaes_plan
    from .plan_encoding import PlanEncoder

    body_idx, body_dict, scenario_name = task
    cfg: PlanDatasetConfig = _WORKER["cfg"]
    body = BodyParams(**body_dict)
    evaluator, scenario, nozzle = _evaluator_for(scenario_name, body, cfg)
    limits = plan_limits(cfg)
    encoder = PlanEncoder(scenario, limits)
    seed = cfg.seed * 100000 + body_idx
    t0 = time.perf_counter()

    # 1) 단일 자세 최적 — 따뜻한 시작점이자 고정 회전 기준선(P1opt_10)의 자세다.
    pose_res = run_cmaes(evaluator, body, scenario, nozzle, max_evals=cfg.pose_max_evals,
                         seed=seed, popsize=cfg.popsize, sigma0=cfg.sigma0,
                         tol_stagnation_gens=cfg.tol_stagnation_gens,
                         starts=cli.parse_starts(cfg.starts))
    best_pose = pose_res.best_pose

    # 2) 계획 최적화 — 그 자세를 모든 단계에 넣은 계획에서 작은 스텝으로 출발한다.
    result = run_cmaes_plan(
        evaluator, body, scenario, nozzle, limits=limits,
        starts=[Start("warm", best_pose, cfg.warm_start_sigma0)],
        max_evals=cfg.max_evals, seed=seed, popsize=cfg.popsize, sigma0=cfg.sigma0,
        tol_stagnation_gens=cfg.tol_stagnation_gens,
        record_candidates=cfg.candidate_k > 0,
    )
    plan = result.best_plan
    metrics = evaluate_plan_row(evaluator, plan, nozzle, body, scenario)

    row: dict = {"body_idx": body_idx, "scenario": scenario_name}
    row.update({f"body_{k}": float(v) for k, v in body_dict.items()})
    row.update(dict(zip(plan_columns(encoder),
                        (float(v) for v in encoder.raw_vector(encoder.normalize(plan))))))
    row["score"] = float(metrics["score"])
    row["total_removal"] = float(metrics["total_removal"])
    row["energy"] = float(metrics.get("energy", np.nan))
    row["duration_s"] = float(plan.duration_s)
    row["discomfort"] = float(metrics["discomfort"])
    row["n_evals"] = int(result.n_evals)
    row["n_infeasible"] = int(result.n_infeasible)
    row["elapsed_s"] = round(time.perf_counter() - t0, 3)
    row.update(candidate_columns(result.candidates or [], encoder, result.best_score,
                                 k=cfg.candidate_k, tol=cfg.candidate_tol,
                                 min_dist=cfg.candidate_min_dist))

    if cfg.baselines:
        # 같은 체형·시나리오에서 "그냥 도는" 두 방식 — 추천이 이보다 나은지 행 단위로 본다.
        for name, pose in (("p1_10", PoseParams()), ("p1opt_10", best_pose)):
            rot = rotation_plan(pose, scenario, limits, ROTATION_STEPS)
            row[f"score_{name}"] = float(
                evaluate_plan_row(evaluator, rot, nozzle, body, scenario)["score"])
        row.update({f"pose_{k}": float(getattr(best_pose, k)) for k in POSE_KEYS})
        row["pose_score"] = float(pose_res.best_score)
    return row


def stamp_for(cfg: PlanDatasetConfig) -> dict:
    """행마다 붙는 도장 10개 + 한도 6개."""
    from airis.sim.scenario import zone_nozzle_counts

    from . import cli, explog
    from .plan_encoding import plan_kinetics_stamp, plan_physics_cfg

    nozzle, _ = cli.resolve_nozzles()
    physics = plan_physics_cfg()
    kin = plan_kinetics_stamp(physics)
    limits = plan_limits(cfg)
    lo, hi = limits.duration_bounds_s
    return {
        "nozzle_layout_hash": cli.nozzle_hash(nozzle),
        "physics_hash": explog.file_hash(explog.ROOT / "configs" / "physics.yaml"),
        "commit": explog.git_commit(),
        "body_model": configured_body_model(),
        "patches_per_m2": cfg.patches_per_m2,
        "kinetics_enabled": bool(kin["kinetics_enabled"]),
        "time_constant_s": float(kin["time_constant_s"]),
        # 목록이 아니라 문자열이다 — F 의 도장 비교가 문자열 기준이다 (합의 10-06).
        "zone_nozzle_counts": ";".join(f"{v:g}" for v in zone_nozzle_counts(nozzle)),
        "n_phases": cfg.n_phases,
        "energy_weight": cfg.energy_weight,
        "duration_lo_s": float(lo),
        "duration_hi_s": float(hi),
        "min_phase_s": float(limits.min_phase_s),
        "transition_s": float(limits.transition_s),
        "s_max": float(limits.s_max),
        "cap_ratio": float(limits.cap_ratio),
        # 재현용 (자세 데이터셋과 같은 열 이름).
        "body_seed": cfg.body_seed,
        "max_evals": cfg.max_evals,
        "popsize": cfg.popsize,
        "evaluator": cfg.evaluator,
        "candidate_k": cfg.candidate_k,
        "warm_start_sigma0": cfg.warm_start_sigma0,
        "tol_stagnation_gens": cfg.tol_stagnation_gens,
        "candidate_tol": cfg.candidate_tol,
        "candidate_min_dist": cfg.candidate_min_dist,
        "pose_max_evals": cfg.pose_max_evals,
    }


def build_plan_dataset(
    out_path: Path | str,
    *,
    n_bodies: int,
    scenarios: list[str],
    cfg: PlanDatasetConfig = PlanDatasetConfig(),
    processes: int = 8,
    flush_every: int = 20,
    log=print,
) -> dict:
    """계획 데이터셋을 만들거나 이어 만든다. {"path", "n_rows", "n_new", "n_skipped"}."""
    import multiprocessing as mp

    import pandas as pd

    from .dataset import _read, _write_atomic

    out_path = Path(out_path)
    body_model = configured_body_model()
    if cfg.body_model is not None and cfg.body_model != body_model:
        raise ValueError(f"체형 분포 몸 모델({cfg.body_model})이 physics.yaml body.model"
                         f"({body_model})과 다르다. 섞지 않는다.")
    stamp = stamp_for(cfg)

    existing = _read(out_path)
    done: set = set()
    if existing is not None and len(existing):
        for col in CONSISTENCY_COLS:
            vals = set(existing[col].astype(str))
            if vals != {str(stamp[col])}:
                raise ValueError(
                    f"{out_path} 의 {col} 가 {sorted(vals)} 라 이번 실행({stamp[col]})과 섞을 수 "
                    f"없다. 다른 파일로 만들거나 기존 파일을 옮겨라.")
        done = set(zip(existing["body_idx"].astype(int), existing["scenario"].astype(str)))

    bodies = sample_bodies(n_bodies, cfg.body_seed, body_model)
    tasks = [(i, asdict(b), s) for i, b in enumerate(bodies) for s in scenarios if (i, s) not in done]
    n_skipped = n_bodies * len(scenarios) - len(tasks)
    log(f"[plan] {out_path.name}: 몸 {body_model}, 단계 {cfg.n_phases}, 가중 {cfg.energy_weight:g}, "
        f"작업 {len(tasks)}개 (건너뜀 {n_skipped}), 프로세스 {processes}")

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
        if n_new % 5 == 0 or n_new == len(tasks):
            rate = n_new / max(time.perf_counter() - t0, 1e-9)
            log(f"[plan] {n_new}/{len(tasks)}  {rate * 60:.2f} 행/분")

    try:                                    # 작업자가 죽어도 모은 행은 저장한다 (flush 주기 ≤ 20행)
        if processes <= 1:
            _worker_init(cfg)
            for task in tasks:
                consume(run_task(task))
        elif tasks:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes, initializer=_worker_init, initargs=(cfg,)) as pool:
                for row in pool.imap_unordered(run_task, tasks, chunksize=1):
                    consume(row)
    finally:
        flush()

    n_rows = len(frames[0]) if frames else 0
    return {"path": out_path, "n_rows": n_rows, "n_new": n_new, "n_skipped": n_skipped}
