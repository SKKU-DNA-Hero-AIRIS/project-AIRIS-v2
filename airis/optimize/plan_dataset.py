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

from airis.sim import ZONE_NAMES, BodyParams, PoseParams

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
        # run_cmaes 가 남긴 x 는 **탐색 공간 값**이고 [-1, 1] 밖으로도 나간다. decode 가 그것을
        # 잘라 내므로 x 끼리는 멀어 보여도 **계획은 같은** 쌍이 생긴다. 대칭 정규화(거울·앞뒤·
        # 정면 동률)도 x 에는 안 들어가 있다. 그래서 둘 다 남고, 저장할 때 정규화되면서 거의
        # 같은 값이 된다 (60행 실측: 저장된 쌍거리 최소 0.097 < 임계 0.15).
        # encode() 가 clip + normalize 를 하므로 **정규 형태로 바꿔** 거리를 잰다.
        x = encoder.encode(encoder.decode(np.asarray(cand["x"], dtype=np.float64)))
        if any(float(np.linalg.norm(x - px)) / math.sqrt(dim) < min_dist for px in picked):
            continue
        chosen.append(cand)
        picked.append(x)
        if len(chosen) >= k:
            break
    # 저장도 같은 정규 형태를 원래 단위로 되돌린 값이다 (decode 는 encode 의 역이라 추가
    # normalize 가 필요 없다 — x 가 이미 정규 형태다).
    return {
        "cand_plan_raw": [[float(v) for v in encoder.raw_vector(encoder.decode(x))] for x in picked],
        "cand_score": [float(c["score"]) for c in chosen],
        "n_candidates": len(chosen),
    }


def plan_from_raw(encoder, raw):
    """`cand_plan_raw` 한 줄(원래 단위 벡터) → Plan. `raw_vector` 의 역이다."""
    from airis.sim import Phase, Plan

    from .plan_encoding import plan_keys

    n = encoder.limits.n_phases
    v = dict(zip(plan_keys(n), (float(x) for x in raw)))
    poses = [PoseParams(**{p: v[f"p{k}_{p}"] for p in POSE_KEYS}) for k in range(1, n + 1)]
    times = encoder.durations(v["duration_s"], [v[f"share_{k}"] for k in range(1, n)])
    return Plan([Phase(p, t) for p, t in zip(poses, times)], [v[f"zone_{z}"] for z in ZONE_NAMES])


def refilter_candidates(raws, scores, encoder, *, min_dist: float, k: int | None = None) -> dict:
    """저장된 후보를 **읽는 쪽에서 다시 거른다** (점수 높은 쪽을 남긴다).

    2026-10-08 이전에 만든 파일은 거리를 **탐색 공간 값**으로 재서 중복이 남아 있을 수 있다.
    `run_cmaes` 의 x 는 [-1, 1] 밖으로도 나가고 `decode` 가 그것을 자르므로, x 끼리는 멀어
    보여도 잘린 뒤에는 같은 계획인 쌍이 생긴다(위 `candidate_columns` 주석).

    저장된 벡터는 이미 정규 형태라 다시 encode 해 같은 임계로 거르면 새 코드와 같은 결과가
    된다 — **단, 후보가 `candidate_k` 상한에 닿았던 행은 예외다.** 그런 행은 상한 때문에
    버려진 후보가 있어, 새로 만들었다면 중복 대신 다른 후보가 들어왔을 수 있다. 중복이
    사라지는 것은 같고 **개수만 적을 수 있다.**
    """
    order = sorted(range(len(raws)), key=lambda i: -float(scores[i]))
    keep: list[int] = []
    picked: list[np.ndarray] = []
    for i in order:
        x = encoder.encode(plan_from_raw(encoder, raws[i]))
        if any(float(np.linalg.norm(x - px)) / math.sqrt(encoder.dim) < min_dist for px in picked):
            continue
        keep.append(i)
        picked.append(x)
        if k is not None and len(keep) >= k:
            break
    return {
        "cand_plan_raw": [[float(v) for v in raws[i]] for i in keep],
        "cand_score": [float(scores[i]) for i in keep],
        "n_candidates": len(keep),
    }


def encoder_for_row(row):
    """데이터셋 행 하나가 쓴 `PlanEncoder`. 한도는 **행의 열에서** 읽는다(설정 파일이 아니라).

    실행이 총 시간 하한을 `N × min_phase_s` 로 올리므로 설정 파일 값과 다를 수 있다.
    """
    import dataclasses

    from airis.sim.scenario import load_scenarios

    from .plan_encoding import PlanEncoder, PlanLimits

    limits = PlanLimits(
        n_phases=int(row["n_phases"]),
        duration_bounds_s=(float(row["duration_lo_s"]), float(row["duration_hi_s"])),
        min_phase_s=float(row["min_phase_s"]), transition_s=float(row["transition_s"]),
        s_max=float(row["s_max"]), cap_ratio=float(row["cap_ratio"]))
    # 구역 노즐 수도 행의 도장에서 읽는다 ("한도는 행에서 읽는다"와 같은 이유). 풍량 한도
    # (Σ s_m ≤ cap_ratio · M)가 이 값에 걸리므로, 배치가 바뀐 파일을 지금 설정으로 읽으면
    # 다른 계획이 된다. 지금 설정에서는 값이 같아 결과가 바뀌지 않는다.
    counts = [float(v) for v in str(row["zone_nozzle_counts"]).split(";") if v]
    return PlanEncoder(load_scenarios()[str(row["scenario"])], dataclasses.replace(limits),
                       zone_nozzle_counts=counts or None)


def refilter_candidate_frame(df, *, min_dist: float = 0.15, k: int | None = None):
    """데이터셋 DataFrame → **후보를 다시 거른 새 DataFrame** (원본은 바꾸지 않는다).

    `cand_plan_raw`·`cand_score`·`n_candidates` 세 열을 **함께** 갱신한다. 점수가 높은 후보를
    남기므로 **최적 계획은 항상 남고**, 후보가 0개가 되는 행은 생기지 않는다(원래 0개였던
    행은 그대로 0개다).

    2026-10-08 이전에 만든 파일은 거리 필터가 탐색 공간 값으로 돌아 **같은 계획이 중복으로**
    남아 있다(60행 기준 후보의 약 1/3). 읽은 직후 한 번 부르면 새 코드와 같은 결과가 된다.
    한도·시나리오는 행의 열에서 읽으므로 파일마다 N 이 달라도 된다.
    """
    out = df.copy()
    cache: dict = {}
    raws, scores, counts = [], [], []
    for _, row in df.iterrows():
        key = (str(row["scenario"]), int(row["n_phases"]), float(row["duration_lo_s"]),
               str(row["zone_nozzle_counts"]))
        enc = cache.get(key) or cache.setdefault(key, encoder_for_row(row))
        got = refilter_candidates(list(row["cand_plan_raw"]), list(row["cand_score"]), enc,
                                  min_dist=min_dist, k=k)
        raws.append(got["cand_plan_raw"])
        scores.append(got["cand_score"])
        counts.append(got["n_candidates"])
    out["cand_plan_raw"] = raws
    out["cand_score"] = scores
    out["n_candidates"] = counts
    return out


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
    bodies: list[tuple[int, BodyParams]] | None = None,
    log=print,
) -> dict:
    """계획 데이터셋을 만들거나 이어 만든다. {"path", "n_rows", "n_new", "n_skipped"}.

    `bodies` 를 주면 `(번호, 체형)` 쌍을 그대로 쓴다(번호는 시드 수열의 진짜 번호여야 한다) —
    꼬리 구간만 보강할 때 쓴다. 주지 않으면 `sample_bodies(n_bodies)` 앞부분을 쓴다.
    """
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
        missing = [c for c in CONSISTENCY_COLS if c not in existing.columns]
        if missing:
            # 옛 코드로 만든 파일이다. 이어 붙이면 그 열이 NaN 인 행이 섞여 도장 검사가 무의미해진다.
            raise ValueError(
                f"{out_path} 에 도장 열이 없다: {missing}. 옛 코드로 만든 파일이라 이어 만들 수 "
                f"없다 — 다른 파일로 만들거나 기존 파일을 옮겨라.")
        for col in CONSISTENCY_COLS:
            vals = set(existing[col].astype(str))
            if vals != {str(stamp[col])}:
                raise ValueError(
                    f"{out_path} 의 {col} 가 {sorted(vals)} 라 이번 실행({stamp[col]})과 섞을 수 "
                    f"없다. 다른 파일로 만들거나 기존 파일을 옮겨라.")
        done = set(zip(existing["body_idx"].astype(int), existing["scenario"].astype(str)))

    picked = (list(enumerate(sample_bodies(n_bodies, cfg.body_seed, body_model)))
              if bodies is None else [(int(i), b) for i, b in bodies])
    tasks = [(i, asdict(b), s) for i, b in picked for s in scenarios if (i, s) not in done]
    n_skipped = len(picked) * len(scenarios) - len(tasks)
    where = "" if bodies is None else f", 체형 번호 {min(i for i, _ in picked)}~{max(i for i, _ in picked)}"
    log(f"[plan] {out_path.name}: 몸 {body_model}, 단계 {cfg.n_phases}, 가중 {cfg.energy_weight:g}, "
        f"작업 {len(tasks)}개 (건너뜀 {n_skipped}), 프로세스 {processes}{where}")

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


# ------------------------------------------------------------------- 다듬기 한 바퀴 (polish)

#: 다듬기가 **더하는** 열. 원래 열(score · plan_* · cand_*)은 하나도 바꾸지 않는다.
POLISH_COLS = ("polish_score", "polish_plan_raw", "polish_source", "polish_gain",
               "polish_own_score", "polish_n_evals", "polish_starts", "polish_max_evals",
               "polish_k_neighbours", "polish_elapsed_s")


@dataclass
class PolishConfig:
    """다듬기 설정. 기본값은 총괄 승인(10-11) 기준: 이웃 1개 + 원래 계획, 총 예산 3,000."""
    k_neighbours: int = 16       # 재채점할 이웃 수 (재채점은 평가 1회라 싸다)
    n_starts: int = 1            # 그중 몇 개를 따뜻한 시작으로 쓸지 (원래 계획은 항상 더한다)
    max_evals: int = 3000        # 시작점들이 **나눠 쓰는** 총 예산 (run_cmaes 규칙)
    sigma0: float = 0.2
    seed: int = 7000


def config_from_row(row) -> PlanDatasetConfig:
    """행의 도장에서 그 행을 만든 설정을 되살린다 (평가기·에너지 가중을 같게 쓰려고).

    한도는 설정 파일이 아니라 행에서 읽어야 하므로(`encoder_for_row`) 여기서는 평가기 설정만
    되살린다. 후보 기록·기준선은 다듬기에서 쓰지 않아 끈다.
    """
    return PlanDatasetConfig(
        n_phases=int(row["n_phases"]), energy_weight=float(row["energy_weight"]),
        evaluator=str(row["evaluator"]), patches_per_m2=float(row["patches_per_m2"]),
        popsize=int(row["popsize"]), tol_stagnation_gens=int(row["tol_stagnation_gens"]),
        warm_start_sigma0=float(row["warm_start_sigma0"]), body_seed=int(row["body_seed"]),
        candidate_k=0, baselines=False)


def neighbour_indices(df, scenario: str, body_idx: int, k: int) -> list[int]:
    """같은 시나리오에서 체형이 가장 가까운 행 k 개의 `body_idx`.

    거리는 체형 5변수를 그 시나리오 행들의 표준편차로 나눈 z 거리다 (F 의 kNN 과 같은 뜻이고,
    여기서는 "따뜻한 시작으로 쓸 이웃"을 고르는 데만 쓴다).
    """
    d = df[df["scenario"].astype(str) == str(scenario)]
    cols = [f"body_{key}" for key in BODY_KEYS]
    x = d[cols].to_numpy(dtype=np.float64)
    idx = d["body_idx"].to_numpy(dtype=int)
    me = np.flatnonzero(idx == int(body_idx))
    if not len(me) or len(idx) < 2:
        return []
    sd = x.std(axis=0, ddof=0)
    sd[sd <= 0] = 1.0
    dist = np.linalg.norm((x - x[me[0]]) / sd, axis=1)
    order = np.argsort(dist, kind="stable")
    return [int(idx[i]) for i in order if int(idx[i]) != int(body_idx)][:max(0, k)]


def _polish_init(pol: PolishConfig) -> None:
    _WORKER["polish"] = pol


def polish_task(task: tuple[dict, list]) -> dict:
    """행 하나를 다듬는다 (Pool 작업자). `KEY_COLS` + `POLISH_COLS` 를 돌려준다."""
    from .baselines import evaluate_plan_row
    from .cmaes_runner import Start, run_cmaes_plan

    row, neighbours = task
    pol: PolishConfig = _WORKER["polish"]
    cfg = config_from_row(row)
    body = BodyParams(**{key: float(row[f"body_{key}"]) for key in BODY_KEYS})
    enc = encoder_for_row(row)
    counts = [float(v) for v in str(row["zone_nozzle_counts"]).split(";") if v]
    evaluator, scenario, nozzle = _evaluator_for(str(row["scenario"]), body, cfg)
    t0 = time.perf_counter()

    own = plan_from_raw(enc, [float(row[c]) for c in plan_columns(enc)])
    own_score = float(evaluate_plan_row(evaluator, own, nozzle, body, scenario)["score"])

    # 이웃 계획을 **이 체형에서** 재채점한다 (부스 밖이면 버린다). 평가 1회씩이라 싸다.
    scored: list[tuple[float, int, object]] = []
    for nb_idx, raw in neighbours:
        plan = plan_from_raw(enc, [float(v) for v in raw])
        m = evaluate_plan_row(evaluator, plan, nozzle, body, scenario)
        if not m["infeasible"]:
            scored.append((float(m["score"]), int(nb_idx), plan))
    scored.sort(key=lambda t: -t[0])

    starts = [Start("own", own.phases[0].pose, pol.sigma0, plan=own)]
    starts += [Start(f"nb{nb_idx}", plan.phases[0].pose, pol.sigma0, plan=plan)
               for _, nb_idx, plan in scored[:max(0, pol.n_starts)]]
    res = run_cmaes_plan(evaluator, body, scenario, nozzle, limits=enc.limits,
                         zone_nozzle_counts=counts or None, starts=starts,
                         max_evals=pol.max_evals, popsize=cfg.popsize, sigma0=pol.sigma0,
                         tol_stagnation_gens=cfg.tol_stagnation_gens,
                         seed=pol.seed * 100000 + int(row["body_idx"]))

    # 셋 중 가장 높은 것을 쓴다 — 원래 계획 · 재채점한 이웃 계획 · 다듬은 계획.
    # 원래 계획을 시작점에 넣었으니 보통 다듬은 쪽이 이기지만, 수치로 확인해 고른다.
    best_plan, best_score, source = own, own_score, "own"
    options = [(float(res.best_score), res.best_plan, "polish")]
    options += [(s, p, f"neighbour:{i}") for s, i, p in scored[:1]]
    for score, plan, src in options:
        if plan is not None and score > best_score:
            best_plan, best_score, source = plan, score, src

    return {
        "body_idx": int(row["body_idx"]), "scenario": str(row["scenario"]),
        "polish_score": float(best_score),
        # cand_plan_raw 와 같은 규약: 대칭 정규화 뒤 **원래 단위** 벡터 (8N + 5).
        "polish_plan_raw": [float(v) for v in enc.raw_vector(enc.normalize(best_plan))],
        "polish_source": str(source),
        "polish_gain": float(best_score - own_score),
        "polish_own_score": float(own_score),
        "polish_n_evals": int(res.n_evals),
        "polish_starts": ";".join(s.name for s in starts),
        "polish_max_evals": int(pol.max_evals),
        "polish_k_neighbours": int(pol.k_neighbours),
        "polish_elapsed_s": round(time.perf_counter() - t0, 3),
    }


def _polish_payload(row, plan_cols: list[str]) -> dict:
    """작업자에게 보낼 열만 담는다 (후보 열은 보내지 않는다 — 크고 쓰지 않는다)."""
    keys = ["body_idx", "scenario", "score", "n_phases", "duration_lo_s", "duration_hi_s",
            "min_phase_s", "transition_s", "s_max", "cap_ratio", "zone_nozzle_counts",
            "energy_weight", "evaluator", "patches_per_m2", "popsize", "tol_stagnation_gens",
            "warm_start_sigma0", "body_seed"]
    keys += [f"body_{key}" for key in BODY_KEYS] + list(plan_cols)
    return {k: row[k] for k in keys}


def polish_plan_dataset(
    in_path: Path | str,
    out_path: Path | str,
    *,
    pol: PolishConfig = PolishConfig(),
    scenarios: list[str] | None = None,
    processes: int = 8,
    flush_every: int = 20,
    log=print,
) -> dict:
    """계획 데이터셋에 **다듬은 계획 열을 더해 새 파일로** 쓴다.

    행마다 이웃 체형 k 명의 계획을 그 체형에서 재채점하고, 가장 좋은 이웃 계획과 원래 계획을
    따뜻한 시작으로 CMA-ES 를 한 바퀴 더 돈다. `polish_score` 는 그 셋 중 최대라
    **`polish_score >= score` 가 구조적으로 보장된다.**

    왜 (F 5-fold 10-11): 이웃 체형의 계획이 그 체형의 최적보다 높게 채점되는 행이 **90.7%**,
    2% 넘게 높은 행이 53.3% 였다. 행별 최적이 예산 소진(272/300)으로 2~3% 덜 수렴했다는 뜻이다.

    **주의 — 측정과 목표를 섞지 말 것.** 다듬은 열은 다른 행(다른 폴드일 수 있다)의 계획에서
    출발해 얻은 값이라 "이웃에서 옮겨 쓸 수 있는가"를 재는 측정에는 쓸 수 없다. 그 측정은 원래
    `score` 열로 하고, 학습 목표에는 `polish_*` 를 쓴다.

    중단되면 같은 명령을 다시 돌린다 — `out_path` 에 이미 다듬어진 행은 건너뛴다.
    """
    import multiprocessing as mp

    import pandas as pd

    from .dataset import _read, _write_atomic
    from .plan_encoding import plan_keys

    in_path, out_path = Path(in_path), Path(out_path)
    if out_path == in_path:
        raise ValueError("다듬은 결과는 다른 파일에 쓴다 (F 가 읽는 중일 수 있다)")
    base = _read(in_path)
    if base is None or not len(base):
        raise ValueError(f"{in_path} 에 행이 없다")
    missing = [c for c in CONSISTENCY_COLS if c not in base.columns]
    if missing:
        raise ValueError(f"{in_path} 에 도장 열이 없다: {missing}. 옛 코드로 만든 파일이다")
    if scenarios:
        base = base[base["scenario"].astype(str).isin(list(scenarios))].reset_index(drop=True)

    got: dict[tuple[int, str], dict] = {}
    prev = _read(out_path)
    if prev is not None and "polish_score" in prev.columns:
        for _, r in prev.iterrows():
            if pd.notna(r.get("polish_score")):
                key = (int(r["body_idx"]), str(r["scenario"]))
                got[key] = {c: r[c] for c in ("body_idx", "scenario", *POLISH_COLS)
                            if c in prev.columns}

    tasks = []
    for _, row in base.iterrows():
        key = (int(row["body_idx"]), str(row["scenario"]))
        if key in got:
            continue
        plan_cols = [f"plan_{k}" for k in plan_keys(int(row["n_phases"]))]
        same = base[base["scenario"].astype(str) == key[1]].set_index("body_idx")
        nb = neighbour_indices(base, key[1], key[0], pol.k_neighbours)
        raws = [(i, [float(same.loc[i, c]) for c in plan_cols]) for i in nb]
        tasks.append((_polish_payload(row, plan_cols), raws))

    log(f"[polish] {in_path.name} -> {out_path.name}: 행 {len(base)}개 중 할 일 {len(tasks)}개 "
        f"(건너뜀 {len(got)}), 이웃 {pol.k_neighbours}·시작 {1 + pol.n_starts}개·예산 "
        f"{pol.max_evals}, 프로세스 {processes}")

    n_new = 0
    t0 = time.perf_counter()

    def flush():
        if not got:
            return
        add = pd.DataFrame(list(got.values()))
        _write_atomic(base.merge(add, on=list(KEY_COLS), how="left"), out_path)

    def consume(res: dict) -> None:
        nonlocal n_new
        got[(int(res["body_idx"]), str(res["scenario"]))] = res
        n_new += 1
        if n_new % max(1, flush_every) == 0:
            flush()
        if n_new % 5 == 0 or n_new == len(tasks):
            rate = n_new / max(time.perf_counter() - t0, 1e-9)
            log(f"[polish] {n_new}/{len(tasks)}  {rate * 60:.2f} 행/분")

    try:                                    # 작업자가 죽어도 모은 행은 저장한다
        if processes <= 1:
            _polish_init(pol)
            for task in tasks:
                consume(polish_task(task))
        elif tasks:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes, initializer=_polish_init, initargs=(pol,)) as pool:
                for res in pool.imap_unordered(polish_task, tasks, chunksize=1):
                    consume(res)
    finally:
        flush()

    gains = [float(r["polish_gain"]) for r in got.values()]
    return {"path": out_path, "n_rows": len(base), "n_new": n_new, "n_done": len(got),
            "gain_mean": float(np.mean(gains)) if gains else 0.0,
            "gain_rows": int(sum(1 for g in gains if g > 0))}
