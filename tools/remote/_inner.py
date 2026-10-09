"""실행 폴더(고정 커밋) 안에서 가상환경 Python 으로 도는 보조 코드. plan_share.py 가 부른다."""
from __future__ import annotations

import json
import sys
import time


def _eval_worker(args):
    scenario_name, n_plan, n_pose = args
    from airis.optimize import plan_dataset as pd_
    from airis.optimize.baselines import rotation_plan
    from airis.optimize.dataset import configured_body_model, sample_bodies
    from airis.optimize.plan_encoding import PlanEncoder
    from airis.sim import PoseParams

    cfg = pd_.PlanDatasetConfig()
    body = sample_bodies(1, 0, configured_body_model())[0]
    evaluator, scenario, nozzle = pd_._evaluator_for(scenario_name, body, cfg)
    limits = pd_.plan_limits(cfg)
    PlanEncoder(scenario, limits)
    plan = rotation_plan(PoseParams(), scenario, limits, cfg.n_phases)
    evaluator.evaluate_plan(plan, nozzle, body, scenario)          # 첫 호출(준비 비용)은 재지 않는다
    t0 = time.perf_counter()
    for _ in range(n_plan):
        evaluator.evaluate_plan(plan, nozzle, body, scenario)
    t_plan = (time.perf_counter() - t0) / max(n_plan, 1)
    t0 = time.perf_counter()
    for _ in range(n_pose):
        evaluator.evaluate(PoseParams(), nozzle, body, scenario)
    t_pose = (time.perf_counter() - t0) / max(n_pose, 1)
    return t_plan, t_pose


def info() -> dict:
    """설정 식별값과 체형 샘플 (기준 노트북과 같은지 비교용)."""
    from dataclasses import asdict

    from airis.optimize import plan_dataset as pd_
    from airis.optimize.dataset import configured_body_model, sample_bodies

    cfg = pd_.PlanDatasetConfig()
    stamp = pd_.stamp_for(cfg)
    bodies = sample_bodies(3, 0, configured_body_model())
    return {"physics_hash": stamp["physics_hash"], "nozzle_layout_hash": stamp["nozzle_layout_hash"],
            "commit": stamp["commit"], "body_model": stamp["body_model"],
            "bodies": [asdict(b) for b in bodies]}


def bench(scenario: str, processes: int, n_plan: int, n_pose: int) -> dict:
    """프로세스 processes 개가 동시에 돌 때의 평가 1회 시간 (실제 실행과 같은 혼잡 조건)."""
    import multiprocessing as mp

    with mp.Pool(processes) as pool:
        got = pool.map(_eval_worker, [(scenario, n_plan, n_pose)] * processes)
    t_plan = sum(g[0] for g in got) / len(got)
    t_pose = sum(g[1] for g in got) / len(got)
    return {"processes": processes, "t_plan_s": t_plan, "t_pose_s": t_pose}


def run_range(out: str, scenario: str, lo: int, hi: int, processes: int, flush_every: int,
              cli_args: list[str]) -> dict:
    """체형 번호 lo~hi(양 끝 포함) × 시나리오 하나만 만든다.

    `build_plan_dataset` 의 반복문과 같은 일을 하되 작업 목록만 구간으로 줄였다 (고정 커밋의 CLI 에는
    체형 시작 번호 옵션이 없다). 행의 계산(`run_task`)·도장(`stamp_for`)·저장(`_write_atomic`)·
    설정(`run_plan_dataset.build_parser`)은 고정 커밋의 것을 그대로 쓴다. 시드는 체형 번호로 정해지므로
    구간으로 나눠도 같은 행이 나온다.
    """
    import multiprocessing as mp
    from dataclasses import asdict
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, str(Path.cwd() / "scripts"))
    import run_plan_dataset as cli_mod

    from airis.optimize import plan_dataset as pd_
    from airis.optimize.dataset import _read, _write_atomic, configured_body_model, sample_bodies

    args = cli_mod.build_parser().parse_args(["--n-bodies", str(hi + 1), *cli_args])
    cfg = pd_.PlanDatasetConfig(
        n_phases=args.n_phases, energy_weight=args.energy_weight, evaluator=args.evaluator,
        patches_per_m2=args.patches_per_m2, max_evals=args.max_evals,
        pose_max_evals=args.pose_max_evals, popsize=args.popsize, sigma0=args.sigma0,
        warm_start_sigma0=args.warm_start_sigma0, tol_stagnation_gens=args.tol_stagnation_gens,
        seed=args.seed, body_seed=args.body_seed, candidate_k=args.candidate_k,
        candidate_tol=args.candidate_tol, candidate_min_dist=args.candidate_min_dist,
        baselines=not args.no_baselines)
    out_path = Path(out)
    stamp = pd_.stamp_for(cfg)
    existing = _read(out_path)
    done: set = set()
    if existing is not None and len(existing):
        for col in pd_.CONSISTENCY_COLS:
            vals = set(existing[col].astype(str))
            if vals != {str(stamp[col])}:
                raise SystemExit(f"{out_path} 의 {col} 가 {sorted(vals)} 라 이번 실행({stamp[col]})과 섞을 수 없다.")
        done = set(zip(existing["body_idx"].astype(int), existing["scenario"].astype(str)))
    bodies = sample_bodies(hi + 1, cfg.body_seed, configured_body_model())
    tasks = [(i, asdict(bodies[i]), scenario) for i in range(lo, hi + 1) if (i, scenario) not in done]
    n_skipped = (hi - lo + 1) - len(tasks)
    print(f"[plan] {out_path.name}: {scenario} 체형 {lo}~{hi}, 단계 {cfg.n_phases}, "
          f"작업 {len(tasks)}개 (건너뜀 {n_skipped}), 프로세스 {processes}", flush=True)

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
        merged = merged.sort_values(list(pd_.KEY_COLS), kind="mergesort").reset_index(drop=True)
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
        rate = n_new / max(time.perf_counter() - t0, 1e-9)
        print(f"[plan] {n_new}/{len(tasks)}  {rate * 60:.2f} 행/분", flush=True)

    try:
        if processes <= 1:
            pd_._worker_init(cfg)
            for task in tasks:
                consume(pd_.run_task(task))
        elif tasks:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes, initializer=pd_._worker_init, initargs=(cfg,)) as pool:
                for row in pool.imap_unordered(pd_.run_task, tasks, chunksize=1):
                    consume(row)
    finally:
        flush()
    print(f"[plan] 끝: 새로 {n_new}행", flush=True)
    return {"n_new": n_new, "n_skipped": n_skipped}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "info":
        out = info()
    elif cmd == "bench":
        out = bench(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif cmd == "run-range":
        # run-range <out> <scenario> <lo> <hi> <processes> <flush_every> -- <run_plan_dataset 인자...>
        rest = sys.argv[sys.argv.index("--") + 1:]
        out = run_range(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]),
                        int(sys.argv[7]), rest)
    else:
        raise SystemExit(f"모르는 명령: {cmd}")
    print("JSON:" + json.dumps(out, ensure_ascii=False))
