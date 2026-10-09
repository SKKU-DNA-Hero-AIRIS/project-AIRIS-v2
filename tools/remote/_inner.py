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


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "info":
        out = info()
    elif cmd == "bench":
        out = bench(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    else:
        raise SystemExit(f"모르는 명령: {cmd}")
    print("JSON:" + json.dumps(out, ensure_ascii=False))
