"""DummyEvaluator 로 CMA-ES 루프를 검증한다. 소유자: C. docs/tracks/C_optimize.md 단계 10."""
import json

import numpy as np
import pytest

from airis.optimize.cmaes_runner import INFEASIBLE_SCORE, cma_seed, run_cmaes
from airis.optimize.dummy import DummyEvaluator
from airis.optimize.encoding import PoseEncoder
from airis.sim import BodyParams, PoseParams
from airis.sim.scenario import load_nozzles, load_scenarios

pytest.importorskip("cma", reason="cma 패키지 필요 (pip install cma)")

TARGET = PoseParams(
    shoulder_abduction=95.0, shoulder_flexion=30.0, elbow_flexion=20.0,
    torso_pitch=5.0, torso_yaw=10.0, hip_flexion=5.0, knee_flexion=5.0,
)


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios()


def _run(scenario, **kwargs):
    evaluator = DummyEvaluator(TARGET, scenario)
    result = run_cmaes(
        evaluator, BodyParams(), scenario, load_nozzles(),
        **{"max_evals": 1500, "popsize": 20, "seed": 0, **kwargs},
    )
    return evaluator, result


@pytest.mark.parametrize("name", ["default", "wheelchair"])
def test_converges_to_target(scenarios, name):
    scenario = scenarios[name]
    evaluator, result = _run(scenario)
    enc = PoseEncoder(scenario)
    distance = float(np.linalg.norm(enc.encode(result.best_pose) - enc.encode(evaluator.target)))
    assert distance < 0.05, f"정규화 공간 거리 {distance}"
    assert result.n_evals <= 1500
    assert result.history


def test_same_seed_gives_identical_history(scenarios):
    scenario = scenarios["default"]
    _, first = _run(scenario, max_evals=600)
    _, second = _run(scenario, max_evals=600)
    assert first.history == second.history
    assert first.n_evals == second.n_evals
    assert first.best_score == second.best_score


def test_different_seed_diverges(scenarios):
    scenario = scenarios["default"]
    _, a = _run(scenario, max_evals=600, seed=0)
    _, b = _run(scenario, max_evals=600, seed=1)
    assert a.history != b.history


def test_cma_seed_never_zero():
    # cma 는 seed 0/None 을 시각 기반으로 해석해 재현성이 깨진다.
    for seed in (-1, 0, 1, 12345):
        assert cma_seed(seed) >= 1


def test_wheelchair_keeps_fixed_pose(scenarios):
    _, result = _run(scenarios["wheelchair"], max_evals=600)
    assert result.best_pose.hip_flexion == 90
    assert result.best_pose.knee_flexion == 90
    assert result.free_keys == [
        "shoulder_abduction", "shoulder_flexion", "elbow_flexion", "torso_pitch", "torso_yaw",
    ]


def test_history_columns(scenarios):
    _, result = _run(scenarios["default"], max_evals=200)
    for i, row in enumerate(result.history, start=1):
        assert set(row) == {"gen", "evals", "best", "mean", "sigma", "infeasible_frac"}
        assert row["infeasible_frac"] == 0.0   # 더미는 불가 판정을 내지 않는다
        assert row["gen"] == i
    best_values = [row["best"] for row in result.history]
    assert best_values == sorted(best_values), "누적 best 는 단조 증가해야 한다"


class _BoothWallEvaluator(DummyEvaluator):
    """shoulder_abduction 이 LIMIT 를 넘으면 부스 밖이라 불가로 판정하는 더미.

    목표(TARGET, 벌림 95°)는 불가 구간에 있다. 불가 후보를 best 로 고르지 않으면
    best 는 LIMIT 이하에 머문다.
    """

    LIMIT = 60.0

    def evaluate(self, pose, nozzle, body, scenario):
        result = super().evaluate(pose, nozzle, body, scenario)
        if pose.shoulder_abduction > self.LIMIT:
            result.score = INFEASIBLE_SCORE
            result.extra["infeasible"] = True
        return result


class _BatchBoothWallEvaluator(_BoothWallEvaluator):
    """batch_evaluate 를 오버라이드한 평가기(입자판 흉내). 점수만 돌려준다."""

    def batch_evaluate(self, candidates, body, scenario):
        return np.array(
            [self.evaluate(p, n, body, scenario).score for p, n in candidates], dtype=np.float32,
        )


@pytest.mark.parametrize("cls", [_BoothWallEvaluator, _BatchBoothWallEvaluator])
def test_infeasible_candidates_excluded_from_best(scenarios, cls):
    scenario = scenarios["default"]
    evaluator = cls(TARGET, scenario)
    result = run_cmaes(
        evaluator, BodyParams(), scenario, load_nozzles(),
        max_evals=600, popsize=20, seed=0,
    )
    fracs = [row["infeasible_frac"] for row in result.history]
    assert all(0.0 <= f <= 1.0 for f in fracs)
    assert any(f > 0 for f in fracs), "목표가 불가 구간이라 불가 후보가 나와야 한다"
    assert result.n_infeasible == round(sum(f * 20 for f in fracs))
    assert result.best_pose.shoulder_abduction <= cls.LIMIT
    assert result.best_score > INFEASIBLE_SCORE
    assert not result.best_result.extra.get("infeasible", False)
    assert all(row["best"] > INFEASIBLE_SCORE for row in result.history)


class _AlwaysInfeasibleEvaluator(DummyEvaluator):
    def evaluate(self, pose, nozzle, body, scenario):
        result = super().evaluate(pose, nozzle, body, scenario)
        result.score = INFEASIBLE_SCORE
        result.extra["infeasible"] = True
        return result


def test_all_infeasible_has_no_best(scenarios):
    scenario = scenarios["default"]
    result = run_cmaes(
        _AlwaysInfeasibleEvaluator(TARGET, scenario), BodyParams(), scenario, load_nozzles(),
        max_evals=100, popsize=20, seed=0,
    )
    assert all(row["infeasible_frac"] == 1.0 for row in result.history)
    assert all(row["best"] == -np.inf for row in result.history)
    assert result.n_infeasible == result.n_evals


def test_run_optimize_writes_three_files(tmp_path):
    """완료 기준: --evaluator dummy 가 끝까지 돌고 outputs/<exp_id>/ 에 세 파일이 생긴다."""
    from scripts.run_optimize import main

    code = main([
        "--evaluator", "dummy", "--scenario", "default",
        "--max-evals", "200", "--popsize", "20", "--seed", "0",
        "--log-dir", str(tmp_path), "--tag", "test",
    ])
    assert code == 0

    runs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(runs) == 1
    run_dir = runs[0]
    assert run_dir.name.startswith("test_")
    for filename in ("meta.json", "history.csv", "best.json"):
        assert (run_dir / filename).is_file(), filename

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["scenario"]["name"] == "default"
    assert meta["seed"] == 0
    assert meta["n_evals"] > 0
    # B 병합 후에는 configs/nozzles.yaml 의 실제 배치(16개)를 쓴다.
    assert meta["args"]["nozzle_source"] == "config"

    best = json.loads((run_dir / "best.json").read_text(encoding="utf-8"))
    assert set(best["best_pose"]) == {
        "shoulder_abduction", "shoulder_flexion", "elbow_flexion",
        "torso_pitch", "torso_yaw", "hip_flexion", "knee_flexion",
    }

    header, *rows = (run_dir / "history.csv").read_text(encoding="utf-8").strip().splitlines()
    assert header == "gen,evals,best,mean,sigma,infeasible_frac"
    assert len(rows) == len(set(rows)) and rows

    index = (tmp_path / "index.csv").read_text(encoding="utf-8").strip().splitlines()
    assert index[0] == "exp_id,date,scenario,best_score,commit"
    assert len(index) == 2


class _NotImplementedEvaluator:
    """클래스는 있지만 evaluate 가 미구현인 평가기 (트랙 병합 전 스텁을 흉내 낸다)."""

    def __init__(self, physics_cfg):
        self.cfg = physics_cfg

    def evaluate(self, pose, nozzle, body, scenario):
        raise NotImplementedError("테스트용 미구현 평가기")


# 실제 트랙 병합 상태와 무관하게 두 실패 경로를 모두 검사한다.
@pytest.mark.parametrize("dotted, cause", [
    ("airis.sim.nonexistent_module.Nope", "ModuleNotFoundError"),
    (f"{__name__}._NotImplementedEvaluator", "NotImplementedError"),
])
def test_run_optimize_reports_unmerged_track(tmp_path, capsys, monkeypatch, dotted, cause):
    from airis.optimize import cli
    from scripts.run_optimize import main

    monkeypatch.setitem(cli._OWNER, "patch", ("D", dotted, "docs/tracks/D_patch_baseline.md"))
    code = main([
        "--evaluator", "patch", "--scenario", "default",
        "--max-evals", "100", "--log-dir", str(tmp_path),
    ])
    assert code == 3
    err = capsys.readouterr().err
    assert "해당 트랙 미병합" in err
    assert cause in err
    assert not any(p.is_dir() for p in tmp_path.iterdir()), "실패 시 실험 폴더를 만들면 안 된다"


def test_patch_evaluator_runs_in_loop(scenarios):
    """D 병합 후: cli 가 실제 PatchEvaluator 를 만들고 CMA-ES 루프가 돈다 (짧게)."""
    from airis.optimize import cli
    from airis.sim.patch_baseline import PatchEvaluator

    scenario = scenarios["wheelchair"]
    nozzle = load_nozzles()
    evaluator = cli.make_evaluator("patch", scenario, body=BodyParams(), nozzle=nozzle)
    assert isinstance(evaluator, PatchEvaluator)

    result = run_cmaes(evaluator, BodyParams(), scenario, nozzle,
                       max_evals=8, popsize=4, seed=0)
    assert result.n_evals == 8
    assert np.isfinite(result.best_score)
    assert result.best_pose.hip_flexion == 90
    assert result.best_pose.knee_flexion == 90
    assert result.best_result.removal_by_part.shape == (5,)
