"""재채점 단축 옵션(중복 제거·저밀도 선별·스레드) 단위 테스트. 소유자: F. 총괄 2026-09-30.

predict 에 붙인 통합 테스트는 tests/test_pose_flow.py (학습한 모델이 필요하다). 여기는 torch 없이 돈다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from airis.model import predict as pred
from airis.optimize.dummy import DummyEvaluator
from airis.sim import BodyParams, PoseParams
from airis.sim.scenario import load_scenarios

ROOT = Path(__file__).resolve().parents[1]


def _pose(abd=90.0, yaw=45.0, **kw):
    return PoseParams(shoulder_abduction=abd, torso_yaw=yaw, **kw)


def test_dedup_keeps_order_and_first_of_each_group():
    cands = [_pose(10), _pose(11), _pose(40), _pose(12.5), _pose(43.1)]
    srcs = ["flow", "flow", "knn", "knn", "knn"]
    out, out_s, n = pred.dedup_candidates(cands, srcs, 3.0)
    assert out == [cands[0], cands[2], cands[4]] and out_s == ["flow", "knn", "knn"] and n == 2
    assert out[0] is cands[0], "남긴 후보는 같은 객체 (E 가 identity 로 찾는다)"


def test_dedup_never_drops_extra_and_drops_others_close_to_extra():
    extra = _pose(175.0)
    cands = [_pose(174.0), _pose(10), extra, _pose(10.5)]
    srcs = ["flow", "knn", "extra", "extra"]
    out, out_s, n = pred.dedup_candidates(cands, srcs, 3.0)
    assert out_s.count("extra") == 2, "고정 후보는 서로 겹쳐도 전부 남긴다 (E 가 extra 순서로 표와 짝짓는다)"
    assert cands[0] not in out, "고정 후보와 겹치는 flow 후보를 뺀다"
    assert cands[1] not in out, "고정 후보(10.5°)와 0.5° 차이인 kNN 후보도 뺀다"
    assert n == 2


def test_dedup_folds_yaw_and_zero_is_noop():
    a, b = _pose(yaw=10.0), _pose(yaw=-11.0)          # 좌우 거울: 접으면 11°, a 와 1° 차 → 중복
    c = _pose(yaw=171.0)                               # 앞뒤 등가: 접으면 9°, a 와 1° 차 → 중복
    d = _pose(yaw=150.0)                               # 접으면 30° → 남는다
    out, _, n = pred.dedup_candidates([a, b, c, d], ["flow"] * 4, 3.0)
    assert out == [a, d] and n == 2
    same, _, n0 = pred.dedup_candidates([a, a, a], ["flow"] * 3, 0.0)
    assert len(same) == 3 and n0 == 0


def test_threaded_scoring_matches_sequential():
    sc = load_scenarios()["default"]
    ev = DummyEvaluator(_pose(120.0, 30.0), sc)
    rng = np.random.default_rng(0)
    poses = [_pose(float(rng.uniform(0, 180)), float(rng.uniform(0, 90))) for _ in range(17)]
    seq = pred.score_candidates(ev, poses, object(), BodyParams(), sc)
    for n in (2, 3, 8):
        par = pred.score_candidates(ev, poses, object(), BodyParams(), sc, n_threads=n)
        assert np.array_equal(seq[0], par[0]) and np.array_equal(seq[1], par[1]), n


def test_threaded_scoring_matches_sequential_on_patch_evaluator():
    """실제 패치판(캡슐, 저밀도)에서도 스레드 결과가 순차와 같다 (D 확인을 F 쪽에서도 고정)."""
    from airis.optimize import cli

    sc = load_scenarios()["default"]
    ev = pred._rescore_evaluator("capsule", 200.0)
    nozzle = cli.resolve_nozzles()[0]
    poses = [_pose(a, y) for a, y in ((10, 30), (175, 60), (90, 45), (40, 80), (150, 10), (60, 70))]
    seq = pred.score_candidates(ev, poses, nozzle, BodyParams(), sc)
    par = pred.score_candidates(ev, poses, nozzle, BodyParams(), sc, n_threads=3)
    assert np.array_equal(seq[0], par[0]) and np.array_equal(seq[1], par[1])


def test_run_e5_variant_kwargs():
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_e5_flow

    args = run_e5_flow.build_parser().parse_args(["--dedup-deg", "2", "--screen-b", "500:5", "--threads", "2"])
    assert run_e5_flow.variant_kwargs("hybrid", args) == {}
    assert run_e5_flow.variant_kwargs("hybrid-a", args) == {"dedup_deg": 2.0}
    assert run_e5_flow.variant_kwargs("hybrid-b", args) == {"dedup_deg": 2.0, "screen_density": 500.0,
                                                            "screen_top": 5}
    assert run_e5_flow.variant_kwargs("hybrid-c", args)["screen_density"] == 800.0
    assert run_e5_flow.variant_kwargs("hybrid-d", args) == {"dedup_deg": 2.0, "n_threads": 2}
    assert run_e5_flow.variant_kwargs("hybrid-e", args) == {"dedup_deg": 2.0, "screen_density": 500.0,
                                                            "screen_top": 5, "n_threads": 2}
    assert run_e5_flow.variant_kwargs("hybrid-a1", args) == {"dedup_deg": 1.0}
    assert run_e5_flow.variant_kwargs("hybrid-a2", args) == {"dedup_deg": 2.0}
    assert run_e5_flow.variant_kwargs("hybrid-bx", args) == {"dedup_deg": 2.0, "screen_density": 500.0,
                                                             "screen_top": 5, "screen_keep_extra": True}
    assert run_e5_flow.variant_kwargs("hybrid-f", args) == {"dedup_deg": 2.0, "screen_density": 500.0,
                                                            "screen_top": 5, "screen_keep_extra": True,
                                                            "n_threads": 2}
    assert {"hybrid-a", "hybrid-a1", "hybrid-a2", "hybrid-b", "hybrid-bx", "hybrid-c", "hybrid-d",
            "hybrid-e", "hybrid-f"} <= set(run_e5_flow.METHODS)


def test_threads_fall_back_to_sequential_for_batch_evaluators():
    """batch_evaluate 를 재정의한 평가기(입자판 같은)는 스레드 동일성을 확인하지 않았으므로 순차로 채점한다."""
    from airis.sim import Evaluator

    sc = load_scenarios()["default"]

    class _Batch(Evaluator):
        def evaluate(self, pose, nozzle, body, scenario):
            raise AssertionError("batch 경로만 써야 한다")

        def batch_evaluate(self, items, body, scenario):
            return [p.shoulder_abduction / 180.0 for p, _ in items]

    poses = [_pose(a) for a in (10.0, 90.0, 170.0)]
    seq = pred.score_candidates(_Batch(), poses, object(), BodyParams(), sc)
    with pytest.warns(RuntimeWarning, match="순차"):
        par = pred.score_candidates(_Batch(), poses, object(), BodyParams(), sc, n_threads=3)
    assert np.array_equal(seq[0], par[0]) and np.array_equal(seq[1], par[1])
