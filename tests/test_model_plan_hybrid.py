"""혼합 계획 추천(flow + kNN + 고정 회전 계획 → 재채점) 테스트. 소유자: F.

총괄 2026-10-06: 계획 데이터셋이 나오기 전에 학습·추천 코드를 미리 만든다. 단계 수 N 은 데이터에서 읽으므로
여기서는 N = 3 합성 데이터로 본다 (설정 파일의 n_phases 2 와 일부러 다르게).
"""
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))      # 합성 계획 데이터 도우미(test_model_plan_knn)를 함께 쓴다

torch = pytest.importorskip("torch", reason="flow matching 모델은 torch 필요")
pd = pytest.importorskip("pandas")

from test_model_plan_knn import MIN_PHASE, limits_for, plan_df   # noqa: E402

from airis.model import flow                                   # noqa: E402
from airis.model import predict as pred                        # noqa: E402
from airis.model.knn import PlanKNN                            # noqa: E402
from airis.optimize.plan_encoding import PlanEncoder           # noqa: E402
from airis.sim import ZONE_NAMES, BodyParams, EvalResult, Evaluator, Phase, Plan, PoseParams   # noqa: E402
from airis.sim.scenario import load_scenarios                  # noqa: E402

N = 3
CFG = flow.FlowConfig(hidden=48, layers=2, steps=500, batch_size=128, lr=2e-3, seed=0)


class PlanScorer(Evaluator):
    """가짜 계획 평가기. 점수는 주어진 함수로, 불가는 "키 1.85 m 초과인 몸의 만세"로 정한다."""

    def __init__(self, score=lambda plan: 0.5):
        self.score, self.calls = score, []

    def evaluate(self, pose, nozzle, body, scenario):
        raise NotImplementedError

    def evaluate_plan(self, plan, nozzle, body, scenario):
        self.calls.append((plan, body.height_m))
        bad = body.height_m > 1.85 and any(ph.pose.shoulder_abduction >= 120.0 for ph in plan.phases)
        return EvalResult(score=-2.0 if bad else float(self.score(plan)), removal_by_part=np.zeros(5),
                          total_removal=0.0, discomfort=0.0, extra={"infeasible": bad, "energy": 0.0})


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios()


@pytest.fixture(autouse=True)
def matching_stamp(monkeypatch):
    """합성 데이터의 도장(n0·p0)을 지금 설정으로 본다. 표를 처음 읽는 테스트가 무엇이든 도장 경고가 나지 않게."""
    monkeypatch.setattr(pred, "current_stamp", lambda: {"nozzle_layout_hash": "n0", "physics_hash": "p0",
                                                        "kinetics_enabled": True, "time_constant_s": 2.0})


@pytest.fixture(scope="module")
def artifacts(scenarios, tmp_path_factory):
    """N = 3 합성 계획 데이터로 만든 flow 산출물과 kNN 표."""
    df = plan_df(N)
    lim = limits_for(N)
    space = flow.PlanSpace.from_scenarios([scenarios[s] for s in ("default", "wheelchair")], n_phases=N,
                                          duration_bounds_s=lim.duration_bounds_s, min_phase_s=lim.min_phase_s)
    model = flow.train_flow(df, scenarios, CFG, space=space, meta={"body_model": "capsule", "patches_per_m2": 400.0})
    d = tmp_path_factory.mktemp("plan_artifacts")
    return {"df": df, "model": model, "flow": model.save(d / "plan_flow.pt"),
            "knn": PlanKNN.from_dataset(df).save(d / "plan_knn.parquet"), "none": d / "none"}


def kw(artifacts, **more):
    return dict(path=artifacts["flow"], knn_path=artifacts["knn"], nozzle=object(), **more)


# ---------- 한도는 산출물에서 ----------

def test_limits_come_from_artifacts_not_config(artifacts):
    model, knn = artifacts["model"], PlanKNN.load(artifacts["knn"])
    assert pred.plan_limits_for().n_phases == 2, "아무 산출물도 없으면 설정 파일 (n_phases 2)"
    for lim in (pred.plan_limits_for(model, knn), pred.plan_limits_for(model), pred.plan_limits_for(None, knn)):
        assert lim.n_phases == N and lim.min_phase_s == MIN_PHASE
        assert lim.duration_bounds_s == pytest.approx((6.0, 20.0)), "N 단계면 하한이 N × min_phase_s"
        assert lim.s_max == pytest.approx(1.0)
    # 한도 열이 없는 kNN 표: 하한을 N × min_phase_s 로 올린다 (설정 파일의 5 s 그대로면 인코더가 거부한다)
    bare = PlanKNN.from_dataset(plan_df(9, n_bodies=4, stamps=False))
    lim9 = pred.plan_limits_for(None, bare)
    assert lim9.n_phases == 9 and lim9.duration_bounds_s == pytest.approx((18.0, 20.0))
    PlanEncoder(load_scenarios()["default"], lim9)             # 인코더가 받아들인다
    # 데이터셋의 한도 열이 우선한다
    odd = PlanKNN.from_dataset(plan_df(N, n_bodies=4).assign(duration_lo_s=8.0, cap_ratio=0.7, transition_s=1.0))
    lim_odd = pred.plan_limits_for(None, odd)
    assert lim_odd.duration_bounds_s == pytest.approx((8.0, 20.0)) and lim_odd.cap_ratio == 0.7
    assert lim_odd.transition_s == 1.0
    with pytest.raises(ValueError, match="단계 수가 다르다"):
        pred.plan_limits_for(model, PlanKNN.from_dataset(plan_df(5, n_bodies=4)))


# ---------- 혼합 후보 ----------

def test_hybrid_gathers_flow_knn_and_fixed_rotations(artifacts, scenarios):
    sc = scenarios["default"]
    ev = PlanScorer()
    best_pose = PoseParams(shoulder_abduction=170.0, torso_yaw=40.0)
    p = pred.predict_plan_candidates(BodyParams(), sc, evaluator=ev, n_flow=4, n_knn=3, rotation_pose=best_pose,
                                     **kw(artifacts))
    assert p.sources == ["flow"] * 4 + ["knn"] * 3 + ["fixed"] * 2
    assert [len(c.phases) for c in p.candidates] == [N] * 7 + [pred.ROTATION_STEPS] * 2
    assert p.scores.shape == (9,) and len(ev.calls) == 9, "후보를 전부 한 번씩 채점한다"
    rot_best, rot_default = p.candidates[7], p.candidates[8]
    assert all(ph.pose.shoulder_abduction == pytest.approx(170.0) for ph in rot_best.phases), "제안 자세 그대로 회전"
    assert all(ph.pose.shoulder_abduction == PoseParams().shoulder_abduction for ph in rot_default.phases)
    yaws = [ph.pose.torso_yaw for ph in rot_default.phases]
    assert len(set(np.round(yaws, 6))) == pred.ROTATION_STEPS, "10방향으로 돈다"
    assert rot_default.duration_s == pytest.approx(20.0)
    assert all(ph.duration_s == pytest.approx(2.0) for ph in rot_default.phases), "10단계면 단계당 2.0 s"
    assert p.plan is p.candidates[0] and p.source == "flow", "동률이면 앞 후보"

    # rotation_pose 를 주지 않으면 기본 자세 회전만 넣는다
    q = pred.predict_plan_candidates(BodyParams(), sc, evaluator=PlanScorer(), n_flow=2, n_knn=2, **kw(artifacts))
    assert q.sources == ["flow"] * 2 + ["knn"] * 2 + ["fixed"]


def test_fixed_rotation_wins_when_it_scores_best(artifacts, scenarios):
    """평가기가 회전(단계가 많은 계획)을 높게 매기면 고정 회전 계획이 뽑힌다. 모델 후보보다 나빠지지 않는다."""
    sc = scenarios["default"]
    ev = PlanScorer(lambda plan: len(plan.phases) + plan.phases[0].pose.shoulder_abduction / 1000.0)
    p = pred.predict_plan_candidates(BodyParams(), sc, evaluator=ev, rotation_pose=PoseParams(shoulder_abduction=90.0),
                                     **kw(artifacts))
    assert p.source == "fixed" and p.plan is p.candidates[p.sources.index("fixed")], "제안 자세 회전이 가장 높다"
    assert p.scores.max() == pytest.approx(10.09)
    assert isinstance(pred.predict_plan(BodyParams(), sc, evaluator=ev, **kw(artifacts)), Plan)


def test_candidates_are_clipped_to_artifact_limits(artifacts, scenarios):
    """flow·kNN 후보는 산출물의 한도(N = 3, 6~20 s)로 투영된다. 설정 파일의 n_phases 2 로는 투영조차 못 한다."""
    wc = scenarios["wheelchair"]
    p = pred.predict_plan_candidates(BodyParams(), wc, evaluator=PlanScorer(), n_flow=8, n_knn=8, **kw(artifacts))
    lo, hi = wc.pose_bounds["torso_yaw"]
    for plan, src in zip(p.candidates, p.sources):
        assert all(ph.pose.hip_flexion == 90.0 and lo - 1e-9 <= ph.pose.torso_yaw <= hi + 1e-9 for ph in plan.phases)
        assert np.all((plan.zone_strengths >= 0) & (plan.zone_strengths <= 1.0 + 1e-12))
        if src != "fixed":
            assert len(plan.phases) == N and 6.0 - 1e-9 <= plan.duration_s <= 20.0 + 1e-9
            assert min(ph.duration_s for ph in plan.phases) >= MIN_PHASE - 1e-9
    # 쾌적 상한이 있는 시나리오: 모델 후보도 고정 회전 계획도 상한을 지킨다
    capped = replace(scenarios["default"], nozzle_strength_cap={"chest_low": 0.6, "chest_high": 0.6})
    chest = [ZONE_NAMES.index(z) for z in ("chest_low", "chest_high")]
    c = pred.predict_plan_candidates(BodyParams(), capped, evaluator=PlanScorer(), **kw(artifacts))
    assert all(np.all(plan.zone_strengths[chest] <= 0.6 + 1e-12) for plan in c.candidates)


def test_extra_candidates_keep_or_clip_by_phase_count(artifacts, scenarios):
    sc = scenarios["default"]

    def pose(yaw):
        return PoseParams(shoulder_abduction=15.0, torso_yaw=yaw)

    same_n = Plan([Phase(pose(10.0 * k), 30.0) for k in range(N)], np.full(len(ZONE_NAMES), 2.0))   # 범위 밖
    other_n = Plan([Phase(pose(10.0 * k), 4.0) for k in range(5)], np.ones(len(ZONE_NAMES)))
    p = pred.predict_plan_candidates(BodyParams(), sc, evaluator=PlanScorer(), n_flow=1, n_knn=1,
                                     extra_candidates=[same_n, other_n], **kw(artifacts))
    assert p.sources == ["flow", "knn", "fixed", "extra", "extra"]
    clipped, kept = p.candidates[3], p.candidates[4]
    assert clipped.duration_s == pytest.approx(20.0) and np.all(clipped.zone_strengths <= 1.0), "단계 수가 같으면 투영"
    assert kept is other_n, "단계 수가 다른 계획은 그대로 채점한다"


# ---------- 폴백과 예외 ----------

def test_falls_back_to_available_artifact_with_warning(artifacts, scenarios):
    sc = scenarios["default"]
    with pytest.warns(RuntimeWarning, match="kNN 표 없음"):
        only_flow = pred.predict_plan_candidates(BodyParams(), sc, evaluator=PlanScorer(), path=artifacts["flow"],
                                                 knn_path=artifacts["none"], nozzle=object())
    assert set(only_flow.sources) == {"flow", "fixed"} and only_flow.sources.count("flow") == pred.N_FLOW
    with pytest.warns(RuntimeWarning, match="flow 산출물 없음"):
        only_knn = pred.predict_plan_candidates(BodyParams(), sc, evaluator=PlanScorer(), path=artifacts["none"],
                                                knn_path=artifacts["knn"], nozzle=object())
    assert set(only_knn.sources) == {"knn", "fixed"} and len(only_knn.candidates[0].phases) == N
    with warnings.catch_warnings():
        warnings.simplefilter("error")                          # backend 를 정했으면 경고 없이 그쪽만
        k = pred.predict_plan_candidates(BodyParams(), sc, evaluator=PlanScorer(), backend="knn", n_knn=5,
                                         path=artifacts["none"], knn_path=artifacts["knn"], nozzle=object())
    assert k.sources == ["knn"] * 5, "backend 를 정하면 고정 회전 계획은 넣지 않는다 (fixed=True 로 켤 수 있다)"
    with pytest.raises(FileNotFoundError, match="계획 모델 산출물이 없다"):
        pred.predict_plan(BodyParams(), sc, path=artifacts["none"], knn_path=artifacts["none"])
    with pytest.raises(FileNotFoundError):
        pred.predict_plan(BodyParams(), sc, backend="knn", knn_path=artifacts["none"])
    with pytest.raises(KeyError, match="pregnant"):
        pred.predict_plan(BodyParams(), scenarios["pregnant"], evaluator=PlanScorer(), **kw(artifacts))
    with pytest.raises(ValueError, match="backend"):
        pred.predict_plan(BodyParams(), sc, backend="best", **kw(artifacts))
    with pytest.raises(ValueError, match="n_samples"):
        pred.predict_plan(BodyParams(), sc, backend="hybrid", n_samples=4, **kw(artifacts))


def test_legacy_n_samples_and_no_rescore(artifacts, scenarios):
    """n_samples 를 주면 예전 동작(flow 만, 고정 계획 없음). rescore=False 면 첫 후보를 투영만 해서 돌려준다."""
    sc = scenarios["default"]
    ev = PlanScorer()
    p = pred.predict_plan_candidates(BodyParams(), sc, n_samples=6, evaluator=ev, **kw(artifacts))
    assert p.sources == ["flow"] * 6 and len(ev.calls) == 6
    with_fixed = pred.predict_plan_candidates(BodyParams(), sc, n_samples=6, fixed=True, evaluator=PlanScorer(),
                                              **kw(artifacts))
    assert with_fixed.sources == ["flow"] * 6 + ["fixed"]
    raw = pred.predict_plan_candidates(BodyParams(), sc, rescore=False, **kw(artifacts))
    assert raw.scores is None and raw.plan is raw.candidates[0] and "fixed" not in raw.sources
    assert len(raw.plan.phases) == N and 6.0 - 1e-9 <= raw.plan.duration_s <= 20.0 + 1e-9


# ---------- 부스 안 판정 여유 ----------

def test_feasibility_margin_applies_to_plans(artifacts, scenarios):
    """키운 체형에서 한 단계라도 천장에 닿는 계획(만세)을 넘기고, 다음 순위 계획을 고른다."""
    sc = scenarios["default"]
    body = BodyParams(height_m=1.80)
    score = lambda plan: plan.phases[0].pose.shoulder_abduction / 180.0          # noqa: E731  만세가 높다
    base = pred.predict_plan_candidates(body, sc, evaluator=PlanScorer(score), **kw(artifacts))
    assert base.plan.phases[0].pose.shoulder_abduction >= 120.0
    assert (base.n_margin_checks, base.n_margin_rejected, base.margin_fallback) == (0, 0, False)

    ev = PlanScorer(score)
    safe = pred.predict_plan_candidates(body, sc, evaluator=ev, feasibility_margin=0.05, **kw(artifacts))
    n_up = sum(any(ph.pose.shoulder_abduction >= 120.0 for ph in c.phases) for c in safe.candidates)
    assert all(ph.pose.shoulder_abduction < 120.0 for ph in safe.plan.phases)
    assert safe.n_margin_rejected == n_up and safe.n_margin_checks == n_up + 1 and not safe.margin_fallback
    assert np.array_equal(safe.scores, base.scores) and safe.source == safe.sources[safe.candidates.index(safe.plan)]
    heights = sorted({round(h, 3) for _, h in ev.calls})
    assert heights == [1.80, 1.89], "점수는 추정 체형으로, 여유 확인만 키운 체형으로"
    with pytest.raises(ValueError, match="여유 방식"):
        pred.predict_plan_candidates(body, sc, evaluator=PlanScorer(), feasibility_margin=0.05,
                                     feasibility_margin_mode="tall", **kw(artifacts))


def test_default_rescorer_for_plans_uses_plan_physics(artifacts, scenarios, monkeypatch):
    """평가기를 주지 않으면 계획 채점 설정(kinetics 켬)의 재채점기를 쓴다. kNN 표만 있어도 된다."""
    seen = {}

    def fake(model, *, plan=False):
        seen["plan"], seen["type"] = plan, type(model).__name__
        return PlanScorer(), object()

    monkeypatch.setattr(pred, "default_rescorer", fake)
    pred.predict_plan_candidates(BodyParams(), scenarios["default"], backend="knn", n_knn=2, knn_path=artifacts["knn"])
    assert seen == {"plan": True, "type": "PlanKNN"}
