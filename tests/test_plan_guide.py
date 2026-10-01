"""운전 계획 추천·장비 제어 출력·계획 그림 테스트. 소유자: E. (`airis/realtime/plan_guide.py`, `airis/viz/plan_view.py`)

계획 모델 산출물(`data/models/plan_flow.pt`)이 있든 없든 건너뛰는 것 없이 통과해야 한다. 표를 기대하는 테스트는
`no_plan_model` 픽스처로 기본 경로를 없는 파일로 돌려 놓는다. 캡슐 기준 테스트는 `model="capsule"`을 명시한다
(전역 `body.model` 전환과 독립).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from airis.optimize.plan_encoding import plan_physics_cfg
from airis.realtime import plan_guide as pg
from airis.sim.body import build_body
from airis.sim.human_mesh import MESH_DEFAULT_BODY
from airis.sim.patch_baseline import outside_booth
from airis.sim.scenario import (apply_zone_strengths, load_nozzle_layout, load_nozzles, load_physics,
                                load_scenarios, zone_nozzle_counts, zone_strength_caps)
from airis.sim.types import ZONE_NAMES, BodyParams, Phase, Plan, PoseParams
from airis.viz.plan_view import figure_plan_phases, figure_timeline, figure_zone_strengths, phase_panels

SCENARIOS = load_scenarios()
BOOTH = load_nozzle_layout()["booth"]
NOZZLE = load_nozzles()
CAPSULE_BODY = BodyParams(1.70, 0.42, 0.22, 0.62, 0.85)
#: 관절 중심 정의의 1.70 m 체형 (메시 시절 비율). 캡슐 몸 휠체어에서도 만세 계획이 부스 안이다.
PLAN_BODY = BodyParams(1.70, 0.342, 0.194, 0.463, 0.883)
CAP = dict(model="capsule")


@pytest.fixture
def no_plan_model(monkeypatch, tmp_path):
    from airis.model import predict
    monkeypatch.setattr(predict, "DEFAULT_PLAN_MODEL_PATH", tmp_path / "없음_plan_flow.pt")
    return predict


# ---------------------------------------------------------------------------
# 계획 후보표
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_stub_plans_are_already_inside_plan_limits(scenario):
    """표는 E7 P5 결과를 반올림한 값이다. 인코더 투영이 값을 거의 바꾸지 않아야 표가 E7 과 같은 계획이다."""
    sc = SCENARIOS[scenario]
    enc = pg.plan_encoder(sc)
    for entry in pg.PLAN_STUB_TABLE[scenario]:
        clipped = enc.clip_plan(entry.plan)
        assert len(clipped.phases) == pg.plan_limits().n_phases
        np.testing.assert_allclose(clipped.zone_strengths, entry.plan.zone_strengths, atol=2e-3)
        for a, b in zip(clipped.phases, entry.plan.phases):
            assert a.duration_s == pytest.approx(b.duration_s, abs=0.02)
            np.testing.assert_allclose(a.pose.to_vector(), b.pose.to_vector(), atol=1e-6)


def test_pregnant_stub_respects_chest_comfort_cap():
    caps = zone_strength_caps(SCENARIOS["pregnant"])
    s = pg.PLAN_STUB_TABLE["pregnant"][0].plan.zone_strengths
    assert np.all(s <= np.where(np.isfinite(caps), caps, np.inf) + 1e-9)
    assert s[ZONE_NAMES.index("chest_low")] == pytest.approx(0.6)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_stub_plans_stay_inside_booth_for_mesh_default_body(scenario):
    sc = SCENARIOS[scenario]
    for entry in pg.PLAN_STUB_TABLE[scenario]:
        for ph in entry.plan.phases:
            state = build_body(MESH_DEFAULT_BODY, ph.pose, sc, patches_per_m2=200, model="mesh")
            assert not outside_booth(state.patch_pos, BOOTH), (scenario, ph.pose)


# ---------------------------------------------------------------------------
# 추천
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_recommend_plan_picks_best_scored_candidate(scenario, no_plan_model):
    rec = pg.recommend_plan(PLAN_BODY, SCENARIOS[scenario], **CAP)
    assert rec.source.startswith(("stub:", "baseline: P1"))
    assert any("plan_flow.pt" in n for n in rec.notes)
    feasible = [c.score for c in rec.candidates if not c.infeasible]
    assert rec.result.score == pytest.approx(max(feasible))
    assert not rec.result.extra.get("infeasible")
    # 고른 계획은 후보 중 하나 그대로다 (다시 투영하거나 바꾸지 않는다).
    assert any(c.plan is rec.plan for c in rec.candidates)


def test_recommend_plan_scores_with_plan_physics(no_plan_model):
    """계획 경로는 시간 항(kinetics)을 켠 설정으로 채점한다. 꺼진 설정이면 시간이 점수에 영향이 없다."""
    ev = pg.plan_evaluator("capsule")
    assert ev.cfg["adhesion"]["kinetics"]["enabled"] is True
    assert plan_physics_cfg()["adhesion"]["kinetics"] == ev.cfg["adhesion"]["kinetics"]


def test_product_rotation_wins_when_it_scores_higher(monkeypatch, no_plan_model):
    """P1 이 표의 2단계 계획보다 높으면 P1 을 그대로 안내하고 그 사실을 알린다."""
    sc = SCENARIOS["default"]
    weak = Plan([Phase(PoseParams(), 2.5), Phase(PoseParams(), 2.5)], np.full(len(ZONE_NAMES), 0.2))
    monkeypatch.setitem(pg.PLAN_STUB_TABLE, "default", [pg.PlanStubEntry("약한 계획", weak, "test")])
    rec = pg.recommend_plan(CAPSULE_BODY, sc, **CAP)
    assert rec.source == "baseline: P1"
    assert len(rec.plan.phases) == 12
    assert any("12방향 회전" in n for n in rec.notes)


def test_recommend_plan_without_product_rotation_uses_table(no_plan_model):
    rec = pg.recommend_plan(CAPSULE_BODY, SCENARIOS["pregnant"], include_product_rotation=False, **CAP)
    assert rec.source.startswith("stub:")
    assert all(c.source != "P1" for c in rec.candidates)


def test_recommend_plan_uses_model_candidates(monkeypatch):
    """계획 모델이 있으면 그 후보도 같은 풀에서 채점하고, 이기면 출처가 model 이다."""
    from airis.model import predict as P

    sc = SCENARIOS["default"]
    strong = pg.PLAN_STUB_TABLE["default"][0].plan
    seen = {}

    def fake(body, scenario, **kw):
        seen.update(kw)
        return SimpleNamespace(candidates=[strong])

    monkeypatch.setattr(P, "predict_plan_candidates", fake)
    monkeypatch.setitem(pg.PLAN_STUB_TABLE, "default", [pg.PlanStubEntry(
        "약한 계획", Plan([Phase(PoseParams(), 2.5), Phase(PoseParams(), 2.5)],
                       np.full(len(ZONE_NAMES), 0.2)), "test")])
    rec = pg.recommend_plan(CAPSULE_BODY, sc, include_product_rotation=False, **CAP)
    assert rec.source == "model: plan_flow"
    assert seen["rescore"] is False and seen["n_samples"] == pg.PLAN_MODEL_SAMPLES
    assert [c.source for c in rec.candidates] == ["model", "stub"]


@pytest.mark.parametrize("exc, word", [(FileNotFoundError("x"), "plan_flow.pt"),
                                       (ImportError("no torch"), "torch"),
                                       (KeyError("wheelchair"), "KeyError")])
def test_recommend_plan_falls_back_with_note(monkeypatch, exc, word):
    from airis.model import predict as P

    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(P, "predict_plan_candidates", boom)
    rec = pg.recommend_plan(CAPSULE_BODY, SCENARIOS["default"], **CAP)
    assert rec.source.startswith(("stub:", "baseline: P1"))
    assert any(word in n for n in rec.notes)


def test_recommend_plan_falls_back_to_p0_when_everything_is_outside(monkeypatch, no_plan_model):
    sc = SCENARIOS["default"]
    real = pg.PatchEvaluator.evaluate_plan
    calls = {"n": 0}

    def outside_first(self, plan, nozzle, body, scenario):
        calls["n"] += 1
        r = real(self, plan, nozzle, body, scenario)
        if calls["n"] <= 2:                  # 표 1개 + P1: 전부 부스 밖이라고 본다
            r.extra["infeasible"] = True
        return r

    monkeypatch.setattr(pg.PatchEvaluator, "evaluate_plan", outside_first)
    rec = pg.recommend_plan(CAPSULE_BODY, sc, **CAP)
    assert rec.source == "fallback: P0"
    assert len(rec.plan.phases) == 1
    assert any("P0" in n for n in rec.notes)


def test_unevaluable_candidate_is_dropped_with_note(no_plan_model):
    """캡슐 몸 휠체어(이 체형)는 yaw −30° 에서 패치 수가 바뀌어 P1 을 못 잰다 (B 보장 위반). 화면은 멈추지 않는다.

    이 체형은 팔이 길어 표의 만세 계획도 부스 밖이라, 남는 후보가 없어 P0 으로 안내한다.
    """
    rec = pg.recommend_plan(CAPSULE_BODY, SCENARIOS["wheelchair"], **CAP)
    p1 = [c for c in rec.candidates if c.source == "P1"]
    assert p1 and p1[0].infeasible
    assert any("평가할 수 없어 제외" in n for n in rec.notes)
    assert rec.source == "fallback: P0"
    assert not rec.result.extra.get("infeasible")


def test_compare_plan_rows_and_current_operation_energy(no_plan_model):
    sc = SCENARIOS["default"]
    rec = pg.recommend_plan(CAPSULE_BODY, sc, **CAP)
    rows = pg.compare_plan_with_baselines(CAPSULE_BODY, sc, rec, **CAP)
    assert [r.name for r in rows] == ["추천", "P0 현행 운전", "P1 제품 안내 (12방향 회전)"]
    by = {r.name: r for r in rows}
    # 현행 운전(전 팬 1, 20 s)의 에너지는 정의상 1 이다 (00_common 4.7).
    assert by["P0 현행 운전"].energy == pytest.approx(1.0)
    assert by["P0 현행 운전"].duration_s == pytest.approx(20.0)
    # 추천은 같은 채점기의 풀에서 고른 최고라 P1 보다 낮을 수 없다.
    assert by["추천"].score >= by["P1 제품 안내 (12방향 회전)"].score - 1e-9


# ---------------------------------------------------------------------------
# 안내 문장
# ---------------------------------------------------------------------------
def test_plan_instructions_per_phase_and_side_switch():
    plan = pg.PLAN_STUB_TABLE["default"][0].plan
    steps = pg.plan_instructions(plan, SCENARIOS["default"])
    assert [t for t, _ in steps] == ["1단계 · 7.2초", "2단계 · 7.0초"]
    assert steps[1][1][0] == "반대쪽 벽을 보도록 몸을 돌리세요."
    assert any("만세" in line for line in steps[0][1])


def test_plan_instructions_compress_rotation():
    p1 = pg.product_rotation_plan(SCENARIOS["default"])
    steps = pg.plan_instructions(p1, SCENARIOS["default"])
    assert len(steps) == 1 and "12단계" in steps[0][0]


# ---------------------------------------------------------------------------
# 장비 제어 출력
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_device_control_speeds_match_zone_mapping(scenario):
    sc = SCENARIOS[scenario]
    plan = pg.PLAN_STUB_TABLE[scenario][0].plan
    out = pg.device_control(plan, sc)
    json.dumps(out)                                          # 표준 JSON 만
    phases = [s for s in out["segments"] if s["kind"] == "phase"]
    assert len(phases) == len(plan.phases)
    for seg, ph in zip(phases, plan.phases):
        expect = apply_zone_strengths(NOZZLE, plan.zone_strengths, ph.pose.torso_yaw).strengths
        np.testing.assert_allclose(seg["speed_ratio"], expect, atol=1e-4)
        rated = load_physics()["fan"]["rated_flow_m3_min"]
        assert seg["total_flow_m3_min"] == pytest.approx(sum(seg["speed_ratio"]) * rated, abs=0.02)
        assert len(seg["nozzle_zone"]) == NOZZLE.count
    # 총 풍량은 한도(cap_ratio × 노즐 수 × 정격) 안이다.
    limits = pg.plan_limits()
    for seg in phases:
        assert sum(seg["speed_ratio"]) <= limits.cap_ratio * zone_nozzle_counts(NOZZLE).sum() + 1e-6


def test_device_control_switches_chest_wall_between_phases():
    plan = pg.PLAN_STUB_TABLE["default"][0].plan              # yaw +47.7° → −55.6°
    out = pg.device_control(plan, SCENARIOS["default"])
    p1, p2 = [s for s in out["segments"] if s["kind"] == "phase"]
    assert (p1["chest_wall"], p2["chest_wall"]) == ("+y", "-y")
    # 같은 물리 노즐이 단계마다 가슴 쪽 ↔ 등 쪽 구역으로 바뀐다.
    swapped = [(a, b) for a, b in zip(p1["nozzle_zone"], p2["nozzle_zone"]) if a != b]
    assert swapped and all({a.split("_")[0], b.split("_")[0]} == {"chest", "back"} for a, b in swapped)


def test_device_control_timeline_and_transition():
    plan = pg.PLAN_STUB_TABLE["default"][0].plan
    out = pg.device_control(plan, SCENARIOS["default"])
    kinds = [s["kind"] for s in out["segments"]]
    assert kinds == ["phase", "transition", "phase"]
    starts = [s["start_s"] for s in out["segments"]]
    assert starts == sorted(starts)
    for a, b in zip(out["segments"], out["segments"][1:]):
        assert a["end_s"] == pytest.approx(b["start_s"], abs=0.011)
    tr = out["segments"][1]
    assert tr["end_s"] - tr["start_s"] == pytest.approx(pg.plan_limits().transition_s, abs=0.011)
    assert all(v == 0.0 for v in tr["speed_ratio"])            # 기본: 전환 동안 팬 끔
    assert out["timeline_s"] == pytest.approx(plan.duration_s + pg.plan_limits().transition_s, abs=0.02)
    held = pg.device_control(plan, SCENARIOS["default"], transition_fans="hold")
    assert held["segments"][1]["speed_ratio"] == held["segments"][0]["speed_ratio"]
    with pytest.raises(ValueError):
        pg.device_control(plan, SCENARIOS["default"], transition_fans="maybe")


def test_device_control_rotation_has_no_transitions():
    p1 = pg.product_rotation_plan(SCENARIOS["default"])
    out = pg.device_control(p1, SCENARIOS["default"])
    assert all(s["kind"] == "phase" for s in out["segments"])
    assert out["timeline_s"] == pytest.approx(20.0, abs=0.02)


def test_device_control_copies_evaluator_metrics(no_plan_model):
    sc = SCENARIOS["default"]
    rec = pg.recommend_plan(CAPSULE_BODY, sc, **CAP)
    out = pg.device_control(rec.plan, sc, result=rec.result)
    assert out["energy"] == pytest.approx(rec.result.extra["energy"], abs=1e-4)
    assert out["score"] == pytest.approx(rec.result.score, abs=1e-4)
    assert pg.device_control(rec.plan, sc)["energy"] is None


# ---------------------------------------------------------------------------
# 그림
# ---------------------------------------------------------------------------
def test_phase_panels_sample_long_plans():
    p1 = pg.product_rotation_plan(SCENARIOS["default"])
    assert phase_panels(p1) == [0, 4, 7, 11]
    assert phase_panels(pg.PLAN_STUB_TABLE["default"][0].plan) == [0, 1]


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_plan_figures_build_and_serialize(scenario):
    sc = SCENARIOS[scenario]
    plan = pg.PLAN_STUB_TABLE[scenario][0].plan
    figs = [figure_plan_phases(CAPSULE_BODY, plan, sc, patches_per_m2=150, **CAP),
            figure_zone_strengths(plan, sc, s_max=1.0),
            figure_timeline(pg.device_control(plan, sc))]
    for fig in figs:
        json.dumps(fig.to_dict(), default=str)
    caps = [t for t in figs[1].data if t.name == "쾌적 상한 (시나리오)"]
    assert bool(caps) == (scenario == "pregnant")
