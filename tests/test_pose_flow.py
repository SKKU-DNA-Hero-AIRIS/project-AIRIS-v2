"""조건부 flow matching 자세 모델 테스트. 소유자: F. docs/proposals/flow_matching.md.

합성 데이터로 (1) 봉우리 두 개를 평균 내지 않고 둘 다 뽑는지, (2) 체형 조건에 따라 봉우리가 바뀌는지,
(3) interfaces.md 회귀 계약(clip_pose, fixed_pose, 예외 규약)을 지키는지 본다.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="flow matching 모델은 torch 필요")
pd = pytest.importorskip("pandas")

from airis.model import flow                                   # noqa: E402
from airis.model import predict as pred                        # noqa: E402
from airis.optimize import dataset                             # noqa: E402
from airis.optimize.dummy import DummyEvaluator                # noqa: E402
from airis.sim import BodyParams, PoseParams                   # noqa: E402
from airis.sim.scenario import load_scenarios                  # noqa: E402

UP, DOWN = 175.0, 10.0          # 팔 벌림 두 봉우리 (만세 / 팔 내림)
TALL, SHORT = BodyParams(height_m=1.85), BodyParams(height_m=1.52)


def _row(i, body, scenario, abd, yaw, hip=0.0, knee=5.0):
    r = {"body_idx": i, "scenario": scenario, "score": 0.5,
         "arm_class": "hands_up" if abd >= 90 else "arms_down"}
    r.update({f"body_{k}": getattr(body, k) for k in flow.BODY_KEYS})
    pose = dict(shoulder_abduction=abd, shoulder_flexion=-4.0, elbow_flexion=2.0, torso_pitch=0.0,
                torso_yaw=yaw, hip_flexion=hip, knee_flexion=knee)
    r.update({f"pose_{k}": v for k, v in pose.items()})
    return r


def synthetic_df(n=300, seed=0):
    """default: 체형과 무관하게 봉우리 반반 / pregnant: 키 1.68 m 이상이면 만세 / wheelchair: 하체 고정."""
    rng = np.random.default_rng(seed)
    bodies = dataset.sample_bodies(n, seed=seed)
    rows = []
    for i, b in enumerate(bodies):
        up = rng.random() < 0.5
        rows.append(_row(i, b, "default", UP if up else DOWN, 72 + rng.normal(0, 2)))
        rows.append(_row(i, b, "pregnant", UP if b.height_m >= 1.68 else DOWN, 78 + rng.normal(0, 2)))
        rows.append(_row(i, b, "wheelchair", UP, 36 + rng.normal(0, 2), hip=90.0, knee=90.0))
    return pd.DataFrame(rows)


# 1,500 스텝이면 봉우리 사이에 샘플이 약 17% 남는다 (덜 학습). 3,000 스텝부터 약 2%.
SMALL_CFG = flow.FlowConfig(hidden=64, layers=3, steps=3000, batch_size=256, lr=2e-3, seed=0)


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios()


@pytest.fixture(scope="module")
def model(scenarios):
    return flow.train_pose_flow(synthetic_df(), scenarios, SMALL_CFG,
                                meta={"body_model": "capsule", "patches_per_m2": 400.0})


def _abd(model, body, scenario, n=400, seed=0):
    return np.array([p.shoulder_abduction for p in model.sample_poses(body, scenario, n, seed=seed)])


def test_pose_space_bounds_and_mask(scenarios):
    space = flow.PoseSpace.from_scenarios([scenarios[s] for s in ("default", "pregnant", "wheelchair")])
    b = space.to_dict()
    assert b["torso_yaw"] == [0.0, 90.0], "yaw 는 0~90° 로 접은 범위"
    assert b["shoulder_abduction"] == [0.0, 180.0]
    assert b["torso_pitch"] == [-20.0, 45.0], "시나리오 범위의 합집합"
    m = space.mask(scenarios["wheelchair"])
    assert m[flow.POSE_KEYS.index("hip_flexion")] == 0 and m[flow.POSE_KEYS.index("knee_flexion")] == 0
    assert space.mask(scenarios["default"]).all()
    deg = np.array([[90.0, 0.0, 70.0, 12.5, 45.0, 15.0, 15.0]])
    assert np.allclose(space.decode(space.encode(deg)), deg)


def test_samples_both_modes_without_averaging(model, scenarios):
    """default 는 체형과 무관하게 반반이다. 평균 회귀라면 약 92° 를 내는 자리."""
    abd = _abd(model, BodyParams(), scenarios["default"])
    up, down = np.mean(abd >= 150), np.mean(abd <= 45)
    assert up > 0.25 and down > 0.25, (up, down)
    assert np.mean((abd > 60) & (abd < 150)) < 0.1, "두 봉우리 사이(60~150°)는 거의 없어야 한다"


def test_mode_follows_body_condition(model, scenarios):
    assert np.mean(_abd(model, TALL, scenarios["pregnant"]) >= 150) > 0.8
    assert np.mean(_abd(model, SHORT, scenarios["pregnant"]) <= 45) > 0.8


def test_fixed_pose_and_bounds_applied(model, scenarios):
    wc = scenarios["wheelchair"]
    poses = model.sample_poses(BodyParams(), wc, 64)
    assert all(p.hip_flexion == 90.0 and p.knee_flexion == 90.0 for p in poses)
    for p in poses:
        for key, (lo, hi) in wc.pose_bounds.items():
            if key not in wc.fixed_pose:
                assert lo <= getattr(p, key) <= hi, key
    yaw = np.array([p.torso_yaw for p in poses])
    assert np.median(yaw) == pytest.approx(36, abs=6)


def test_unknown_scenario_raises_key_error(model, scenarios):
    from dataclasses import replace

    with pytest.raises(KeyError):
        model.sample_poses(BodyParams(), replace(scenarios["default"], name="stroller"), 4)


def test_save_load_reproducible(model, scenarios, tmp_path):
    path = model.save(tmp_path / "m.pt")
    again = flow.PoseFlow.load(path)
    a = model.sample(BodyParams(), "default", 32, seed=7)
    b = again.sample(BodyParams(), "default", 32, seed=7)
    assert np.allclose(a, b)
    assert again.meta["body_model"] == "capsule"
    assert not np.allclose(a, model.sample(BodyParams(), "default", 32, seed=8)), "시드가 다르면 다른 샘플"


def test_predict_contract_and_rescoring(model, scenarios, tmp_path):
    path = model.save(tmp_path / "m.pt")
    sc = scenarios["default"]
    with pytest.raises(FileNotFoundError):     # flow 산출물이 없을 때 (backend="flow")
        pred.predict_pose(BodyParams(), sc, backend="flow", path=tmp_path / "none.pt")

    # 재채점: 평가기가 팔 내림 봉우리를 선호하면 반반 분포에서 팔 내림을 고른다.
    target = PoseParams(shoulder_abduction=DOWN, torso_yaw=72.0, knee_flexion=5.0, elbow_flexion=2.0,
                        shoulder_flexion=-4.0)
    ev = DummyEvaluator(target, sc)
    p = pred.predict(BodyParams(), sc, backend="flow", n_samples=32, path=path, evaluator=ev, nozzle=object())
    assert p.pose.shoulder_abduction <= 45
    assert len(p.candidates) == 32 and p.scores.shape == (32,)
    assert p.pose == p.candidates[int(np.argmax(p.scores))]

    only = pred.predict(BodyParams(), sc, backend="flow", n_samples=8, rescore=False, path=path)
    assert only.scores is None and only.pose == only.candidates[0]

    from dataclasses import replace
    with pytest.raises(KeyError):
        pred.predict_pose(BodyParams(), replace(sc, name="stroller"), path=path, evaluator=ev, nozzle=object())


def test_candidate_rows_expand_with_group_weights(scenarios):
    df = synthetic_df(n=4)
    for k in flow.POSE_KEYS:
        vals: list = [None] * len(df)
        vals[0] = [float(df.at[0, f"pose_{k}"])] * 3              # 첫 행만 후보 3개
        df[f"cand_pose_{k}"] = vals
    space = flow.PoseSpace.from_scenarios([scenarios[s] for s in ("default", "pregnant", "wheelchair")])
    arr = flow.training_arrays(df, space, scenarios)
    assert len(arr["x"]) == len(df) + 2
    sums = np.bincount(arr["group"], weights=arr["weight"])
    assert np.allclose(sums, 1.0), "묶음(체형 × 시나리오)마다 가중치 합 1"


def test_dataset_candidate_columns(tmp_path):
    pytest.importorskip("cma")
    pytest.importorskip("pyarrow")
    cfg = dataset.DatasetConfig(evaluator="dummy", max_evals=60, popsize=10, dataset_id="t",
                                candidate_k=4, candidate_tol=0.5, candidate_min_dist=0.05)
    out = tmp_path / "ds.parquet"
    dataset.build_dataset(out, n_bodies=2, scenarios=["default", "wheelchair"], cfg=cfg, log=lambda *_: None)
    df = pd.read_parquet(out)
    assert (df["candidate_k"] == 4).all()
    for _, row in df.iterrows():
        n = int(row["n_candidates"])
        assert 1 <= n <= 4
        scores = np.asarray(row["cand_score"])
        assert np.all(np.diff(scores) <= 0), "점수 내림차순"
        assert np.all(np.asarray(row["cand_pose_torso_yaw"]) >= 0) and np.all(np.asarray(row["cand_pose_torso_yaw"]) <= 90)
        if row["scenario"] == "wheelchair":
            assert np.all(np.asarray(row["cand_pose_hip_flexion"]) == 90)
    # 후보 설정이 다르면 섞지 않는다
    with pytest.raises(ValueError, match="candidate_k"):
        dataset.build_dataset(out, n_bodies=3, scenarios=["default"],
                              cfg=dataset.DatasetConfig(evaluator="dummy", max_evals=60, popsize=10,
                                                        dataset_id="t"), log=lambda *_: None)
    # 학습 배열로 펼치면 후보 수만큼 점이 된다
    space = flow.PoseSpace.from_scenarios(list(load_scenarios().values()))
    arr = flow.training_arrays(df, space, load_scenarios())
    assert len(arr["x"]) == int(df["n_candidates"].sum())


def test_recommend_uses_flow_model(model, scenarios, tmp_path, monkeypatch):
    """E 의 recommend 가 C 의 predict_pose 를 찾아 쓰고, 산출물이 없으면 스텁으로 폴백한다 (캡슐 몸)."""
    from airis.realtime import recommend as rec

    # 두 산출물(flow·kNN)이 모두 없어야 스텁으로 폴백한다.
    monkeypatch.setattr(pred, "DEFAULT_MODEL_PATH", tmp_path / "missing.pt")
    monkeypatch.setattr(pred, "DEFAULT_KNN_PATH", tmp_path / "missing.parquet")
    stub = rec.recommend(BodyParams(), scenarios["default"], model="capsule", rank_by_score=False)
    assert stub.source.startswith("stub")

    monkeypatch.setattr(pred, "DEFAULT_MODEL_PATH", model.save(tmp_path / "pose_flow.pt"))
    r = rec.recommend(BodyParams(), scenarios["wheelchair"], model="capsule")
    assert r.source.startswith("model")          # "model" 또는 "model: hybrid (flow|knn|extra)" (E #95)
    assert r.pose.hip_flexion == 90.0 and r.pose.knee_flexion == 90.0


def test_predict_rescore_options_default_unchanged_and_dedup(model, scenarios, tmp_path):
    """단축 옵션 기본값은 지금 동작 그대로. 중복 제거는 후보를 줄이고 남은 후보만 채점한다."""
    path = model.save(tmp_path / "m.pt")
    sc = scenarios["default"]
    ev = DummyEvaluator(PoseParams(shoulder_abduction=DOWN, torso_yaw=72.0), sc)
    base = pred.predict(BodyParams(), sc, backend="flow", n_samples=32, path=path, evaluator=ev, nozzle=object())
    assert base.n_dropped == 0 and base.screen_scores is None and base.n_rescored == 32

    d = pred.predict(BodyParams(), sc, backend="flow", n_samples=32, path=path, evaluator=ev, nozzle=object(),
                     dedup_deg=5.0)
    assert d.n_dropped > 0 and len(d.candidates) == 32 - d.n_dropped == d.n_rescored
    assert d.scores.shape == (len(d.candidates),) and d.pose is d.candidates[int(np.argmax(d.scores))]
    assert all(c in base.candidates for c in d.candidates), "남은 후보는 원래 후보의 부분 집합 (순서 유지)"


def test_predict_threads_pick_same_as_sequential(model, scenarios, tmp_path):
    path = model.save(tmp_path / "m.pt")
    sc = scenarios["default"]
    ev = DummyEvaluator(PoseParams(shoulder_abduction=UP, torso_yaw=72.0), sc)
    kw = dict(backend="flow", n_samples=24, path=path, evaluator=ev, nozzle=object(), dedup_deg=3.0)
    a = pred.predict(BodyParams(), sc, **kw)
    d = pred.predict(BodyParams(), sc, n_threads=3, **kw)
    assert d.pose == a.pose and np.array_equal(d.scores, a.scores) and d.sources == a.sources


def test_predict_screening_rescores_only_top(model, scenarios, tmp_path):
    """선별: 모든 후보를 선별 평가기로 채점하고 상위 k 개만 최종 평가기로 다시 채점한다. 동률은 앞 후보부터."""
    from airis.sim import EvalResult, Evaluator

    path = model.save(tmp_path / "m.pt")
    sc = scenarios["default"]
    final_ev = DummyEvaluator(PoseParams(shoulder_abduction=DOWN, torso_yaw=72.0), sc)
    calls = []

    class _Screen(Evaluator):                    # 만세 쪽을 높게 매기는 (최종과 다른) 선별 평가기
        def evaluate(self, pose, nozzle, body, scenario):
            calls.append(pose)
            return EvalResult(score=pose.shoulder_abduction / 180.0, removal_by_part=np.zeros(5),
                              total_removal=0.0, discomfort=0.0, extra={})

    p = pred.predict(BodyParams(), sc, backend="flow", n_samples=16, path=path, evaluator=final_ev, nozzle=object(),
                     screen_evaluator=_Screen(), screen_top=3)
    assert len(calls) == 16 and p.screen_scores.shape == (16,) and p.n_rescored == 3
    top = np.argsort(-p.screen_scores, kind="stable")[:3]
    assert np.isfinite(p.scores[top]).all() and np.isnan(np.delete(p.scores, top)).all()
    assert p.pose is p.candidates[top[np.argmax(p.scores[top])]], "최종 점수로 상위 3개 중 최고"
    assert min(p.candidates[i].shoulder_abduction for i in top) >= np.median(
        [c.shoulder_abduction for c in p.candidates]), "선별 평가기 기준 상위만 남는다"

    class _Flat(_Screen):                        # 선별 점수 동률 → 앞 후보부터
        def evaluate(self, pose, nozzle, body, scenario):
            return EvalResult(score=0.5, removal_by_part=np.zeros(5), total_removal=0.0, discomfort=0.0, extra={})

    flat = pred.predict(BodyParams(), sc, backend="flow", n_samples=8, path=path, evaluator=final_ev,
                        nozzle=object(), screen_evaluator=_Flat(), screen_top=2)
    assert np.isfinite(flat.scores[:2]).all() and np.isnan(flat.scores[2:]).all()
    threaded = pred.predict(BodyParams(), sc, backend="flow", n_samples=16, path=path, evaluator=final_ev,
                            nozzle=object(), screen_evaluator=_Screen(), screen_top=3, n_threads=3)
    assert threaded.pose == p.pose and np.array_equal(threaded.screen_scores, p.screen_scores), \
        "선별 + 스레드(hybrid-e)는 선별(hybrid-b)과 같은 자세를 고른다"
    assert np.array_equal(np.isnan(threaded.scores), np.isnan(p.scores))
    small = pred.predict(BodyParams(), sc, backend="flow", n_samples=2, path=path, evaluator=final_ev,
                         nozzle=object(), screen_evaluator=_Flat(), screen_top=3)
    assert small.screen_scores is None and small.n_rescored == 2, "후보가 상위 개수 이하면 선별하지 않는다"


def test_predict_screening_keep_extra_and_infeasible_fallback(model, scenarios, tmp_path):
    """선별 + 표 유지: 고정 후보는 선별에서 떨어져도 최종 채점에 들어간다 (표보다 나빠지지 않는다).
    선별 상위가 최종 밀도에서 전부 불가면 나머지 후보를 채점해 가능한 후보를 고른다."""
    from airis.sim import EvalResult, Evaluator

    path = model.save(tmp_path / "m.pt")
    sc = scenarios["default"]

    class _Screen(Evaluator):                    # 만세 쪽을 높게 매긴다 (표 후보 90° 는 선별에서 떨어진다)
        def evaluate(self, pose, nozzle, body, scenario):
            return EvalResult(score=pose.shoulder_abduction / 180.0, removal_by_part=np.zeros(5),
                              total_removal=0.0, discomfort=0.0, extra={})

    extra = PoseParams(shoulder_abduction=90.0, torso_yaw=20.0)
    final_ev = DummyEvaluator(extra, sc)         # 최종 평가기는 표 후보를 가장 높게 매긴다
    common = dict(backend="flow", n_samples=12, path=path, evaluator=final_ev, nozzle=object(),
                  extra_candidates=[extra], screen_evaluator=_Screen(), screen_top=3)
    lost = pred.predict(BodyParams(), sc, **common)
    kept = pred.predict(BodyParams(), sc, screen_keep_extra=True, **common)
    j = kept.sources.index("extra")
    assert np.isnan(lost.scores[j]) and lost.source != "extra", "표 후보가 선별에서 떨어지면 표보다 나빠질 수 있다"
    assert np.isfinite(kept.scores[j]) and kept.source == "extra" and kept.n_rescored == 4

    class _Ceiling(Evaluator):                   # 팔 120° 이상은 최종 밀도에서 불가 (천장)
        def evaluate(self, pose, nozzle, body, scenario):
            bad = pose.shoulder_abduction >= 120.0
            return EvalResult(score=-2.0 if bad else 1.0 - pose.shoulder_abduction / 180.0,
                              removal_by_part=np.zeros(5), total_removal=0.0, discomfort=0.0,
                              extra={"infeasible": bad})

    p = pred.predict(BodyParams(), sc, backend="flow", n_samples=16, path=path, evaluator=_Ceiling(),
                     nozzle=object(), screen_evaluator=_Screen(), screen_top=3)
    top = np.argsort(-p.screen_scores, kind="stable")[:3]
    assert all(p.candidates[t].shoulder_abduction >= 120.0 for t in top), "선별 상위는 전부 만세(불가)"
    assert p.pose.shoulder_abduction < 120.0 and not p.infeasible[p.candidates.index(p.pose)], \
        "나머지 후보로 되돌아가 가능한 자세를 고른다"
    assert p.n_rescored > 3


def test_extra_candidates_are_rescored_with_samples(model, scenarios, tmp_path):
    """고정 후보(E 의 후보표 등)를 함께 재채점하면 결과가 그 후보보다 나빠지지 않는다."""
    path = model.save(tmp_path / "m.pt")
    sc = scenarios["default"]
    target = PoseParams(shoulder_abduction=95.0, torso_yaw=20.0)      # 모델이 거의 내지 않는 자세
    ev = DummyEvaluator(target, sc)
    p = pred.predict(BodyParams(), sc, backend="flow", n_samples=8, path=path, evaluator=ev, nozzle=object(),
                     extra_candidates=[target])
    assert len(p.candidates) == 9 and p.scores.shape == (9,)
    assert p.pose == p.candidates[-1], "평가기가 가장 높게 매기는 고정 후보가 뽑혀야 한다"
    alone = pred.predict(BodyParams(), sc, backend="flow", n_samples=8, path=path, evaluator=ev, nozzle=object())
    assert p.scores.max() >= alone.scores.max()


def test_artifact_without_space_kind_loads_as_pose(model, tmp_path):
    """space_kind 를 넣기 전 산출물(#79 첫 판)도 자세 모델로 읽힌다."""
    path = model.save(tmp_path / "m.pt")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    ck.pop("space_kind"), ck.pop("space_extra")
    torch.save(ck, tmp_path / "old.pt")
    again = flow.PoseFlow.load(tmp_path / "old.pt")
    assert isinstance(again.space, flow.PoseSpace) and again.out_dim == 7
    assert np.allclose(again.sample(BodyParams(), "default", 4, seed=1),
                       model.sample(BodyParams(), "default", 4, seed=1))


# ---------- 계획 모델 (docs/plan_extension.md) ----------

from airis.sim import ZONE_NAMES, EvalResult, Evaluator, Phase, Plan      # noqa: E402

ZONES_A = np.array([1.0, 1.0, 0.3, 0.3, 0.9])       # 봉우리 A: 가슴 쪽 벽 강하게 + 만세
ZONES_B = np.array([0.4, 0.4, 1.0, 1.0, 0.1])       # 봉우리 B: 등 쪽 벽 강하게 + 팔 내림


@pytest.fixture(scope="module")
def plan_space(scenarios):
    return flow.PlanSpace.from_scenarios([scenarios[s] for s in ("default", "wheelchair")])


def _plan(abd, zones, t1=6.0, t2=9.0, yaw1=72.0, yaw2=-70.0, hip=0.0, knee=5.0):
    def pose(yaw):
        return PoseParams(shoulder_abduction=abd, shoulder_flexion=-4.0, elbow_flexion=2.0, torso_pitch=0.0,
                          torso_yaw=yaw, hip_flexion=hip, knee_flexion=knee)
    return Plan([Phase(pose(yaw1), t1), Phase(pose(yaw2), t2)], np.asarray(zones, dtype=np.float64))


def synthetic_plan_df(space, n=160, seed=0):
    """default: 체형과 무관하게 두 봉우리 반반 (자세와 구역 세기가 함께 바뀐다) / wheelchair: 봉우리 A, 하체 고정."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, b in enumerate(dataset.sample_bodies(n, seed=seed)):
        for scenario in ("default", "wheelchair"):
            a = scenario == "wheelchair" or rng.random() < 0.5
            seat = dict(hip=90.0, knee=90.0, yaw1=36.0, yaw2=-36.0) if scenario == "wheelchair" else {}
            plan = _plan(UP if a else DOWN, ZONES_A if a else ZONES_B,
                         t1=6.0 + rng.normal(0, 0.3), t2=9.0 + rng.normal(0, 0.3), **seat)
            r = {"body_idx": i, "scenario": scenario, "score": 0.5}
            r.update({f"body_{k}": getattr(b, k) for k in flow.BODY_KEYS})
            r.update({f"plan_{k}": v for k, v in zip(space.keys, space.from_plan(plan))})
            rows.append(r)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def plan_model(scenarios, plan_space):
    return flow.train_flow(synthetic_plan_df(plan_space), scenarios, SMALL_CFG, space=plan_space,
                           meta={"body_model": "capsule", "patches_per_m2": 400.0})


def test_plan_space_keys_mask_and_roundtrip(scenarios, plan_space):
    assert plan_space.dim == 2 * 7 + 1 + 1 + len(ZONE_NAMES) == 21
    assert plan_space.keys[:7] == [f"p1_{k}" for k in flow.POSE_KEYS]
    assert plan_space.keys[-len(ZONE_NAMES):] == [f"zone_{z}" for z in ZONE_NAMES]
    b = plan_space.to_dict()
    assert b["p1_torso_yaw"] == [0.0, 90.0], "1단계 yaw 만 0~90° 로 접는다"
    assert b["p2_torso_yaw"] == [-180.0, 180.0], "나머지 단계는 시나리오 범위 그대로"
    assert b["duration_s"] == [5.0, 20.0] and b["zone_top"] == [0.0, 1.0]
    m = plan_space.mask(scenarios["wheelchair"])
    fixed = [k for k, v in zip(plan_space.keys, m) if v == 0]
    assert fixed == ["p1_hip_flexion", "p1_knee_flexion", "p2_hip_flexion", "p2_knee_flexion"]

    plan = _plan(UP, ZONES_A)
    back = plan_space.to_plan(plan_space.encode(plan_space.from_plan(plan))[0], scenarios["default"])
    assert [ph.duration_s for ph in back.phases] == pytest.approx([6.0, 9.0])
    for got, want in zip(back.phases, plan.phases):
        assert np.allclose(got.pose.to_vector(), want.pose.to_vector(), atol=1e-4)
    assert np.allclose(back.zone_strengths, ZONES_A)
    assert back.duration_s == pytest.approx(15.0)


def test_plan_durations_always_feasible(scenarios, plan_space):
    rng = np.random.default_rng(0)
    for x in rng.uniform(-1.5, 1.5, size=(200, plan_space.dim)):
        plan = plan_space.to_plan(x, scenarios["wheelchair"])
        assert all(ph.duration_s >= plan_space.min_phase_s - 1e-9 for ph in plan.phases)
        assert 5.0 - 1e-9 <= plan.duration_s <= 20.0 + 1e-9
        assert np.all((plan.zone_strengths >= 0) & (plan.zone_strengths <= 1))
        assert all(ph.pose.hip_flexion == 90.0 and ph.pose.knee_flexion == 90.0 for ph in plan.phases)
    with pytest.raises(ValueError, match="총 시간 하한"):
        flow.PlanSpace.from_scenarios([scenarios["default"]], n_phases=3, min_phase_s=2.0,
                                      duration_bounds_s=(5.0, 20.0))


def test_plan_flow_keeps_joint_modes(plan_model, scenarios):
    """같은 코드가 21차원 계획을 학습한다. 자세 봉우리와 구역 세기가 짝을 이룬 채로 나와야 한다."""
    assert plan_model.out_dim == 21
    plans = plan_model.sample_plans(BodyParams(), scenarios["default"], 300, seed=0)
    abd = np.array([p.phases[0].pose.shoulder_abduction for p in plans])
    chest = np.array([p.zone_strengths[0] for p in plans])
    up, down = abd >= 150, abd <= 45
    assert up.mean() > 0.25 and down.mean() > 0.25, (up.mean(), down.mean())
    assert np.mean(~up & ~down) < 0.15, "두 봉우리 사이 자세는 거의 없어야 한다"
    assert np.median(chest[up]) > 0.8 and np.median(chest[down]) < 0.6, "자세와 세기가 같은 봉우리에서 나온다"
    wc = plan_model.sample_plans(BodyParams(), scenarios["wheelchair"], 64, seed=0)
    assert np.mean([p.phases[0].pose.shoulder_abduction >= 150 for p in wc]) > 0.8
    with pytest.raises(TypeError):
        plan_model.sample_poses(BodyParams(), scenarios["default"], 4)


class _PlanEvaluator(Evaluator):
    """봉우리 B(등 쪽 벽 강하게 + 팔 내림)를 선호하는 가짜 계획 평가기."""

    def evaluate(self, pose, nozzle, body, scenario):
        raise NotImplementedError

    def evaluate_plan(self, plan, nozzle, body, scenario):
        s = -float(np.abs(plan.zone_strengths - ZONES_B).mean()) \
            - abs(plan.phases[0].pose.shoulder_abduction - DOWN) / 180.0
        return EvalResult(score=s, removal_by_part=np.zeros(5), total_removal=0.0, discomfort=0.0,
                          extra={"energy": 0.0, "duration_s": plan.duration_s})


def test_predict_plan_contract_and_rescoring(plan_model, model, scenarios, tmp_path):
    path = plan_model.save(tmp_path / "plan.pt")
    sc = scenarios["default"]
    again = flow.PoseFlow.load(path)
    assert isinstance(again.space, flow.PlanSpace) and again.space.n_phases == 2
    assert np.allclose(again.sample(BodyParams(), "default", 8, seed=3),
                       plan_model.sample(BodyParams(), "default", 8, seed=3))

    with pytest.raises(FileNotFoundError):
        pred.predict_plan(BodyParams(), sc, path=tmp_path / "none.pt")
    with pytest.raises(ValueError, match="pose 모델"):
        pred.predict_plan(BodyParams(), sc, path=model.save(tmp_path / "pose.pt"))
    with pytest.raises(ValueError, match="plan 모델"):
        pred.predict_pose(BodyParams(), sc, path=path)

    p = pred.predict_plan_candidates(BodyParams(), sc, n_samples=32, path=path,
                                     evaluator=_PlanEvaluator(), nozzle=object())
    assert p.plan.phases[0].pose.shoulder_abduction <= 45 and p.plan.zone_strengths[2] > 0.8
    assert len(p.candidates) == 32 and p.scores.shape == (32,)
    assert isinstance(pred.predict_plan(BodyParams(), sc, path=path, evaluator=_PlanEvaluator(),
                                        nozzle=object()), Plan)

    # 고정 계획을 함께 재채점: 평가기의 이상형을 넣으면 그것이 뽑힌다
    ideal = _plan(DOWN, ZONES_B)
    q = pred.predict_plan_candidates(BodyParams(), sc, n_samples=8, path=path, evaluator=_PlanEvaluator(),
                                     nozzle=object(), extra_candidates=[ideal])
    assert np.allclose(plan_model.space.from_plan(q.plan), plan_model.space.from_plan(ideal)), \
        "범위 안의 고정 계획은 투영해도 그대로이고, 그것이 뽑힌다"

    # evaluate_plan 을 구현하지 않은 평가기는 NotImplementedError 를 그대로 올린다
    # (DummyEvaluator 는 계획용 더미를 갖고 있으므로 자세만 구현한 평가기로 확인한다)
    from dataclasses import replace

    from airis.sim import Evaluator

    class _PoseOnly(Evaluator):
        def evaluate(self, pose, nozzle, body, scenario):
            return DummyEvaluator(PoseParams(), sc).evaluate(pose, nozzle, body, scenario)

    with pytest.raises(NotImplementedError):
        pred.predict_plan(BodyParams(), sc, path=path, evaluator=_PoseOnly(), nozzle=object())
    with pytest.raises(KeyError):
        pred.predict_plan(BodyParams(), replace(sc, name="stroller"), path=path,
                          evaluator=_PlanEvaluator(), nozzle=object())


def test_plan_rescorer_uses_plan_physics_cfg(plan_model, scenarios, tmp_path, monkeypatch):
    """계획 재채점은 C 의 plan_physics_cfg (kinetics 켬), 자세 재채점은 설정 파일 그대로 (docs/plan_extension.md 4절)."""
    from airis.sim.scenario import load_physics

    meta = {"body_model": "capsule", "patches_per_m2": 400.0}
    plan_ev, _ = pred.default_rescorer(meta, plan=True)
    pose_ev, _ = pred.default_rescorer(meta)
    assert plan_ev.cfg["adhesion"]["kinetics"]["enabled"] is True
    assert pose_ev.cfg["adhesion"]["kinetics"]["enabled"] == load_physics()["adhesion"]["kinetics"]["enabled"]
    assert plan_ev is not pose_ev

    seen = {}

    def fake_rescorer(model, *, plan=False):
        seen["plan"] = plan
        return _PlanEvaluator(), object()

    monkeypatch.setattr(pred, "default_rescorer", fake_rescorer)
    pred.predict_plan_candidates(BodyParams(), scenarios["default"], n_samples=4,
                                 path=plan_model.save(tmp_path / "plan.pt"))
    assert seen == {"plan": True}, "predict_plan 의 기본 재채점기는 계획 설정"


def test_stamp_mismatch_compares_kinetics_only_when_stamped():
    now = {"nozzle_layout_hash": "1e500000", "physics_hash": "p0", "kinetics_enabled": True, "time_constant_s": 2.0}
    assert pred.stamp_mismatch({"nozzle_layout_hash": "1e500000", "physics_hash": "p0"}, now) == [], \
        "자세 산출물(kinetics 도장 없음)은 kinetics 를 비교하지 않는다"
    same = {"nozzle_layout_hash": "1e500000", "physics_hash": "p0",
            "kinetics_enabled": np.bool_(True), "time_constant_s": 2}
    assert pred.stamp_mismatch(same, now) == []
    assert pred.stamp_mismatch({**same, "kinetics_enabled": "True", "time_constant_s": "2.0"}, now) == []
    assert pred.stamp_mismatch({**same, "kinetics_enabled": False}, now) == ["kinetics_enabled"]
    assert pred.stamp_mismatch({**same, "time_constant_s": 5.0}, now) == ["time_constant_s"]
    assert pred.stamp_mismatch({**same, "nozzle_layout_hash": "2e500000"}, now) == ["nozzle_layout_hash"], \
        "해시는 문자열로 비교한다 (16진 해시를 숫자로 읽으면 둘 다 inf 가 된다)"


def test_current_stamp_carries_plan_kinetics():
    from airis.sim.scenario import load_physics

    now = pred.current_stamp()
    assert now["kinetics_enabled"] is True
    assert now["time_constant_s"] == pytest.approx(load_physics()["adhesion"]["kinetics"]["time_constant_s"])


def test_predict_plan_clips_before_rescoring(plan_model, scenarios, tmp_path):
    """계획 후보(flow 샘플 + 고정 계획)는 채점 전에 전부 PlanEncoder.clip_plan 을 거친다."""
    from dataclasses import replace

    from airis.optimize.plan_encoding import PlanLimits

    path = plan_model.save(tmp_path / "plan.pt")
    # 구역 이름 키만 쓴 쾌적 상한 (옛 부위 키 torso_front 없이도 걸려야 한다, #92).
    # plan_model 은 default·wheelchair 로만 학습했으므로 이름은 default 로 둔다.
    capped = replace(scenarios["default"], nozzle_strength_cap={"chest_low": 0.6, "chest_high": 0.6})
    chest = [ZONE_NAMES.index(z) for z in ("chest_low", "chest_high")]
    raw = plan_model.sample_plans(BodyParams(), capped, 16, seed=0)
    assert any(np.any(p.zone_strengths[chest] > 0.6) for p in raw), "투영 전에는 쾌적 상한을 넘는 샘플이 있다"

    seen: list[Plan] = []

    class _Recorder(_PlanEvaluator):
        def evaluate_plan(self, plan, nozzle, body, scenario):
            seen.append(plan)
            return super().evaluate_plan(plan, nozzle, body, scenario)

    wild = _plan(UP, np.ones(len(ZONE_NAMES)), t1=30.0, t2=30.0)          # 총 60 s, 가슴 1.0
    pred.predict_plan_candidates(BodyParams(), capped, n_samples=16, path=path, evaluator=_Recorder(),
                                 nozzle=object(), extra_candidates=[wild])
    assert len(seen) == 17
    for plan in seen:
        assert np.all(plan.zone_strengths[chest] <= 0.6 + 1e-12)
        assert 5.0 - 1e-9 <= plan.duration_s <= 20.0 + 1e-9
        assert min(ph.duration_s for ph in plan.phases) >= 2.0 - 1e-9
    assert seen[-1].duration_s == pytest.approx(20.0), "범위 밖 고정 계획도 투영한다"

    only = pred.predict_plan_candidates(BodyParams(), capped, n_samples=4, rescore=False, path=path)
    assert np.all(only.plan.zone_strengths[chest] <= 0.6 + 1e-12), "재채점을 꺼도 투영한다"

    # 풍량 한도 보수: Σ_구역 노즐 수 × 세기 ≤ cap_ratio × 노즐 수. 노즐 수는 실제 장비 구성이 기본이다.
    from airis.sim.scenario import zone_nozzle_counts

    counts = zone_nozzle_counts()
    tight = pred.predict_plan_candidates(BodyParams(), scenarios["default"], n_samples=8, path=path,
                                         evaluator=_PlanEvaluator(), nozzle=object(),
                                         limits=PlanLimits(cap_ratio=0.5))
    assert all(counts @ c.zone_strengths <= 0.5 * counts.sum() + 1e-9 for c in tight.candidates)
    assert any(counts @ p.zone_strengths > 0.5 * counts.sum() for p in raw), "투영 전에는 한도를 넘는 샘플이 있다"
    even = pred.predict_plan_candidates(BodyParams(), scenarios["default"], n_samples=8, path=path,
                                        evaluator=_PlanEvaluator(), nozzle=object(),
                                        limits=PlanLimits(cap_ratio=0.5), zone_nozzle_counts=np.ones(len(ZONE_NAMES)))
    assert all(c.zone_strengths.sum() <= 0.5 * len(ZONE_NAMES) + 1e-9 for c in even.candidates)


# ---------- 혼합 추천 (flow + kNN + 고정 후보), 총괄 2026-09-30 ----------

def _knn_table(tmp_path, df=None, exclude=()):
    """합성 데이터로 kNN 표 산출물을 만든다."""
    from airis.model.knn import PoseKNN

    d = synthetic_df() if df is None else df
    if exclude:
        d = d[~d["body_idx"].isin(exclude)]
    d = d.assign(nozzle_layout_hash="n0", physics_hash="p0", body_model="capsule",
                 patches_per_m2=400.0, commit="c0")
    return PoseKNN.from_dataset(d).save(tmp_path / "pose_knn.parquet")


def test_artifact_status_reports_stamps_without_warning(model, tmp_path, monkeypatch):
    """E 대시보드용 산출물 상태: 경로·존재·도장·지금 설정과 일치 여부. _check_stamp 와 같은 기준."""
    import warnings

    monkeypatch.setattr(pred, "current_stamp", lambda: {"nozzle_layout_hash": "n0", "physics_hash": "p1"})
    knn = _knn_table(tmp_path)                                  # 도장 n0 · p0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        st = pred.artifact_status(model_path=tmp_path / "none.pt", knn_path=knn)
    assert st["current"] == {"nozzle_layout_hash": "n0", "physics_hash": "p1"} and st["current_error"] is None
    assert st["flow"]["exists"] is False and st["flow"]["match"] is None
    k = st["knn"]
    assert k["exists"] and k["error"] is None and k["stamp"]["commit"] == "c0"
    assert k["match"] is False and k["mismatched"] == ["physics_hash"]

    path = model.save(tmp_path / "m.pt")                        # 도장이 없는 산출물은 확인 불가 (일치로 보지 않는다)
    flow_st = pred.artifact_status(model_path=path, knn_path=knn)["flow"]
    assert flow_st["exists"] and flow_st["error"] is None
    assert flow_st["match"] is None and flow_st["mismatched"] == []
    assert flow_st["stamp"]["physics_hash"] is None

    def stamped(name, **stamp):
        m = flow.PoseFlow.load(model.save(tmp_path / name))
        m.meta.update(stamp)
        return pred.artifact_status(model_path=m.save(tmp_path / name), knn_path=knn)["flow"]

    assert stamped("m2.pt", nozzle_layout_hash="n0", physics_hash="p1")["match"] is True   # 전부 같을 때만 True
    part = stamped("m3.pt", nozzle_layout_hash="n0")            # 일부만 찍혔고 그 키는 같다 → 확인 불가
    assert part["match"] is None and part["mismatched"] == []
    part_bad = stamped("m4.pt", nozzle_layout_hash="n9")        # 일부만 찍혔어도 다른 키가 있으면 False
    assert part_bad["match"] is False and part_bad["mismatched"] == ["nozzle_layout_hash"]

    def broken():
        raise OSError("설정 없음")
    monkeypatch.setattr(pred, "current_stamp", broken)
    st = pred.artifact_status(model_path=path, knn_path=knn)
    assert st["current"] is None and "설정 없음" in st["current_error"] and st["knn"]["match"] is None


def test_knn_candidates_are_nearest_and_deterministic(tmp_path, scenarios):
    from airis.model.knn import PoseKNN

    table = PoseKNN.load(_knn_table(tmp_path))
    sc = scenarios["wheelchair"]
    bodies = dataset.sample_bodies(300, seed=0)
    me = bodies[7]
    cands = table.candidates(me, sc, 3)
    assert len(cands) == 3
    # 학습 체형 자신을 물으면 그 체형의 자세가 첫 후보다.
    assert cands[0].shoulder_abduction == pytest.approx(UP, abs=1e-6)
    assert cands[0].hip_flexion == 90 and cands[0].knee_flexion == 90, "fixed_pose 적용"
    assert [p.torso_yaw for p in cands] == [p.torso_yaw for p in table.candidates(me, sc, 3)]
    with pytest.raises(KeyError):
        table.candidates(me, __import__("dataclasses").replace(sc, name="stroller"), 2)


def test_hybrid_uses_all_three_sources(tmp_path, model, scenarios):
    path = model.save(tmp_path / "m.pt")
    knn_path = _knn_table(tmp_path)
    sc = scenarios["default"]
    stub = [PoseParams(shoulder_abduction=DOWN, torso_yaw=72.0, knee_flexion=5.0, elbow_flexion=2.0,
                       shoulder_flexion=-4.0)]
    ev = DummyEvaluator(stub[0], sc)
    p = pred.predict(BodyParams(), sc, n_flow=3, n_knn=4, path=path, knn_path=knn_path,
                     evaluator=ev, nozzle=object(), extra_candidates=stub)
    assert p.sources == ["flow"] * 3 + ["knn"] * 4 + ["extra"]
    assert len(p.candidates) == 8 and p.scores.shape == (8,)
    assert p.pose == p.candidates[int(np.argmax(p.scores))]
    assert p.source == p.sources[int(np.argmax(p.scores))]
    # 평가기가 고정 후보 자세를 목표로 삼으므로 그 후보가 뽑힌다 (extra 가 결과를 나쁘게 하지 않는다).
    assert p.source == "extra"

    same = pred.predict(BodyParams(), sc, n_flow=3, n_knn=4, path=path, knn_path=knn_path,
                        evaluator=ev, nozzle=object(), extra_candidates=stub)
    assert [c.shoulder_abduction for c in same.candidates] == [c.shoulder_abduction for c in p.candidates]


def test_hybrid_falls_back_to_one_source(tmp_path, model, scenarios):
    path = model.save(tmp_path / "m.pt")
    knn_path = _knn_table(tmp_path)
    sc = scenarios["default"]
    ev = DummyEvaluator(PoseParams(shoulder_abduction=UP, torso_yaw=72.0), sc)

    # flow 산출물만 없으면 kNN + extra 로 돈다 (경고 1회).
    with pytest.warns(RuntimeWarning, match="일부만"):
        only_knn = pred.predict(BodyParams(), sc, n_flow=3, n_knn=2, path=tmp_path / "none.pt",
                                knn_path=knn_path, evaluator=ev, nozzle=object())
    assert set(only_knn.sources) == {"knn"} and len(only_knn.candidates) == 2

    # kNN 표만 없으면 flow + extra 로 돈다.
    with pytest.warns(RuntimeWarning, match="일부만"):
        only_flow = pred.predict(BodyParams(), sc, n_flow=3, n_knn=2, path=path,
                                 knn_path=tmp_path / "none.parquet", evaluator=ev, nozzle=object())
    assert set(only_flow.sources) == {"flow"} and len(only_flow.candidates) == 3

    # 둘 다 없으면 FileNotFoundError (E 가 스텁으로 폴백한다).
    with pytest.raises(FileNotFoundError):
        pred.predict(BodyParams(), sc, path=tmp_path / "none.pt", knn_path=tmp_path / "none.parquet",
                     evaluator=ev, nozzle=object())


def test_hybrid_avoids_infeasible_with_extra_candidate(tmp_path, model, scenarios):
    """이웃(만세)이 전부 불가인 큰 체형에서도 고정 후보 덕분에 가능한 자세를 고른다."""
    sc = scenarios["default"]
    knn_path = _knn_table(tmp_path)          # 합성 데이터의 wheelchair·default 최적은 만세 포함
    safe = PoseParams(shoulder_abduction=DOWN, torso_yaw=72.0, knee_flexion=5.0, elbow_flexion=2.0,
                      shoulder_flexion=-4.0)

    class CeilingEvaluator(DummyEvaluator):
        """벌림 90° 이상은 천장에 걸려 불가."""

        def evaluate(self, pose, nozzle, body, scenario):
            r = super().evaluate(pose, nozzle, body, scenario)
            if pose.shoulder_abduction >= 90:
                r.score = -1.0 - 0.1 * (pose.shoulder_abduction - 90) / 90
                r.extra["infeasible"] = True
            return r

    ev = CeilingEvaluator(safe, sc)
    p = pred.predict(BodyParams(height_m=1.93), sc, backend="knn", n_samples=8, knn_path=knn_path,
                     evaluator=ev, nozzle=object(), extra_candidates=[safe])
    assert p.source == "extra"
    assert not p.infeasible[int(np.argmax(np.where(p.infeasible, -np.inf, p.scores)))]
    assert p.pose.shoulder_abduction < 90


def test_hybrid_rejects_bad_backend(tmp_path, model, scenarios):
    path = model.save(tmp_path / "m.pt")
    with pytest.raises(ValueError, match="backend"):
        pred.predict(BodyParams(), scenarios["default"], backend="magic", path=path)
    with pytest.raises(ValueError, match="n_samples"):
        pred.predict(BodyParams(), scenarios["default"], backend="hybrid", n_samples=4, path=path)


def test_build_pose_knn_script(tmp_path):
    from airis.model.knn import PoseKNN
    from scripts.build_pose_knn import main

    df = synthetic_df(n=20).assign(nozzle_layout_hash="n0", physics_hash="p0", body_model="capsule",
                                   patches_per_m2=400.0, commit="c0")
    src = tmp_path / "ds.parquet"
    df.to_parquet(src, index=False)
    out = tmp_path / "knn.parquet"
    assert main(["--dataset", str(src), "--out", str(out), "--exclude-bodies", "0", "1"]) == 0
    table = PoseKNN.load(out)
    assert set(table.table["body_idx"]) == set(range(2, 20))
    assert table.meta["physics_hash"] == "p0"
    assert sorted(table.scenario_names) == ["default", "pregnant", "wheelchair"]
