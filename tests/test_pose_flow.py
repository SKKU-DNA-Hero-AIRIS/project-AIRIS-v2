"""조건부 flow matching 자세 모델 테스트. 소유자: C. docs/proposals/flow_matching.md.

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
    with pytest.raises(FileNotFoundError):
        pred.predict_pose(BodyParams(), sc, path=tmp_path / "none.pt")

    # 재채점: 평가기가 팔 내림 봉우리를 선호하면 반반 분포에서 팔 내림을 고른다.
    target = PoseParams(shoulder_abduction=DOWN, torso_yaw=72.0, knee_flexion=5.0, elbow_flexion=2.0,
                        shoulder_flexion=-4.0)
    ev = DummyEvaluator(target, sc)
    p = pred.predict(BodyParams(), sc, n_samples=32, path=path, evaluator=ev, nozzle=object())
    assert p.pose.shoulder_abduction <= 45
    assert len(p.candidates) == 32 and p.scores.shape == (32,)
    assert p.pose == p.candidates[int(np.argmax(p.scores))]

    only = pred.predict(BodyParams(), sc, n_samples=8, rescore=False, path=path)
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

    monkeypatch.setattr(pred, "DEFAULT_MODEL_PATH", tmp_path / "missing.pt")
    stub = rec.recommend(BodyParams(), scenarios["default"], model="capsule", rank_by_score=False)
    assert stub.source.startswith("stub")

    monkeypatch.setattr(pred, "DEFAULT_MODEL_PATH", model.save(tmp_path / "pose_flow.pt"))
    r = rec.recommend(BodyParams(), scenarios["wheelchair"], model="capsule")
    assert r.source == "model"
    assert r.pose.hip_flexion == 90.0 and r.pose.knee_flexion == 90.0


def test_extra_candidates_are_rescored_with_samples(model, scenarios, tmp_path):
    """고정 후보(E 의 후보표 등)를 함께 재채점하면 결과가 그 후보보다 나빠지지 않는다."""
    path = model.save(tmp_path / "m.pt")
    sc = scenarios["default"]
    target = PoseParams(shoulder_abduction=95.0, torso_yaw=20.0)      # 모델이 거의 내지 않는 자세
    ev = DummyEvaluator(target, sc)
    p = pred.predict(BodyParams(), sc, n_samples=8, path=path, evaluator=ev, nozzle=object(),
                     extra_candidates=[target])
    assert len(p.candidates) == 9 and p.scores.shape == (9,)
    assert p.pose == p.candidates[-1], "평가기가 가장 높게 매기는 고정 후보가 뽑혀야 한다"
    alone = pred.predict(BodyParams(), sc, n_samples=8, path=path, evaluator=ev, nozzle=object())
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
    assert q.plan is ideal

    # evaluate_plan 을 구현하지 않은 평가기는 NotImplementedError 를 그대로 올린다
    from dataclasses import replace
    with pytest.raises(NotImplementedError):
        pred.predict_plan(BodyParams(), sc, path=path, evaluator=DummyEvaluator(PoseParams(), sc),
                          nozzle=object())
    with pytest.raises(KeyError):
        pred.predict_plan(BodyParams(), replace(sc, name="stroller"), path=path,
                          evaluator=_PlanEvaluator(), nozzle=object())
