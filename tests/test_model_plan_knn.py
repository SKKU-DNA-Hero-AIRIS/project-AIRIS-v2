"""계획 kNN 표와 계획 벡터 변환 테스트. 소유자: F. torch 없이 돈다.

단계 수 N 은 데이터셋 열에서 읽는다 (C 의 설계 결정값, 5 또는 9 예상). 그래서 N = 2·5·9 를 모두 본다.
"""
import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from airis.model import flow                                   # noqa: E402
from airis.model.knn import PLAN_LIMIT_COLS, PLAN_STAMP_COLS, PlanKNN   # noqa: E402
from airis.optimize import dataset                             # noqa: E402
from airis.optimize.plan_encoding import PlanEncoder, PlanLimits   # noqa: E402
from airis.sim import ZONE_NAMES, BodyParams, Phase, Plan, PoseParams   # noqa: E402
from airis.sim.scenario import load_scenarios                  # noqa: E402

MIN_PHASE = 2.0


def limits_for(n_phases: int) -> PlanLimits:
    """N 단계 한도. C 의 run_e7 처럼 총 시간 하한을 N × min_phase_s 로 올린다."""
    return PlanLimits(n_phases=n_phases, duration_bounds_s=(max(5.0, n_phases * MIN_PHASE), 20.0),
                      min_phase_s=MIN_PHASE)


def make_plan(n_phases: int, abd: float, zones, rng, scenario) -> Plan:
    """단계마다 yaw 를 돌리는 N 단계 계획 (정규화·투영을 거친, 데이터셋 행과 같은 형태)."""
    enc = PlanEncoder(scenario, limits_for(n_phases))
    total = 20.0
    w = rng.dirichlet(np.ones(n_phases) * 4)
    phases = [Phase(PoseParams(shoulder_abduction=abd, torso_yaw=30.0 + 15.0 * k, elbow_flexion=2.0), total * w[k])
              for k in range(n_phases)]
    return enc.clip_plan(enc.normalize(Plan(phases, np.asarray(zones, dtype=np.float64))))


def plan_df(n_phases: int, n_bodies: int = 40, seed: int = 0, scenarios=("default", "wheelchair"),
            stamps: bool = True):
    """합성 계획 데이터셋. 체형 번호가 짝수면 만세·가슴 쪽 강하게, 홀수면 팔 내림·등 쪽 강하게."""
    rng = np.random.default_rng(seed)
    scs = load_scenarios()
    keys = flow.PlanSpace.plan_keys(n_phases)
    lim = limits_for(n_phases)
    rows = []
    for i, b in enumerate(dataset.sample_bodies(n_bodies, seed=seed)):
        for name in scenarios:
            up = i % 2 == 0
            plan = make_plan(n_phases, 170.0 if up else 15.0,
                             [1.0, 1.0, 0.3, 0.3, 0.9] if up else [0.4, 0.4, 1.0, 1.0, 0.2], rng, scs[name])
            raw = PlanEncoder(scs[name], lim).raw_vector(plan)
            r = {"body_idx": i, "scenario": name, "score": 0.5 + 0.001 * i}
            r.update({f"body_{k}": getattr(b, k) for k in flow.BODY_KEYS})
            r.update({f"plan_{k}": float(v) for k, v in zip(keys, raw)})
            if stamps:
                r.update(nozzle_layout_hash="n0", physics_hash="p0", body_model="capsule", patches_per_m2=400.0,
                         commit="c0", kinetics_enabled=True, time_constant_s=2.0, zone_nozzle_counts="2;2;2;2;4",
                         n_phases=n_phases, energy_weight=0.03, duration_lo_s=lim.duration_bounds_s[0],
                         duration_hi_s=lim.duration_bounds_s[1], min_phase_s=lim.min_phase_s,
                         transition_s=lim.transition_s, s_max=lim.s_max, cap_ratio=lim.cap_ratio)
            rows.append(r)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios()


# ---------- 계획 벡터 변환 ----------

@pytest.mark.parametrize("n_phases", [1, 2, 5, 9])
def test_phase_count_and_keys_come_from_columns(n_phases):
    df = plan_df(n_phases, n_bodies=2)
    assert flow.plan_phase_count(df.columns) == n_phases
    keys = flow.PlanSpace.plan_keys(n_phases)
    assert len(keys) == 8 * n_phases + 5, "자세 7N + 총 시간 1 + 몫 N−1 + 구역 5"
    assert all(f"plan_{k}" in df.columns for k in keys)


def test_phase_count_rejects_gaps_and_missing():
    with pytest.raises(ValueError, match="단계 수"):
        flow.plan_phase_count(["plan_p1_torso_yaw", "plan_p3_torso_yaw"])
    with pytest.raises(ValueError, match="단계 수"):
        flow.plan_phase_count(["pose_torso_yaw", "plan_duration_s"])
    assert flow.plan_phase_count(["cand_plan_p1_torso_yaw", "cand_plan_p2_torso_yaw"], prefix="cand_plan_") == 2


@pytest.mark.parametrize("n_phases", [2, 5, 9])
def test_plan_from_raw_inverts_raw_vector(scenarios, n_phases):
    """C 의 PlanEncoder.raw_vector 로 만든 벡터(= 데이터셋 행)가 같은 계획으로 돌아온다."""
    rng = np.random.default_rng(n_phases)
    sc = scenarios["default"]
    enc = PlanEncoder(sc, limits_for(n_phases))
    for _ in range(20):
        plan = make_plan(n_phases, float(rng.uniform(0, 180)), rng.uniform(0, 1, len(ZONE_NAMES)), rng, sc)
        raw = enc.raw_vector(plan)
        back = flow.plan_from_raw(raw, sc, n_phases, MIN_PHASE)
        assert len(back.phases) == n_phases
        assert [p.duration_s for p in back.phases] == pytest.approx([p.duration_s for p in plan.phases])
        for a, b in zip(back.phases, plan.phases):
            assert np.allclose(a.pose.to_vector(), b.pose.to_vector(), atol=1e-5)
        assert np.allclose(back.zone_strengths, plan.zone_strengths)
        assert np.allclose(enc.raw_vector(back), raw, atol=1e-6)
    with pytest.raises(ValueError, match="계획 벡터 길이"):
        flow.plan_from_raw(np.zeros(21), sc, 3, MIN_PHASE)


def test_plan_durations_match_encoder_for_any_phase_count(scenarios):
    rng = np.random.default_rng(0)
    for n in (2, 5, 9):
        enc = PlanEncoder(scenarios["default"], limits_for(n))
        for _ in range(20):
            total = rng.uniform(n * MIN_PHASE, 20.0)
            shares = list(rng.uniform(-0.1, 0.4, n - 1))
            assert flow.plan_durations(total, shares, n, MIN_PHASE) == pytest.approx(enc.durations(total, shares))


def test_planspace_to_plan_uses_same_conversion(scenarios):
    """PlanSpace.to_plan 은 plan_from_raw 를 그대로 쓴다 (모델 샘플과 kNN 후보가 같은 변환을 거친다)."""
    sc = scenarios["default"]
    space = flow.PlanSpace.from_scenarios([sc], n_phases=5, duration_bounds_s=(10.0, 20.0), min_phase_s=MIN_PHASE)
    assert space.dim == 45
    x = np.random.default_rng(1).uniform(-1, 1, space.dim)
    a, b = space.to_plan(x, sc), flow.plan_from_raw(space.decode(x)[0], sc, 5, MIN_PHASE)
    assert np.allclose(space.from_plan(a), space.from_plan(b))
    assert sum(p.duration_s for p in a.phases) >= 10.0 - 1e-9 and min(p.duration_s for p in a.phases) >= MIN_PHASE - 1e-9


# ---------- 계획 kNN 표 ----------

@pytest.mark.parametrize("n_phases", [2, 5, 9])
def test_plan_knn_returns_nearest_bodies_plans(scenarios, n_phases):
    df = plan_df(n_phases)
    table = PlanKNN.from_dataset(df)
    assert table.n_phases == n_phases and table.min_phase_s == MIN_PHASE
    assert table.scenario_names == ["default", "wheelchair"]
    bodies = dataset.sample_bodies(40, seed=0)
    sc = scenarios["default"]
    got = table.candidates(bodies[7], sc, 3)
    assert len(got) == 3 and all(isinstance(p, Plan) and len(p.phases) == n_phases for p in got)
    # 학습 체형 자신을 물으면 그 체형의 계획이 첫 후보다
    own = df[(df.body_idx == 7) & (df.scenario == "default")].iloc[0]
    want = flow.plan_from_raw([own[f"plan_{k}"] for k in table.keys], sc, n_phases, MIN_PHASE)
    enc = PlanEncoder(sc, limits_for(n_phases))
    assert np.allclose(enc.raw_vector(got[0]), enc.raw_vector(want), atol=1e-6)
    assert got[0].phases[0].pose.shoulder_abduction == pytest.approx(15.0), "체형 7(홀수)은 팔 내림"
    again = table.candidates(bodies[7], sc, 3)
    assert all(np.allclose(enc.raw_vector(a), enc.raw_vector(b)) for a, b in zip(got, again)), "같은 입력이면 같은 후보"
    assert table.candidates(bodies[7], sc, 0) == [] and len(table.candidates(bodies[7], sc, 999)) == 40
    with pytest.raises(KeyError, match="pregnant"):
        table.candidates(bodies[7], scenarios["pregnant"], 3)


def test_plan_knn_applies_scenario_pose_constraints(scenarios):
    """휠체어: 돌려주는 계획의 모든 단계가 하체 고정(hip·knee 90°)과 yaw 범위를 지킨다."""
    table = PlanKNN.from_dataset(plan_df(5))
    wc = scenarios["wheelchair"]
    lo, hi = wc.pose_bounds["torso_yaw"]
    for plan in table.candidates(BodyParams(), wc, 6):
        for ph in plan.phases:
            assert ph.pose.hip_flexion == 90.0 and ph.pose.knee_flexion == 90.0
            assert lo - 1e-9 <= ph.pose.torso_yaw <= hi + 1e-9
        assert sum(p.duration_s for p in plan.phases) == pytest.approx(20.0)


def test_plan_knn_save_load_keeps_stamps_and_limits(tmp_path):
    df = plan_df(9)
    df.loc[df.index[-3:], "commit"] = "c1"                     # 생성이 중단·재개된 경우
    path = PlanKNN.from_dataset(df).save(tmp_path / "plan_knn.parquet")
    table = PlanKNN.load(path)
    assert table.n_phases == 9 and table.min_phase_s == MIN_PHASE
    m = table.meta
    assert m["physics_hash"] == "p0" and m["n_phases"] == 9 and m["energy_weight"] == pytest.approx(0.03)
    assert m["zone_nozzle_counts"] == "2;2;2;2;4" and bool(m["kinetics_enabled"]) is True
    assert m["duration_lo_s"] == pytest.approx(18.0) and m["duration_hi_s"] == pytest.approx(20.0)
    assert m["commit"] == "c0+c1", "출처 열은 여러 값을 전부 적는다"
    assert set(PLAN_STAMP_COLS) | set(PLAN_LIMIT_COLS) <= set(table.table.columns)
    assert not any(c.startswith("cand_") for c in table.table.columns)
    with pytest.raises(FileNotFoundError, match="계획 kNN 표"):
        PlanKNN.load(tmp_path / "none.parquet")


def test_plan_knn_without_limit_columns_uses_argument_then_config(tmp_path):
    """한도 열이 없는 데이터셋: 단계 최소 시간은 인자 → 설정 파일 순. 저장하면 표에 남아 다시 읽어도 같다."""
    df = plan_df(2, n_bodies=6, stamps=False)
    assert PlanKNN.from_dataset(df).min_phase_s == PlanLimits().min_phase_s
    table = PlanKNN.from_dataset(df, min_phase_s=1.5)
    assert table.min_phase_s == 1.5 and table.meta["n_phases"] == 2
    assert PlanKNN.load(table.save(tmp_path / "k.parquet")).min_phase_s == 1.5
    with pytest.raises(ValueError, match="열이 없다"):
        PlanKNN.from_dataset(df.drop(columns=["plan_duration_s"]))
    with pytest.raises(ValueError, match="단계 수"):
        PlanKNN.from_dataset(df[[c for c in df.columns if not c.startswith("plan_p")]])


# ---------- 후보 벡터 열 (cand_plan_raw) ----------

def test_candidate_matrix_reads_2d_and_flat_lists():
    """parquet 에서 읽은 2차원 목록(배열의 배열)과 행 우선으로 편 1차원 목록을 같은 배열로 읽는다."""
    want = np.arange(12, dtype=np.float64).reshape(3, 4)
    nested = np.empty(3, dtype=object)
    for i in range(3):
        nested[i] = want[i].copy()
    for value in (want.tolist(), nested, want.reshape(-1), want.reshape(-1).tolist(), [want[0].tolist()]):
        got = flow.candidate_matrix(value, 4)
        assert np.array_equal(got, want[:len(got)]) and got.shape[1] == 4
    with pytest.raises(ValueError, match="배수"):
        flow.candidate_matrix(list(range(10)), 4)
    with pytest.raises(ValueError, match="다르다"):
        flow.candidate_matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0, 8.0]], 4)


@pytest.mark.parametrize("flat", [False, True])
def test_training_arrays_expand_cand_plan_raw(scenarios, tmp_path, flat):
    """계획 데이터셋의 후보 열(cand_plan_raw)이 학습 점으로 펼쳐진다. parquet 을 거쳐도 같다."""
    n = 5
    df = plan_df(n, n_bodies=4, scenarios=("default",))
    keys = flow.PlanSpace.plan_keys(n)
    best = df[[f"plan_{k}" for k in keys]].to_numpy(dtype=np.float64)
    cands, counts = [], [3, 1, 0, 2]
    for i, c in enumerate(counts):
        m = np.stack([best[i] + 0.01 * j for j in range(c)]) if c else np.zeros((0, len(keys)))
        cands.append((m.reshape(-1) if flat else [row for row in m]) if c else None)
    df["cand_plan_raw"] = pd.Series(cands, dtype=object)
    df["n_candidates"] = counts
    path = tmp_path / "plan_ds.parquet"
    df.to_parquet(path)
    back = pd.read_parquet(path)
    lim = limits_for(n)
    space = flow.PlanSpace.from_scenarios([scenarios["default"]], n_phases=n, duration_bounds_s=lim.duration_bounds_s,
                                          min_phase_s=lim.min_phase_s)
    for frame in (df, back):
        arr = flow.training_arrays(frame, space, scenarios)
        assert len(arr["x"]) == 3 + 1 + 1 + 2, "후보가 있으면 후보들을, 없으면 최적 계획 하나를 쓴다"
        assert np.allclose(np.bincount(arr["group"], weights=arr["weight"]), 1.0), "묶음마다 가중치 합 1"
        assert arr["x"].shape[1] == space.dim == 8 * n + 5 and np.all(np.abs(arr["x"]) <= 1.0)
        # 범위 안에 있는 값(1단계 어깨 벌림 170°)으로 본다. 상한에 붙은 값(세기 1.0)은 범위 밖 후보가 잘린다.
        abd = space.decode(arr["x"][:3])[:, space.keys.index("p1_shoulder_abduction")]
        assert abd == pytest.approx([170.0, 170.01, 170.02], abs=1e-4), "후보 순서와 값이 그대로"
