"""계획 산출물 갱신 절차(데이터셋 → 학습 → kNN 표 → 체형 K-fold → 판정 → 설치) 테스트. 소유자: F.

계획 데이터셋이 나오기 전이라 합성 데이터(N = 3)와 가짜 계획 평가기로 처음부터 끝까지 돌려 본다.
실제 수치는 데이터셋이 나온 뒤 docs/experiments_model.md 4절에 적는다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "scripts", Path(__file__).resolve().parent):   # 합성 계획 데이터 도우미(test_model_plan_knn)를 함께 쓴다
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

torch = pytest.importorskip("torch", reason="flow matching 모델은 torch 필요")
pd = pytest.importorskip("pandas")

import run_e5_plan                                             # noqa: E402
import train_plan_models as tpl                                # noqa: E402
from test_model_plan_knn import MIN_PHASE, plan_df             # noqa: E402

from airis.model import flow, plan_data                        # noqa: E402
from airis.model import predict as pred                        # noqa: E402
from airis.model.knn import PlanKNN                            # noqa: E402
from airis.optimize.plan_encoding import PlanEncoder           # noqa: E402
from airis.sim import EvalResult, Evaluator                    # noqa: E402
from airis.sim.scenario import load_scenarios                  # noqa: E402

N = 3
NOW = {"nozzle_layout_hash": "n0", "physics_hash": "p0", "kinetics_enabled": True, "time_constant_s": 2.0}


class ConstScorer(Evaluator):
    """가짜 계획 평가기: 모든 계획에 같은 점수. 합성 데이터의 score(0.5 + 0.001 × 체형 번호)와 비교된다."""

    def __init__(self, score: float = 0.6):
        self.score, self.calls = score, 0

    def evaluate(self, pose, nozzle, body, scenario):
        raise NotImplementedError

    def evaluate_plan(self, plan, nozzle, body, scenario):
        self.calls += 1
        return EvalResult(score=self.score, removal_by_part=np.zeros(5), total_removal=0.0, discomfort=0.0,
                          extra={"infeasible": False, "energy": 0.0})


@pytest.fixture(autouse=True)
def matching_stamp(monkeypatch):
    """합성 데이터의 도장(n0·p0)을 지금 설정으로 본다."""
    monkeypatch.setattr(pred, "current_stamp", lambda: dict(NOW))


@pytest.fixture
def scorer(monkeypatch):
    ev = ConstScorer()
    monkeypatch.setattr(pred, "default_rescorer", lambda model, *, plan=False: (ev, object()))
    # 응답 시간 판정은 CPU 혼잡에 좌우되므로 이 테스트들에서는 풀어 둔다 (기준 자체는 test_gate_… 가 본다)
    judge = tpl.judge
    monkeypatch.setattr(tpl, "judge", lambda stats: judge(stats, tpl.PlanGate(max_mean_s=float("inf"))))
    return ev


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory):
    """N = 3 합성 계획 데이터셋 (체형 12개 × 시나리오 2개)과 단일 자세 최적 열."""
    df = plan_df(N, n_bodies=12)
    for k in flow.POSE_KEYS:
        df[f"pose_{k}"] = df[f"plan_p1_{k}"]
    df["score_p1_10"], df["score_p1opt_10"] = 0.3, 0.4
    path = tmp_path_factory.mktemp("plan_dataset") / "plan_n3.parquet"
    df.to_parquet(path)
    return path


# ---------- 데이터셋에서 단계 수·한도·도장 ----------

def test_limits_and_space_come_from_dataset():
    df = plan_df(N, n_bodies=4).assign(duration_lo_s=8.0, cap_ratio=0.7, transition_s=1.0)
    lim = plan_data.plan_limits_from_dataset(df)
    assert lim.n_phases == N and lim.min_phase_s == MIN_PHASE
    assert lim.duration_bounds_s == pytest.approx((8.0, 20.0)) and lim.cap_ratio == 0.7 and lim.transition_s == 1.0
    scs = load_scenarios()
    space = plan_data.plan_space_from_dataset(df, [scs["default"], scs["wheelchair"]])
    assert space.dim == 8 * N + 5 and space.n_phases == N
    assert space.to_dict()["duration_s"] == pytest.approx((8.0, 20.0))

    # 한도 열이 없으면 설정 파일에서 읽되 하한을 N × min_phase_s 로 올린다 (인코더가 받아들인다)
    bare = plan_data.plan_limits_from_dataset(plan_df(9, n_bodies=2, stamps=False))
    assert bare.n_phases == 9 and bare.duration_bounds_s[0] == pytest.approx(9 * bare.min_phase_s)
    PlanEncoder(scs["default"], bare)


def test_dataset_with_mixed_conditions_is_rejected():
    df = plan_df(N, n_bodies=4)
    mixed = df.copy()
    mixed.loc[mixed.index[:2], "energy_weight"] = 0.1
    with pytest.raises(ValueError, match="energy_weight 가 여러 값"):
        plan_data.check_plan_dataset(mixed)
    for col, other in (("candidate_tol", 0.05), ("candidate_k", 8), ("pose_max_evals", 2000)):
        blend = df.copy()                                       # 데이터셋 생성 설정이 섞여도 거부한다 (C, PR #159)
        blend.loc[blend.index[:2], col] = other
        with pytest.raises(ValueError, match=f"{col} 가 여러 값"):
            plan_data.check_plan_dataset(blend)
    with pytest.raises(ValueError, match="단계 수"):
        plan_data.plan_limits_from_dataset(df.assign(n_phases=5))
    with pytest.raises(ValueError, match="보다 짧다"):
        plan_data.plan_limits_from_dataset(df.assign(duration_lo_s=4.0))


def test_stamp_check_and_meta(monkeypatch):
    df = plan_df(N, n_bodies=4)
    stamp, warns = plan_data.check_plan_dataset(df)
    assert warns == [] and stamp["physics_hash"] == "p0" and stamp["n_phases"] == N
    assert stamp["zone_nozzle_counts"] == "2;2;2;2;4"
    assert (stamp["candidate_k"], stamp["candidate_tol"], stamp["pose_max_evals"]) == (16, 0.02, 3000)
    old = df.drop(columns=["candidate_tol", "warm_start_sigma0"])   # 생성 설정 열이 없는 예전 데이터셋: 알리고 진행
    _, warns_old = plan_data.check_plan_dataset(old)
    assert len(warns_old) == 1 and "candidate_tol" in warns_old[0] and "candidate_k" not in warns_old[0]

    # 출처(commit)는 여러 값이어도 경고만, meta 에는 전부 적는다
    two = df.copy()
    two.loc[two.index[:2], "commit"] = "c1"
    stamp2, warns2 = plan_data.check_plan_dataset(two)
    assert len(warns2) == 1 and "commit" in warns2[0]
    meta = plan_data.plan_meta_from_dataset(two)
    assert "c0" in meta["dataset_commit"] and "c1" in meta["dataset_commit"] and "commit" not in meta
    assert meta["n_phases"] == N and meta["energy_weight"] == pytest.approx(0.03)
    assert (meta["duration_lo_s"], meta["duration_hi_s"], meta["min_phase_s"]) == pytest.approx((6.0, 20.0, MIN_PHASE))
    json.dumps(meta)                                            # 산출물 meta 에 그대로 넣을 수 있다 (기본 타입)

    # 지금 설정과 다르면 경고 (물리 해시, 계획 채점 설정)
    monkeypatch.setattr(pred, "current_stamp", lambda: {**NOW, "physics_hash": "p1", "time_constant_s": 3.0})
    _, warns3 = plan_data.check_plan_dataset(df)
    assert sorted(w.split(":")[0] for w in warns3) == ["physics_hash", "time_constant_s"]

    # 도장·한도 열에 빈 값(NaN)이 있으면 거부한다: 정상 경로에서는 비지 않는다 (옛 파일을 이어 만든 신호)
    monkeypatch.setattr(pred, "current_stamp", lambda: dict(NOW))
    half = df.copy()
    half.loc[half.index[:2], "energy_weight"] = np.nan
    for bad, col in ((df.assign(duration_lo_s=np.nan), "duration_lo_s"), (half, "energy_weight"),
                     (df.assign(candidate_tol=np.nan), "candidate_tol")):
        for fn in (plan_data.check_plan_dataset, plan_data.plan_meta_from_dataset):
            with pytest.raises(ValueError, match=f"{col} 에 빈 값"):
                fn(bad)
    with pytest.raises(ValueError, match="duration_lo_s 에 빈 값"):
        plan_data.plan_limits_from_dataset(df.assign(duration_lo_s=np.nan))
    plan_data.check_plan_dataset(df.assign(energy=np.nan))      # 지표 열은 비어도 된다 (도장이 아니다)

    # 도장·한도 열이 없는 데이터셋: 무엇이 없는지 알린다
    _, warns4 = plan_data.check_plan_dataset(plan_df(N, n_bodies=2, stamps=False))
    assert len(warns4) == 1 and "energy_weight" in warns4[0] and "duration_lo_s" in warns4[0]


# ---------- 체형 K-fold 평가 ----------

def base_model(dataset_path, out_dir):
    df = pd.read_parquet(dataset_path)
    scs = load_scenarios()
    space = plan_data.plan_space_from_dataset(df, [scs[n] for n in ("default", "wheelchair")])
    cfg = flow.FlowConfig(hidden=32, layers=2, steps=60, batch_size=64, lr=2e-3, seed=0)
    return flow.train_flow(df, scs, cfg, space=space, meta=plan_data.plan_meta_from_dataset(df)).save(
        out_dir / "plan_flow.pt")


def test_plan_cv_scores_every_body_once_per_method(dataset_path, tmp_path, scorer):
    model = base_model(dataset_path, tmp_path)
    out = tmp_path / "cv.csv"
    rc = run_e5_plan.main(["--model", str(model), "--dataset", str(dataset_path), "--folds", "2", "--n-flow", "3",
                           "--n-knn", "2", "--methods", "hybrid,hybrid-nofixed,flow+fixed,knn+fixed,fixed",
                           "--out", str(out)])
    assert rc == 0
    res = pd.read_csv(out)
    assert len(res) == 12 * 2 * 5, "체형 12 × 시나리오 2 × 방법 5"
    assert not res.duplicated(["body_idx", "scenario", "method"]).any(), "체형마다 한 번씩만 평가한다"
    assert set(res["fold"]) == {0, 1}
    evals = res.groupby("method")["n_evals"].first().to_dict()
    assert evals == {"hybrid": 6, "hybrid-nofixed": 5, "flow+fixed": 6, "knn+fixed": 6, "fixed": 1}
    assert scorer.calls == int(res["n_evals"].sum())
    fixed = res[res["method"] == "fixed"]
    assert (fixed["source"] == "fixed").all() and (fixed["plan_phases"] == pred.ROTATION_STEPS).all()
    assert (res[res["method"] == "hybrid-nofixed"]["plan_phases"] == N).all()
    # 점수 비율 = 추천 점수 / 데이터셋 score, 기준선과의 비율도 나란히
    r = res.iloc[0]
    assert r["ratio"] == pytest.approx(0.6 / (0.5 + 0.001 * r["body_idx"]))
    assert r["ratio_vs_p1_10"] == pytest.approx(2.0) and r["ratio_vs_p1opt_10"] == pytest.approx(1.5)
    stats = pd.read_csv(tmp_path / "cv_overall.csv")
    assert list(stats["method"]) == ["hybrid", "hybrid-nofixed", "flow+fixed", "knn+fixed", "fixed"]
    assert stats.set_index("method").loc["fixed", "fixed_picked"] == 1.0
    assert stats.set_index("method").loc["fixed", "plan_duration_s"] == pytest.approx(20.0)
    # fold 산출물은 학습 체형만으로 만든다 (평가 체형이 표에 없다)
    for f in (0, 1):
        knn = PlanKNN.load(tmp_path / f"_fold{f}_plan_knn.parquet")
        held = set(res[res["fold"] == f]["body_idx"])
        assert held and not held & set(knn.table["body_idx"])


def test_plan_cv_rerun_in_same_process_reloads_fold_artifacts(dataset_path, tmp_path, scorer):
    """같은 --out 으로 다시 돌려도 앞 실행의 fold 산출물(경로별 캐시)을 다시 쓰지 않는다."""
    other = plan_df(N, n_bodies=12, seed=1)
    other.to_parquet(tmp_path / "other.parquet")
    out = tmp_path / "cv.csv"
    for path in (dataset_path, tmp_path / "other.parquet"):
        assert run_e5_plan.main(["--model", str(base_model(path, tmp_path)), "--dataset", str(path), "--folds", "2",
                                 "--n-flow", "1", "--n-knn", "1", "--methods", "knn+fixed", "--limit", "1",
                                 "--out", str(out)]) == 0
    fold1 = tmp_path / "_fold1_plan_knn.parquet"
    cached, on_disk = pred.load_plan_knn(fold1).table, PlanKNN.load(fold1).table
    assert np.allclose(cached["body_height_m"], on_disk["body_height_m"])
    first = pd.read_parquet(dataset_path)
    assert not set(np.round(on_disk["body_height_m"], 9)) <= set(np.round(first["body_height_m"], 9)), "둘째 데이터셋의 표"


def test_plan_cv_rescore_top_counts_only_scored_plans(dataset_path, tmp_path, scorer):
    model = base_model(dataset_path, tmp_path)
    out = tmp_path / "top.csv"
    assert run_e5_plan.main(["--model", str(model), "--dataset", str(dataset_path), "--folds", "2", "--limit", "2",
                             "--n-flow", "3", "--n-knn", "2", "--methods", "hybrid,hybrid-nofixed,knn+fixed,fixed",
                             "--rescore-top", "2", "--n-threads", "2", "--out", str(out)]) == 0
    res = pd.read_csv(out)
    evals = res.groupby("method")["n_evals"].first().to_dict()
    assert evals == {"hybrid": 2, "hybrid-nofixed": 2, "knn+fixed": 2, "fixed": 1}, "채점한 수만 센다"
    assert scorer.calls == int(res["n_evals"].sum())
    assert (res["rescore_top"] == 2).all() and (res["n_threads"] == 2).all()


def test_plan_cv_rotation_pose_and_bad_arguments(dataset_path, tmp_path, scorer, capsys):
    model = base_model(dataset_path, tmp_path)
    common = ["--model", str(model), "--dataset", str(dataset_path), "--folds", "2", "--limit", "2"]
    out = tmp_path / "rot.csv"
    assert run_e5_plan.main(common + ["--methods", "knn+fixed,fixed", "--n-flow", "1", "--n-knn", "1",
                                      "--rotation-pose", "dataset", "--out", str(out)]) == 0
    res = pd.read_csv(out)
    assert len(res) == 2 * 2 * 2, "--limit 은 fold 마다 평가 행을 자른다"
    assert res.groupby("method")["n_evals"].first().to_dict() == {"knn+fixed": 4, "fixed": 2}, "제안 자세 회전이 더해진다"
    assert (res["rotation_pose"] == "dataset").all()
    assert "직접 최적화한 값" in capsys.readouterr().out, "새는 값을 썼다고 알린다"
    assert not list(tmp_path.glob("_fold*_plan_flow.pt")), "flow 를 쓰지 않는 방법만이면 학습하지 않는다"

    assert run_e5_plan.main(common + ["--methods", "hybrid,best"]) == 2
    assert run_e5_plan.main(common + ["--rotation-pose", "model"]) == 2
    assert run_e5_plan.main(common[:4] + ["--folds", "1"]) == 2
    mixed = pd.read_parquet(dataset_path)
    mixed.loc[mixed.index[:2], "energy_weight"] = 0.1
    mixed.to_parquet(tmp_path / "mixed.parquet")
    assert run_e5_plan.main(["--model", str(model), "--dataset", str(tmp_path / "mixed.parquet")]) == 2


# ---------- 판정 ----------

def stats_row(**over):
    row = {"method": "hybrid", "n": 100, "n_evals": 17.0, "median": 1.0, "p05": 0.98, "min": 0.9, "below_095": 0.01,
           "infeasible": 0.0, "mean_s": 1.2, "p95_s": 1.4, "plan_duration_s": 19.0, "fixed_picked": 0.25}
    return pd.DataFrame([{**row, **over}])


def test_gate_is_p05_infeasible_and_response_time():
    assert tpl.judge(stats_row())["passed"]
    assert tpl.judge(stats_row(p05=0.97))["passed"], "하위 5% 0.97 이상"
    for bad in ({"p05": 0.9699}, {"infeasible": 0.001}, {"mean_s": 1.51}):
        v = tpl.judge(stats_row(**bad))
        assert not v["passed"] and [k for k, c in v["checks"].items() if not c[2]] == list(bad)
    assert tpl.judge(stats_row(below_095=0.5))["passed"], "0.95 미만 비율은 계획 판정에 없다 (완성도 F2)"
    none = tpl.judge(stats_row(method="fixed"))
    assert not none["passed"] and "결과 없음" in none["reason"]
    md = tpl.markdown_table(stats_row(), tpl.judge(stats_row()),
                            {"dataset": "d.parquet", "rows": 100, "n_phases": 9, "energy_weight": 0.03, "commit": "abc",
                             "folds": 5, "rotation_pose": "none", "device": "cpu", "out_dir": "outputs/x"})
    assert "**합격**" in md and "단계 9개" in md and "| 25% |" in md


# ---------- 처음부터 끝까지 ----------

def run_refresh(dataset_path, out_dir, model_dir, *more):
    return tpl.main(["--dataset", str(dataset_path), "--out-dir", str(out_dir), "--model-dir", str(model_dir),
                     "--steps", "60", "--hidden", "32", "--layers", "2", "--batch-size", "64", "--folds", "2",
                     "--n-flow", "2", "--n-knn", "2", "--methods", "hybrid,fixed", *more])


def test_refresh_end_to_end_then_install(dataset_path, tmp_path, scorer):
    out_dir, model_dir = tmp_path / "run", tmp_path / "models"
    assert run_refresh(dataset_path, out_dir, model_dir, "--no-install") == 0
    assert all((out_dir / n).is_file() for n in (*tpl.NAMES, "e5plan.csv", "e5plan_overall.csv", "gate.md"))
    assert not model_dir.exists() or not list(model_dir.iterdir()), "--no-install 이면 설치 폴더를 건드리지 않는다"
    report = json.loads((out_dir / "gate.json").read_text(encoding="utf-8"))
    assert report["verdict"]["passed"] and report["installed"] == []
    assert report["limits"]["n_phases"] == N and report["space_dim"] == 8 * N + 5
    assert report["dataset_stamp"]["energy_weight"] == pytest.approx(0.03)

    # 산출물은 데이터셋의 단계 수·한도·도장을 가진다
    model = pred.load_model(out_dir / tpl.FLOW_FILE, kind="plan")
    knn = PlanKNN.load(out_dir / tpl.KNN_FILE)
    assert model.meta["n_phases"] == N and model.meta["dataset_commit"] == "c0" and model.meta["train_rows"] == 24
    lim = pred.plan_limits_for(model, knn)
    assert lim.n_phases == N and lim.duration_bounds_s == pytest.approx((6.0, 20.0))

    # 설치만 따로. 두 번 하면 이력이 쌓이고 앞 산출물이 .prev 로 남는다
    assert tpl.main(["--install-from", str(out_dir), "--model-dir", str(model_dir)]) == 0
    assert sorted(p.name for p in model_dir.iterdir()) == sorted(tpl.NAMES)
    assert tpl.main(["--install-from", str(out_dir), "--model-dir", str(model_dir)]) == 0
    assert sorted(p.name for p in model_dir.iterdir()) == sorted([*tpl.NAMES, *(n + ".prev" for n in tpl.NAMES)])
    report = json.loads((out_dir / "gate.json").read_text(encoding="utf-8"))
    assert len(report["installs"]) == 2 and all(i["error"] is None and not i["forced"] for i in report["installs"])
    # 설치한 산출물로 추천이 된다
    scs = load_scenarios()
    p = pred.predict_plan_candidates(pred.BodyParams(), scs["default"], evaluator=ConstScorer(), nozzle=object(),
                                     path=model_dir / tpl.FLOW_FILE, knn_path=model_dir / tpl.KNN_FILE)
    assert p.sources.count("flow") == pred.N_FLOW and "knn" in p.sources and "fixed" in p.sources


def test_failed_gate_does_not_install(dataset_path, tmp_path, scorer):
    scorer.score = 0.3                                          # 데이터셋 score(0.5~)의 0.6배 → 하위 5% 미달
    out_dir, model_dir = tmp_path / "run", tmp_path / "models"
    assert run_refresh(dataset_path, out_dir, model_dir) == 1
    report = json.loads((out_dir / "gate.json").read_text(encoding="utf-8"))
    assert not report["verdict"]["passed"] and not report["verdict"]["checks"]["p05"][2]
    assert "**불합격**" in (out_dir / "gate.md").read_text(encoding="utf-8")
    assert not model_dir.exists() or not list(model_dir.iterdir())
    assert tpl.main(["--install-from", str(out_dir), "--model-dir", str(model_dir)]) == 1
    assert not model_dir.exists() or not list(model_dir.iterdir())
    assert tpl.main(["--install-from", str(out_dir), "--model-dir", str(model_dir), "--install"]) == 0, "강제 설치"
    report = json.loads((out_dir / "gate.json").read_text(encoding="utf-8"))
    assert report["installs"][-1]["forced"] and (model_dir / tpl.FLOW_FILE).is_file()


def test_refresh_guards(dataset_path, tmp_path, scorer):
    model_dir = tmp_path / "models"
    # 빠른 점검(--limit)은 점수가 좋아도 판정 무효라 설치하지 않는다
    assert run_refresh(dataset_path, tmp_path / "quick", model_dir, "--limit", "2") == 0
    quick = json.loads((tmp_path / "quick" / "gate.json").read_text(encoding="utf-8"))
    assert not quick["verdict"]["passed"] and "판정 무효" in quick["verdict"]["reason"]
    assert not model_dir.exists() or not list(model_dir.iterdir())
    # 5-fold 를 건너뛰면 설치 여부를 정해 줘야 한다
    assert run_refresh(dataset_path, tmp_path / "skip", model_dir, "--skip-cv") == 2
    assert run_refresh(dataset_path, tmp_path / "skip", model_dir, "--skip-cv", "--no-install") == 0
    assert not (tmp_path / "skip" / "gate.md").exists() and scorer.calls > 0
    assert tpl.main(["--install-from", str(tmp_path / "skip"), "--model-dir", str(model_dir)]) == 1, "판정이 없으면 불합격 취급"
    # 입력이 잘못된 경우
    assert tpl.main(["--model-dir", str(model_dir)]) == 2
    assert tpl.main(["--install-from", str(tmp_path / "nowhere"), "--model-dir", str(model_dir)]) == 2
    # 앞선 설치가 남긴 백업이 있으면 시작하지 않는다
    model_dir.mkdir(exist_ok=True)
    (model_dir / (tpl.FLOW_FILE + ".prev.new")).write_bytes(b"old")
    assert run_refresh(dataset_path, tmp_path / "blocked", model_dir) == 2
    assert not (tmp_path / "blocked").exists()
    assert tpl.main(["--install-from", str(tmp_path / "quick"), "--model-dir", str(model_dir), "--install"]) == 2
