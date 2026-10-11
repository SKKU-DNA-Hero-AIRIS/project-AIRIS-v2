"""데이터셋 생성 테스트. 소유자: C. docs/tracks/C_optimize.md 단계 9."""
import numpy as np
import pytest

from airis.optimize import dataset
from airis.sim import BodyParams

pytest.importorskip("cma", reason="cma 패키지 필요 (pip install cma)")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

SMALL = dataset.DatasetConfig(evaluator="dummy", max_evals=40, popsize=10, dataset_id="t")


def test_sample_bodies_deterministic_and_prefix_stable():
    a = dataset.sample_bodies(5, seed=3)
    b = dataset.sample_bodies(5, seed=3)
    c = dataset.sample_bodies(8, seed=3)
    assert a == b
    assert c[:5] == a, "n 을 늘려도 앞부분은 같다 (이어 만들기 전제)"
    assert dataset.sample_bodies(5, seed=4) != a


def test_sample_bodies_distribution():
    bodies = dataset.sample_bodies(4000, seed=0)
    h = np.array([b.height_m for b in bodies])
    assert h.min() >= 1.45 and h.max() <= 1.95
    assert h.mean() == pytest.approx(1.68, abs=0.01)
    assert h.std() == pytest.approx(0.09, abs=0.01)
    arm = np.array([b.arm_length_m for b in bodies])
    resid = arm - 0.365 * h
    assert resid.mean() == pytest.approx(0.0, abs=0.003)
    assert resid.std() == pytest.approx(0.02, abs=0.003)
    assert all(isinstance(b, BodyParams) for b in bodies[:3])


def test_build_dataset_schema_and_resume(tmp_path):
    out = tmp_path / "ds.parquet"
    info = dataset.build_dataset(out, n_bodies=2, scenarios=["default", "wheelchair"], cfg=SMALL,
                                 flush_every=1, log=lambda *_: None)
    assert (info["n_rows"], info["n_new"], info["n_skipped"]) == (4, 4, 0)
    df = pd.read_parquet(out)

    # interfaces.md 스키마 열 + 회귀 계약 열
    for col in (["body_idx", "scenario", "score", "total_removal", "discomfort", "nozzle_layout_hash",
                 "exp_id", "commit", "seed", "arm_class", "peak_gap", "hands_up_feasible",
                 "pose_torso_yaw_raw", "physics_hash"]
                + [f"body_{k}" for k in dataset.BODY_KEYS] + [f"pose_{k}" for k in dataset.POSE_KEYS]):
        assert col in df.columns, col
    assert df["pose_torso_yaw"].between(0, 90).all(), "yaw 는 0~90° 로 접는다"
    assert np.allclose(df["pose_torso_yaw"], 90 - (90 - df["pose_torso_yaw_raw"].abs()).abs())
    assert set(df["arm_class"]) <= {"hands_up", "arms_down"}
    wc = df[df["scenario"] == "wheelchair"]
    assert (wc["pose_hip_flexion"] == 90).all() and (wc["pose_knee_flexion"] == 90).all()
    assert list(zip(df["body_idx"], df["scenario"])) == [
        (0, "default"), (0, "wheelchair"), (1, "default"), (1, "wheelchair")]

    # 같은 설정으로 체형을 늘리면 새 체형만 계산한다.
    info2 = dataset.build_dataset(out, n_bodies=3, scenarios=["default", "wheelchair"], cfg=SMALL,
                                  log=lambda *_: None)
    assert (info2["n_rows"], info2["n_new"], info2["n_skipped"]) == (6, 2, 4)
    df2 = pd.read_parquet(out)
    pd.testing.assert_frame_equal(df2[df2["body_idx"] < 2].reset_index(drop=True), df,
                                  check_dtype=False)


def test_build_dataset_refuses_mixed_settings(tmp_path):
    out = tmp_path / "ds.parquet"
    dataset.build_dataset(out, n_bodies=1, scenarios=["default"], cfg=SMALL, log=lambda *_: None)
    other = dataset.DatasetConfig(evaluator="dummy", max_evals=60, popsize=10, dataset_id="t")
    with pytest.raises(ValueError, match="max_evals"):
        dataset.build_dataset(out, n_bodies=2, scenarios=["default"], cfg=other, log=lambda *_: None)


def test_build_dataset_pool_matches_serial(tmp_path):
    """프로세스 2개로 만든 행이 순차 실행과 같다 (시드가 작업마다 고정)."""
    serial = dataset.build_dataset(tmp_path / "a.parquet", n_bodies=2, scenarios=["default"],
                                   cfg=SMALL, log=lambda *_: None)
    pooled = dataset.build_dataset(tmp_path / "b.parquet", n_bodies=2, scenarios=["default"],
                                   cfg=SMALL, processes=2, log=lambda *_: None)
    a = pd.read_parquet(serial["path"]).drop(columns=["elapsed_s"])
    b = pd.read_parquet(pooled["path"]).drop(columns=["elapsed_s"])
    pd.testing.assert_frame_equal(a, b)


def test_height_bin_summary():
    df = pd.DataFrame({
        "scenario": ["default"] * 4,
        "body_height_m": [1.55, 1.58, 1.90, 1.92],
        "arm_class": ["hands_up", "hands_up", "arms_down", "hands_up"],
        "hands_up_feasible": [True, True, False, True],
        "peak_gap": [0.02, 0.01, -0.03, 0.0],
        "score": [0.5, 0.5, 0.4, 0.45],
        "pose_torso_yaw": [90.0, 90.0, 85.0, 95.0],
    })
    s = dataset.height_bin_summary(df)
    assert list(s["n"]) == [2, 2]
    assert list(s["hands_up_frac"]) == [1.0, 0.5]
    assert list(s["hands_up_feasible_frac"]) == [1.0, 0.5]


def test_run_dataset_script(tmp_path):
    from scripts.run_dataset import main

    out = tmp_path / "s.parquet"
    assert main(["--evaluator", "dummy", "--n-bodies", "2", "--scenarios", "default",
                 "--max-evals", "40", "--popsize", "10", "--out", str(out)]) == 0
    assert out.exists() and (tmp_path / "s_height_bins.csv").exists()
    assert main(["--evaluator", "dummy", "--n-bodies", "2", "--scenarios", "default",
                 "--max-evals", "60", "--popsize", "10", "--out", str(out)]) == 2


def test_mesh_distribution_matches_mesh_default_body():
    """메시 분포의 비율은 MakeHuman 기본 체형(MESH_DEFAULT_BODY)을 키 1.70 으로 나눈 값이다."""
    from airis.sim.human_mesh import MESH_DEFAULT_BODY

    for key, (ratio, sigma) in dataset.BODY_DISTRIBUTIONS["mesh"].items():
        assert ratio * MESH_DEFAULT_BODY.height_m == pytest.approx(getattr(MESH_DEFAULT_BODY, key))
        cap_ratio, cap_sigma = dataset.BODY_DISTRIBUTIONS["capsule"][key]
        # 변동계수(σ / 평균)를 캡슐 분포와 비슷하게 유지한다.
        assert sigma / ratio == pytest.approx(cap_sigma / cap_ratio, rel=0.05)


def test_sample_bodies_mesh_model():
    cap = dataset.sample_bodies(2000, seed=0)
    mesh = dataset.sample_bodies(2000, seed=0, model="mesh")
    assert [b.height_m for b in cap] == [b.height_m for b in mesh], "같은 시드면 키가 같다"
    h = np.array([b.height_m for b in mesh])
    arm = np.array([b.arm_length_m for b in mesh])
    assert (arm - 0.463 / 1.70 * h).std() == pytest.approx(0.015, abs=0.002)
    assert np.mean([b.shoulder_width_m for b in mesh]) == pytest.approx(0.342 / 1.70 * h.mean(), abs=0.002)
    with pytest.raises(ValueError):
        dataset.sample_bodies(1, seed=0, model="stick")


@pytest.mark.parametrize("model, other", [("capsule", "mesh"), ("mesh", "capsule")])
def test_build_dataset_stamps_and_checks_body_model(tmp_path, monkeypatch, model, other):
    """physics.yaml body.model 이 무엇이든(monkeypatch 로 고정) 그 모델로 찍고, 다른 모델과는 섞지 않는다."""
    monkeypatch.setattr(dataset, "configured_body_model", lambda: model)
    out = tmp_path / "ds.parquet"
    dataset.build_dataset(out, n_bodies=1, scenarios=["default"], cfg=SMALL, log=lambda *_: None)
    df = pd.read_parquet(out)
    assert set(df["body_model"]) == {model}
    expected = dataset.sample_bodies(1, seed=SMALL.body_seed, model=model)[0]
    assert df["body_shoulder_width_m"].iloc[0] == pytest.approx(expected.shoulder_width_m)

    # 평가기(physics.yaml)와 다른 몸 모델 분포는 거부한다.
    other_cfg = dataset.DatasetConfig(evaluator="dummy", max_evals=40, popsize=10, body_model=other)
    with pytest.raises(ValueError, match="body.model"):
        dataset.build_dataset(tmp_path / "o.parquet", n_bodies=1, scenarios=["default"], cfg=other_cfg,
                              log=lambda *_: None)

    # physics 몸 모델이 바뀐 뒤에는 이전 파일에 이어 쓰지 않는다.
    monkeypatch.setattr(dataset, "configured_body_model", lambda: other)
    with pytest.raises(ValueError, match="body_model"):
        dataset.build_dataset(out, n_bodies=2, scenarios=["default"], cfg=SMALL, log=lambda *_: None)


def _legacy_file(tmp_path, monkeypatch):
    """body_model 열이 생기기 전(캡슐판) 파일을 흉내 낸다."""
    monkeypatch.setattr(dataset, "configured_body_model", lambda: "capsule")
    out = tmp_path / "old.parquet"
    dataset.build_dataset(out, n_bodies=1, scenarios=["default"], cfg=SMALL, log=lambda *_: None)
    pd.read_parquet(out).drop(columns=["body_model"]).to_parquet(out, index=False)
    return out


def test_legacy_file_without_body_model_is_capsule(tmp_path, monkeypatch):
    out = _legacy_file(tmp_path, monkeypatch)
    info = dataset.build_dataset(out, n_bodies=2, scenarios=["default"], cfg=SMALL, log=lambda *_: None)
    assert info["n_new"] == 1
    assert set(pd.read_parquet(out)["body_model"]) == {"capsule"}


def test_legacy_capsule_file_refused_under_mesh(tmp_path, monkeypatch):
    """body.model 이 mesh 면 열 없는 옛 파일(캡슐판)에 이어 쓰지 않는다 (의도된 거부)."""
    out = _legacy_file(tmp_path, monkeypatch)
    monkeypatch.setattr(dataset, "configured_body_model", lambda: "mesh")
    with pytest.raises(ValueError, match="body_model"):
        dataset.build_dataset(out, n_bodies=2, scenarios=["default"], cfg=SMALL, log=lambda *_: None)


# ---------- 꼬리 보강 샘플러 (총괄 승인 10-11: 키 큰 체형 10개) ----------

def test_sample_bodies_taller_than_returns_true_sequence_indices():
    """키 하한을 넘는 체형만 주고, 번호는 시드 수열의 **진짜 번호**다."""
    from airis.optimize.dataset import sample_bodies, sample_bodies_taller_than

    got = sample_bodies_taller_than(5, 0, "mesh", min_height_m=1.85)
    assert len(got) == 5
    assert all(b.height_m >= 1.85 for _, b in got)
    assert [i for i, _ in got] == sorted(i for i, _ in got)

    # 번호가 수열 번호라면 그 번호의 체형과 같아야 한다 ("같은 body_idx = 같은 체형").
    last = max(i for i, _ in got)
    seq = sample_bodies(last + 1, 0, "mesh")
    for i, body in got:
        assert body == seq[i]

    # 같은 인수면 같은 결과다.
    assert sample_bodies_taller_than(5, 0, "mesh", min_height_m=1.85) == got


def test_sample_bodies_taller_than_skips_existing_range():
    """start_idx 앞쪽(이미 만든 구간)은 건너뛴다 — 이어 만들 때 겹치지 않게."""
    from airis.optimize.dataset import sample_bodies_taller_than

    first = sample_bodies_taller_than(3, 0, "mesh", min_height_m=1.85)
    after = sample_bodies_taller_than(3, 0, "mesh", min_height_m=1.85, start_idx=100)
    assert all(i >= 100 for i, _ in after), "이미 만든 구간의 번호는 주지 않는다"
    # 앞쪽 100 안에 있던 번호는 빠지고, 100 이상이던 번호는 그대로 다시 나온다.
    assert [i for i, _ in first if i < 100] and not [i for i, _ in after if i < 100]
    assert set(i for i, _ in first if i >= 100) <= set(i for i, _ in after)


def test_sample_bodies_taller_than_refuses_impossible_bound():
    """분포 상한(1.95 m)을 넘는 하한이면 조용히 적게 주지 않고 거부한다."""

    from airis.optimize.dataset import sample_bodies_taller_than

    with pytest.raises(ValueError, match="하한을 낮춰라"):
        sample_bodies_taller_than(2, 0, "mesh", min_height_m=2.5, max_scan=2000)


def test_run_plan_dataset_tall_option(tmp_path):
    """--tall-bodies 는 키 하한을 넘는 체형만 만들고 번호는 수열 번호다 (더미 평가기)."""
    import pandas as pd

    from scripts.run_plan_dataset import main as plan_main

    out = tmp_path / "tall.parquet"
    rc = plan_main([
        "--n-bodies", "0", "--tall-bodies", "2", "--tall-min-height", "1.80",
        "--scenarios", "default", "--evaluator", "dummy", "--n-phases", "3",
        "--max-evals", "200", "--pose-max-evals", "100", "--popsize", "10",
        "--candidate-k", "0", "--processes", "1", "--no-baselines",
        "--out", str(out)])
    assert rc == 0
    df = pd.read_parquet(out)
    assert len(df) == 2
    assert (df["body_height_m"] >= 1.80).all()
    assert df["body_idx"].min() >= 0 and df["body_idx"].nunique() == 2
