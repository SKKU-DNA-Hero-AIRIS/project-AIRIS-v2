"""학습 곡선(scripts/run_e5_learning_curve.py)과 fold 규칙(scripts/run_e5_flow.py --fold-rule) 테스트. 소유자: F.

저장소의 함수를 직접 부른다 (팀원 제안 3ab035e 의 테스트는 테스트 안에서 fold 함수를 다시 정의해 검사했다).
학습·재채점 자체는 다른 테스트가 본다. 여기서는 분할 규칙, 실행 계획, 단계 연결을 가짜 단계로 확인한다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_e5_flow                                             # noqa: E402
import run_e5_learning_curve as lc                             # noqa: E402


# ---------- fold 규칙 ----------

def test_permutation_rule_is_default_and_unchanged():
    """기본 규칙은 지금까지의 분할 그대로다 (docs/experiments_model.md 의 기존 수치가 이 분할)."""
    ids = np.repeat(np.arange(300), 3)                         # 체형마다 시나리오 3행
    folds = run_e5_flow.fold_indices(ids, 5, 0)
    want = [np.sort(part) for part in np.array_split(np.random.default_rng(0).permutation(np.arange(300)), 5)]
    assert all(np.array_equal(a, b) for a, b in zip(folds, want))
    assert all(np.array_equal(a, b) for a, b in
               zip(folds, run_e5_flow.fold_indices(ids, 5, 0, "permutation")))
    assert run_e5_flow.build_parser().parse_args([]).fold_rule == "permutation"


def test_modulo_rule_partitions_and_is_stable_when_bodies_are_added():
    small = run_e5_flow.fold_indices(np.arange(100), 5, 0, "modulo")
    big = run_e5_flow.fold_indices(np.arange(1000), 5, 0, "modulo")
    assert np.array_equal(np.sort(np.concatenate(big)), np.arange(1000)), "빠짐·겹침 없이 전 체형"
    assert [len(f) for f in big] == [200] * 5
    for f_small, f_big in zip(small, big):
        assert set(f_small) <= set(f_big), "체형을 늘려도 같은 체형은 같은 fold"
    # 순열 규칙은 체형 목록이 바뀌면 fold 가 바뀐다 (그래서 학습 곡선에는 modulo 를 쓴다)
    p_small = run_e5_flow.fold_indices(np.arange(100), 5, 0)
    p_big = run_e5_flow.fold_indices(np.arange(1000), 5, 0)
    assert not all(set(a) <= set(b) for a, b in zip(p_small, p_big))
    shifted = run_e5_flow.fold_indices(np.arange(100), 5, 2, "modulo")
    assert np.array_equal(shifted[0], np.arange(100)[(np.arange(100) + 2) % 5 == 0]), "fold = (body_idx + seed) % K"
    gaps = run_e5_flow.fold_indices(np.array([3, 10, 11, 42, 97]), 5, 0, "modulo")
    assert [f.tolist() for f in gaps] == [[10], [11], [42, 97], [3], []], "번호가 비어 있어도 번호로 정한다"
    with pytest.raises(ValueError, match="fold 규칙"):
        run_e5_flow.fold_indices(np.arange(10), 5, 0, "random")


def test_fold_rule_reaches_fold_indices_from_cli(monkeypatch):
    """--fold-rule 이 실제로 fold_indices 까지 전달된다 (런타임 바꿔치기 없이 정식 인자로)."""
    seen = {}

    class _Stop(Exception):
        pass

    def spy(body_idx, folds, seed, rule="permutation"):
        seen.update(folds=folds, seed=seed, rule=rule)
        raise _Stop

    class _Model:
        meta = {"dataset": "unused"}

    from airis.model import predict as pred

    monkeypatch.setattr(run_e5_flow, "fold_indices", spy)
    monkeypatch.setattr(pred, "load_model", lambda *a, **k: _Model())
    monkeypatch.setattr(pred, "default_rescorer", lambda m: (object(), object()))
    monkeypatch.setattr(pd, "read_parquet", lambda p: pd.DataFrame({"body_idx": [0, 1, 2]}))
    with pytest.raises(_Stop):
        run_e5_flow.main(["--model", "m.pt", "--dataset", "d.parquet", "--folds", "5", "--fold-rule", "modulo",
                          "--fold-seed", "3", "--methods", "stub"])
    assert seen == {"folds": 5, "seed": 3, "rule": "modulo"}


# ---------- 실행 계획 ----------

def test_plan_holdout_is_nested_and_never_trains_on_eval_bodies():
    runs = lc.plan_runs(np.repeat(np.arange(1000), 3), [600, 100, 300, 800], holdout=200)
    assert [r["size"] for r in runs] == [100, 300, 600, 800] and all(r["repeat"] == 0 for r in runs)
    test = runs[0]["test_ids"]
    assert np.array_equal(test, np.arange(800, 1000)), "번호가 가장 큰 200체형이 공통 평가 집합"
    for r in runs:
        assert np.array_equal(r["test_ids"], test), "모든 크기가 같은 평가 집합"
        assert not set(r["train_ids"]) & set(test), "평가 체형은 어느 크기의 학습에도 없다"
        assert len(r["train_ids"]) == r["size"]
    for a, b in zip(runs, runs[1:]):
        assert set(a["train_ids"]) <= set(b["train_ids"]), "작은 크기는 큰 크기의 부분 집합"


def test_plan_holdout_repeats_only_small_sizes_and_is_deterministic():
    kw = dict(holdout=200, repeats=3, repeat_max_size=300, seed=7)
    runs = lc.plan_runs(np.arange(1000), [100, 300, 600], **kw)
    assert [(r["size"], r["repeat"]) for r in runs] == [(100, 0), (100, 1), (100, 2), (300, 0), (300, 1),
                                                         (300, 2), (600, 0)]
    r100 = [r["train_ids"] for r in runs if r["size"] == 100]
    assert not np.array_equal(r100[0], r100[1]) and not np.array_equal(r100[1], r100[2]), "반복마다 다른 학습 체형"
    assert all(len(np.unique(t)) == 100 and t.max() < 800 for t in r100), "반복도 평가 체형을 쓰지 않는다"
    again = lc.plan_runs(np.arange(1000), [100, 300, 600], **kw)
    assert all(np.array_equal(a["train_ids"], b["train_ids"]) for a, b in zip(runs, again)), "같은 인자면 같은 분할"
    other = lc.plan_runs(np.arange(1000), [100], holdout=200, repeats=2, seed=8)
    assert not np.array_equal(other[1]["train_ids"], r100[1]), "시드가 다르면 다른 뽑기"
    full = lc.plan_runs(np.arange(1000), [800], holdout=200, repeats=3, repeat_max_size=800)
    assert len(full) == 1, "학습 후보 전부를 쓰는 크기는 다르게 뽑을 수 없으므로 반복하지 않는다"


def test_plan_cv_uses_first_bodies_without_holdout():
    runs = lc.plan_runs(np.arange(1000), [100, 1000], design="cv", repeats=3)
    assert [(r["size"], r["repeat"]) for r in runs] == [(100, 0), (1000, 0)]
    assert all(r["test_ids"] is None for r in runs)
    assert np.array_equal(runs[0]["train_ids"], np.arange(100))


def test_plan_rejects_impossible_sizes():
    with pytest.raises(ValueError, match="학습에 쓸 수 있는 체형 수"):
        lc.plan_runs(np.arange(1000), [900], holdout=200)
    with pytest.raises(ValueError, match="데이터셋 체형 수"):
        lc.plan_runs(np.arange(300), [600], design="cv")
    with pytest.raises(ValueError, match="holdout 체형 수"):
        lc.plan_runs(np.arange(100), [50], holdout=100)
    with pytest.raises(ValueError, match="양수"):
        lc.plan_runs(np.arange(100), [0], holdout=10)


def test_defaults_come_from_existing_scripts():
    """상수를 따로 정하지 않고 기존 스크립트의 기본값을 쓴다."""
    import train_pose_models

    args = lc.build_parser().parse_args(["--dataset", "d.parquet"])
    e5 = run_e5_flow.build_parser().parse_args([])
    tpm = train_pose_models.build_parser().parse_args(["--dataset", "d.parquet"])
    assert (args.n_flow, args.n_knn, args.fold_seed) == (e5.n_flow, e5.n_knn, e5.fold_seed)
    assert (args.steps, args.folds, args.device) == (tpm.steps, tpm.folds, tpm.device)
    assert args.design == "holdout"


# ---------- 단계 연결 (가짜 학습·평가) ----------

def _dataset(tmp_path, n_bodies=60):
    rows = [{"body_idx": b, "scenario": s, "score": 0.5, "physics_hash": "p0", "nozzle_layout_hash": "n0"}
            for b in range(n_bodies) for s in ("default", "wheelchair")]
    path = tmp_path / "ds.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


@pytest.fixture
def fake_pipeline(monkeypatch):
    """학습과 평가를 가짜로 바꾼다. 점수 비율은 학습 체형 수가 늘수록 좋아지게 한다."""
    import train_pose_models
    from airis.model import predict as pred

    calls = []

    def fake_train(train_df, tag, work, args):
        calls.append(("train", tag, sorted(train_df["body_idx"].unique().tolist())))
        return work / f"{tag}_flow.pt", work / f"{tag}_knn.parquet", 1.5

    def fake_holdout(train_df, test_df, flow_path, knn_path, methods, args):
        calls.append(("eval", sorted(test_df["body_idx"].unique().tolist())))
        ratio = 1.0 - 1.0 / train_df["body_idx"].nunique()
        return pd.DataFrame([{"body_idx": int(r["body_idx"]), "scenario": r["scenario"], "method": m,
                              "ratio": ratio, "infeasible": False, "n_evals": 18, "ms": 500.0, "boundary": False}
                             for _, r in test_df.iterrows() for m in methods])

    def fake_cv(tag, flow_path, work, methods, args):
        calls.append(("cv", tag))
        return pd.DataFrame([{"body_idx": 0, "scenario": "default", "method": m, "ratio": 1.0,
                              "infeasible": False, "n_evals": 18, "ms": 500.0, "boundary": False, "fold": 0}
                             for m in methods])

    monkeypatch.setattr(lc, "train_artifacts", fake_train)
    monkeypatch.setattr(lc, "evaluate_holdout", fake_holdout)
    monkeypatch.setattr(lc, "evaluate_cv", fake_cv)
    monkeypatch.setattr(train_pose_models, "trained_device", lambda p: "cpu")
    stamp = {"nozzle_layout_hash": "n0", "physics_hash": "p0"}
    monkeypatch.setattr(train_pose_models, "current_stamp", lambda: stamp)
    monkeypatch.setattr(pred, "current_stamp", lambda: stamp)
    return calls


def test_holdout_run_writes_summary_and_manifest(fake_pipeline, tmp_path):
    ds, out = _dataset(tmp_path), tmp_path / "out"
    rc = lc.main(["--dataset", str(ds), "--sizes", "10,30", "--holdout", "20", "--repeats", "2",
                  "--repeat-max-size", "10", "--methods", "hybrid,knn+stub", "--out-dir", str(out)])
    assert rc == 0
    evals = [c for c in fake_pipeline if c[0] == "eval"]
    assert len(evals) == 3 and all(e[1] == list(range(40, 60)) for e in evals), "매번 같은 평가 체형"
    trains = [c for c in fake_pipeline if c[0] == "train"]
    assert [t[1] for t in trains] == ["N0010_r0", "N0010_r1", "N0030_r0"]
    assert all(max(t[2]) < 40 for t in trains), "평가 체형은 학습에 없다"

    summary = pd.read_csv(out / "learning_curve_summary.csv")
    assert set(summary["method"]) == {"hybrid", "knn+stub"} and len(summary) == 6
    assert {"design", "size", "repeat", "p05", "below_095", "infeasible", "median", "mean_s",
            "n_train_bodies", "train_s", "eval_s"} <= set(summary.columns)
    by_size = summary[summary["method"] == "hybrid"].groupby("size")["p05"].mean()
    assert by_size[30] > by_size[10], "가짜 평가는 체형이 늘수록 좋아진다 → 요약에 그대로"
    assert (summary["n"] == 40).all(), "평가 행 = 체형 20 × 시나리오 2"
    rows = pd.read_csv(out / "rows_N0010_r1.csv")
    assert (rows["size"] == 10).all() and (rows["repeat"] == 1).all() and (rows["design"] == "holdout").all()

    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert "3ab035e" in m["source"] and m["design"] == "holdout" and m["fold_rule"] is None
    assert m["holdout_bodies"] == {"n": 20, "body_idx_min": 40, "body_idx_max": 59}
    assert m["dataset_stamp"]["physics_hash"] == "p0" and m["current_stamp"]["physics_hash"] == "p0"
    assert m["devices"] == ["cpu"] and m["sizes"] == [10, 30] and m["commit"]
    assert [(r["size"], r["repeat"]) for r in m["runs"]] == [(10, 0), (10, 1), (30, 0)]


def test_cv_design_and_resume(fake_pipeline, tmp_path):
    ds, out = _dataset(tmp_path), tmp_path / "out"
    assert lc.main(["--dataset", str(ds), "--design", "cv", "--sizes", "20,60", "--out-dir", str(out)]) == 0
    assert [c[1] for c in fake_pipeline if c[0] == "cv"] == ["N0020_r0", "N0060_r0"]
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["design"] == "cv" and m["fold_rule"] == "modulo" and m["holdout_bodies"] is None

    fake_pipeline.clear()
    assert lc.main(["--dataset", str(ds), "--design", "cv", "--sizes", "20,60", "--out-dir", str(out),
                    "--resume"]) == 0
    assert fake_pipeline == [], "--resume 이면 이미 있는 (크기, 반복)은 다시 돌리지 않는다"
    assert len(pd.read_csv(out / "learning_curve_summary.csv")) == 6


def test_bad_inputs_return_2(fake_pipeline, tmp_path, capsys):
    ds = _dataset(tmp_path)
    assert lc.main(["--dataset", str(ds), "--sizes", "50", "--holdout", "20", "--out-dir", str(tmp_path / "o")]) == 2
    assert "학습에 쓸 수 있는 체형 수" in capsys.readouterr().err
    assert lc.main(["--dataset", str(ds), "--sizes", "10", "--holdout", "20", "--methods", "nope",
                    "--out-dir", str(tmp_path / "o")]) == 2
    assert fake_pipeline == [] and not (tmp_path / "o").exists(), "잘못된 입력이면 아무것도 만들지 않는다"
