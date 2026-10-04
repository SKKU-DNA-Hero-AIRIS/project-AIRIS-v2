"""체형 입력 오차 내성 평가(scripts/run_e5_body_noise.py, run_e5_flow.py --body-noise) 테스트. 소유자: F.

핵심 규약: 추천(후보 생성·재채점)은 오차를 넣은 추정 체형으로, 고른 자세의 채점은 진짜 체형으로 한다.
"""
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_e5_body_noise as bn                                 # noqa: E402
import run_e5_flow                                             # noqa: E402
from airis.sim import BodyParams, EvalResult, Evaluator        # noqa: E402
from airis.sim.scenario import load_scenarios                  # noqa: E402

BODY_KEYS = [f.name for f in fields(BodyParams)]


def test_estimated_body_is_bounded_deterministic_and_identity_at_zero():
    body = BodyParams()
    assert run_e5_flow.estimated_body(body, 0.0, 0, 3, "default") is body, "오차 0 이면 진짜 체형 그대로"
    est = run_e5_flow.estimated_body(body, 0.05, 0, 3, "default")
    ratios = np.array([getattr(est, k) / getattr(body, k) for k in BODY_KEYS])
    assert np.all(np.abs(ratios - 1.0) <= 0.05 + 1e-12) and np.any(np.abs(ratios - 1.0) > 1e-4)
    assert len(set(np.round(ratios, 9))) > 1, "체형 값마다 독립으로 흔든다"
    assert est == run_e5_flow.estimated_body(body, 0.05, 0, 3, "default"), "같은 인자면 같은 추정 체형"
    for other in (dict(seed=1), dict(body_idx=4), dict(scenario="wheelchair"), dict(noise=0.10)):
        kw = dict(noise=0.05, seed=0, body_idx=3, scenario="default") | other
        assert run_e5_flow.estimated_body(body, kw["noise"], kw["seed"], kw["body_idx"], kw["scenario"]) != est
    many = [run_e5_flow.estimated_body(body, 0.10, 0, i, "default").height_m / body.height_m for i in range(400)]
    assert 0.90 <= min(many) < 0.92 and 1.08 < max(many) <= 1.10, "±10% 범위를 고르게 쓴다"
    assert abs(np.mean(many) - 1.0) < 0.01


class _HeightEvaluator(Evaluator):
    """채점에 쓰인 체형을 기록한다. 점수는 자세만으로 정해진다."""

    def __init__(self):
        self.heights: list[float] = []

    def evaluate(self, pose, nozzle, body, scenario):
        self.heights.append(body.height_m)
        return EvalResult(score=0.5 + pose.shoulder_abduction / 1000.0, removal_by_part=np.zeros(5),
                          total_removal=0.0, discomfort=0.0, extra={})


def _test_rows(n=4):
    rows = []
    for i in range(n):
        r = {"body_idx": i, "scenario": "default", "score": 0.6, "arm_class": "hands_up"}
        r.update({f"body_{k}": getattr(BodyParams(), k) for k in BODY_KEYS})
        rows.append(r)
    return pd.DataFrame(rows)


def test_run_split_recommends_with_estimate_and_scores_with_truth():
    """stub 방법: 후보 재채점은 추정 체형으로, 마지막 채점은 진짜 체형으로 한다."""
    scenarios = load_scenarios()
    true_h = BodyParams().height_m
    test = _test_rows()
    for noise in (0.0, 0.05):
        ev = _HeightEvaluator()
        args = run_e5_flow.build_parser().parse_args(["--body-noise", str(noise), "--body-noise-seed", "2"])
        rows = run_e5_flow.run_split(test, test, None, ["stub"], args, ev, object(), scenarios)
        per_row = len(ev.heights) // len(test)                  # 후보 재채점 n 번 + 진짜 체형 채점 1 번
        assert per_row >= 2 and len(ev.heights) == per_row * len(test)
        for i, r in enumerate(rows):
            calls = ev.heights[i * per_row:(i + 1) * per_row]
            assert calls[-1] == true_h, "고른 자세의 점수는 진짜 체형으로 잰다"
            assert r["body_noise"] == noise and r["height_m"] == true_h
            if noise == 0.0:
                assert all(h == true_h for h in calls) and r["est_height_m"] == true_h
            else:
                assert all(h == r["est_height_m"] for h in calls[:-1]), "추천(재채점)은 추정 체형으로 한다"
                assert r["est_height_m"] != true_h and abs(r["est_height_m"] / true_h - 1) <= 0.05 + 1e-12
        assert all(r["ratio"] == pytest.approx(r["score"] / 0.6) for r in rows)


def test_parse_levels_always_includes_zero():
    assert bn.parse_levels("0.05,0.02") == [0.0, 0.02, 0.05]
    assert bn.parse_levels("0,0.1,0.1") == [0.0, 0.1]
    assert bn.build_parser().parse_args(["--dataset", "d", "--model", "m"]).levels == "0,0.02,0.05,0.10"
    for bad in ("-0.01", "1.0", "0.05,2"):
        with pytest.raises(ValueError, match="오차 비율"):
            bn.parse_levels(bad)


def _rows(level, ratios, infeasible=None, method="hybrid"):
    n = len(ratios)
    return pd.DataFrame({"body_idx": range(n), "scenario": "default", "method": method, "body_noise": level,
                         "ratio": ratios, "infeasible": infeasible if infeasible is not None else [False] * n,
                         "n_evals": 18, "ms": 500.0, "boundary": False})


def test_summary_reports_change_against_zero_noise():
    base = np.full(100, 1.0)
    noisy = base.copy()
    noisy[:10] = 0.90                                           # 10행이 0.10 씩 떨어진다
    bad = [True] * 3 + [False] * 97
    rows = pd.concat([_rows(0.0, base), _rows(0.05, noisy, bad), _rows(0.0, base, method="stub"),
                      _rows(0.05, base, method="stub")])
    s = bn.summarize(rows)
    h0 = s[(s["method"] == "hybrid") & (s["body_noise"] == 0.0)].iloc[0]
    h5 = s[(s["method"] == "hybrid") & (s["body_noise"] == 0.05)].iloc[0]
    assert h0["d_p05"] == 0 and h0["paired_ratio_mean"] == 0 and h0["rows_changed"] == 0
    assert h5["below_095"] == pytest.approx(0.10) and h5["d_below_095"] == pytest.approx(0.10)
    assert h5["infeasible"] == pytest.approx(0.03) and h5["d_infeasible"] == pytest.approx(0.03)
    assert h5["paired_ratio_mean"] == pytest.approx(-0.01) and h5["paired_ratio_min"] == pytest.approx(-0.10)
    assert h5["rows_changed"] == pytest.approx(0.10)
    s5 = s[(s["method"] == "stub") & (s["body_noise"] == 0.05)].iloc[0]
    assert s5["paired_ratio_mean"] == 0 and s5["rows_changed"] == 0, "방법마다 자기 기준(p = 0)과 비교한다"
    md = bn.markdown_table(s, {"dataset": "d.parquet", "rows": 300, "folds": 5, "commit": "abc",
                               "noise_seed": 0, "device": "cpu"})
    assert "| hybrid | ±5% |" in md and "| stub | ±0% |" in md and md.count("\n| ") == 5


def test_main_runs_every_level_on_the_same_fold_models(tmp_path, monkeypatch):
    """fold 마다 학습은 한 번, 오차 수준만 바꿔 평가한다. 결과 파일과 manifest."""
    import json

    import train_pose_models
    from airis.model import knn, predict as pred

    calls = {"train": 0, "levels": []}

    class _Model:
        meta = {"device": "cpu"}

        def save(self, path):
            return Path(path)

    def fake_train(train, scenarios, base, meta):
        calls["train"] += 1
        return _Model()

    def fake_split(test, train, model, methods, args, evaluator, nozzle, scenarios, *, fold=None,
                   knn_path=None, model_path=None):
        calls["levels"].append((fold, args.body_noise, args.body_noise_seed))
        return [{"body_idx": int(r["body_idx"]), "scenario": r["scenario"], "method": m, "body_noise": args.body_noise,
                 "ratio": 1.0 - args.body_noise, "infeasible": False, "n_evals": 18, "ms": 500.0, "boundary": False,
                 "fold": fold} for _, r in test.iterrows() for m in methods]

    class _Knn:
        def save(self, path):
            return Path(path)

    monkeypatch.setattr(run_e5_flow, "train_fold_model", fake_train)
    monkeypatch.setattr(run_e5_flow, "run_split", fake_split)
    monkeypatch.setattr(pred, "load_model", lambda *a, **k: _Model())
    monkeypatch.setattr(pred, "default_rescorer", lambda m: (object(), object()))
    monkeypatch.setattr(knn.PoseKNN, "from_dataset", classmethod(lambda cls, df: _Knn()))
    stamp = {"nozzle_layout_hash": "n0", "physics_hash": "p0"}
    monkeypatch.setattr(train_pose_models, "current_stamp", lambda: stamp)
    monkeypatch.setattr(pred, "current_stamp", lambda: stamp)

    ds = tmp_path / "ds.parquet"
    pd.DataFrame([{"body_idx": b, "scenario": s, "score": 0.5, "physics_hash": "p0", "nozzle_layout_hash": "n0"}
                  for b in range(10) for s in ("default", "wheelchair")]).to_parquet(ds)
    out = tmp_path / "out"
    rc = bn.main(["--dataset", str(ds), "--model", "m.pt", "--levels", "0.05,0.10", "--noise-seed", "7",
                  "--methods", "hybrid,stub", "--out-dir", str(out)])
    assert rc == 0 and calls["train"] == 5, "fold 마다 학습 한 번"
    assert calls["levels"] == [(f, lv, 7) for f in range(5) for lv in (0.0, 0.05, 0.10)], "0 은 항상 포함"
    rows = pd.read_csv(out / "rows.csv")
    assert len(rows) == 20 * 2 * 3                              # 행 20 × 방법 2 × 수준 3
    s = pd.read_csv(out / "summary.csv")
    assert len(s) == 6 and set(s["body_noise"].round(2)) == {0.0, 0.05, 0.10}
    assert s[s["body_noise"] == 0.10]["paired_ratio_mean"].iloc[0] == pytest.approx(-0.10)
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["levels"] == [0.0, 0.05, 0.10] and m["noise_seed"] == 7 and m["devices"] == ["cpu"]
    assert m["dataset_stamp"]["physics_hash"] == "p0" and m["commit"] and (out / "summary.md").exists()
    assert bn.main(["--dataset", str(ds), "--model", "m.pt", "--levels", "1.5", "--out-dir", str(tmp_path / "x")]) == 2
    assert not (tmp_path / "x").exists()
