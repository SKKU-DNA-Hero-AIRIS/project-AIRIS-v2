"""산출물 갱신 명령(scripts/train_pose_models.py) 테스트. 소유자: F.

학습·5-fold 자체는 각 스크립트 테스트가 본다. 여기서는 판정 계산, 설치(불합격이면 덮지 않음), 도장 확인,
단계 순서를 가짜 단계로 확인한다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_pose_models as tpm                                # noqa: E402


def _rows(method, ratios, infeasible=None, ms=500.0, boundary=None):
    n = len(ratios)
    return pd.DataFrame({
        "method": method, "ratio": ratios, "n_evals": 18,
        "infeasible": infeasible if infeasible is not None else [False] * n,
        "ms": ms, "boundary": boundary if boundary is not None else [False] * n,
    })


def test_stats_pool_scenarios_and_judge_thresholds():
    good = np.r_[np.full(99, 1.0), 0.94]                     # 0.95 미만 1%
    res = pd.concat([_rows("hybrid", good), _rows("stub", np.r_[np.full(80, 1.0), np.full(20, 0.9)])])
    stats = tpm.method_stats(res)
    h = stats[stats["method"] == "hybrid"].iloc[0]
    assert h["n"] == 100 and h["below_095"] == pytest.approx(0.01) and h["mean_s"] == pytest.approx(0.5)
    v = tpm.judge(stats)
    assert v["passed"] and all(c[2] for c in v["checks"].values())

    assert not tpm.judge(stats, method="stub")["passed"]                        # 0.95 미만 20%
    slow = tpm.method_stats(_rows("hybrid", good, ms=1600.0))
    assert not tpm.judge(slow)["passed"] and not tpm.judge(slow)["checks"]["mean_s"][2]
    bad = tpm.method_stats(_rows("hybrid", good, infeasible=[True] + [False] * 99))
    assert not tpm.judge(bad)["checks"]["infeasible"][2]
    assert not tpm.judge(tpm.method_stats(_rows("knn", good)))["passed"], "혼합 결과가 없으면 불합격"


def test_markdown_marks_gate_method():
    stats = tpm.method_stats(pd.concat([_rows("hybrid", np.ones(20)), _rows("stub", np.ones(20))]))
    md = tpm.markdown_table(stats, tpm.judge(stats), {"dataset": "d.parquet", "rows": 20, "commit": "abc",
                                                       "folds": 5, "out_dir": "outputs/x", "stamp_text": "k=v"})
    assert "**혼합 (flow 8 + kNN 8 + 고정 2)**" in md and "**합격**" in md
    assert md.count("\n| ") == 3                                              # 머리행 + 방법 2개


def test_check_dataset_rejects_mixed_stamps_and_warns_on_hash(monkeypatch):
    monkeypatch.setattr(tpm, "current_stamp", lambda: {"nozzle_layout_hash": "n1", "physics_hash": "p1"})
    ok = pd.DataFrame({"physics_hash": ["p0", "p0"], "nozzle_layout_hash": ["n1", "n1"], "body_model": "mesh"})
    stamp, warns = tpm.check_dataset(ok)
    assert stamp["physics_hash"] == "p0" and len(warns) == 1 and "physics_hash" in warns[0]
    with pytest.raises(ValueError, match="physics_hash"):
        tpm.check_dataset(ok.assign(physics_hash=["p0", "p2"]))


def test_install_keeps_previous(tmp_path):
    src, dst = tmp_path / "new", tmp_path / "models"
    src.mkdir(), dst.mkdir()
    for name in (tpm.FLOW_FILE, tpm.KNN_FILE):
        (src / name).write_text("new")
    (dst / tpm.FLOW_FILE).write_text("old")
    tpm.install(src, dst)
    assert (dst / tpm.FLOW_FILE).read_text() == "new" and (dst / tpm.KNN_FILE).read_text() == "new"
    assert (dst / (tpm.FLOW_FILE + ".prev")).read_text() == "old"
    assert not list(dst.glob("*.new"))


def test_install_failure_keeps_old_artifacts(tmp_path):
    """두 번째 파일 복사가 실패하면 첫 번째도 바꾸지 않는다 (flow·kNN 이 섞이지 않게)."""
    src, dst = tmp_path / "new", tmp_path / "models"
    src.mkdir(), dst.mkdir()
    (src / tpm.FLOW_FILE).write_text("new")                  # kNN 표가 없어 두 번째 복사가 실패한다
    for name in (tpm.FLOW_FILE, tpm.KNN_FILE):
        (dst / name).write_text("old")
    with pytest.raises(FileNotFoundError):
        tpm.install(src, dst)
    assert (dst / tpm.FLOW_FILE).read_text() == "old" and (dst / tpm.KNN_FILE).read_text() == "old"
    assert not list(dst.glob("*.new")) and not list(dst.glob("*.prev"))


@pytest.fixture
def fake_steps(monkeypatch, tmp_path):
    """학습·표·5-fold 를 가짜로 바꾼다. ratios 로 5-fold 결과를 정한다."""
    import build_pose_knn
    import run_e5_flow
    import train_pose_flow

    calls: list[str] = []
    state = {"ratios": np.ones(40)}

    def _out(argv):
        return Path(argv[argv.index("--out") + 1])

    def fake_train(argv):
        calls.append("train")
        assert "--holdout-frac" in argv and argv[argv.index("--holdout-frac") + 1] == "0"
        _out(argv).write_text("flow")
        return 0

    def fake_knn(argv):
        calls.append("knn")
        _out(argv).write_text("knn")
        return 0

    def fake_cv(argv):
        calls.append("cv")
        assert argv[argv.index("--model") + 1].endswith(tpm.FLOW_FILE)
        assert argv[argv.index("--folds") + 1] == "5"
        _rows("hybrid", state["ratios"]).assign(scenario="default").to_csv(_out(argv), index=False)
        return 0

    monkeypatch.setattr(train_pose_flow, "main", fake_train)
    monkeypatch.setattr(build_pose_knn, "main", fake_knn)
    monkeypatch.setattr(run_e5_flow, "main", fake_cv)
    monkeypatch.setattr(tpm, "current_stamp", lambda: {"nozzle_layout_hash": "n0", "physics_hash": "p0"})
    ds = tmp_path / "ds.parquet"
    pd.DataFrame({"body_idx": [0, 1], "physics_hash": "p0", "nozzle_layout_hash": "n0"}).to_parquet(ds)
    return calls, state, ds


def test_refresh_installs_only_when_gate_passes(fake_steps, tmp_path):
    calls, state, ds = fake_steps
    models = tmp_path / "models"
    assert tpm.main(["--dataset", str(ds), "--out-dir", str(tmp_path / "ok"), "--model-dir", str(models)]) == 0
    assert calls == ["train", "knn", "cv"]
    assert (models / tpm.FLOW_FILE).read_text() == "flow"
    report = json.loads((tmp_path / "ok" / "gate.json").read_text(encoding="utf-8"))
    assert report["verdict"]["passed"] and report["installed"]
    assert (tmp_path / "ok" / "gate.md").exists()

    (models / tpm.FLOW_FILE).write_text("working")
    state["ratios"] = np.r_[np.ones(30), np.full(10, 0.9)]                   # 0.95 미만 25%
    assert tpm.main(["--dataset", str(ds), "--out-dir", str(tmp_path / "bad"), "--model-dir", str(models)]) == 1
    assert (models / tpm.FLOW_FILE).read_text() == "working", "불합격 모델은 설치하지 않는다"

    state["ratios"] = np.ones(40)
    assert tpm.main(["--dataset", str(ds), "--out-dir", str(tmp_path / "repro"), "--model-dir", str(models),
                     "--no-install"]) == 0
    assert (models / tpm.FLOW_FILE).read_text() == "working", "--no-install 이면 합격해도 두지 않는다"


def test_skip_cv_requires_explicit_install_choice(fake_steps, tmp_path):
    calls, _, ds = fake_steps
    assert tpm.main(["--dataset", str(ds), "--skip-cv", "--out-dir", str(tmp_path / "o")]) == 2
    assert tpm.main(["--dataset", str(ds), "--skip-cv", "--install", "--out-dir", str(tmp_path / "o"),
                     "--model-dir", str(tmp_path / "m")]) == 0
    assert "cv" not in calls and (tmp_path / "m" / tpm.KNN_FILE).exists()
