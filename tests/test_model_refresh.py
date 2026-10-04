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


def _dirs(tmp_path, *, prev=False):
    src, dst = tmp_path / "new", tmp_path / "models"
    src.mkdir(), dst.mkdir()
    for name in (tpm.FLOW_FILE, tpm.KNN_FILE):
        (src / name).write_text("new")
        (dst / name).write_text("old")
        if prev:
            (dst / (name + ".prev")).write_text("older")
    return src, dst


def _state(dst):
    return {p.name: p.read_text() for p in sorted(dst.iterdir())}


def test_install_partial_copy_is_cleaned_up(tmp_path, monkeypatch):
    """쓰는 도중 실패(디스크 부족·Ctrl+C)해도 부분 .new 가 남지 않고 기존 산출물은 그대로다."""
    src, dst = _dirs(tmp_path, prev=True)
    before = _state(dst)
    real = tpm.shutil.copy2

    def flaky(a, b, *args, **kw):
        if str(b).endswith(tpm.KNN_FILE + ".new"):
            Path(b).write_text("partial")
            raise KeyboardInterrupt
        return real(a, b, *args, **kw)

    monkeypatch.setattr(tpm.shutil, "copy2", flaky)
    with pytest.raises(KeyboardInterrupt):
        tpm.install(src, dst)
    assert _state(dst) == before


def test_install_backup_failure_changes_nothing(tmp_path, monkeypatch):
    """옛 파일을 .prev.new 로 복사하다 실패하면 아무것도 바뀌지 않는다 (.prev 세대도 그대로)."""
    src, dst = _dirs(tmp_path, prev=True)
    before = _state(dst)
    real = tpm.shutil.copy2

    def flaky(a, b, *args, **kw):
        if str(b).endswith(tpm.KNN_FILE + ".prev.new"):
            raise OSError("디스크 부족")
        return real(a, b, *args, **kw)

    monkeypatch.setattr(tpm.shutil, "copy2", flaky)
    with pytest.raises(OSError):
        tpm.install(src, dst)
    assert _state(dst) == before


@pytest.mark.parametrize("had_old", [True, False])
def test_install_replace_failure_rolls_back(tmp_path, monkeypatch, had_old):
    """두 번째 교체가 실패하면(대시보드가 파일을 열고 있음 등) 첫 번째 교체를 되돌린다. 새 flow + 옛 kNN 이 남지 않는다."""
    src, dst = _dirs(tmp_path, prev=True)
    if not had_old:
        for p in dst.iterdir():
            p.unlink()
    before = _state(dst)
    real = tpm.os.replace

    def locked(a, b):
        if str(b).endswith(tpm.KNN_FILE):
            raise PermissionError("다른 프로세스가 사용 중")
        return real(a, b)

    monkeypatch.setattr(tpm.os, "replace", locked)
    with pytest.raises(PermissionError):
        tpm.install(src, dst)
    assert _state(dst) == before


def test_install_prev_commit_failure_keeps_pair_consistent(tmp_path, monkeypatch):
    """모델 교체 뒤 .prev 확정이 실패하면 모델은 새것이고, 세대가 어긋난 .prev 쌍은 남기지 않는다."""
    src, dst = _dirs(tmp_path, prev=True)
    real = tpm.os.replace

    def locked(a, b):
        if str(b).endswith(tpm.KNN_FILE + ".prev"):
            raise PermissionError("잠김")
        return real(a, b)

    monkeypatch.setattr(tpm.os, "replace", locked)
    tpm.install(src, dst)
    assert _state(dst) == {tpm.FLOW_FILE: "new", tpm.KNN_FILE: "new"}


def test_install_success_rotates_prev_as_pair(tmp_path):
    src, dst = _dirs(tmp_path, prev=True)
    tpm.install(src, dst)
    assert _state(dst) == {tpm.FLOW_FILE: "new", tpm.FLOW_FILE + ".prev": "old",
                           tpm.KNN_FILE: "new", tpm.KNN_FILE + ".prev": "old"}


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


def test_install_double_failure_keeps_backup(tmp_path, monkeypatch, capsys):
    """교체도 되돌리기도 실패하면(두 파일 모두 잠김) 옛 산출물 백업(.prev.new)을 지우지 않고 남긴다."""
    src, dst = _dirs(tmp_path, prev=True)
    real = tpm.os.replace

    def locked(a, b):
        if str(b).endswith(tpm.KNN_FILE) or (str(a).endswith(".prev.new") and str(b).endswith(tpm.FLOW_FILE)):
            raise PermissionError("잠김")
        return real(a, b)

    monkeypatch.setattr(tpm.os, "replace", locked)
    with pytest.raises(tpm.InstallRollbackError) as err:
        tpm.install(src, dst)
    assert isinstance(err.value.__cause__, PermissionError), "원래 예외(교체 실패)가 원인으로 이어진다"
    state = _state(dst)
    assert state[tpm.FLOW_FILE + ".prev.new"] == "old", "되돌리지 못한 옛 flow 는 백업으로 남는다"
    assert state[tpm.KNN_FILE] == "old" and state[tpm.FLOW_FILE + ".prev"] == "older"
    assert not any(k.endswith(".new") and not k.endswith(".prev.new") for k in state)
    assert "되돌리기도 실패" in capsys.readouterr().err


def test_install_ctrl_c_right_after_replace_is_rolled_back(tmp_path, monkeypatch):
    """os.replace 직후(기록 전) Ctrl+C 가 와도 그 파일을 되돌린다."""
    src, dst = _dirs(tmp_path, prev=True)
    before = _state(dst)
    real = tpm.os.replace

    def interrupt_after(a, b):
        real(a, b)
        if str(b).endswith(tpm.FLOW_FILE) and str(a).endswith(".new") and not str(a).endswith(".prev.new"):
            raise KeyboardInterrupt

    monkeypatch.setattr(tpm.os, "replace", interrupt_after)
    with pytest.raises(KeyboardInterrupt):
        tpm.install(src, dst)
    assert _state(dst) == before


def test_install_locked_prev_does_not_raise_and_reports(tmp_path, monkeypatch, capsys):
    """.prev 가 잠겨 확정도 지우기도 안 되면: 모델은 새 쌍, 예외 없음, 남은 묵은 .prev 를 경고에 적는다."""
    src, dst = _dirs(tmp_path, prev=True)
    real_replace, real_unlink = tpm.os.replace, Path.unlink
    knn_prev = dst / (tpm.KNN_FILE + ".prev")

    def locked_replace(a, b):
        if Path(b) == knn_prev:
            raise PermissionError("잠김")
        return real_replace(a, b)

    def locked_unlink(self, *args, **kw):
        if self == knn_prev:
            raise PermissionError("잠김")
        return real_unlink(self, *args, **kw)

    monkeypatch.setattr(tpm.os, "replace", locked_replace)
    monkeypatch.setattr(Path, "unlink", locked_unlink)
    out = tpm.install(src, dst)
    assert [p.name for p in out] == [tpm.FLOW_FILE, tpm.KNN_FILE]
    assert _state(dst) == {tpm.FLOW_FILE: "new", tpm.KNN_FILE: "new", tpm.KNN_FILE + ".prev": "older"}
    assert tpm.KNN_FILE + ".prev" in capsys.readouterr().err


def test_install_ctrl_c_during_prev_commit(tmp_path, monkeypatch):
    """.prev 확정 중 Ctrl+C: 모델은 새 쌍, .prev 는 세대가 섞이지 않게 모두 지우고, Ctrl+C 는 다시 올린다."""
    src, dst = _dirs(tmp_path, prev=True)
    real = tpm.os.replace

    def interrupt(a, b):
        if str(b).endswith(tpm.KNN_FILE + ".prev"):
            raise KeyboardInterrupt
        return real(a, b)

    monkeypatch.setattr(tpm.os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        tpm.install(src, dst)
    assert _state(dst) == {tpm.FLOW_FILE: "new", tpm.KNN_FILE: "new"}


def test_install_drops_stale_prev_when_old_missing(tmp_path):
    """옛 파일이 한쪽만 있으면, 없던 쪽의 묵은 .prev 는 지운다 (.prev 는 늘 같은 세대의 쌍)."""
    src, dst = _dirs(tmp_path, prev=True)
    (dst / tpm.KNN_FILE).unlink()
    tpm.install(src, dst)
    assert _state(dst) == {tpm.FLOW_FILE: "new", tpm.FLOW_FILE + ".prev": "old", tpm.KNN_FILE: "new"}


def test_refresh_records_install_failure(fake_steps, tmp_path, monkeypatch):
    """설치가 실패해도 gate.json 을 남기고 종료 코드 1."""
    calls, _, ds = fake_steps

    def boom(*a, **kw):
        raise PermissionError("대시보드가 열고 있음")

    monkeypatch.setattr(tpm, "install", boom)
    out = tmp_path / "o"
    assert tpm.main(["--dataset", str(ds), "--out-dir", str(out), "--model-dir", str(tmp_path / "m")]) == 1
    report = json.loads((out / "gate.json").read_text(encoding="utf-8"))
    assert report["verdict"]["passed"] and report["installed"] == []
    assert "PermissionError" in report["install_error"]


def test_install_refuses_when_backup_left_over(tmp_path, monkeypatch):
    """이중 실패가 남긴 백업(.prev.new)이 있으면 덮어쓰지 않고 멈춘다 (그대로 재실행하면 옛 flow 를 잃는다)."""
    src, dst = _dirs(tmp_path, prev=True)
    real = tpm.os.replace

    def locked(a, b):
        if str(b).endswith(tpm.KNN_FILE) or (str(a).endswith(".prev.new") and str(b).endswith(tpm.FLOW_FILE)):
            raise PermissionError("잠김")
        return real(a, b)

    monkeypatch.setattr(tpm.os, "replace", locked)
    with pytest.raises(tpm.InstallRollbackError):
        tpm.install(src, dst)
    monkeypatch.setattr(tpm.os, "replace", real)          # 잠김이 풀린 뒤 복구 없이 다시 돌린 경우
    after_failure = _state(dst)
    with pytest.raises(RuntimeError, match="prev.new"):
        tpm.install(src, dst)
    assert _state(dst) == after_failure, "멈출 때 아무것도 바꾸지 않는다"
    assert after_failure[tpm.FLOW_FILE + ".prev.new"] == "old"


def test_install_ctrl_c_during_rollback_keeps_backup(tmp_path, monkeypatch):
    """되돌리는 도중 Ctrl+C 가 와도 백업을 지우지 않고, Ctrl+C 를 올린다."""
    src, dst = _dirs(tmp_path, prev=True)
    real = tpm.os.replace

    def flaky(a, b):
        if str(b).endswith(tpm.KNN_FILE):
            raise PermissionError("잠김")
        if str(a).endswith(".prev.new") and str(b).endswith(tpm.FLOW_FILE):
            raise KeyboardInterrupt
        return real(a, b)

    monkeypatch.setattr(tpm.os, "replace", flaky)
    with pytest.raises(KeyboardInterrupt):
        tpm.install(src, dst)
    state = _state(dst)
    assert state[tpm.FLOW_FILE + ".prev.new"] == "old" and state[tpm.KNN_FILE] == "old"


def test_refresh_records_ctrl_c_during_install(fake_steps, tmp_path, monkeypatch):
    """설치 중 Ctrl+C 도 gate.json 에 남기고(설치 폴더 실제 상태 포함) Ctrl+C 를 올린다."""
    _, _, ds = fake_steps
    models = tmp_path / "m"
    real = tpm.install

    def half_then_interrupt(src, dst, *a, **kw):
        real(src, dst, *a, **kw)                          # 설치는 끝났는데
        raise KeyboardInterrupt                           # 그 뒤 Ctrl+C

    monkeypatch.setattr(tpm, "install", half_then_interrupt)
    out = tmp_path / "o"
    with pytest.raises(KeyboardInterrupt):
        tpm.main(["--dataset", str(ds), "--out-dir", str(out), "--model-dir", str(models)])
    report = json.loads((out / "gate.json").read_text(encoding="utf-8"))
    assert report["install_error"].startswith("KeyboardInterrupt")
    assert report["install_state"] == {tpm.FLOW_FILE: "new", tpm.KNN_FILE: "new"}


def test_check_dataset_ignores_kinetics_for_pose(monkeypatch):
    """자세 데이터셋의 kinetics 열(False)은 계획 설정(True)과 달라도 경고하지 않는다."""
    monkeypatch.setattr(tpm, "current_stamp", lambda: {"nozzle_layout_hash": "n0", "physics_hash": "p0",
                                                       "kinetics_enabled": True, "time_constant_s": 2.0})
    from airis.model import knn

    monkeypatch.setattr(knn, "STAMP_COLS", knn.STAMP_COLS + ("kinetics_enabled",))   # 도장 열이 늘어난 경우
    df = pd.DataFrame({"physics_hash": ["p0"], "nozzle_layout_hash": ["n0"], "kinetics_enabled": [False]})
    stamp, warns = tpm.check_dataset(df)
    assert stamp["kinetics_enabled"] == "False" and warns == []
    _, warns = tpm.check_dataset(df.assign(physics_hash=["p9"]))
    assert len(warns) == 1 and warns[0].startswith("physics_hash")


def test_refresh_stops_early_when_backup_left_over(fake_steps, tmp_path):
    """설치 폴더에 .prev.new 가 남아 있으면 학습·5-fold 전에 멈춘다. --no-install 이면 그대로 돈다."""
    calls, _, ds = fake_steps
    models = tmp_path / "m"
    models.mkdir()
    (models / (tpm.FLOW_FILE + ".prev.new")).write_text("old")
    assert tpm.main(["--dataset", str(ds), "--out-dir", str(tmp_path / "o"), "--model-dir", str(models)]) == 2
    assert calls == [], "학습도 5-fold 도 시작하지 않는다"
    assert tpm.main(["--dataset", str(ds), "--out-dir", str(tmp_path / "o2"), "--model-dir", str(models),
                     "--no-install"]) == 0
    assert calls == ["train", "knn", "cv"]


def test_check_dataset_allows_mixed_commit_but_not_mixed_config(monkeypatch):
    """commit 은 출처 기록이라 여러 값이어도 진행한다 (생성 중단·재개). 설정 해시가 섞이면 여전히 거부한다."""
    monkeypatch.setattr(tpm, "current_stamp", lambda: {"nozzle_layout_hash": "n0", "physics_hash": "p0"})
    df = pd.DataFrame({"physics_hash": ["p0"] * 5, "nozzle_layout_hash": ["n0"] * 5, "body_model": "mesh",
                       "commit": ["aaa", "aaa", "aaa", "bbb", "bbb"]})
    stamp, warns = tpm.check_dataset(df)
    assert stamp["commit"] == "aaa+bbb", "많은 순으로 전부 적는다"
    assert stamp["physics_hash"] == "p0" and stamp["body_model"] == "mesh"
    assert len(warns) == 1 and warns[0].startswith("commit") and "'aaa': 3" in warns[0]
    with pytest.raises(ValueError, match="physics_hash"):
        tpm.check_dataset(df.assign(physics_hash=["p0", "p0", "p0", "p1", "p1"]))
    with pytest.raises(ValueError, match="body_model"):
        tpm.check_dataset(df.assign(body_model=["mesh"] * 4 + ["capsule"]))


def test_knn_table_accepts_mixed_commit(tmp_path, capsys):
    """kNN 표 생성도 commit 이 섞인 데이터셋을 받고, 표 meta 에 전부 적는다."""
    import build_pose_knn
    from airis.model.flow import BODY_KEYS, POSE_KEYS
    from airis.model.knn import PoseKNN
    from airis.sim import BodyParams, PoseParams

    rows = []
    for i in range(6):
        r = {"body_idx": i, "scenario": "default", "score": 0.5, "arm_class": "hands_up",
             "nozzle_layout_hash": "n0", "physics_hash": "p0", "body_model": "mesh", "patches_per_m2": 1500.0,
             "commit": "aaa" if i < 4 else "bbb"}
        r.update({f"body_{k}": getattr(BodyParams(), k) + 0.01 * i for k in BODY_KEYS})
        r.update({f"pose_{k}": getattr(PoseParams(), k) for k in POSE_KEYS})
        rows.append(r)
    df = pd.DataFrame(rows)
    ds, out = tmp_path / "ds.parquet", tmp_path / "knn.parquet"
    df.to_parquet(ds)
    assert build_pose_knn.main(["--dataset", str(ds), "--out", str(out)]) == 0
    assert "commit" in capsys.readouterr().out
    table = PoseKNN.load(out)
    assert table.meta["commit"] == "aaa+bbb" and table.meta["physics_hash"] == "p0"
    df.assign(physics_hash=["p0"] * 5 + ["p1"]).to_parquet(ds)
    assert build_pose_knn.main(["--dataset", str(ds), "--out", str(tmp_path / "bad.parquet")]) == 2
    assert not (tmp_path / "bad.parquet").exists(), "설정 해시가 섞이면 여전히 거부한다"


def _finished_run(tmp_path, passed=True):
    """--no-install 로 끝난 실행의 결과 폴더."""
    run = tmp_path / "run"
    run.mkdir(parents=True)
    for name in (tpm.FLOW_FILE, tpm.KNN_FILE):
        (run / name).write_text("new")
    (run / "gate.json").write_text(json.dumps({"verdict": {"method": "hybrid", "passed": passed, "checks": {}},
                                               "installed": [], "device": "cpu"}), encoding="utf-8")
    return run


def test_install_from_installs_finished_run_only_when_gate_passed(tmp_path, capsys):
    """5-fold 뒤 따로 설치: 합격한 실행만 설치하고 결과를 gate.json 에 덧붙인다. 데이터셋·학습은 필요 없다."""
    run, models = _finished_run(tmp_path), tmp_path / "models"
    models.mkdir()
    (models / tpm.FLOW_FILE).write_text("old")
    assert tpm.main(["--install-from", str(run), "--model-dir", str(models)]) == 0
    assert _state(models) == {tpm.FLOW_FILE: "new", tpm.FLOW_FILE + ".prev": "old", tpm.KNN_FILE: "new"}
    report = json.loads((run / "gate.json").read_text(encoding="utf-8"))
    assert len(report["installed"]) == 2 and report["install_error"] is None and report["installed_at"]
    assert report["install_forced"] is False and report["verdict"]["passed"] and report["device"] == "cpu"

    failed = _finished_run(tmp_path / "f", passed=False)
    before = _state(models)
    assert tpm.main(["--install-from", str(failed), "--model-dir", str(models)]) == 1
    assert _state(models) == before and "합격이 아니라" in capsys.readouterr().err
    assert tpm.main(["--install-from", str(failed), "--model-dir", str(models), "--install"]) == 0
    assert json.loads((failed / "gate.json").read_text(encoding="utf-8"))["install_forced"] is True


def test_install_from_rejects_bad_inputs(tmp_path, capsys):
    run, models = _finished_run(tmp_path), tmp_path / "models"
    models.mkdir()
    (models / (tpm.KNN_FILE + ".prev.new")).write_text("old")          # 앞선 설치의 남은 백업
    assert tpm.main(["--install-from", str(run), "--model-dir", str(models)]) == 2
    assert "prev.new" in capsys.readouterr().err and not (models / tpm.FLOW_FILE).exists()
    (run / tpm.KNN_FILE).unlink()
    assert tpm.main(["--install-from", str(run), "--model-dir", str(tmp_path / "m2")]) == 2
    assert tpm.KNN_FILE in capsys.readouterr().err
    assert tpm.main(["--model-dir", str(tmp_path / "m3")]) == 2, "--dataset 도 --install-from 도 없으면 거부"


def test_refresh_records_commit_taken_at_start(fake_steps, tmp_path, monkeypatch):
    """실행 중에 HEAD 가 바뀌어도(같은 작업 폴더에서 커밋·병합) 시작할 때의 커밋을 기록한다."""
    _, _, ds = fake_steps
    heads = iter(["start111", "later222", "later333"])
    monkeypatch.setattr(tpm.explog, "git_commit", lambda: next(heads))
    out = tmp_path / "o"
    assert tpm.main(["--dataset", str(ds), "--out-dir", str(out), "--no-install"]) == 0
    assert json.loads((out / "gate.json").read_text(encoding="utf-8"))["commit"] == "start111"
    assert "커밋 `start111`" in (out / "gate.md").read_text(encoding="utf-8")
