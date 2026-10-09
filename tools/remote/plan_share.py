"""계획 데이터셋 분담 실행 도구 (다른 노트북에서 사용자 유형 하나를 맡아 돌린다).

사용법은 docs/remote_run.md 에 있다. 요약:

    python tools/remote/plan_share.py setup                 # 가상환경 + 실행 폴더 + 점검
    python tools/remote/plan_share.py bench                 # 이 노트북의 예상 속도
    python tools/remote/plan_share.py run --segment W2       # 구간표(docs/remote_plan_board.md)의 구간 하나
    python tools/remote/plan_share.py status                # 진행 확인
    python tools/remote/plan_share.py stop --yes            # 멈춤 (저장된 행은 남는다)

기준 노트북(총괄)에서 쓰는 것:

    python tools/remote/plan_share.py segments <파일 ...>    # 구간별 완료 행 수
    python tools/remote/plan_share.py export-seed --from <데이터셋> --scenario wheelchair --out <파일>
    python tools/remote/plan_share.py merge --out <합친 파일> <파일 1> <파일 2> ...

이 파일은 표준 라이브러리만으로 시작한다 (setup 이 가상환경을 만든 뒤에는 그 Python 으로 다시 실행한다).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

#: 기준 노트북이 데이터셋을 만들고 있는 커밋. 같은 코드로 만들어야 한 파일로 합칠 수 있다.
PIN_COMMIT = "4f9f310f171e9d5657968f834781c0aeae06600e"
#: 기준 노트북의 설정 식별값. 다르면 설정 파일이 다른 것이라 실행하지 않는다.
EXPECT_HASH = {"physics_hash": "c5aea26d", "nozzle_layout_hash": "3b99200e", "body_model": "mesh"}
#: 기준 노트북에서 뽑힌 체형 0~2번의 키 (체형 샘플러가 같은 값을 내는지 확인).
EXPECT_HEIGHTS = (1.6913157198984052, 1.7125435549418535, 1.6239052983716382)
#: 기준 노트북과 같은 생성 설정 (총괄 채택 10-06). 바꾸면 합칠 수 없다.
RUN_ARGS = ["--n-phases", "9", "--energy-weight", "0.1", "--warm-start-sigma0", "0.2",
            "--max-evals", "12000", "--tol-stagnation-gens", "30", "--pose-max-evals", "3000",
            "--patches-per-m2", "1500", "--candidate-k", "16", "--candidate-min-dist", "0.15"]
#: 몇 행마다 저장할지 (끊겼을 때 잃는 양을 줄이려고 기준 노트북의 20 보다 작게 잡았다).
FLUSH_EVERY = 5
#: 구간표. 이름 → (사용자 유형, 체형 번호 처음, 끝(포함)). docs/remote_plan_board.md 와 같아야 한다.
#: 2026-10-09 기준: 선 자세 0~20, 임산부 0~19, 휠체어 0~18 은 이미 끝나 있어 구간에 넣지 않았다.
SEGMENTS = {
    "D1": ("default", 21, 99),
    "P1": ("pregnant", 20, 39), "P2": ("pregnant", 40, 59),
    "P3": ("pregnant", 60, 79), "P4": ("pregnant", 80, 99),
    "W1": ("wheelchair", 19, 39), "W2": ("wheelchair", 40, 59),
    "W3": ("wheelchair", 60, 79), "W4": ("wheelchair", 80, 99),
}
SCENARIOS = ("default", "pregnant", "wheelchair")
SCENARIO_KO = {"default": "선 자세", "pregnant": "임산부", "wheelchair": "휠체어"}
#: 행 하나의 평가 횟수 (기준 노트북 60행 평균: 계획 11,378 + 단일 자세 3,000).
EVALS_PLAN, EVALS_POSE = 11400, 3000
#: 합칠 때 모든 파일에서 같아야 하는 열 (airis/optimize/plan_dataset.py CONSISTENCY_COLS 와 같다).
CONSISTENCY_COLS = ("nozzle_layout_hash", "physics_hash", "body_seed", "max_evals", "popsize",
                    "patches_per_m2", "evaluator", "body_model", "n_phases", "energy_weight",
                    "kinetics_enabled", "time_constant_s", "zone_nozzle_counts",
                    "duration_lo_s", "duration_hi_s", "min_phase_s",
                    "candidate_k", "candidate_tol", "candidate_min_dist",
                    "warm_start_sigma0", "pose_max_evals")
KEY_COLS = ("body_idx", "scenario")

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
VENV = ROOT / ".venv-remote"
RUN_DIR = ROOT.parent / f"{ROOT.name}-run-{PIN_COMMIT[:7]}"
WORK = ROOT / "outputs" / "remote_share"
STATE = WORK / "run_state.json"
IS_WIN = os.name == "nt"


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")


def say(msg: str = "") -> None:
    print(msg, flush=True)


def fail(msg: str) -> "NoReturn":  # noqa: F821
    say(f"\n[실패] {msg}")
    raise SystemExit(1)


def run(cmd, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], **kw)


def inner_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(RUN_DIR)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def inner(*args: str, timeout: float | None = None) -> dict:
    """실행 폴더 안에서 보조 코드를 돌리고 결과(JSON)를 받는다."""
    got = run([venv_python(), HERE / "_inner.py", *args], cwd=RUN_DIR, env=inner_env(),
              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    for line in got.stdout.splitlines():
        if line.startswith("JSON:"):
            return json.loads(line[5:])
    fail(f"보조 코드가 결과를 내지 않았다 ({' '.join(args)}):\n{got.stdout[-1500:]}\n{got.stderr[-3000:]}")


def out_path(label: str) -> Path:
    return WORK / f"plan_dataset_n9_{label}.parquet"


def parse_bodies(text: str) -> tuple[int, int]:
    try:
        lo, hi = (int(v) for v in text.split("-"))
    except ValueError:
        fail(f"체형 구간은 '40-59' 처럼 적는다: {text!r}")
    if not 0 <= lo <= hi:
        fail(f"체형 구간이 이상하다: {text!r}")
    return lo, hi


def resolve_target(args) -> tuple[str, int, int, str]:
    """(사용자 유형, 체형 처음, 끝, 파일 이름에 쓸 표지)."""
    if getattr(args, "segment", None):
        name = args.segment.upper()
        if name not in SEGMENTS:
            fail(f"없는 구간: {args.segment} (가능: {', '.join(SEGMENTS)})")
        if args.scenario or args.bodies:
            fail("--segment 를 쓰면 --scenario·--bodies 는 주지 않는다.")
        scenario, lo, hi = SEGMENTS[name]
        return scenario, lo, hi, name
    if not args.scenario:
        fail("--segment 또는 --scenario 중 하나를 준다.")
    if args.bodies:
        lo, hi = parse_bodies(args.bodies)
        return args.scenario, lo, hi, f"{args.scenario}_b{lo}-{hi}"
    return args.scenario, 0, args.n_bodies - 1, args.scenario


# --------------------------------------------------------------------------------------- setup

def cmd_setup(args) -> int:
    say(f"[1/4] Python 확인: {sys.version.split()[0]}")
    if sys.version_info < (3, 12):
        fail("Python 3.12 이상이 필요하다 (기준 노트북은 3.13). python.org 에서 3.13 을 설치한 뒤 다시 실행.")
    if sys.version_info[:2] != (3, 13):
        say("      주의: 기준 노트북은 3.13 이다. 패키지 설치가 실패하면 3.13 으로 다시 해 본다.")
    if shutil.which("git") is None:
        fail("git 이 없다. https://git-scm.com 에서 설치한 뒤 다시 실행.")

    say(f"[2/4] 가상환경: {VENV}")
    if not venv_python().exists():
        run([sys.executable, "-m", "venv", VENV], check=True)
    run([venv_python(), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=True)
    got = run([venv_python(), "-m", "pip", "install", "--quiet", "-r", HERE / "requirements-plan.txt"])
    if got.returncode != 0:
        fail("패키지 설치 실패. 위 오류를 총괄에게 보내 달라 (Python 버전·운영체제와 함께).")

    say(f"[3/4] 실행 폴더(고정 커밋 {PIN_COMMIT[:7]}): {RUN_DIR}")
    if not (RUN_DIR / "scripts" / "run_plan_dataset.py").exists():
        has = run(["git", "-C", ROOT, "cat-file", "-e", PIN_COMMIT + "^{commit}"], capture_output=True)
        if has.returncode != 0:
            run(["git", "-C", ROOT, "fetch", "--quiet", "origin"], check=True)
        got = run(["git", "-C", ROOT, "worktree", "add", "--detach", RUN_DIR, PIN_COMMIT])
        if got.returncode != 0:
            fail("실행 폴더를 만들지 못했다. 저장소를 zip 이 아니라 git clone 으로 받았는지 확인.")
    WORK.mkdir(parents=True, exist_ok=True)

    say("[4/4] 점검")
    return subprocess.call([str(venv_python()), str(Path(__file__).resolve()), "check"],
                           env=dict(os.environ, AIRIS_REMOTE_REEXEC="1", PYTHONUTF8="1",
                                    PYTHONIOENCODING="utf-8"))


def cmd_check(args) -> int:
    need_venv()
    info = inner("info", timeout=600)
    bad = [f"{k}: 여기 {info.get(k)} / 기준 {v}" for k, v in EXPECT_HASH.items() if info.get(k) != v]
    if not str(info.get("commit", "")).startswith(PIN_COMMIT[:7]):
        bad.append(f"commit: 여기 {info.get('commit')} / 기준 {PIN_COMMIT[:7]}")
    heights = [b["height_m"] for b in info["bodies"]]
    if any(abs(a - b) > 1e-12 for a, b in zip(heights, EXPECT_HEIGHTS)):
        bad.append(f"체형 샘플: 여기 {heights} / 기준 {list(EXPECT_HEIGHTS)}")
    if bad:
        fail("기준 노트북과 설정이 다르다. 이대로 만들면 합칠 수 없다:\n  " + "\n  ".join(bad))
    say("  설정 식별값·커밋·체형 샘플이 기준 노트북과 같다.")

    say("  작은 실행 1건 (1~3분)...")
    tmp = WORK / "_smoke.parquet"
    tmp.unlink(missing_ok=True)
    t0 = time.perf_counter()
    got = run([venv_python(), "-u", HERE / "_inner.py", "run-range", tmp, "default", "1", "1", "1", "1",
               "--", "--n-phases", "2", "--max-evals", "200", "--pose-max-evals", "100",
               "--popsize", "10", "--patches-per-m2", "300", "--candidate-k", "4"],
              cwd=RUN_DIR, env=inner_env(), capture_output=True, text=True,
              encoding="utf-8", errors="replace")
    if got.returncode != 0 or not tmp.exists():
        fail(f"작은 실행이 실패했다:\n{got.stdout[-1500:]}\n{got.stderr[-3000:]}")
    import pandas as pd

    small = pd.read_parquet(tmp)
    tmp.unlink(missing_ok=True)
    if len(small) != 1 or int(small["body_idx"].iloc[0]) != 1:
        fail(f"작은 실행의 결과가 이상하다: 행 {len(small)}, 체형 번호 {list(small['body_idx'])}")
    say(f"  작은 실행 통과 ({time.perf_counter() - t0:.0f}초).")
    say("\n[통과] 이 노트북에서 돌릴 수 있다. 다음: python tools/remote/plan_share.py bench")
    return 0


# --------------------------------------------------------------------------------------- bench

def cmd_bench(args) -> int:
    need_venv()
    cores = os.cpu_count() or 1
    say(f"논리 코어 {cores}개. 프로세스 {args.processes}개가 동시에 돌 때의 속도를 잰다 (2~5분)...")
    got = inner("bench", args.scenario, str(args.processes), str(args.n_plan), str(args.n_pose),
                timeout=3600)
    row_min = (EVALS_PLAN * got["t_plan_s"] + EVALS_POSE * got["t_pose_s"]) / 60.0
    per_hour = args.processes * 60.0 / row_min
    say(f"  계획 평가 1회 {got['t_plan_s']:.3f}초, 자세 평가 1회 {got['t_pose_s']:.3f}초")
    say(f"  예상: 행 하나 약 {row_min:.0f}분, 시간당 약 {per_hour:.1f}행 (프로세스 {args.processes}개)")
    say(f"  100행(사용자 유형 하나 전체)이면 약 {100 / per_hour:.0f}시간")
    say("  어림값이다 (실제와 수십 % 다를 수 있다). 다른 프로그램을 닫고 재야 한다.")
    say("  참고: 기준 노트북 실측은 행 하나 122분, 시간당 3.9행 (프로세스 8개).")
    say("  프로세스 수를 바꿔 다시 재 볼 수 있다: bench --processes 4  (많다고 빨라지지 않는다)")
    return 0


# --------------------------------------------------------------------------------------- run

def read_state() -> dict | None:
    if not STATE.exists():
        return None
    return json.loads(STATE.read_text(encoding="utf-8"))


def alive(state: dict | None):
    """실행 중이면 psutil.Process, 아니면 None."""
    if not state:
        return None
    import psutil

    try:
        p = psutil.Process(int(state["pid"]))
        if abs(p.create_time() - float(state["create_time"])) < 2 and p.is_running():
            return p
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return None


def check_frame(df, where: str) -> None:
    missing = [c for c in CONSISTENCY_COLS + KEY_COLS if c not in df.columns]
    if missing:
        fail(f"{where}: 열 없음 {missing}")
    for k in ("physics_hash", "nozzle_layout_hash", "body_model"):
        vals = set(df[k].astype(str))
        if vals != {EXPECT_HASH[k]}:
            fail(f"{where}: {k} 가 {sorted(vals)} (기준 {EXPECT_HASH[k]})")


def cmd_run(args) -> int:
    need_venv()
    import pandas as pd
    import psutil

    if not (RUN_DIR / "scripts" / "run_plan_dataset.py").exists():
        fail("실행 폴더가 없다. 먼저: python tools/remote/plan_share.py setup")
    scenario, lo, hi, label = resolve_target(args)
    state = read_state()
    if alive(state):
        fail(f"이미 실행 중이다 (pid {state['pid']}, {state.get('label', state['scenario'])}). "
             "상태: status / 멈춤: stop --yes")

    WORK.mkdir(parents=True, exist_ok=True)
    out = out_path(label)
    if args.seed_file:
        seed = pd.read_parquet(args.seed_file)
        check_frame(seed, "받은 파일")
        seed = seed[(seed["scenario"].astype(str) == scenario)
                    & seed["body_idx"].astype(int).between(lo, hi)]
        if out.exists():
            have = pd.read_parquet(out)
            keys = set(zip(have["body_idx"].astype(int), have["scenario"].astype(str)))
            add = seed[[(int(i), str(s)) not in keys for i, s in zip(seed["body_idx"], seed["scenario"])]]
            merged = pd.concat([have, add], ignore_index=True)
            say(f"받은 파일에서 {len(add)}행을 더했다 (이미 있던 {len(have)}행은 그대로).")
        else:
            merged = seed
            say(f"받은 파일의 {scenario} 체형 {lo}~{hi} {len(seed)}행을 건너뛸 목록으로 넣었다.")
        merged = merged.sort_values(list(KEY_COLS), kind="mergesort").reset_index(drop=True)
        tmp = out.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp, index=False)
        os.replace(tmp, out)
    elif not out.exists() and not getattr(args, "segment", None):
        say("주의: 받은 파일(--seed-file) 없이 시작한다. 기준 노트북이 이미 만든 행도 다시 계산한다.")

    log, err = WORK / f"run_{label}.log", WORK / f"run_{label}.err"
    if log.exists():
        shutil.copyfile(log, WORK / f"run_{label}_{time.strftime('%m%d_%H%M%S')}.log")
    cmd = [str(venv_python()), "-u", str(HERE / "_inner.py"), "run-range", str(out), scenario,
           str(lo), str(hi), str(args.processes), str(FLUSH_EVERY), "--", *RUN_ARGS]
    flags = {}
    if IS_WIN:
        flags["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        flags["start_new_session"] = True
    with open(log, "w", encoding="utf-8") as fo, open(err, "w", encoding="utf-8") as fe:
        proc = subprocess.Popen(cmd, cwd=RUN_DIR, env=inner_env(), stdout=fo, stderr=fe,
                                stdin=subprocess.DEVNULL, **flags)
    state = {"pid": proc.pid, "create_time": psutil.Process(proc.pid).create_time(),
             "scenario": scenario, "label": label, "lo": lo, "hi": hi,
             "out": str(out), "log": str(log), "err": str(err),
             "processes": args.processes, "n_bodies": hi - lo + 1,
             "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    say(f"시작했다: {label} — {SCENARIO_KO[scenario]}, 체형 {lo}~{hi}번 (pid {proc.pid}). "
        "이 창을 닫아도 계속 돈다. 40초 뒤 첫 줄을 확인한다...")
    time.sleep(40)
    if not alive(state):
        fail(f"시작 직후 끝났다. 오류:\n{err.read_text(encoding='utf-8', errors='replace')[-3000:]}\n"
             f"{log.read_text(encoding='utf-8', errors='replace')[-1500:]}")
    say(log.read_text(encoding="utf-8", errors="replace").strip()[-400:])
    say("\n진행 확인: python tools/remote/plan_share.py status")
    say("노트북이 절전·종료되면 멈춘다. 전원 어댑터를 꽂고 절전을 꺼 둔다 (docs/remote_run.md 4절).")
    return 0


def cmd_status(args) -> int:
    need_venv()
    import pandas as pd

    state = read_state()
    if not state:
        say("실행 기록이 없다. 시작: python tools/remote/plan_share.py run --scenario <유형>")
        return 0
    p = alive(state)
    lo, hi = int(state.get("lo", 0)), int(state.get("hi", state["n_bodies"] - 1))
    label = state.get("label", state["scenario"])
    say(f"구간: {label} — {SCENARIO_KO.get(state['scenario'], state['scenario'])}, 체형 {lo}~{hi}번, "
        f"시작 {state['started']}")
    say(f"실행 상태: {'도는 중 (pid %s)' % state['pid'] if p else '멈춰 있음'}")
    out = Path(state["out"])
    n = 0
    if out.exists():
        df = pd.read_parquet(out)
        n = int(((df["scenario"].astype(str) == state["scenario"])
                 & df["body_idx"].astype(int).between(lo, hi)).sum())
        say(f"저장된 행: {n} / {state['n_bodies']}  (파일 {out}, 수정 {time.strftime('%m-%d %H:%M', time.localtime(out.stat().st_mtime))})")
    else:
        say("저장된 행: 아직 없음 (5행마다 저장한다)")
    log = Path(state["log"])
    if log.exists():
        lines = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        for line in lines[-3:]:
            say("  " + line)
        rate = None
        for line in reversed(lines):
            if "행/분" in line:
                try:
                    rate = float(line.split("행/분")[0].split()[-1])
                except ValueError:
                    pass
                break
        if rate and p:
            left = max(state["n_bodies"] - n, 0)
            say(f"예상 남은 시간: 약 {left / rate / 60:.0f}시간 (지금 속도 {rate * 60:.1f}행/시간)")
    err = Path(state["err"])
    if err.exists():
        text = err.read_text(encoding="utf-8", errors="replace")
        if "Traceback" in text or "Error" in text:        # 경고(UserWarning)만 있으면 보이지 않는다
            say("오류 로그 끝:\n" + text[-1500:])
    if not p and n < state["n_bodies"]:
        target = (f"--segment {label}" if label in SEGMENTS else
                  f"--scenario {state['scenario']} --bodies {lo}-{hi}")
        say(f"\n이어서 돌리기: python tools/remote/plan_share.py run {target} --processes {state['processes']}")
    if n >= state["n_bodies"]:
        say(f"\n끝났다. 이 파일을 총괄에게 보낸다: {out}")
        say("보낸 뒤 구간표(docs/remote_plan_board.md)에서 다음 구간을 받아 run 을 다시 하면 된다.")
    return 0


def cmd_stop(args) -> int:
    need_venv()
    import psutil

    state = read_state()
    p = alive(state)
    if not p:
        say("도는 실행이 없다.")
        return 0
    if not args.yes:
        fail("멈추면 계산 중이던 행(저장 전)은 사라진다. 저장된 행은 남는다. 멈추려면: stop --yes")
    procs = p.children(recursive=True) + [p]
    for q in procs:
        try:
            q.terminate()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=20)
    say("멈췄다. 저장된 행은 남아 있고, run 을 다시 하면 이어서 돈다.")
    return 0


# --------------------------------------------------------------------------------------- 기준 노트북용

def cmd_export_seed(args) -> int:
    import pandas as pd

    df = pd.read_parquet(args.src)
    check_frame(df, str(args.src))
    part = df[df["scenario"].astype(str) == args.scenario]
    if args.bodies:
        lo, hi = parse_bodies(args.bodies)
        part = part[part["body_idx"].astype(int).between(lo, hi)]
    part = part.reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    part.to_parquet(args.out, index=False)
    say(f"{args.scenario} {len(part)}행 → {args.out} (체형 번호 {sorted(part['body_idx'].astype(int))[:5]} ...)")
    return 0


def cmd_segments(args) -> int:
    """구간표와, 파일을 주면 구간별 완료 행 수."""
    done: set = set()
    if args.files:
        import pandas as pd

        for f in args.files:
            df = pd.read_parquet(f)
            check_frame(df, str(f))
            done |= set(zip(df["scenario"].astype(str), df["body_idx"].astype(int)))
    say("구간  사용자 유형        체형 번호   행 수" + ("   완료" if args.files else ""))
    for name, (scenario, lo, hi) in SEGMENTS.items():
        total = hi - lo + 1
        line = f"{name:<4}  {scenario:<10} {SCENARIO_KO[scenario]:<4}  {lo:>3}~{hi:<3}    {total:>3}"
        if args.files:
            n = sum((scenario, i) in done for i in range(lo, hi + 1))
            line += f"    {n:>3}" + ("  끝" if n == total else "")
        say(line)
    return 0


def cmd_merge(args) -> int:
    import pandas as pd

    frames = []
    for f in args.files:
        df = pd.read_parquet(f)
        check_frame(df, str(f))
        say(f"  {f}: {len(df)}행 {df['scenario'].value_counts().to_dict()} commit {sorted(set(df['commit'].astype(str)))}")
        frames.append(df)
    cols = [set(d.columns) for d in frames]
    if any(c != cols[0] for c in cols):
        diff = set.union(*cols) - set.intersection(*cols)
        fail(f"파일마다 열이 다르다: {sorted(diff)}")
    merged = pd.concat(frames, ignore_index=True)
    for col in CONSISTENCY_COLS:
        vals = set(merged[col].astype(str))
        if len(vals) != 1:
            fail(f"{col} 가 파일마다 다르다: {sorted(vals)}. 합치지 않는다.")
    dup = int(merged.duplicated(list(KEY_COLS)).sum())
    merged = merged.drop_duplicates(list(KEY_COLS), keep="first")
    merged = merged.sort_values(list(KEY_COLS), kind="mergesort").reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    merged.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    say(f"합침: {len(merged)}행 {merged['scenario'].value_counts().to_dict()} (겹쳐서 버린 행 {dup}, 앞 파일 우선)")
    full = merged.groupby("body_idx")["scenario"].nunique()
    say(f"세 유형이 모두 있는 체형 {int((full == len(SCENARIOS)).sum())}개 → {out}")
    return 0


# --------------------------------------------------------------------------------------- main

def need_venv() -> None:
    """setup 이 만든 가상환경의 Python 으로 다시 실행한다 (패키지가 거기에만 있다)."""
    vp = venv_python()
    if not vp.exists():
        fail("가상환경이 없다. 먼저: python tools/remote/plan_share.py setup")
    if Path(sys.executable).resolve() != vp.resolve() and os.environ.get("AIRIS_REMOTE_REEXEC") != "1":
        env = dict(os.environ, AIRIS_REMOTE_REEXEC="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        raise SystemExit(subprocess.call([str(vp), str(Path(__file__).resolve()), *sys.argv[1:]], env=env))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="계획 데이터셋 분담 실행 도구 (docs/remote_run.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="가상환경·실행 폴더를 만들고 점검한다").set_defaults(fn=cmd_setup)
    sub.add_parser("check", help="설정이 기준 노트북과 같은지, 작은 실행이 되는지 본다").set_defaults(fn=cmd_check)
    b = sub.add_parser("bench", help="이 노트북의 예상 속도를 잰다")
    b.add_argument("--scenario", choices=SCENARIOS, default="wheelchair")
    b.add_argument("--processes", type=int, default=8)
    b.add_argument("--n-plan", type=int, default=30)
    b.add_argument("--n-pose", type=int, default=60)
    b.set_defaults(fn=cmd_bench)
    r = sub.add_parser("run", help="사용자 유형 하나를 맡아 돌린다 (창을 닫아도 계속 돈다)")
    r.add_argument("--segment", default=None, help="구간표의 구간 이름 (예: W2). docs/remote_plan_board.md")
    r.add_argument("--scenario", choices=SCENARIOS, default=None, help="구간표에 없는 범위를 직접 줄 때")
    r.add_argument("--bodies", default=None, help="체형 번호 구간 '40-59' (--scenario 와 함께)")
    r.add_argument("--seed-file", default=None, help="총괄에게 받은 파일 (이미 만든 행을 건너뛴다)")
    r.add_argument("--processes", type=int, default=8)
    r.add_argument("--n-bodies", type=int, default=100)
    r.set_defaults(fn=cmd_run)
    sub.add_parser("status", help="진행 확인").set_defaults(fn=cmd_status)
    s = sub.add_parser("stop", help="멈춘다 (저장된 행은 남는다)")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(fn=cmd_stop)
    e = sub.add_parser("export-seed", help="[기준 노트북] 한 유형의 완료 행을 떼어 낸다")
    e.add_argument("--from", dest="src", required=True)
    e.add_argument("--scenario", choices=SCENARIOS, required=True)
    e.add_argument("--bodies", default=None, help="체형 번호 구간 '40-59'")
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_export_seed)
    g = sub.add_parser("segments", help="구간표를 보여 준다 (파일을 주면 구간별 완료 행 수)")
    g.add_argument("files", nargs="*")
    g.set_defaults(fn=cmd_segments)
    m = sub.add_parser("merge", help="[기준 노트북] 여러 파일을 하나로 합친다")
    m.add_argument("--out", required=True)
    m.add_argument("files", nargs="+")
    m.set_defaults(fn=cmd_merge)
    return ap


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
