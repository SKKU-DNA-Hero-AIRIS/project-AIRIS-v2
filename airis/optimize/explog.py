"""실험 로그. 소유자: C. docs/tracks/C_optimize.md 단계 4.

저장 구조:
    outputs/<exp_id>/meta.json     시나리오, 체형, 설정 해시, 커밋, 시드, 인자, 평가 횟수, 최고점
    outputs/<exp_id>/history.csv   gen, evals, best, mean, sigma, infeasible_frac, start
    outputs/<exp_id>/best.json     best_pose, removal_by_part, discomfort
    outputs/index.csv              exp_id, date, scenario, best_score, commit  (append)
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from airis.sim import PART_NAMES

ROOT = Path(__file__).resolve().parents[2]
HISTORY_COLUMNS = ["gen", "evals", "best", "mean", "sigma", "infeasible_frac", "start"]
INDEX_COLUMNS = ["exp_id", "date", "scenario", "best_score", "commit"]


def new_exp_id(prefix: str) -> str:
    """"{prefix}_{YYYYmmdd_HHMMSS}_{6자리 해시}"."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:6]}"


def git_commit() -> str:
    """현재 커밋 짧은 해시. git 이 없거나 저장소가 아니면 "nogit"."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "nogit"
    if out.returncode != 0:
        return "nogit"
    return out.stdout.strip() or "nogit"


def file_hash(path: Path | str) -> str:
    """설정 파일의 내용 해시 앞 8자리. 없으면 "nofile".

    YAML(.yaml/.yml)은 **파싱한 내용**을 정렬 직렬화(JSON, sort_keys, 구분자 고정)해 sha256 한다
    (총괄 확정 2026-09-30). 그래서 주석·줄바꿈(CRLF/LF)·키 순서가 달라도 같은 해시가 나오고,
    값이 바뀌면 달라진다. 0.50 과 0.5 처럼 표기만 다른 수도 파싱 값이 같아 같은 해시다.
    그 밖의 파일은 예전처럼 바이트 sha256 이다.

    경계: 파싱한 값으로 재는 것이라 정수 1 과 실수 1.0 은 **다른** 해시다(JSON 직렬화가 "1" 과 "1.0").
    설정에서 1 → 1.0 으로 표기를 바꾸면 해시가 달라지니 값 자체를 바꾼 것과 구별되지 않는다.
    YAML 파싱이 실패하면 예외를 그대로 올린다 (망가진 설정을 조용히 바이트 해시로 넘기지 않는다).
    """
    p = Path(path)
    if not p.is_file():
        return "nofile"
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        blob = json.dumps(data, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:8]
    return hashlib.sha256(p.read_bytes()).hexdigest()[:8]


def _jsonable(obj: Any) -> Any:
    """dataclass / numpy / Path 를 json 이 먹을 수 있는 형태로.

    nan·inf 는 **null** 로 바꾼다. 파이썬 json 은 이것들을 NaN·Infinity 로 쓰는데 표준 JSON 이
    아니라 다른 도구가 읽다 깨진다 (시작점이 하나면 peak_gap 이 nan 이다, 통합 2026-10-01 지적).
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run_dir(log_dir: Path | str, exp_id: str) -> Path:
    d = Path(log_dir) / exp_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_history(log_dir: Path | str, exp_id: str, history: Iterable[dict]) -> Path:
    """history.csv 를 처음부터 다시 쓴다."""
    path = run_dir(log_dir, exp_id) / "history.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k) for k in HISTORY_COLUMNS})
    return path


def append_history_row(log_dir: Path | str, exp_id: str, row: dict) -> None:
    """세대마다 history.csv 에 한 줄씩 덧붙인다 (긴 실행이 중간에 죽어도 남도록)."""
    path = run_dir(log_dir, exp_id) / "history.csv"
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in HISTORY_COLUMNS})


def append_index(log_dir: Path | str, *, exp_id: str, scenario: str,
                 best_score: float, commit: str) -> Path:
    """outputs/index.csv 에 한 줄 append. 기계용 원장."""
    path = Path(log_dir) / "index.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=INDEX_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "exp_id": exp_id,
            "date": datetime.now().isoformat(timespec="seconds"),
            "scenario": scenario,
            "best_score": f"{best_score:.6f}",
            "commit": commit,
        })
    return path


def write_run(log_dir: Path | str, exp_id: str, *, scenario, body,
              nozzle_hash: str, physics_hash: str, commit: str, seed: int,
              args: dict, result) -> Path:
    """meta.json, history.csv, best.json 을 쓰고 index.csv 에 append 한다."""
    d = run_dir(log_dir, exp_id)

    meta = {
        "exp_id": exp_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scenario": _jsonable(scenario),
        "body": _jsonable(body),
        "nozzle_hash": nozzle_hash,
        "physics_hash": physics_hash,
        "commit": commit,
        "seed": int(seed),
        "args": _jsonable(args),
        "n_evals": int(result.n_evals),
        "n_infeasible": int(result.n_infeasible),
        "per_start": _jsonable(result.per_start),
        # float() 로 바로 넣으면 nan 이 JSON 에 NaN 리터럴로 새므로 _jsonable 을 거친다.
        "best_score": _jsonable(float(result.best_score)),
    }
    (d / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_history(d.parent, exp_id, result.history)

    best = {
        "exp_id": exp_id,
        "best_pose": _jsonable(result.best_pose),
        "best_score": _jsonable(float(result.best_score)),
        "removal_by_part": _jsonable(result.best_result.removal_by_part),
        "part_names": list(PART_NAMES),
        "total_removal": _jsonable(float(result.best_result.total_removal)),
        "discomfort": _jsonable(float(result.best_result.discomfort)),
        "extra": _jsonable(result.best_result.extra),
    }
    (d / "best.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    append_index(
        log_dir,
        exp_id=exp_id,
        scenario=getattr(scenario, "name", str(scenario)),
        best_score=float(result.best_score),
        commit=commit,
    )
    return d
