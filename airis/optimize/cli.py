"""실행 스크립트 공용 헬퍼. 소유자: C.

run_optimize.py 와 run_baselines.py 가 함께 쓰는 평가기 선택, 노즐 폴백, 표 출력.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import fields
from typing import Any

import numpy as np

from airis.sim import (
    PART_NAMES, BodyParams, Evaluator, NozzleConfig, PoseParams, Scenario,
)
from airis.sim.scenario import load_nozzles, load_physics

from .dummy import DummyEvaluator

EVALUATOR_CHOICES = ("dummy", "patch", "particle")

#: 평가기별 담당 트랙과 구현 위치. 미병합 안내에 쓴다.
_OWNER = {
    "patch": ("D", "airis.sim.patch_baseline.PatchEvaluator", "docs/tracks/D_patch_baseline.md"),
    "particle": ("A", "airis.sim.particles.ParticleEvaluator", "docs/tracks/A_particles.md"),
}

#: --evaluator dummy 의 기본 목표 자세. E4 의 B1(업계 권장, 팔 들고 정면)과 같은 값.
DEFAULT_DUMMY_TARGET = PoseParams(shoulder_abduction=90.0, elbow_flexion=0.0)


def enable_utf8_stdout() -> None:
    """윈도우 콘솔 기본 코드페이지에서 한글이 깨지지 않게 한다."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


class TrackNotMerged(RuntimeError):
    """해당 트랙이 아직 병합되지 않아 평가기를 쓸 수 없다."""


def dataclass_from_json(cls, raw: str | None):
    """'{"height_m":1.6}' 같은 JSON 으로 dataclass 기본값을 덮어쓴다."""
    obj = cls()
    if not raw:
        return obj
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"JSON 파싱 실패: {raw!r} ({exc})") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"JSON 객체여야 한다: {raw!r}")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise SystemExit(
            f"{cls.__name__} 에 없는 키: {sorted(unknown)} (가능: {sorted(known)})"
        )
    for k, v in data.items():
        setattr(obj, k, float(v))
    return obj


def resolve_nozzles() -> tuple[NozzleConfig, str]:
    """실제 노즐 배치를 시도하고, B 미병합이면 임시 배치로 폴백한다.

    Returns: (NozzleConfig, source) — source 는 "config" 또는 "fake".
    """
    try:
        return load_nozzles(), "config"
    except NotImplementedError:
        # load_nozzles 가 미구현인 브랜치용 폴백. 가짜는 tests/fakes.py 한 곳에만 둔다 (00_common.md 3절).
        from tests.fakes import fake_nozzles
        return fake_nozzles(), "fake"


def nozzle_hash(nozzle: NozzleConfig) -> str:
    """노즐 배치 내용 해시 앞 8자리. 설정판이든 임시판이든 같은 방식으로 찍힌다."""
    h = hashlib.sha256()
    for arr in (nozzle.positions, nozzle.directions, nozzle.strengths):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()[:8]


def make_evaluator(
    name: str,
    scenario: Scenario,
    *,
    body: BodyParams,
    nozzle: NozzleConfig,
    dummy_target: PoseParams | None = None,
) -> Evaluator:
    """--evaluator 이름으로 평가기를 만든다.

    patch / particle 은 클래스가 있어도 내부가 NotImplementedError 일 수 있으므로
    기본 자세로 한 번 시험 평가해 본다. 미구현이면 TrackNotMerged 를 던진다.
    """
    if name == "dummy":
        return DummyEvaluator(dummy_target or DEFAULT_DUMMY_TARGET, scenario)

    if name not in _OWNER:
        raise SystemExit(f"알 수 없는 평가기: {name} (가능: {', '.join(EVALUATOR_CHOICES)})")

    track, dotted, doc = _OWNER[name]
    module_path, cls_name = dotted.rsplit(".", 1)
    try:
        module = __import__(module_path, fromlist=[cls_name])
        evaluator = getattr(module, cls_name)(load_physics())
    except (ImportError, AttributeError, NotImplementedError) as exc:
        raise TrackNotMerged(_not_merged_msg(name, track, dotted, doc, exc)) from exc

    try:
        evaluator.evaluate(PoseParams(), nozzle, body, scenario)
    except NotImplementedError as exc:
        raise TrackNotMerged(_not_merged_msg(name, track, dotted, doc, exc)) from exc
    return evaluator


def _not_merged_msg(name: str, track: str, dotted: str, doc: str, exc: BaseException) -> str:
    return (
        f"해당 트랙 미병합: --evaluator {name} 은 트랙 {track} 의 {dotted} 가 필요하다.\n"
        f"  원인: {type(exc).__name__}: {exc}\n"
        f"  참고: {doc}\n"
        f"  지금 파이프라인을 확인하려면 --evaluator dummy 로 실행한다."
    )


def pose_dict(pose: PoseParams) -> dict[str, float]:
    return {f.name: float(getattr(pose, f.name)) for f in fields(PoseParams)}


def format_pose_table(pose: PoseParams, scenario: Scenario, free_keys: list[str] | None = None) -> str:
    """자세를 표로. 고정 변수는 fixed 로 표시한다."""
    lines = [f"{'자세 변수':<20} {'값(deg)':>10}  비고", "-" * 46]
    for key, value in pose_dict(pose).items():
        if key in scenario.fixed_pose:
            note = "fixed"
        elif free_keys is not None and key not in free_keys:
            note = "고정"
        else:
            note = ""
        lines.append(f"{key:<20} {value:>10.2f}  {note}")
    return "\n".join(lines)


def format_removal_table(result: Any) -> str:
    """부위별 제거율 표."""
    values = np.asarray(result.removal_by_part, dtype=float).reshape(-1)
    lines = [f"{'부위':<14} {'제거율':>8}", "-" * 24]
    for part, value in zip(PART_NAMES, values):
        lines.append(f"{part:<14} {value:>8.4f}")
    lines.append("-" * 24)
    lines.append(f"{'total':<14} {float(result.total_removal):>8.4f}")
    lines.append(f"{'discomfort':<14} {float(result.discomfort):>8.4f}")
    return "\n".join(lines)
