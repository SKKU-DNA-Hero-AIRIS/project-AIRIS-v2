"""입자 애니메이션 (plotly 프레임 슬라이더). 소유자: E. (`docs/tracks/E_realtime.md` 단계 2)

A의 프레임 덤프(`docs/interfaces.md` "시각화용 상태 덤프")를 읽는다.

    outputs/<exp_id>/frames/<step>.npz
      계약 키: pos (P,3) f32, attached (P,) bool, part (P,) int, candidate (P,) int
      A가 더 넣는 키(있으면 쓴다): state (P,) i8 = 0 부착 / 1 부유 / 2 제거, t_s, step

로더는 계약 키만 검사한다. `state`가 없으면 부착 여부와 부스 경계로 부유·제거를 나눈다.
A 병합 전 개발·테스트용으로 **합성 프레임**(`synthetic_frames`)을 만든다. 합성 프레임은
물리 계산이 아니다: 패치에서 출발해 +y 벽으로 날아가는 가짜 궤적이다.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from ..sim.scenario import load_nozzle_layout
from ..sim.types import PART_NAMES, BodyState, NozzleConfig
from .pose_view import (PART_COLORS, _camera_buttons, _scene_layout, body_traces,
                        booth_traces, nozzle_traces)

#: 덤프 계약 키 (`docs/interfaces.md`).
REQUIRED_KEYS = ("pos", "attached", "part", "candidate")
ATTACHED, AIRBORNE, REMOVED = 0, 1, 2
ATTACHED_COLOR = "#8a8a8a"
REMOVED_COLOR = "#e53935"

Frame = tuple[int, dict[str, np.ndarray]]


# ---------------------------------------------------------------------------
# 로더
# ---------------------------------------------------------------------------
def frames_dir(path: str | Path) -> Path:
    """`outputs/<exp_id>` 또는 `outputs/<exp_id>/frames` 어느 쪽을 받아도 frames 디렉터리를 돌려준다."""
    p = Path(path)
    if (p / "frames").is_dir():
        return p / "frames"
    return p


def load_frames(path: str | Path) -> list[Frame]:
    """`<step>.npz`를 스텝 순으로 읽는다 → [(step, {키: 배열})].

    계약 키가 빠진 파일이 있으면 ValueError. 스텝은 파일 이름(정수)에서 읽는다.
    """
    d = frames_dir(path)
    files = [f for f in d.glob("*.npz") if f.stem.isdigit()]
    if not files:
        raise FileNotFoundError(f"프레임 파일(<step>.npz)이 없다: {d}")
    frames: list[Frame] = []
    for f in sorted(files, key=lambda f: int(f.stem)):
        with np.load(f) as z:
            data = {k: z[k] for k in z.files}
        missing = [k for k in REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(f"{f.name}: 덤프 계약 키 누락 {missing} (docs/interfaces.md)")
        n = data["pos"].shape[0]
        if data["pos"].shape != (n, 3) or any(data[k].shape != (n,) for k in REQUIRED_KEYS[1:]):
            raise ValueError(f"{f.name}: 배열 형태가 계약과 다르다 (pos (P,3), 나머지 (P,))")
        frames.append((int(f.stem), data))
    return frames


def candidates_in(frames: Sequence[Frame]) -> list[int]:
    """덤프에 들어 있는 후보 번호 (입력 순서 번호, 부스 밖 후보는 빠져 있을 수 있다)."""
    return sorted(int(c) for c in np.unique(frames[0][1]["candidate"]))


def particle_status(frame: Mapping[str, np.ndarray], booth: Mapping) -> np.ndarray:
    """(P,) int8: 0 부착 / 1 부유 / 2 제거.

    A의 `state` 키가 있으면 그대로 쓴다. 없으면 부착이 아니고 부스 밖(x 문 밖, |y| 벽 밖,
    z 천장 위)이면 제거, 나머지는 부유로 본다. 바닥은 출구가 아니다 (발 캡슐 뚜껑은 z < 0 까지 내려간다).
    """
    if "state" in frame:
        return np.asarray(frame["state"]).astype(np.int8)
    pos = np.asarray(frame["pos"], dtype=np.float64)
    attached = np.asarray(frame["attached"]).astype(bool)
    lx, w, h = float(booth["length_m"]), float(booth["width_m"]), float(booth["height_m"])
    outside = ((pos[:, 0] < 0.0) | (pos[:, 0] > lx) | (np.abs(pos[:, 1]) > w / 2.0)
               | (pos[:, 2] > h))
    status = np.full(pos.shape[0], AIRBORNE, dtype=np.int8)
    status[attached] = ATTACHED
    status[~attached & outside] = REMOVED
    return status


# ---------------------------------------------------------------------------
# 합성 프레임 (A 병합 전 개발용 가짜 데이터)
# ---------------------------------------------------------------------------
def synthetic_frames(state: BodyState, booth: Mapping | None = None, *,
                     n_particles: int = 2000, n_frames: int = 31, dump_every: int = 10,
                     dt_s: float = 0.002, detach_fraction: float = 0.4,
                     speed_mps: float = 1.5, n_candidates: int = 1,
                     seed: int = 0, include_state: bool = False) -> list[Frame]:
    """패치 위에서 시작해 +y 벽으로 날아가는 가짜 입자 프레임. 물리 계산이 아니다.

    - 입자는 패치 면적 비례로 뿌린다. `detach_fraction`만큼이 무작위 시각에 이탈해
      +y(와 약간의 위아래 흔들림)로 `speed_mps`로 움직이고, 벽(|y| > width/2)을 넘으면
      그 자리에서 멈춘다(제거, "마지막 위치").
    - 후보 `n_candidates`개를 이어붙인다(후보마다 시드가 달라 궤적이 다르다).
    - 키는 계약 키 4개. `include_state=True`면 A처럼 `state`도 넣는다.
    """
    booth = booth if booth is not None else load_nozzle_layout()["booth"]
    rng = np.random.default_rng(seed)
    area = np.asarray(state.patch_area, dtype=np.float64)
    pos0_all, part_all, cand_all, t_det_all, vel_all = [], [], [], [], []
    duration = (n_frames - 1) * dump_every * dt_s
    for c in range(n_candidates):
        idx = rng.choice(area.size, size=n_particles, p=area / area.sum())
        pos0_all.append(np.asarray(state.patch_pos, dtype=np.float64)[idx])
        part_all.append(np.asarray(state.patch_part, dtype=np.int32)[idx])
        cand_all.append(np.full(n_particles, c, dtype=np.int32))
        detach = rng.random(n_particles) < detach_fraction
        t_det_all.append(np.where(detach, rng.uniform(0.0, 0.6 * duration, n_particles), np.inf))
        v = np.zeros((n_particles, 3))
        v[:, 0] = rng.normal(0.0, 0.2, n_particles) * speed_mps
        v[:, 1] = speed_mps * rng.uniform(0.6, 1.4, n_particles)
        v[:, 2] = rng.normal(0.0, 0.3, n_particles) * speed_mps
        vel_all.append(v)
    pos0 = np.concatenate(pos0_all)
    part = np.concatenate(part_all)
    cand = np.concatenate(cand_all)
    t_det = np.concatenate(t_det_all)
    vel = np.concatenate(vel_all)

    half_w = float(booth["width_m"]) / 2.0
    frames: list[Frame] = []
    for i in range(n_frames):
        step = i * dump_every
        t = step * dt_s
        fly = np.clip(t - t_det, 0.0, None)[:, None]            # inf → 0 (부착 유지)
        fly = np.where(np.isfinite(fly), fly, 0.0)
        # 벽(y = width/2)을 넘는 시각에서 멈추고 벽 바로 밖(+2 cm)에 둔다: 그 뒤로는 마지막 위치 유지
        t_wall = np.where(vel[:, 1] > 0, (half_w - pos0[:, 1]) / vel[:, 1], np.inf)
        stopped = fly[:, 0] >= t_wall
        pos = pos0 + vel * np.minimum(fly, t_wall[:, None])
        pos[stopped, 1] += 0.02
        attached = t < t_det
        frame = {
            "pos": pos.astype(np.float32),
            "attached": attached,
            "part": part.copy(),
            "candidate": cand.copy(),
        }
        if include_state:
            st = np.where(attached, ATTACHED, AIRBORNE).astype(np.int8)
            st[~attached & (np.abs(pos[:, 1]) > half_w)] = REMOVED
            frame["state"] = st
        frames.append((step, frame))
    return frames


def write_frames(frames: Sequence[Frame], out_dir: str | Path) -> Path:
    """프레임을 A와 같은 형식(`<step:05d>.npz`)으로 쓴다. 반환: frames 디렉터리."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for step, data in frames:
        np.savez(d / f"{step:05d}.npz", **data)
    return d


# ---------------------------------------------------------------------------
# 애니메이션
# ---------------------------------------------------------------------------
def _subsample(n: int, max_points: int) -> np.ndarray:
    """균등 서브샘플 인덱스 (모든 프레임에서 같은 입자를 따라간다)."""
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def _frame_indices(n: int, max_frames: int) -> np.ndarray:
    """프레임이 많으면 균등하게 고른다. 처음과 마지막은 항상 포함."""
    if n <= max_frames:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_frames).round().astype(int))


def _particle_traces(pos: np.ndarray, status: np.ndarray, part: np.ndarray,
                     marker_size: float) -> list[go.Scatter3d]:
    part_colors = np.array([PART_COLORS[name] for name in PART_NAMES])
    traces = []
    for code, name, color in ((ATTACHED, "부착", ATTACHED_COLOR),
                              (AIRBORNE, "부유", None),
                              (REMOVED, "제거", REMOVED_COLOR)):
        m = status == code
        p = pos[m]
        if code == AIRBORNE:
            c = part_colors[np.clip(part[m], 0, len(PART_NAMES) - 1)].tolist()
        else:
            c = color
        traces.append(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", name=name,
            marker=dict(size=marker_size * (0.7 if code == ATTACHED else 1.0), color=c,
                        opacity=0.3 if code == ATTACHED else 0.9),
            hoverinfo="skip"))
    return traces


def status_counts(frames: Sequence[Frame], booth: Mapping | None = None,
                  candidate: int = 0) -> list[dict]:
    """프레임별 부착·부유·제거 비율. 애니메이션 부제와 대시보드 표에 쓴다."""
    booth = booth if booth is not None else load_nozzle_layout()["booth"]
    rows = []
    for step, fr in frames:
        m = np.asarray(fr["candidate"]) == candidate
        st = particle_status(fr, booth)[m]
        n = max(int(m.sum()), 1)
        rows.append({"step": step, "t_s": float(fr["t_s"]) if "t_s" in fr else None,
                     "attached": float((st == ATTACHED).sum()) / n,
                     "airborne": float((st == AIRBORNE).sum()) / n,
                     "removed": float((st == REMOVED).sum()) / n})
    return rows


def animation(frames: Sequence[Frame], state: BodyState | None = None,
              booth: Mapping | None = None, candidate: int = 0, max_points: int = 5000, *,
              nozzle: NozzleConfig | None = None, max_frames: int = 60,
              marker_size: float = 2.0, title: str = "", frame_ms: int = 80,
              height: int = 680) -> go.Figure:
    """덤프 프레임 → plotly 애니메이션 (재생 버튼 + 스텝 슬라이더).

    - `candidate`: 볼 후보(덤프의 `candidate` 값). 없으면 ValueError.
    - `max_points`: 입자 균등 서브샘플 상한. `max_frames`: 프레임 균등 서브샘플 상한.
    - 부착 = 회색, 부유 = 부위 색, 제거 = 빨강(마지막 위치). 몸은 `state`의 캡슐 메시.
    """
    if not frames:
        raise ValueError("프레임이 없다")
    booth = booth if booth is not None else load_nozzle_layout()["booth"]
    cand0 = np.asarray(frames[0][1]["candidate"])
    sel = np.flatnonzero(cand0 == candidate)
    if sel.size == 0:
        raise ValueError(f"후보 {candidate} 가 덤프에 없다 (있는 후보: {candidates_in(frames)})")
    sel = sel[_subsample(sel.size, max_points)]
    picked = [frames[i] for i in _frame_indices(len(frames), max_frames)]

    static: list = []
    if state is not None:
        static += body_traces(state, opacity=0.55)
        for tr in static:
            tr.showlegend = False
    static += booth_traces(booth)
    if nozzle is not None:
        static += nozzle_traces(nozzle)

    def traces_for(fr: dict) -> list[go.Scatter3d]:
        st = particle_status(fr, booth)[sel]
        return _particle_traces(np.asarray(fr["pos"])[sel], st, np.asarray(fr["part"])[sel],
                                marker_size)

    def label(step: int, fr: dict) -> str:
        return f"{float(fr['t_s']):.2f} s" if "t_s" in fr else f"step {step}"

    first = traces_for(picked[0][1])
    n_static = len(static)
    particle_idx = list(range(n_static, n_static + len(first)))
    fig = go.Figure(data=static + first)

    counts = status_counts(picked, booth, candidate)
    fig_frames = []
    for (step, fr), c in zip(picked, counts):
        fig_frames.append(go.Frame(
            name=str(step), data=traces_for(fr), traces=particle_idx,
            layout=go.Layout(title=dict(text=(
                f"{title}<br><sub>{label(step, fr)} · 부착 {c['attached']:.0%} · "
                f"부유 {c['airborne']:.0%} · 제거 {c['removed']:.0%}</sub>")))))
    fig.frames = fig_frames

    play = dict(type="buttons", direction="left", x=0.0, y=0.0, xanchor="left", yanchor="top",
                pad=dict(t=40, r=10), showactive=False, buttons=[
                    dict(label="▶ 재생", method="animate",
                         args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                          transition=dict(duration=0), fromcurrent=True)]),
                    dict(label="❚❚ 정지", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate", transition=dict(duration=0))]),
                ])
    slider = dict(active=0, x=0.12, y=0.0, len=0.88, pad=dict(t=30),
                  currentvalue=dict(prefix="", visible=True),
                  steps=[dict(method="animate", label=label(step, fr),
                              args=[[str(step)], dict(mode="immediate",
                                                      frame=dict(duration=0, redraw=True),
                                                      transition=dict(duration=0))])
                         for step, fr in picked])
    c0 = counts[0]
    fig.update_layout(
        title=dict(text=(f"{title}<br><sub>{label(*picked[0])} · 부착 {c0['attached']:.0%} · "
                         f"부유 {c0['airborne']:.0%} · 제거 {c0['removed']:.0%}</sub>"), x=0.5),
        scene=_scene_layout(booth),
        updatemenus=_camera_buttons(["scene"]) + [play],
        sliders=[slider],
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(orientation="v", y=0.92, x=0.0, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.6)"),
        height=height,
    )
    return fig
