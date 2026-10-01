"""운전 계획 그림 (plotly). 소유자: E. (`docs/plan_extension.md` 7절 5번 "계획 시각화")

    from airis.viz.plan_view import figure_plan_phases, figure_zone_strengths, figure_timeline
    fig = figure_plan_phases(body, rec.plan, scenario, model="mesh")      # 단계별 자세 3D
    fig = figure_zone_strengths(rec.plan, scenario)                      # 구역 세기와 상한
    fig = figure_timeline(device_control(rec.plan, scenario))            # 단계·전환 시각표

물리 계산은 하지 않는다. 세기·상한은 계획과 `zone_strength_caps(scenario)` 값을 그대로 그린다.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import plotly.graph_objects as go

from ..sim.scenario import zone_strength_caps
from ..sim.types import ZONE_NAMES, BodyParams, Plan, Scenario
from .pose_view import SLOT_COLOR, figure_compare

ZONE_LABELS = {"chest_low": "가슴 쪽 벽<br>하단", "chest_high": "가슴 쪽 벽<br>상단",
               "back_low": "등 쪽 벽<br>하단", "back_high": "등 쪽 벽<br>상단", "top": "천장"}
CAP_COLOR = "#d95f02"
TRANSITION_COLOR = "#b0bec5"
#: 3D 칸 수 상한. 단계가 더 많으면(P1 12단계) 고르게 골라 그린다.
MAX_PHASE_PANELS = 4


def phase_panels(plan: Plan, max_panels: int = MAX_PHASE_PANELS) -> list[int]:
    """그릴 단계 번호(0부터). 단계가 많으면 처음부터 고르게 `max_panels`개."""
    n = len(plan.phases)
    if n <= max_panels:
        return list(range(n))
    return sorted({int(round(i)) for i in np.linspace(0, n - 1, max_panels)})


def figure_plan_phases(body: BodyParams | None, plan: Plan, scenario: Scenario, *,
                       model: str | None = None, patches_per_m2: float = 400.0,
                       booth: Mapping | None = None, height: int = 520,
                       max_panels: int = MAX_PHASE_PANELS) -> go.Figure:
    """단계별 자세를 나란히. 칸 제목에 단계 시간과 회전 각도를 단다."""
    idx = phase_panels(plan, max_panels)
    poses = {}
    for i in idx:
        ph = plan.phases[i]
        poses[f"{i + 1}단계 · {ph.duration_s:.1f}초 · 회전 {ph.pose.torso_yaw:.0f}°"] = ph.pose
    fig = figure_compare(body, poses, scenario, booth=booth, patches_per_m2=patches_per_m2,
                         height=height, model=model)
    if len(idx) < len(plan.phases):
        fig.add_annotation(text=f"전체 {len(plan.phases)}단계 중 {len(idx)}개만 표시", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=1.12, font=dict(size=12))
    return fig


def figure_zone_strengths(plan: Plan, scenario: Scenario, *, s_max: float | None = None,
                          height: int = 320) -> go.Figure:
    """구역 세기 막대 + 시나리오 쾌적 상한(있는 구역만) + 세기 상한 선."""
    s = np.asarray(plan.zone_strengths, dtype=np.float64)
    caps = zone_strength_caps(scenario)
    labels = [ZONE_LABELS[z] for z in ZONE_NAMES]
    fig = go.Figure(go.Bar(
        x=labels, y=s, name="구역 세기", marker=dict(color=SLOT_COLOR),
        text=[f"{v:.2f}" for v in s], textposition="outside",
        hovertemplate="%{x}<br>세기 %{y:.3f}<extra></extra>"))
    has_cap = np.isfinite(caps)
    if has_cap.any():
        fig.add_trace(go.Scatter(
            x=[lab for lab, ok in zip(labels, has_cap) if ok], y=caps[has_cap], mode="markers",
            name="쾌적 상한 (시나리오)", marker=dict(color=CAP_COLOR, symbol="line-ew-open", size=28,
                                                 line=dict(width=3, color=CAP_COLOR)),
            hovertemplate="%{x}<br>상한 %{y:.2f}<extra></extra>"))
    if s_max is not None:
        # 막대 숫자와 겹치지 않게 글씨는 달지 않는다. 화면 설명(점선 = 장비 상한)은 호출자가 붙인다.
        fig.add_hline(y=s_max, line=dict(color="#607d8b", dash="dot", width=1))
    top = max(1.0, float(np.nanmax(s)), float(s_max or 0.0))
    # 배경·격자 색은 지정하지 않는다 (대시보드 밝은·어두운 테마를 그대로 따른다).
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), bargap=0.45,
                      yaxis=dict(title="출구 속도 배수", range=[0, top * 1.2]),
                      legend=dict(orientation="h", y=1.18, x=0.0))
    return fig


def figure_timeline(control: Mapping, *, height: int = 220) -> go.Figure:
    """`device_control` JSON 의 segments 를 시간 막대로. 단계는 총 풍량을, 전환은 회색으로."""
    fig = go.Figure()
    for seg in control["segments"]:
        start, end = float(seg["start_s"]), float(seg["end_s"])
        if seg["kind"] == "phase":
            label = f"{seg['phase']}단계"
            color = SLOT_COLOR
            hover = (f"{label}<br>{start:.1f}–{end:.1f} s<br>회전 {seg['torso_yaw_deg']:.0f}° · "
                     f"가슴 쪽 벽 {seg['chest_wall']}<br>총 풍량 {seg['total_flow_m3_min']:.1f} m³/min")
        else:
            label, color = "전환", TRANSITION_COLOR
            hover = f"전환<br>{start:.1f}–{end:.1f} s<br>팬 {control.get('transition_fans', 'off')}"
        fig.add_trace(go.Bar(
            x=[end - start], y=["운전"], base=[start], orientation="h", name=label,
            marker=dict(color=color, line=dict(color="white", width=2)),
            text=[label if end - start >= 1.2 else ""], textposition="inside", insidetextanchor="middle",
            hovertemplate=hover + "<extra></extra>", showlegend=False))
    fig.update_layout(barmode="overlay", height=height, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title="시간 (s)", range=[0, float(control["timeline_s"]) * 1.02]),
                      yaxis=dict(showticklabels=False))
    return fig
