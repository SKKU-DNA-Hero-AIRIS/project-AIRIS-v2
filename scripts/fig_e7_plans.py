"""E7(운전 계획) 그림 3장. 소유자: C. 발표용이라 **쉬운 말**만 쓴다.

    python scripts/fig_e7_plans.py                      # docs/figures/ 에 png + svg

읽는 파일은 둘뿐이다 (수치를 손으로 옮기지 않는다):
    docs/e7_reference.json         본 실행 (단계 2개, 가중 4개)
    docs/e7_sweep_reference.json   단계 수 스윕 · N=9 · 회전 기준선 표

용어는 `outputs/presentation/INDEX.md` 대조표를 따른다 — 내부 약어(P1opt, P5, N, w, B1 …)는
축·범례·제목에 쓰지 않고 괄호 안 보조 표기로만 남긴다.

**에너지 가중(w)이 다른 수치를 한 축에 섞지 않는다.** 그림 ①은 w 0.01, 그림 ②는 w 0.1 이고
각 그림에 그 값을 적는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 발표용 이름 (INDEX.md 용어 대조표).
LABELS = {
    "P1": "제조사 안내: 서서 몸 돌리기",
    "P1opt": "제안 자세 그대로 몸 돌리기",
    "P1down": "팔 내린 제안 자세로 몸 돌리기",
    "P5": "자세를 바꿔 가는 계획",
}
SCENARIO_LABELS = {"default": "서서 이용", "pregnant": "임산부", "wheelchair": "휠체어"}
#: 색: 기준선은 회색 계열, 계획은 파랑.
COLORS = {"P1": "#8a8f98", "P1down": "#b08968", "P1opt": "#4c6ef5", "P5": "#0b7285"}


def use_korean_font() -> None:
    import matplotlib
    from matplotlib import font_manager

    matplotlib.rcParams["axes.unicode_minus"] = False
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "Noto Sans KR", "NanumGothic", "Gulim", "Dotum"):
        if name in have:
            matplotlib.rcParams["font.family"] = name
            return
    print("한글 글꼴을 찾지 못했다 — 라벨이 깨질 수 있다", file=sys.stderr)


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} 가 없다. scripts/export_e7_reference.py 로 먼저 만든다")
    return json.loads(path.read_text(encoding="utf-8"))


def pick(rows: list[dict], **kw) -> dict | None:
    """조건에 맞는 행 하나 (없으면 None)."""
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  {out_dir / name}.png / .svg")


# ---------- ① 자세를 바꾸는 횟수와 점수 ----------

def fig_phase_count(sweep: dict, main: dict, out_dir: Path) -> None:
    """가로축 = 자세를 바꾸는 횟수, 세로축 = 점수. 수평선은 몸만 돌리는 두 방식."""
    import matplotlib.pyplot as plt

    rows, main_rows = sweep["rows"], main["rows"]
    counts = [2, 3, 4, 6]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), sharey=False)
    for ax, (key, title) in zip(axes, SCENARIO_LABELS.items()):
        ys = []
        for n in counts:
            if n == 2:                      # 단계 2개는 본 실행에서 (같은 가중 0.01)
                r = pick(main_rows, scenario=key, condition="P5", energy_weight=0.01)
            else:
                r = pick(rows, scenario=key, condition="P5", energy_weight=0.01, n_phases=n)
            ys.append(r["score"] if r else float("nan"))
        ax.plot(counts, ys, "o-", color=COLORS["P5"], lw=2.2, ms=7,
                label=LABELS["P5"], zorder=3)
        for cond in ("P1opt", "P1"):
            r = pick(rows, scenario=key, condition=cond, energy_weight=0.01)
            if r:
                ax.axhline(r["score"], color=COLORS[cond], ls="--", lw=1.6,
                           label=f"{LABELS[cond]} (12방향)")
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("자세를 바꾸는 횟수")
        ax.set_xticks(counts)
        ax.grid(alpha=0.25, ls=":")
    axes[0].set_ylabel("점수 (시뮬레이션 상대값)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("자세를 여러 번 바꿀수록 좋아진다 — 3번부터 제조사 안내를 넘어선다",
                 fontsize=13.5, y=1.02)
    fig.text(0.5, -0.12, "에너지 절약 가중 0.01 로 맞춘 비교. 제안 자세로 도는 방식(파란 선)은 "
                         "여기까지는 넘지 못하고, 9번에서 넘어선다(다음 그림). 점수는 시뮬레이션 "
                         "안의 상대값이며 실제 먼지 제거율(%)이 아니다.",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    save(fig, out_dir, "fig_e7_phase_count")
    plt.close(fig)


# ---------- ② 같은 횟수에서: 한 자세로 돌기 vs 자세를 바꾸기 ----------

def fig_same_count(sweep: dict, main: dict, out_dir: Path) -> None:
    """9번으로 맞춘 비교. 점수 + 제거 효과·에너지·불편도 보조 패널."""
    import matplotlib.pyplot as plt
    import numpy as np

    conds = [("P1_9", "P1"), ("P1opt_9", "P1opt"), ("P5", "P5")]
    metrics = [("score", "점수"), ("total_removal", "먼지 제거 효과"),
               ("energy", "바람 에너지"), ("discomfort", "자세 불편도")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    x = np.arange(len(SCENARIO_LABELS))
    width = 0.26
    for ax, (col, mtitle) in zip(axes, metrics):
        for i, (cond, color_key) in enumerate(conds):
            vals = []
            for key in SCENARIO_LABELS:
                r = pick(sweep["rows"], scenario=key, condition=cond,
                         energy_weight=0.1, n_phases=9)
                vals.append(r[col] if r else float("nan"))
            ax.bar(x + (i - 1) * width, vals, width, color=COLORS[color_key],
                   label=LABELS[color_key] + (" (9방향)" if cond != "P5" else " (9단계)"))
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIO_LABELS.values(), fontsize=10)
        ax.set_title(mtitle, fontsize=12)
        ax.grid(axis="y", alpha=0.25, ls=":")
    axes[0].set_ylabel("시뮬레이션 상대값")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("같은 9번이라도 자세를 바꾸면 더 낫다 — 먼지는 더 떼고 바람은 덜 쓴다",
                 fontsize=13.5, y=1.02)
    fig.text(0.5, -0.12, "에너지 절약 가중 0.1. 자세를 바꾸는 계획은 불편도만 높다. "
                         "참고: 자세를 2번만 바꾸면 점수가 0.85 / 0.45 / 0.55 로 떨어진다.",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    save(fig, out_dir, "fig_e7_same_count")
    plt.close(fig)


# ---------- ③ 몸만 돌릴 때 어떤 자세로 도는가 ----------

def fig_rotation_pose(sweep: dict, out_dir: Path) -> None:
    """같은 10방향 회전을 세 자세로 — 기본 · 팔 내린 제안 자세 · 제안 자세."""
    import matplotlib.pyplot as plt
    import numpy as np

    conds = [("P1_10", "P1"), ("P1down_10", "P1down"), ("P1opt_10", "P1opt")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(SCENARIO_LABELS))
    width = 0.26
    for i, (cond, color_key) in enumerate(conds):
        scores, removals = [], []
        for key in SCENARIO_LABELS:
            r = pick(sweep["rows"], scenario=key, condition=cond, energy_weight=0.0)
            scores.append(r["score"] if r else float("nan"))
            removals.append(r["total_removal"] if r else float("nan"))
        ax.bar(x + (i - 1) * width, scores, width, color=COLORS[color_key], label=LABELS[color_key])
        ax2.bar(x + (i - 1) * width, removals, width, color=COLORS[color_key])
    for a, title in ((ax, "점수"), (ax2, "먼지 제거 효과")):
        a.set_xticks(x)
        a.set_xticklabels(SCENARIO_LABELS.values(), fontsize=10)
        a.set_title(title, fontsize=12)
        a.grid(axis="y", alpha=0.25, ls=":")
    ax.set_ylabel("시뮬레이션 상대값")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("몸을 돌릴 때 어떤 자세로 도는가 — 제안 자세로 돌면 좋아진다 (휠체어는 차이가 작다)",
                 fontsize=13.5, y=1.0)
    fig.text(0.5, -0.12, "10방향 × 2초, 바람 세기 최대, 에너지 절약 가중 0 (세 방식 모두 같은 "
                         "시간·세기라 가중과 무관하게 순서가 같다). 팔 내린 자세는 키가 커서 "
                         "만세가 천장에 닿는 사람을 위한 대체안이다.",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    save(fig, out_dir, "fig_e7_rotation_pose")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E7 그림 3장")
    ap.add_argument("--main-ref", default=str(ROOT / "docs" / "e7_reference.json"))
    ap.add_argument("--sweep-ref", default=str(ROOT / "docs" / "e7_sweep_reference.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "docs" / "figures"))
    ap.add_argument("--only", choices=["count", "same", "pose"], default=None)
    args = ap.parse_args(argv)

    use_korean_font()
    main_ref, sweep_ref = load(Path(args.main_ref)), load(Path(args.sweep_ref))
    out_dir = Path(args.out_dir)
    print("그림 저장")
    if args.only in (None, "count"):
        fig_phase_count(sweep_ref, main_ref, out_dir)
    if args.only in (None, "same"):
        fig_same_count(sweep_ref, main_ref, out_dir)
    if args.only in (None, "pose"):
        fig_rotation_pose(sweep_ref, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
