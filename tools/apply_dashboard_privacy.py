#!/usr/bin/env python3
"""Patch scripts/run_dashboard.py for Skeleton-only privacy UI.

Run from anywhere:
    python tools/apply_dashboard_privacy.py /path/to/project-AIRIS-v2

The script expects the current main structure observed on 2026-09-30.
Review `git diff` after applying.
"""
from __future__ import annotations

import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected 1 match, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: python apply_dashboard_privacy.py /path/to/project-AIRIS-v2",
            file=sys.stderr,
        )
        return 2

    repo = Path(sys.argv[1]).resolve()
    path = repo / "scripts" / "run_dashboard.py"
    text = path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "from airis.realtime import camera, pose_estimate as pe               # noqa: E402\n",
        "from airis.realtime import camera, pose_estimate as pe               # noqa: E402\n"
        "from airis.realtime.privacy import draw_skeleton_only                # noqa: E402\n",
        "privacy import",
    )

    text = replace_once(
        text,
        '        frames = st.session_state.get("local_frames", [])\n',
        '        frames = st.session_state.pop("local_frames", [])\n',
        "local camera session cleanup",
    )

    old = (
        '            st.image(camera.draw_pose(canvas, det), '
        'caption="합성 마네킹 키포인트 (몸 모델 관절을 정면 카메라에 투영한 가짜 입력)",\n'
        '                     **WIDE)\n'
    )
    new = (
        '            st.image(draw_skeleton_only(det), '
        'caption="합성 마네킹 키포인트 (빈 배경 Skeleton)",\n'
        '                     **WIDE)\n'
    )
    text = replace_once(text, old, new, "synthetic skeleton output")

    old = (
        '                    st.image(camera.draw_pose(last_img, last_det), caption=desc,\n'
        '                             **WIDE)\n'
    )
    new = (
        '                    st.image(draw_skeleton_only(\n'
        '                        last_det, image_size=(last_img.shape[1], last_img.shape[0])),\n'
        '                        caption=(desc or "") + " · 원본 미표시 / Skeleton only",\n'
        '                        **WIDE)\n'
    )
    text = replace_once(text, old, new, "camera skeleton output")

    old = (
        '        st.caption("카메라 영상은 저장하지 않습니다. 시나리오(임산부·휠체어)는 영상으로 판별하지 않고 "\n'
        '                   "사용자가 직접 고릅니다.")\n'
    )
    new = (
        '        st.caption("원본 프레임은 YOLO 포즈 추론 동안만 사용하고 화면에는 표시하지 않습니다. "\n'
        '                   "표시는 빈 배경 Skeleton/체형값만 사용합니다. 시나리오(임산부·휠체어)는 "\n'
        '                   "영상으로 판별하지 않고 사용자가 직접 고릅니다.")\n'
    )
    text = replace_once(text, old, new, "privacy caption")

    path.write_text(text, encoding="utf-8")
    print("patched:", path)
    print("review: git diff -- scripts/run_dashboard.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
