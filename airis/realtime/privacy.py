"""Privacy-safe pose visualization for AIRIS.

Raw camera frames are used only for YOLO pose inference. This renderer creates
a fresh solid-color RGB canvas and draws only PoseDetection keypoints/skeleton.
It never receives or copies the original image pixels.
"""
from __future__ import annotations

import numpy as np

from .camera import PoseDetection
from .pose_estimate import MIN_CONF, SKELETON


def draw_skeleton_only(
    det: PoseDetection | None,
    *,
    image_size: tuple[int, int] | None = None,
    min_conf: float = MIN_CONF,
    line_width: int | None = None,
    background: tuple[int, int, int] = (245, 245, 245),
    show_box: bool = False,
) -> np.ndarray:
    """Return an RGB image containing only the detected skeleton.

    Parameters
    ----------
    det:
        YOLO pose result converted to PoseDetection.
    image_size:
        (width, height). Required only when det is None.
    background:
        Solid RGB background.
    show_box:
        Draw a bounding box if desired. False by default for privacy UI.
    """
    from PIL import Image, ImageDraw

    if det is not None:
        w, h = det.image_size
    elif image_size is not None:
        w, h = image_size
    else:
        raise ValueError("det 또는 image_size 가 필요합니다.")

    img = Image.new("RGB", (int(w), int(h)), tuple(int(v) for v in background))
    if det is None:
        return np.asarray(img)

    draw = ImageDraw.Draw(img)
    lw = line_width or max(2, int(round(min(w, h) / 250)))
    kp, conf = det.keypoints, det.conf

    if show_box:
        x1, y1, x2, y2 = (float(v) for v in det.box)
        draw.rectangle([x1, y1, x2, y2], outline=(75, 220, 120), width=lw)

    for a, b in SKELETON:
        if conf[a] >= min_conf and conf[b] >= min_conf:
            draw.line([tuple(kp[a]), tuple(kp[b])], fill=(84, 214, 199), width=lw)

    r = lw * 2
    for (x, y), c in zip(kp, conf):
        color = (255, 190, 70) if c >= min_conf else (160, 160, 160)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    return np.asarray(img)
