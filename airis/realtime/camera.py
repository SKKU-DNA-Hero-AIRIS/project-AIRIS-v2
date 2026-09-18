"""카메라·영상·이미지 입력과 YOLO11-pose 키포인트 추출. 소유자: E. (`docs/tracks/E_realtime.md` 단계 3)

v1(project-AIRIS-MVP, `C:\\Users\\SEONGWOO\\Documents\\AIRIS`)에서 가져온 것:
- 포즈 모델 내려받기·로드: `posture_analysis.load_pose_model` (`attempt_download_asset`)
- 여러 사람 중 가장 큰 사람 고르기: `posture_analysis.PostureProcessor._largest_pose`
- 뼈대 그리기: `posture_analysis._draw_pose`, `SKELETON` (cv2 대신 PIL 로 옮김)
- 영상 프레임 읽기: `vision_demo.analyze_video` 의 `cv2.VideoCapture` 루프
생체·시설 맥락 모듈은 가져오지 않는다.

`ultralytics`·`cv2`는 함수 안에서 지연 import 한다 (테스트·import 가 가볍게). 카메라는 대시보드에서만 연다.
모델 가중치는 `data/models/`(git 밖, `*.pt` 는 .gitignore)에 첫 실행 때 내려받는다.
"""
from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .pose_estimate import MIN_CONF, SKELETON

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "data" / "models"
POSE_WEIGHTS = "yolo11n-pose.pt"


@dataclass
class PoseDetection:
    """사람 한 명의 키포인트. `keypoints` (17,2) px, `conf` (17,), `box` (4,) xyxy px, `image_size` (w, h)."""
    keypoints: np.ndarray
    conf: np.ndarray
    box: np.ndarray
    image_size: tuple[int, int]
    box_conf: float = 1.0


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
def pose_weights_path(weights: str = POSE_WEIGHTS, model_dir: Path | str = MODEL_DIR) -> Path:
    return Path(model_dir) / weights


def load_pose_model(weights: str = POSE_WEIGHTS, model_dir: Path | str = MODEL_DIR):
    """YOLO11-pose 모델. 가중치가 없으면 `data/models/`로 내려받는다 (v1 `load_pose_model`)."""
    from ultralytics import YOLO                                   # 지연 import
    from ultralytics.utils.downloads import attempt_download_asset

    path = pose_weights_path(weights, model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path = Path(attempt_download_asset(path))
    return YOLO(str(path))


def _to_numpy(x: Any) -> np.ndarray:
    """torch 텐서·numpy 모두 받는다."""
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float64)


def detections_from_result(result: Any) -> list[PoseDetection]:
    """ultralytics `Results` 하나 → 사람별 `PoseDetection` 목록 (상자 넓이 큰 순)."""
    kps = getattr(result, "keypoints", None)
    if kps is None or len(kps.xy) == 0:
        return []
    xy = _to_numpy(kps.xy).reshape(-1, 17, 2)
    conf = (_to_numpy(kps.conf).reshape(-1, 17) if getattr(kps, "conf", None) is not None
            else np.ones(xy.shape[:2]))
    boxes = _to_numpy(result.boxes.xyxy).reshape(-1, 4)
    box_conf = (_to_numpy(result.boxes.conf).reshape(-1) if getattr(result.boxes, "conf", None) is not None
                else np.ones(boxes.shape[0]))
    h, w = result.orig_shape[:2]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-area)
    return [PoseDetection(xy[i], conf[i], boxes[i], (int(w), int(h)), float(box_conf[i]))
            for i in order]


def largest_person(result: Any) -> PoseDetection | None:
    """가장 큰 사람 (v1 `_largest_pose`). 게이트 앞에 선 사람이 보통 가장 크다."""
    dets = detections_from_result(result)
    return dets[0] if dets else None


def detect_pose(model, image_bgr: np.ndarray, *, conf: float = 0.25,
                imgsz: int = 640) -> PoseDetection | None:
    """이미지 (H,W,3) BGR uint8 → 가장 큰 사람의 키포인트. 사람이 없으면 None."""
    result = model.predict(image_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
    return largest_person(result)


# ---------------------------------------------------------------------------
# 입력
# ---------------------------------------------------------------------------
def read_image(source: bytes | str | Path | Any) -> np.ndarray:
    """이미지 파일·바이트·파일 객체(streamlit UploadedFile 등) → (H,W,3) BGR uint8.

    PIL 로 읽는다 (EXIF 회전 반영). YOLO 는 cv2 관례대로 BGR 을 받는다.
    """
    from PIL import Image, ImageOps

    if isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    elif hasattr(source, "read"):
        img = Image.open(io.BytesIO(source.read()))
    else:
        img = Image.open(Path(source))
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])


def iter_video_frames(path: str | Path, *, every_n: int = 5,
                      max_frames: int = 30) -> Iterator[np.ndarray]:
    """영상 파일에서 `every_n` 프레임마다 하나씩, 최대 `max_frames`개 (BGR). v1 `analyze_video` 루프."""
    import cv2                                                     # 지연 import

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"영상을 열 수 없다: {path}")
    try:
        idx = yielded = 0
        while yielded < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % every_n == 0:
                yielded += 1
                yield frame
            idx += 1
    finally:
        cap.release()


def capture_frames(device: int = 0, *, n: int = 15, every_n: int = 2,
                   warmup: int = 5) -> list[np.ndarray]:
    """로컬 웹캠에서 프레임 n개 (BGR). 대시보드에서만 부른다 (테스트에서 카메라를 열지 않는다)."""
    import cv2                                                     # 지연 import

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise IOError(f"카메라 {device} 를 열 수 없다")
    frames: list[np.ndarray] = []
    try:
        i = 0
        while len(frames) < n and i < warmup + n * every_n * 3:
            ok, frame = cap.read()
            if ok and i >= warmup and (i - warmup) % every_n == 0:
                frames.append(frame)
            i += 1
    finally:
        cap.release()
    return frames


def detect_marker_scale(image_bgr: np.ndarray, marker_size_m: float,
                        dictionary: str = "DICT_4X4_50") -> float | None:
    """바닥 ArUco 마커가 보이면 m/px (`pose_estimate.scale_from_marker`), 없으면 None. 선택 기능."""
    import cv2                                                     # 지연 import

    from .pose_estimate import scale_from_marker

    if not hasattr(cv2, "aruco"):
        return None
    aruco = cv2.aruco
    detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(getattr(aruco, dictionary)),
                                   aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image_bgr)
    if ids is None or len(corners) == 0:
        return None
    return scale_from_marker(np.asarray(corners[0]).reshape(4, 2), marker_size_m)


# ---------------------------------------------------------------------------
# 그리기
# ---------------------------------------------------------------------------
def draw_pose(image_bgr: np.ndarray, det: PoseDetection | None, *,
              min_conf: float = MIN_CONF, line_width: int | None = None) -> np.ndarray:
    """키포인트·뼈대·상자를 겹친 RGB 이미지 (streamlit `st.image` 용). v1 `_draw_pose` 를 PIL 로."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.ascontiguousarray(image_bgr[:, :, ::-1]))
    if det is None:
        return np.asarray(img)
    draw = ImageDraw.Draw(img)
    lw = line_width or max(2, int(round(min(img.size) / 250)))
    kp, conf = det.keypoints, det.conf
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
