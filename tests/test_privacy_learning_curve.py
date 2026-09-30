from __future__ import annotations

import numpy as np

from airis.realtime.camera import PoseDetection
from airis.realtime.privacy import draw_skeleton_only


def _det() -> PoseDetection:
    kp = np.array(
        [[20 + i * 3, 30 + i * 2] for i in range(17)],
        dtype=np.float64,
    )
    return PoseDetection(
        keypoints=kp,
        conf=np.ones(17, dtype=np.float64),
        box=np.array([10, 10, 100, 100], dtype=np.float64),
        image_size=(160, 120),
    )


def test_skeleton_only_uses_fresh_blank_canvas():
    bg = (17, 19, 23)
    out = draw_skeleton_only(_det(), background=bg)
    assert out.shape == (120, 160, 3)

    flat = out.reshape(-1, 3)
    assert np.any(np.all(flat == np.array(bg), axis=1))
    assert np.any(np.any(flat != np.array(bg), axis=1))


def test_skeleton_only_none_is_blank():
    bg = (3, 5, 7)
    out = draw_skeleton_only(None, image_size=(80, 60), background=bg)
    assert out.shape == (60, 80, 3)
    assert np.all(out == np.array(bg, dtype=np.uint8))


def test_stable_body_fold_is_nested_and_balanced():
    def stable_fold(body_idx: int, folds: int = 5, seed: int = 0) -> int:
        return (int(body_idx) + int(seed)) % int(folds)

    f100 = {i: stable_fold(i) for i in range(100)}
    f300 = {i: stable_fold(i) for i in range(300)}

    assert all(f100[i] == f300[i] for i in f100)
    counts = np.bincount(list(f100.values()), minlength=5)
    np.testing.assert_array_equal(counts, np.full(5, 20))
