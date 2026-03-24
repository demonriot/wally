# core/perception/metrics.py
from __future__ import annotations

import cv2
import numpy as np
from typing import List, Optional, Dict

def frame_diff_mad(prev_frame, curr_frame, resize_wh=(160, 120), blur_ksize=3):
    """
    Mean Absolute Difference (MAD) between two frames.

    Returns a float in roughly [0, 255].
    Higher = more visual change (novelty).

    prev_frame, curr_frame: BGR frames from OpenCV (np.ndarray)
    resize_wh: (width, height) for fast comparison
    blur_ksize: 0 to disable, or odd int like 3/5
    """
    if prev_frame is None or curr_frame is None:
        return 0.0

    # Resize small for speed + noise reduction
    pw, ph = resize_wh
    a = cv2.resize(prev_frame, (pw, ph), interpolation=cv2.INTER_AREA)
    b = cv2.resize(curr_frame, (pw, ph), interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    # Optional tiny blur to suppress RTSP compression noise
    if blur_ksize and blur_ksize >= 3:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        a = cv2.GaussianBlur(a, (blur_ksize, blur_ksize), 0)
        b = cv2.GaussianBlur(b, (blur_ksize, blur_ksize), 0)

    # Mean absolute pixel difference
    diff = cv2.absdiff(a, b)
    return float(np.mean(diff))

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def linear_map(x: float, x0: float, x1: float) -> float:
    if x1 <= x0:
        return 1.0 if x >= x1 else 0.0
    return clamp01((x - x0) / (x1 - x0))


def band_map(x: float, lo: float, hi: float, soft_lo: float, soft_hi: float) -> float:
    """
    Returns high reliability inside [lo, hi], tapers to 0 outside using soft bounds.
    """
    if x < soft_lo or x > soft_hi:
        return 0.0
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        return linear_map(x, soft_lo, lo)
    return linear_map(soft_hi - x, 0.0, soft_hi - hi)


def compute_histogram_stats(gray: np.ndarray) -> Dict[str, float]:
    return {
        "mean_intensity": float(np.mean(gray)),
        "std_intensity": float(np.std(gray)),
    }


def compute_temporal_deviation(
    feature_vector: np.ndarray,
    recent_feature_vectors: Optional[List[np.ndarray]],
) -> Optional[float]:
    """
    Mean absolute deviation from recent local feature mean.
    """
    if not recent_feature_vectors:
        return None

    hist = np.asarray(recent_feature_vectors, dtype=np.float32)
    if hist.ndim != 2 or hist.shape[0] == 0:
        return None

    recent_mean = np.mean(hist, axis=0)
    dev = float(np.mean(np.abs(feature_vector - recent_mean)))
    return dev
