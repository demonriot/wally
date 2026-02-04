# core/perception/metrics.py
import cv2
import numpy as np

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
