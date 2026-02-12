import cv2
import numpy as np

# Create ORB once (faster than recreating every call)
_ORB = cv2.ORB_create()


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def extract_features(frame, cfg):
    """
    Returns:
        features (dict): raw + normalized metrics
        feature_vector (np.ndarray): normalized metrics in [0,1]
            [sharp_q, edge_q, kp_q]
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (cfg.diff_resize_w, cfg.diff_resize_h))

    # Ensure blur kernel is valid: odd and >= 3
    k = int(getattr(cfg, "diff_blur_ksize", 3))
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1

    blurred = cv2.GaussianBlur(resized, (k, k), 0)

    # Sharpness: compute on *unblurred* resized to detect blur properly
    lap = cv2.Laplacian(resized, cv2.CV_64F)
    sharp_raw = float(lap.var())

    # Edges + keypoints: compute on blurred for robustness
    edges = cv2.Canny(blurred, 50, 150)
    edge_raw = float(np.sum(edges > 0) / edges.size)

    keypoints, _descriptors = _ORB.detectAndCompute(blurred, None)
    kp_raw = float(len(keypoints))

    # Normalize (refs must be set sensibly in cfg)
    sharp_ref = float(getattr(cfg, "lap_ref", 1.0)) or 1.0
    edge_ref = float(getattr(cfg, "edge_ref", 1.0)) or 1.0
    kp_ref = float(getattr(cfg, "keypoint_ref", 1.0)) or 1.0

    sharp_q = _clamp01(sharp_raw / sharp_ref)
    edge_q = _clamp01(edge_raw / edge_ref)
    kp_q = _clamp01(kp_raw / kp_ref)

    features = {
        "sharp_raw": sharp_raw,
        "sharp_q": sharp_q,
        "edge_raw": edge_raw,
        "edge_q": edge_q,
        "kp_raw": kp_raw,
        "kp_q": kp_q,
        "blur_ksize_used": k,
    }

    feature_vector = np.array([sharp_q, edge_q, kp_q], dtype=np.float32)
    return features, feature_vector

def is_observation_valid(features, cfg):
    sharp_ok = features["sharp_q"] >= cfg.sharp_min
    structure_ok = (features["edge_q"] >= cfg.edge_min) or (features["kp_q"] >= cfg.kp_min)
    valid = sharp_ok and structure_ok

    if not sharp_ok and not structure_ok:
        reason = "blurry_no_structure"
    elif not sharp_ok:
        reason = "blurry"
    elif not structure_ok:
        reason = "no_structure"
    else:
        reason = "ok"


    

    return valid, reason
