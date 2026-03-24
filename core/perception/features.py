import cv2
import numpy as np

# Create ORB once (faster than recreating every call)
_ORB = cv2.ORB_create()


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _linear_map(x: float, x0: float, x1: float) -> float:
    if x1 <= x0:
        return 1.0 if x >= x1 else 0.0
    return _clamp01((x - x0) / (x1 - x0))


def _band_map(x: float, lo: float, hi: float, soft_lo: float, soft_hi: float) -> float:
    """
    Reliability is 1.0 inside [lo, hi], tapers to 0 outside using soft bounds.
    """
    if x < soft_lo or x > soft_hi:
        return 0.0
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        return _linear_map(x, soft_lo, lo)
    return _linear_map(soft_hi - x, 0.0, soft_hi - hi)


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

    mean_intensity = float(np.mean(resized))
    std_intensity = float(np.std(resized))

    features = {
        "gray": resized,
        "sharp_raw": sharp_raw,
        "sharp_q": sharp_q,
        "edge_raw": edge_raw,
        "edge_q": edge_q,
        "kp_raw": kp_raw,
        "kp_q": kp_q,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "blur_ksize_used": k,
    }

    feature_vector = np.array([sharp_q, edge_q, kp_q], dtype=np.float32)
    return features, feature_vector


def compute_temporal_deviation(feature_vector, recent_feature_vectors):
    """
    Mean absolute deviation from recent local mean.
    Returns None if there is not enough history.
    """
    if recent_feature_vectors is None or len(recent_feature_vectors) == 0:
        return None

    hist = np.asarray(recent_feature_vectors, dtype=np.float32)
    if hist.ndim != 2 or hist.shape[0] == 0:
        return None

    recent_mean = np.mean(hist, axis=0)
    dev = float(np.mean(np.abs(feature_vector - recent_mean)))
    return dev


def compute_reliability(features, feature_vector, recent_feature_vectors, cfg):
    """
    Returns:
        reliability (float): [0,1]
        label (str): reject | suspect | trusted
        reasons (list[str])
        comps (dict): component reliabilities for logging
    """
    reasons = []

    sharp_q = float(features["sharp_q"])
    mean_intensity = float(features["mean_intensity"])
    std_intensity = float(features["std_intensity"])

    # -------------------------
    # Hard vetoes
    # -------------------------
    if sharp_q < cfg.veto_sharp_min:
        return 0.0, "reject", [f"veto_blur({sharp_q:.3f})"], {
            "r_sharp": 0.0,
            "r_histo": 0.0,
            "r_temp": 0.0,
            "temp_dev": None,
        }

    if mean_intensity < cfg.veto_mean_min:
        return 0.0, "reject", [f"veto_dark({mean_intensity:.1f})"], {
            "r_sharp": 0.0,
            "r_histo": 0.0,
            "r_temp": 0.0,
            "temp_dev": None,
        }

    if mean_intensity > cfg.veto_mean_max:
        return 0.0, "reject", [f"veto_bright({mean_intensity:.1f})"], {
            "r_sharp": 0.0,
            "r_histo": 0.0,
            "r_temp": 0.0,
            "temp_dev": None,
        }

    # -------------------------
    # Component reliabilities
    # -------------------------
    r_sharp = _linear_map(sharp_q, cfg.sharp_low, cfg.sharp_high)

    r_mean = _band_map(
        mean_intensity,
        lo=cfg.hist_mean_low,
        hi=cfg.hist_mean_high,
        soft_lo=cfg.veto_mean_min,
        soft_hi=cfg.veto_mean_max,
    )
    r_std = _linear_map(std_intensity, cfg.hist_std_low, cfg.hist_std_high)
    r_histo = _clamp01(0.6 * r_mean + 0.4 * r_std)

    temp_dev = compute_temporal_deviation(feature_vector, recent_feature_vectors)
    if temp_dev is None:
        r_temp = 0.5
        reasons.append("temp_neutral")
    else:
        r_temp = 1.0 - _linear_map(temp_dev, cfg.temp_dev_good, cfg.temp_dev_bad)
        r_temp = _clamp01(r_temp)
        reasons.append(f"temp_dev={temp_dev:.3f}")

    # -------------------------
    # Weighted average
    # -------------------------
    wsum = cfg.rel_w_sharp + cfg.rel_w_histo + cfg.rel_w_temp
    reliability = (
        cfg.rel_w_sharp * r_sharp +
        cfg.rel_w_histo * r_histo +
        cfg.rel_w_temp * r_temp
    ) / max(wsum, 1e-8)
    reliability = _clamp01(reliability)

    if reliability < cfg.rel_reject_thresh:
        label = "reject"
    elif reliability < cfg.rel_suspect_thresh:
        label = "suspect"
    else:
        label = "trusted"

    comps = {
        "r_sharp": r_sharp,
        "r_histo": r_histo,
        "r_temp": r_temp,
        "temp_dev": temp_dev,
    }
    return reliability, label, reasons, comps


def is_observation_valid(features, feature_vector, recent_feature_vectors, cfg):
    """
    New admission-style validity check.
    Returns:
        valid (bool)
        reason (str)
        reliability (float)
        label (str)
        comps (dict)
    """
    reliability, label, reasons, comps = compute_reliability(
        features=features,
        feature_vector=feature_vector,
        recent_feature_vectors=recent_feature_vectors,
        cfg=cfg,
    )

    valid = (label != "reject")
    reason = ",".join(reasons) if reasons else label
    return valid, reason, reliability, label, comps