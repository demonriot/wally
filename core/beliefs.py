from __future__ import annotations
import belief
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

import config
from core.perception.metrics import (
    clamp01,
    linear_map,
    band_map,
    compute_histogram_stats,
    compute_temporal_deviation,
)

def apply_continuous_decay(state, dt: float):
    state.stream_conf = belief.clamp01(state.stream_conf - belief.stream_dec(dt))
    if state.mode == "observe":
        state.map_conf = belief.clamp01(state.map_conf - belief.map_dec(dt))

def apply_stream_evidence(state, stream_status: int):
    state.stream_conf = belief.clamp01(state.stream_conf + belief.stream_inc(stream_status))

def apply_scan_boost(state):
    state.map_conf = belief.clamp01(state.map_conf + belief.map_scan_boost())

@dataclass
class ReliabilityResult:
    reliability: float
    label: str                   # "reject" | "suspect" | "trusted"
    vetoed: bool
    components: Dict[str, float]
    reasons: List[str]


def compute_frame_reliability(
    gray: np.ndarray,
    sharp_q: float,
    feature_vector: np.ndarray,
    recent_feature_vectors: List[np.ndarray] | None,
) -> ReliabilityResult:
    reasons: List[str] = []
    vetoed = False

    hist_stats = compute_histogram_stats(gray)
    mean_intensity = hist_stats["mean_intensity"]
    std_intensity = hist_stats["std_intensity"]

    # -------------------------
    # Hard vetoes
    # -------------------------
    if sharp_q < config.VETO_SHARP_MIN:
        vetoed = True
        reasons.append(f"sharp_q too low ({sharp_q:.3f})")

    if mean_intensity < config.VETO_MEAN_MIN:
        vetoed = True
        reasons.append(f"too dark ({mean_intensity:.1f})")

    if mean_intensity > config.VETO_MEAN_MAX:
        vetoed = True
        reasons.append(f"too bright ({mean_intensity:.1f})")

    if vetoed:
        return ReliabilityResult(
            reliability=0.0,
            label="reject",
            vetoed=True,
            components={"sharp": 0.0, "histo": 0.0, "temp": 0.0},
            reasons=reasons,
        )

    # -------------------------
    # Sharpness reliability
    # -------------------------
    r_sharp = linear_map(sharp_q, config.SHARP_LOW, config.SHARP_HIGH)

    # -------------------------
    # Histogram / exposure reliability
    # -------------------------
    r_mean = band_map(
        mean_intensity,
        lo=config.HIST_MEAN_LOW,
        hi=config.HIST_MEAN_HIGH,
        soft_lo=config.VETO_MEAN_MIN,
        soft_hi=config.VETO_MEAN_MAX,
    )

    r_std = linear_map(std_intensity, config.HIST_STD_LOW, config.HIST_STD_HIGH)

    r_histo = clamp01(0.6 * r_mean + 0.4 * r_std)

    # -------------------------
    # Temporal plausibility
    # -------------------------
    temp_dev = compute_temporal_deviation(feature_vector, recent_feature_vectors)

    if temp_dev is None:
        r_temp = 0.5
        reasons.append("insufficient temporal history")
    else:
        r_temp = 1.0 - linear_map(temp_dev, config.TEMP_DEV_GOOD, config.TEMP_DEV_BAD)
        r_temp = clamp01(r_temp)
        reasons.append(f"temp_dev={temp_dev:.3f}")

    # -------------------------
    # Weighted average
    # -------------------------
    wsum = config.REL_W_SHARP + config.REL_W_HISTO + config.REL_W_TEMP
    reliability = (
        config.REL_W_SHARP * r_sharp
        + config.REL_W_HISTO * r_histo
        + config.REL_W_TEMP * r_temp
    ) / wsum
    reliability = clamp01(reliability)

    # -------------------------
    # Label
    # -------------------------
    if reliability < config.REL_REJECT_THRESH:
        label = "reject"
    elif reliability < config.REL_SUSPECT_THRESH:
        label = "suspect"
    else:
        label = "trusted"

    return ReliabilityResult(
        reliability=reliability,
        label=label,
        vetoed=False,
        components={
            "sharp": r_sharp,
            "histo": r_histo,
            "temp": r_temp,
        },
        reasons=reasons,
    )
