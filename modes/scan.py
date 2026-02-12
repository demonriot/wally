# modes/scan.py
from dataclasses import dataclass, field
from datetime import datetime

from core import beliefs, logger
from core.perception.metrics import frame_diff_mad
from core.perception.features import extract_features
from core.perception.features import is_observation_valid
import numpy as np

@dataclass
class ScanRuntime:
    """Holds scan progress across loop iterations."""
    episode: int = 0
    step_index: int = 0
    phase: str = "idle"
    phase_started_at: float = 0.0

    pause_sampled: bool = False
    prev_pause_frame = None  # will hold an image frame
    diffs: list[float] = field(default_factory=list)  # novelty per pause
    stored_repr = None

def enter(state, now_t: float, cfg):
    """
    Called when mode switches into scan.
    Initializes scan runtime info in the state if absent.
    """
    if not hasattr(state, "scan_rt") or state.scan_rt is None:
        state.scan_rt = ScanRuntime()

    state.scan_rt.phase = "rotate"
    state.scan_rt.phase_started_at = now_t

    # reset counts on entry (optional: you can keep across scan sessions)
    state.scan_rt.episode = 0
    state.scan_rt.step_index = 0

    logger.log_event(state.log_path, datetime.now(), "scan_enter", state, notes="enter scan mode")

    rt = state.scan_rt
    rt.pause_sampled = False
    rt.prev_pause_frame = None
    rt.diffs.clear()



def exit(state, now_t: float, cfg):
    """Called when leaving scan mode."""
    logger.log_event(state.log_path, datetime.now(), "scan_exit", state, notes="exit scan mode")
    # Keep runtime for debugging, or set to None if you prefer:
    # state.scan_rt = None


def step(state, now_t: float, cfg, rotate_fn=None, sample_frame_fn=None):
    """
    Non-blocking scan execution.

    rotate_fn: optional callback like rotate_fn(degrees:int) -> None
              If None, rotation is simulated (no hardware).

    sample_frame_fn: optional callback like sample_frame_fn() -> (ok: bool, frame)
        Recommended: lambda: cam.read_latest(cfg.scan_flush_s)
    """
    rt = state.scan_rt
    if rt is None:
        # safety fallback
        state.scan_rt = ScanRuntime()
        rt = state.scan_rt
        rt.phase = "rotate"
        rt.phase_started_at = now_t
        rt.pause_sampled = False
        rt.prev_pause_frame = None
        rt.diffs.clear()

    # If map_conf already recovered, scan is effectively done
    if state.map_conf >= cfg.scan_exit_thresh:
        logger.log_event(state.log_path, datetime.now(), "scan_done", state, notes="map_conf recovered")
        return "observe"

    # Failsafe: too many episodes and still low
    if rt.episode >= cfg.scan_max_episodes:
        logger.log_event(
            state.log_path, datetime.now(), "scan_failed", state,
            notes=f"max_episodes={cfg.scan_max_episodes} reached"
        )
        return "observe"

    # ---- Phase machine ----
    if rt.phase == "rotate":
        # Perform 90-degree turn (one scan step)
        degrees = cfg.scan_step_degrees

        # Hardware integration point:
        if rotate_fn is not None:
            rotate_fn(degrees)  # (blocking is OK for now)

        logger.log_event(
            state.log_path, datetime.now(),
            "scan_step_rotate",
            state,
            notes=f"episode={rt.episode+1} step={rt.step_index+1}/4 degrees={degrees}"
        )

        # Move to pause phase
        rt.phase = "pause"
        rt.phase_started_at = now_t
        rt.pause_sampled = False  # IMPORTANT: allow one sample in this pause
        return None

    if rt.phase == "pause":
        elapsed = now_t - rt.phase_started_at

        # ---- NEW: sample ONCE during the pause, after settle time ----
        # Requires cfg.scan_settle_s (e.g., 0.25) and cfg.scan_pause_s (e.g., 1.0).
        if (not rt.pause_sampled) and (elapsed >= cfg.scan_settle_s):
            ok, frame = (False, None)
            if sample_frame_fn is not None:
                ok, frame = sample_frame_fn()

            if ok and frame is not None:
                # Minimal novelty placeholder for now:
                features, feature_vector = extract_features(frame, cfg)

                valid, reason = is_observation_valid(features, cfg)
                if valid:
                    if rt.stored_repr is None:
                        rt.stored_repr = feature_vector
                        logger.log_event(state.log_path, datetime.now(), "scan_repr_init", state, notes='initial representation stored')
                    else:
                        e = float(np.mean(np.abs(feature_vector - rt.stored_repr)))
                        logger.log_event(state.log_path, datetime.now(), "scan_pred_error", state, notes=f"pred error e={e:.3f}")
                        rt.stored_repr = ((1-cfg.alpha) * rt.stored_repr) + (cfg.alpha * feature_vector)  # update stored repr with smoothing
                        rt.stored_repr = rt.stored_repr.astype(np.float32, copy=False)
                        logger.log_event(state.log_path, datetime.now(), "scan_repr_update", state, notes=f"stored_repr updated with alpha={cfg.alpha}")
                    notes = (
                            f"episode={rt.episode+1} step={rt.step_index+1}/4 "
                            f"sharp_raw={features['sharp_raw']:.1f} sharp_q={features['sharp_q']:.2f} | "
                            f"edge_raw={features['edge_raw']:.4f} edge_q={features['edge_q']:.2f} | "
                            f"kp_raw={features['kp_raw']:.0f} kp_q={features['kp_q']:.2f} | "
                            f"valid={valid} reason={reason}"
                            )
                    logger.log_event(state.log_path, datetime.now(), "scan_features", state, notes=notes)
                else:
                    notes = (
                            f"episode={rt.episode+1} step={rt.step_index+1}/4 "
                            f"sharp_raw={features['sharp_raw']:.1f} sharp_q={features['sharp_q']:.2f} | "
                            f"edge_raw={features['edge_raw']:.4f} edge_q={features['edge_q']:.2f} | "
                            f"kp_raw={features['kp_raw']:.0f} kp_q={features['kp_q']:.2f} | "
                            f"valid={valid} reason={reason}"
                            )
                    logger.log_event(state.log_path, datetime.now(), "scan_features_invalid", state, notes=notes)

                # first pause sample just initializes prev frame; subsequent samples record a diff entry.
                if rt.prev_pause_frame is None:
                    diff = 0.0
                else:
                    diff = frame_diff_mad(
                        rt.prev_pause_frame, frame,
                        resize_wh=(cfg.diff_resize_w, cfg.diff_resize_h),
                        blur_ksize=cfg.diff_blur_ksize
                    )

                rt.diffs.append(diff)
                rt.prev_pause_frame = frame

                logger.log_event(
                    state.log_path, datetime.now(),
                    "scan_pause_sample",
                    state,
                    notes=f"episode={rt.episode+1} step={rt.step_index+1}/4 diff={diff:.3f}"
                )
            else:
                logger.log_event(
                    state.log_path, datetime.now(),
                    "scan_pause_sample_fail",
                    state,
                    notes=f"episode={rt.episode+1} step={rt.step_index+1}/4 no_frame"
                )

            rt.pause_sampled = True

        # Wait until pause duration has passed
        if elapsed < cfg.scan_pause_s:
            return None

        logger.log_event(
            state.log_path, datetime.now(),
            "scan_step_pause_done",
            state,
            notes=f"episode={rt.episode+1} step={rt.step_index+1}/4 pause_s={cfg.scan_pause_s}"
        )

        # Move to next step
        rt.step_index += 1

        if rt.step_index >= 4:
            # Completed one scan episode (360°)
            rt.step_index = 0
            rt.episode += 1  # now rt.episode is the completed-episode count (1-based)

            # ----- Episode evidence stats -----
            if rt.diffs:
                mean_diff = sum(rt.diffs) / len(rt.diffs)
                max_diff = max(rt.diffs)
                n = len(rt.diffs)
            else:
                mean_diff = 0.0
                max_diff = 0.0
                n = 0

            # ----- Evidence-based boost -----
            q = mean_diff / cfg.scan_diff_norm
            q = max(0.0, min(1.0, q))  # clamp to [0,1]

            boost = cfg.scan_max_boost * q
            state.map_conf = min(1.0, state.map_conf + boost)

            logger.log_event(
                state.log_path, datetime.now(),
                "scan_boost",
                state,
                notes=f"episode={rt.episode} mean_diff={mean_diff:.2f} q={q:.2f} boost={boost:.3f}"
            )

            logger.log_event(
                state.log_path, datetime.now(),
                "scan_episode_summary",
                state,
                notes=f"episode={rt.episode} mean_diff={mean_diff:.2f} max_diff={max_diff:.2f} samples={n}"
            )

            logger.log_event(
                state.log_path, datetime.now(),
                "scan_episode_complete",
                state,
                notes=f"episode={rt.episode} diffs_n={n}"
            )

            # Reset episode-local evidence so next episode is judged on its own
            rt.stored_repr = None
            rt.pause_sampled = False
            rt.prev_pause_frame = None
            rt.diffs.clear()

            # After boosting, if recovered, exit
            if state.map_conf >= cfg.scan_exit_thresh:
                logger.log_event(state.log_path, datetime.now(), "scan_done", state, notes="recovered after episode")
                return "observe"


        # Continue scanning
        rt.phase = "rotate"
        rt.phase_started_at = now_t
        return None

    # Unknown phase fallback
    rt.phase = "rotate"
    rt.phase_started_at = now_t
    rt.pause_sampled = False
    return None
