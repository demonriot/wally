# modes/scan.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from core import logger
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
    stored_repr: Optional[np.ndarray] = None
    last_novelty: float = 0.0
    last_beta: float = 0.0
    last_mem_upd: int = 0


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
    #rt.stored_repr = None


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
        
     # ---- NEW: bootstrap memory? ----
    bootstrap_needed = any(state.memory_bins.get(k) is None for k in range(4))       
    # If map_conf already recovered, scan is effectively done
    # If map_conf already recovered, scan can end — but during bootstrap,
    # force finishing the current 4-step episode so all dirs can be initialized.
    if state.map_conf >= cfg.scan_exit_thresh:
        if not bootstrap_needed:
            logger.log_event(state.log_path, datetime.now(), "scan_done", state, notes="map_conf recovered")
            return "observe"
        else:
            logger.log_event(
                state.log_path, datetime.now(),
                "scan_bootstrap_hold",
                state,
                notes="map_conf recovered but bootstrap_needed=1; finishing episode"
            )

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
                features, feature_vector = extract_features(frame, cfg)
                valid, reason = is_observation_valid(features, cfg)

                ep = rt.episode + 1
                st = rt.step_index + 1

                if not valid:
                    # Single compact log for invalid obs (includes normalized values for tuning)
                    logger.log_event(
                        state.log_path, datetime.now(),
                        "scan_step_obs",
                        state,
                        notes=(
                            f"episode={ep} step={st}/4 "
                            f"valid=0 reason={reason} "
                            f"sharp_q={features['sharp_q']:.2f} edge_q={features['edge_q']:.2f} kp_q={features['kp_q']:.2f}"
                        )
                    )

                else:
                    # Valid observation
                    if rt.stored_repr is None:
                        rt.stored_repr = feature_vector.astype(np.float32, copy=False)

                        # ---- NEW: Directional novelty vs long-term memory bin ----
                        k = rt.step_index
                        M_k = state.memory_bins.get(k, None)

                        if M_k is None:
                            novelty_k = 1.0
                        else:
                            diff = np.abs(feature_vector - M_k)
                            w = np.array([0.4, 0.4, 0.2], dtype=np.float32)
                            novelty_k = float(np.sum(w * diff) / float(np.sum(w)))
                            novelty_k = max(0.0, min(1.0, novelty_k - cfg.novelty_deadband))

                        novelty_k = max(0.0, min(1.0, novelty_k))
                        beta = max(cfg.beta_min, novelty_k)

                        # ---- NEW: gated long-term memory integration ----
                        mem_upd = 0
                        if M_k is None:
                            state.memory_bins[k] = feature_vector.astype(np.float32, copy=False)
                            mem_upd = 1
                        elif state.map_conf >= cfg.memory_stability_thresh:
                            state.memory_bins[k] = ((1.0 - beta) * M_k + beta * feature_vector).astype(np.float32, copy=False)
                            mem_upd = 1

                        rt.last_novelty = novelty_k
                        rt.last_beta = beta
                        rt.last_mem_upd = mem_upd

                        logger.log_event(
                            state.log_path, datetime.now(),
                            "scan_step_obs",
                            state,
                            notes=(
                                f"episode={ep} step={st}/4 valid=1 init=1 "
                                f"dir={k} novelty={novelty_k:.3f} beta={beta:.3f} mem_upd={mem_upd} "
                                f"sharp_q={features['sharp_q']:.2f} edge_q={features['edge_q']:.2f} kp_q={features['kp_q']:.2f}"
                            )
                        )
                    else:
                        # Prediction error (0..1-ish since features are normalized)
                        e = float(np.mean(np.abs(feature_vector - rt.stored_repr)))

                        # EMA update (keep vector type!)
                        alpha = cfg.scan_repr_alpha  # rename in cfg (recommended); or use cfg.alpha
                        rt.stored_repr = ((1.0 - alpha) * rt.stored_repr) + (alpha * feature_vector)
                        rt.stored_repr = rt.stored_repr.astype(np.float32, copy=False)

                        # Belief update (drop then recover)
                        before = float(state.map_conf)
                        state.map_conf -= cfg.scan_k_drop * e
                        state.map_conf += cfg.scan_k_gain * (1.0 - e)
                        state.map_conf = max(0.0, min(1.0, state.map_conf))
                        after = float(state.map_conf)

                        # ---- NEW: Directional novelty vs long-term memory bin ----
                        k = rt.step_index  # 0..3 direction bin aligned with scan steps

                        M_k = state.memory_bins.get(k, None)
                        if M_k is None:
                            novelty_k = 1.0
                        else:
                            diff = np.abs(feature_vector - M_k)
                            w = np.array([0.4, 0.4, 0.2], dtype=np.float32)
                            novelty_k = float(np.sum(w * diff) / float(np.sum(w)))
                            novelty_k = max(0.0, min(1.0, novelty_k - cfg.novelty_deadband))

                        # clamp to [0,1]
                        novelty_k = max(0.0, min(1.0, novelty_k))
                        beta = max(cfg.beta_min, novelty_k)  # choice B: novelty-weighted integration

                       # ---- NEW: gated long-term memory integration ----
                        mem_upd = 0
                        if M_k is None:
                            # bootstrap: always initialize missing bins (no stability gate)
                            state.memory_bins[k] = feature_vector.astype(np.float32, copy=False)
                            mem_upd = 1
                        elif state.map_conf >= cfg.memory_stability_thresh:
                            # stable: integrate proportionally to novelty
                            state.memory_bins[k] = ((1.0 - beta) * M_k + beta * feature_vector).astype(np.float32, copy=False)
                            mem_upd = 1

                        # optional: stash for debugging
                        rt.last_novelty = novelty_k
                        rt.last_beta = beta
                        rt.last_mem_upd = mem_upd

                        logger.log_event(
                        state.log_path, datetime.now(),
                        "scan_step_obs",
                        state,
                        notes=(
                            f"episode={ep} step={st}/4 valid=1 "
                            f"e={e:.3f} alpha={alpha:.2f} "
                            f"map_conf={before:.3f}->{after:.3f} "
                            f"dir={k} novelty={novelty_k:.3f} beta={beta:.3f} mem_upd={mem_upd} "
                            f"sharp_q={features['sharp_q']:.2f} edge_q={features['edge_q']:.2f} kp_q={features['kp_q']:.2f}"

                        )
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

        # Move to next step
        rt.step_index += 1

        if rt.step_index >= 4:
            # Completed one scan episode (360°)
            rt.step_index = 0
            rt.episode += 1  # now rt.episode is the completed-episode count (1-based)

            # Reset episode-local evidence so next episode is judged on its own
            rt.stored_repr = None
            rt.pause_sampled = False
            

            # After boosting, if recovered, exit
            if state.map_conf >= cfg.scan_exit_thresh and not bootstrap_needed:
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
