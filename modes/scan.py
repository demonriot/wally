# modes/scan.py
from dataclasses import dataclass
from datetime import datetime

from core import beliefs, logger


@dataclass
class ScanRuntime:
    """Holds scan progress across loop iterations."""
    episode: int = 0              # 0..max_episodes-1
    step_index: int = 0           # 0..3 (4 steps per episode)
    phase: str = "idle"           # "rotate" -> "pause"
    phase_started_at: float = 0.0 # time.time() when current phase began


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


def exit(state, now_t: float, cfg):
    """Called when leaving scan mode."""
    logger.log_event(state.log_path, datetime.now(), "scan_exit", state, notes="exit scan mode")
    # Keep runtime for debugging, or set to None if you prefer:
    # state.scan_rt = None


def step(state, now_t: float, cfg, rotate_fn=None):
    """
    Non-blocking scan execution.

    rotate_fn: optional callback like rotate_fn(degrees:int) -> None
              If None, rotation is simulated (no hardware).
    """
    rt = state.scan_rt
    if rt is None:
        # safety fallback
        state.scan_rt = ScanRuntime()
        rt = state.scan_rt
        rt.phase = "rotate"
        rt.phase_started_at = now_t

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
            rotate_fn(degrees)  # ideally non-blocking or quick call
      
        logger.log_event(
            state.log_path, datetime.now(),
            "scan_step_rotate",
            state,
            notes=f"episode={rt.episode+1} step={rt.step_index+1}/4 degrees={degrees}"
        )

        # Immediately move to pause phase
        rt.phase = "pause"
        rt.phase_started_at = now_t
        return None

    if rt.phase == "pause":
        # Wait until pause duration has passed
        if (now_t - rt.phase_started_at) < cfg.scan_pause_s:
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
            rt.episode += 1

            # Apply scan boost to map_conf (your "active sensing refresh")
            beliefs.apply_scan_boost(state)

            logger.log_event(
                state.log_path, datetime.now(),
                "scan_episode_complete",
                state,
                notes=f"episode={rt.episode} boost_applied"
            )

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
    return None
