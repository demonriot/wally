# core/decision.py
import os

def should_trigger_scan(state) -> bool:
    """Manual trigger via SCAN_NOW file."""
    return os.path.exists(state.scan_trigger_path)

def update_mode(state, cfg) -> str:
    """
    Pure policy decision based on current mode + beliefs.
    Uses hysteresis: enter scan at low map_conf, exit at high map_conf.
    """
    if state.mode == "observe" and state.map_conf < cfg.scan_enter_thresh:
        return "scan"

    if state.mode == "scan" and state.map_conf > cfg.scan_exit_thresh:
        return "observe"

    return state.mode
