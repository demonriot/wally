# modes/observe.py
from datetime import datetime
from core import logger

def enter(state, now_t, cfg):
    logger.log_event(state.log_path, datetime.now(), "observe_enter", state)

def exit(state, now_t, cfg):
    logger.log_event(state.log_path, datetime.now(), "observe_exit", state)

def step(state, now_t, cfg):
    # No-op for now. Later you can add micro-actions here.
    return None
