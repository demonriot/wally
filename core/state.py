from dataclasses import dataclass, field
from typing import Optional, Dict
import time
import numpy as np


def _default_memory_bins() -> Dict[int, Optional[np.ndarray]]:
    return {0: None, 1: None, 2: None, 3: None}


@dataclass
class AgentState:
    mode: str = "observe"

    stream_conf: float = 0.5
    map_conf: float = 0.5

    last_saved_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)

    session_start_hms: str = ""
    run_folder: str = ""
    frame_folder: str = ""
    log_path: str = ""
    scan_trigger_path: str = ""

    scan_rt: object = None
    motors_enabled: bool = True
    cooldown_until: float = 0.0

    # Long-term directional memory (persistent across runs)
    memory_bins: Dict[int, Optional[np.ndarray]] = field(default_factory=_default_memory_bins)
    memory_last_saved_at: float = field(default_factory=time.time)