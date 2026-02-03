from dataclasses import dataclass, field
import time

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
    scan_rt : object = None
    motors_enabled: bool = True
    #last_turn_command_time: float = 0.0
