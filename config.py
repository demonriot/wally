from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    rtsp_url: str
    save_fps: float = 1.0
    max_images: int = 10

    initial_stream_conf: float = 0.5
    initial_map_conf: float = 0.35

    runs_dir: str = "runs"       # base folder for runs
    scan_trigger_name: str = "SCAN_NOW"  # manual trigger file name
    
    scan_enter_thresh = 0.30
    scan_exit_thresh = 0.60
    scan_step_degrees = 90
    scan_pause_s = 1.0
    scan_max_episodes = 2
