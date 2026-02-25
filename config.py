from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    rtsp_url: str
    save_fps: float = 1.0
    max_images: int = 40

    initial_stream_conf: float = 0.5
    initial_map_conf: float = 0.35

    runs_dir: str = "runs"
    scan_trigger_name: str = "SCAN_NOW"

    # Scan policy params
    scan_enter_thresh: float = 0.30
    scan_exit_thresh: float = 0.60
    scan_step_degrees: int = 90
    scan_pause_s: float = 1.0
    scan_max_episodes: int = 4
    scan_settle_s: float = 0.5
    scan_flush_s: float = 0.25
    scan_cooldown_s: float = 10.0

    # L298N pins (BCM)
    L_IN1: int = 17
    L_IN2: int = 22
    L_ENA: int = 10
    R_IN3: int = 24
    R_IN4: int = 23
    R_ENB: int = 25

    # Turning calibration
    turn_speed: float = 0.5
    turn_seconds_per_degree: float = 0.012
    min_pwm: float = 0.30
    max_turn_time_s: float = 2.0

    # Perception params
    diff_resize_w: int = 160
    diff_resize_h: int = 120
    diff_blur_ksize: int = 3

    # Conf params
    scan_max_boost: float = 0.25
    scan_diff_norm: float = 30.0

    #Image feature params
    lap_ref: float = 5000.0
    edge_ref: float = 0.18
    keypoint_ref: float = 250.0
    sharp_min: float = 0.10
    edge_min: float = 0.10
    kp_min: float = 0.05
    scan_repr_alpha: float = 0.2
    scan_k_drop: float = 0.10
    scan_k_gain: float = 0.05

    # memory params
    memory_stability_thresh: float = 0.50
    memory_save_every_s: float = 5.0
    novelty_deadband: float = 0.10
    beta_min: float = 0.04