import time
import os
from datetime import datetime

from config import Config
from core.state import AgentState
from core.camera import CameraStream
from core import storage, logger, beliefs, decision

from modes import observe, scan


def switch_mode(state, now_t, cfg, new_mode: str):
    """Handles clean enter/exit events for modes."""
    if new_mode == state.mode:
        return

    # exit old mode
    if state.mode == "observe":
        observe.exit(state, now_t, cfg)
    elif state.mode == "scan":
        scan.exit(state, now_t, cfg)

    old = state.mode
    state.mode = new_mode

    logger.log_event(state.log_path, datetime.now(), "mode_switch", state, notes=f"{old}->{new_mode}")

    # enter new mode
    if state.mode == "observe":
        observe.enter(state, now_t, cfg)
    elif state.mode == "scan":
        scan.enter(state, now_t, cfg)


def run(cfg: Config):
    # Setup run paths
    run_folder, frame_folder, log_path = storage.make_run_paths(cfg.runs_dir)
    scan_trigger_path = os.path.join(run_folder, cfg.scan_trigger_name)

    # Init state
    state = AgentState(
        mode="observe",
        stream_conf=cfg.initial_stream_conf,
        map_conf=cfg.initial_map_conf,
    )
    state.run_folder = run_folder
    state.frame_folder = frame_folder
    state.log_path = log_path
    state.scan_trigger_path = scan_trigger_path
    state.session_start_hms = datetime.now().strftime("%H:%M:%S")

    logger.log_session_start(state.log_path, state)

    cam = CameraStream(cfg.rtsp_url)
    if not cam.is_opened():
        with open(state.log_path, "a") as f:
            f.write(f"[{state.session_start_hms}] ERROR: Could not open stream.\n")
        return

    print(f"Logging session started in: {state.run_folder}")

    # Enter initial mode explicitly (optional but clean)
    now_t = time.time()
    observe.enter(state, now_t, cfg)

    try:
        while True:
            now_t = time.time()

            # 1) Continuous belief decay (always)
            dt = now_t - state.last_update_time
            state.last_update_time = now_t
            beliefs.apply_continuous_decay(state, dt)

            # 2) Manual scan trigger = immediate scan boost event (kept from your current system)
            #    This is a "scan completed" signal, not a mode switch.
            if decision.should_trigger_scan(state):
                os.remove(state.scan_trigger_path)
                beliefs.apply_scan_boost(state)
                logger.log_event(state.log_path, datetime.now(), "scan_triggered", state)

            # 3) Read frame (perception)
            ret, frame = cam.read()
            if not ret:
                cam.reconnect(sleep_s=1.0)
                continue

            # 4) Save at configured FPS + apply stream evidence
            if now_t - state.last_saved_time >= (1.0 / cfg.save_fps):
                now_dt = datetime.now()
                filename, _ = storage.save_frame(state.frame_folder, frame, now_dt)

                stream_status = 1
                beliefs.apply_stream_evidence(state, stream_status)

                logger.log_frame_saved(state.log_path, now_dt, filename, stream_status, state)

                storage.cleanup_old_data(state.frame_folder, max_files=cfg.max_images)
                state.last_saved_time = now_t

                print(
                    f"Session {state.session_start_hms} | Saved: {filename} "
                    f"| mode={state.mode} stream_conf={state.stream_conf:.2f} map_conf={state.map_conf:.2f}",
                    end="\r",
                )

            # 5) Decide which mode we SHOULD be in (policy)
            desired_mode = decision.update_mode(state, cfg)
            if desired_mode != state.mode:
                switch_mode(state, now_t, cfg, desired_mode)

            # 6) Run the active mode (behavior execution)
            #    scan.step() may return "observe" if it finishes or fails.
            if state.mode == "observe":
                observe.step(state, now_t, cfg)
            elif state.mode == "scan":
                next_mode = scan.step(state, now_t, cfg, rotate_fn=None)
                if next_mode is not None and next_mode != state.mode:
                    switch_mode(state, now_t, cfg, next_mode)

    except KeyboardInterrupt:
        print("\nStopping Session...")
    finally:
        logger.log_session_end(state.log_path)
        cam.release()


if __name__ == "__main__":
    cfg = Config(rtsp_url="rtsp://10.25.113.245:8080/h264_ulaw.sdp")
    run(cfg)
