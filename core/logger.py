from datetime import datetime

def log_session_start(log_path: str, state):
    with open(log_path, "a") as log_file:
        log_file.write(f"\n{'='*50}\n")
        log_file.write(f"SESSION START: {state.session_start_hms}\n")
        log_file.write(f"initial_stream_conf={state.stream_conf}\n")
        log_file.write(f"initial_map_conf={state.map_conf}\n")
        log_file.write(f"{'='*50}\n")

def log_session_end(log_path: str):
    session_end = datetime.now().strftime("%H:%M:%S")
    with open(log_path, "a") as log_file:
        log_file.write(f"{'-'*50}\n")
        log_file.write(f"SESSION END: {session_end}\n")
        log_file.write(f"{'-'*50}\n\n")

def log_frame_saved(log_path: str, now_dt, filename: str, stream_status: int, state):
    with open(log_path, "a") as log_file:
        log_file.write(
            f"[{now_dt.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Saved {filename} obs=[] mode={state.mode} "
            f"stream_status={stream_status} "
            f"stream_conf={state.stream_conf:.4f} map_conf={state.map_conf:.4f} notes=\n"
        )

def log_event(log_path: str, now_dt, event_name: str, state, notes: str = ""):
    with open(log_path, "a") as log_file:
        log_file.write(
            f"[{now_dt.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"EVENT {event_name} mode={state.mode} "
            f"stream_conf={state.stream_conf:.4f} map_conf={state.map_conf:.4f} notes={notes}\n"
        )
