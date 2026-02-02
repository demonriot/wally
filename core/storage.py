import os
import glob
from datetime import datetime
import cv2

def make_run_paths(runs_dir: str):
    today_str = datetime.now().strftime("%Y-%m-%d")
    run_folder = os.path.join(runs_dir, f"run_{today_str}")
    frame_folder = os.path.join(run_folder, "captured_frames")
    log_path = os.path.join(run_folder, "timeline.log")
    os.makedirs(frame_folder, exist_ok=True)
    return run_folder, frame_folder, log_path

def save_frame(frame_folder: str, frame, now_dt):
    timestamp_str = now_dt.strftime("%H%M%S_%f")[:-3]
    filename = f"frame_{timestamp_str}.jpg"
    file_path = os.path.join(frame_folder, filename)
    cv2.imwrite(file_path, frame)
    return filename, file_path

def cleanup_old_data(frame_folder: str, max_files: int = 100):
    files = sorted(glob.glob(os.path.join(frame_folder, "*.jpg")), key=os.path.getmtime)
    if len(files) > max_files:
        for f in files[:len(files) - max_files]:
            os.remove(f)
