#core/storage.py
import os
import glob
from datetime import datetime
import numpy as np
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



def memory_bins_path(runs_dir: str) -> str:
    return os.path.join(runs_dir, "memory_bins.npz")


def save_memory_bins(path: str, memory_bins: dict[int, object]) -> None:
    # memory_bins values are np.ndarray or None
    np.savez_compressed(
        path,
        bin0=memory_bins.get(0),
        bin1=memory_bins.get(1),
        bin2=memory_bins.get(2),
        bin3=memory_bins.get(3),
    )


def load_memory_bins(path: str) -> dict[int, object]:
    if not os.path.exists(path):
        return {0: None, 1: None, 2: None, 3: None}

    data = np.load(path, allow_pickle=True)

    # np.savez stores None as a 0-d object array; handle that safely
    def _get(name: str):
        if name not in data:
            return None
        v = data[name]
        # If it's a 0-d object array, unwrap it
        if isinstance(v, np.ndarray) and v.shape == () and v.dtype == object:
            v = v.item()
        return v

    return {0: _get("bin0"), 1: _get("bin1"), 2: _get("bin2"), 3: _get("bin3")}