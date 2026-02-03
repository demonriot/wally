import cv2
import time

class CameraStream:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url)

        # Try to reduce latency/buffering (may or may not work depending on backend)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def read_latest(self, flush_s: float = 0.25):
        """
        Drain buffered frames for a short time window and return the latest frame.
        Useful for RTSP streams to avoid stale frames.
        """
        if self.cap is None:
            return False, None

        end_t = time.time() + flush_s
        ok, frame = False, None

        # Prefer grab/retrieve loop (often faster than read)
        while time.time() < end_t:
            grabbed = self.cap.grab()
            if not grabbed:
                break
            ok, frame = self.cap.retrieve()
            if not ok:
                frame = None

        # Fallback: if nothing was retrieved, do a normal read once
        if frame is None:
            return self.read()

        return True, frame

    def reconnect(self, sleep_s: float = 1.0):
        time.sleep(sleep_s)
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = cv2.VideoCapture(self.rtsp_url)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
