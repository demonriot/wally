import cv2
import time

class CameraStream:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url)

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def reconnect(self, sleep_s: float = 1.0):
        time.sleep(sleep_s)
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = cv2.VideoCapture(self.rtsp_url)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
