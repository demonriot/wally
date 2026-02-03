import time
import RPi.GPIO as GPIO
from config import Config  # <-- ensure this exists


class MotorController:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)

        GPIO.setup(self.cfg.L_IN1, GPIO.OUT)
        GPIO.setup(self.cfg.L_IN2, GPIO.OUT)
        GPIO.setup(self.cfg.L_ENA, GPIO.OUT)

        GPIO.setup(self.cfg.R_IN3, GPIO.OUT)
        GPIO.setup(self.cfg.R_IN4, GPIO.OUT)
        GPIO.setup(self.cfg.R_ENB, GPIO.OUT)

        self.left_pwm = GPIO.PWM(self.cfg.L_ENA, 1000)
        self.right_pwm = GPIO.PWM(self.cfg.R_ENB, 1000)
        self.left_pwm.start(0)
        self.right_pwm.start(0)

    def _clamp(self, x: float, lo: float = -1.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, float(x)))

    def set_drive(self, left_speed: float, right_speed: float):
        left_speed = self._clamp(left_speed)
        right_speed = self._clamp(right_speed)

        # LEFT direction
        if left_speed > 0:
            GPIO.output(self.cfg.L_IN1, True)
            GPIO.output(self.cfg.L_IN2, False)
        elif left_speed < 0:
            GPIO.output(self.cfg.L_IN1, False)
            GPIO.output(self.cfg.L_IN2, True)
        else:
            GPIO.output(self.cfg.L_IN1, False)
            GPIO.output(self.cfg.L_IN2, False)

        # RIGHT direction
        if right_speed > 0:
            GPIO.output(self.cfg.R_IN3, True)
            GPIO.output(self.cfg.R_IN4, False)
        elif right_speed < 0:
            GPIO.output(self.cfg.R_IN3, False)
            GPIO.output(self.cfg.R_IN4, True)
        else:
            GPIO.output(self.cfg.R_IN3, False)
            GPIO.output(self.cfg.R_IN4, False)

        # PWM duty with optional deadzone
        left_duty = min(abs(left_speed) * 100.0, 100.0)
        right_duty = min(abs(right_speed) * 100.0, 100.0)

        if getattr(self.cfg, "min_pwm", 0) and left_duty > 0:
            left_duty = max(left_duty, self.cfg.min_pwm * 100.0)
        if getattr(self.cfg, "min_pwm", 0) and right_duty > 0:
            right_duty = max(right_duty, self.cfg.min_pwm * 100.0)

        self.left_pwm.ChangeDutyCycle(left_duty)
        self.right_pwm.ChangeDutyCycle(right_duty)

    def stop(self):
        self.set_drive(0.0, 0.0)

    def rotate_in_place(self, degrees: float):
        turn_time = min(abs(degrees) * self.cfg.turn_seconds_per_degree, self.cfg.max_turn_time_s)
        s = self.cfg.turn_speed

        if degrees > 0:
            self.set_drive(+s, -s)  # rotate right
        else:
            self.set_drive(-s, +s)  # rotate left

        time.sleep(turn_time)
        self.stop()

    def cleanup(self):
        try:
            self.stop()
        finally:
            self.left_pwm.stop()
            self.right_pwm.stop()
            GPIO.cleanup()


if __name__ == "__main__":
    cfg = Config(rtsp_url="rtsp://10.25.113.245:8080/h264_ulaw.sdp")
    mc = MotorController(cfg)
    try:
        mc.rotate_in_place(30)
    finally:
        mc.cleanup()
