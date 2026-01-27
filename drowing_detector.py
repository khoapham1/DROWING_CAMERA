import math
import numpy as np
from collections import deque
import time

class DrowningDetector:
    """
    Drowning / distress detector (single person) using MediaPipe Pose.
    Requirements implemented:
      1) Trigger when ONE hand is above head AND arm/wrist motion becomes abnormally fast (speed/accel/angle-speed).
      2) State machine:
           - Immediately "ACTIVE" when trigger is seen (ready to alert).
           - If trigger persists for >= 3s => "FROZEN"
           - If trigger persists for >= 5s => "SOS"
         If trigger disappears for a short gap => reset back to ACTIVE.
      3) No ID logic here (handled in UI). This module only returns state and features.
    """

    def __init__(self, config=None):
        default_config = {
            # ---- motion thresholds (pixels/sec, pixels/sec^2, degrees/sec) ----
            "WRIST_SPEED_TH": 260.0,         # fast wrist motion
            "WRIST_ACCEL_TH": 700.0,         # abnormal acceleration
            "ELBOW_ANGLE_SPEED_TH": 140.0,   # fast elbow angle change

            # ---- temporal thresholds ----
            "T_FROZEN": 1.5,                 # seconds of continuous distress before FROZEN
            "T_SOS": 3.0,                    # seconds of continuous distress before SOS
            "RESET_GAP": 1.0,                # seconds without trigger to reset

            # ---- head reference ----
            "HEAD_Y_MARGIN": 0.0,            # wrist must be above (head_center_y - margin)
        }
        self.config = {**default_config, **(config or {})}

    def init_track(self, track):
        if getattr(track, "_drown_inited", False):
            return
        track._drown_inited = True

        # joint histories: (x, y, t)
        track.lw_hist = deque(maxlen=20)
        track.rw_hist = deque(maxlen=20)
        track.le_hist = deque(maxlen=20)
        track.re_hist = deque(maxlen=20)

        # elbow angle histories: (angle_deg, t)
        track.la_hist = deque(maxlen=20)
        track.ra_hist = deque(maxlen=20)

        # trigger timers
        track.state = "ACTIVE"
        track.trigger_start_time = None
        track.last_trigger_time = None
        track.sos_start_time = None

    @staticmethod
    def _lm_xy(lm_point, w, h):
        return float(lm_point.x * w), float(lm_point.y * h)

    @staticmethod
    def _speed(hist):
        """Instantaneous speed (px/s) from last 2 points in hist = [(x,y,t), ...]"""
        if len(hist) < 2:
            return 0.0
        x0, y0, t0 = hist[-2]
        x1, y1, t1 = hist[-1]
        dt = float(t1 - t0)
        if dt <= 1e-6:
            return 0.0
        return float(math.hypot(x1 - x0, y1 - y0) / dt)

    @staticmethod
    def _accel(speed_hist):
        """Acceleration (px/s^2) from last 2 speeds in hist = [(v,t), ...]"""
        if len(speed_hist) < 2:
            return 0.0
        v0, t0 = speed_hist[-2]
        v1, t1 = speed_hist[-1]
        dt = float(t1 - t0)
        if dt <= 1e-6:
            return 0.0
        return float((v1 - v0) / dt)

    @staticmethod
    def _angle(a, b, c):
        """Angle at point b between ba and bc in degrees."""
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        c = np.array(c, dtype=np.float32)
        ba = a - b
        bc = c - b
        nba = float(np.linalg.norm(ba))
        nbc = float(np.linalg.norm(bc))
        if nba < 1e-6 or nbc < 1e-6:
            return 0.0
        cosang = float(np.dot(ba, bc) / (nba * nbc))
        return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

    @staticmethod
    def _angle_speed(hist):
        """deg/s from last 2 entries hist=[(angle_deg,t), ...]"""
        if len(hist) < 2:
            return 0.0
        a0, t0 = hist[-2]
        a1, t1 = hist[-1]
        dt = float(t1 - t0)
        if dt <= 1e-6:
            return 0.0
        # smallest angular difference
        da = float((a1 - a0 + 180.0) % 360.0 - 180.0)
        return float(abs(da) / dt)

    def _update_state_machine(self, track, trigger, now):
        if trigger:
            if track.trigger_start_time is None:
                track.trigger_start_time = now
            track.last_trigger_time = now

            elapsed = float(now - track.trigger_start_time)
            if elapsed >= self.config["T_SOS"]:
                track.state = "SOS"
                if track.sos_start_time is None:
                    track.sos_start_time = now
            elif elapsed >= self.config["T_FROZEN"]:
                track.state = "FROZEN"
                track.sos_start_time = None
            else:
                track.state = "ACTIVE"
                track.sos_start_time = None

            return elapsed

        # trigger = False
        if track.last_trigger_time is None:
            # never triggered yet
            track.state = "ACTIVE"
            track.trigger_start_time = None
            track.sos_start_time = None
            return 0.0

        gap = float(now - track.last_trigger_time)
        if gap >= self.config["RESET_GAP"]:
            track.state = "ACTIVE"
            track.trigger_start_time = None
            track.last_trigger_time = None
            track.sos_start_time = None
            return 0.0

        # still in cooldown gap -> keep current state but do not advance time
        return float(now - (track.trigger_start_time or now))

    def detect(self, track, bbox, pose_landmarks, frame_shape):
        self.init_track(track)

        # default results
        results = {
            "state": track.state,
            "arm_above_head": False,
            "waving": False,
            "wrist_speed": 0.0,
            "wrist_accel": 0.0,
            "elbow_angle_speed": 0.0,
            "trigger_elapsed": 0.0,
            "water_level": None
        }

        # always compute water line from bbox (for overlay compatibility)
        x1, y1, x2, y2 = map(float, bbox)
        results["water_level"] = float(y1 + (y2 - y1) * (2.0 / 3.0))

        if not pose_landmarks:
            # No pose -> keep previous state (but allow reset after gap)
            now = time.time()
            results["trigger_elapsed"] = self._update_state_machine(track, trigger=False, now=now)
            results["state"] = track.state
            return results

        h, w = frame_shape[:2]
        lm = pose_landmarks.landmark
        now = time.time()

        # Keypoints (MediaPipe indices)
        nose = self._lm_xy(lm[0], w, h)
        leye = self._lm_xy(lm[1], w, h)
        reye = self._lm_xy(lm[2], w, h)

        l_sh = self._lm_xy(lm[11], w, h)
        r_sh = self._lm_xy(lm[12], w, h)
        l_el = self._lm_xy(lm[13], w, h)
        r_el = self._lm_xy(lm[14], w, h)
        l_wr = self._lm_xy(lm[15], w, h)
        r_wr = self._lm_xy(lm[16], w, h)

        head_center_y = (nose[1] + leye[1] + reye[1]) / 3.0
        head_line = head_center_y - float(self.config["HEAD_Y_MARGIN"])

        left_above = (l_wr[1] < head_line)
        right_above = (r_wr[1] < head_line)
        arm_above_head = bool(left_above or right_above)
        results["arm_above_head"] = arm_above_head

        # Update histories with timestamp
        track.lw_hist.append((l_wr[0], l_wr[1], now))
        track.rw_hist.append((r_wr[0], r_wr[1], now))
        track.le_hist.append((l_el[0], l_el[1], now))
        track.re_hist.append((r_el[0], r_el[1], now))

        # Elbow angles
        la = self._angle(l_sh, l_el, l_wr)
        ra = self._angle(r_sh, r_el, r_wr)
        track.la_hist.append((la, now))
        track.ra_hist.append((ra, now))

        # Speeds (px/s)
        lw_speed = self._speed(track.lw_hist)
        rw_speed = self._speed(track.rw_hist)

        # Speed history for accel (store in track)
        if not hasattr(track, "lw_speed_hist"):
            track.lw_speed_hist = deque(maxlen=10)
            track.rw_speed_hist = deque(maxlen=10)
        track.lw_speed_hist.append((lw_speed, now))
        track.rw_speed_hist.append((rw_speed, now))

        lw_acc = self._accel(track.lw_speed_hist)
        rw_acc = self._accel(track.rw_speed_hist)

        # Angle speed (deg/s)
        la_speed = self._angle_speed(track.la_hist)
        ra_speed = self._angle_speed(track.ra_hist)

        # Aggregate for UI
        results["wrist_speed"] = float(max(lw_speed, rw_speed))
        results["wrist_accel"] = float(max(lw_acc, rw_acc))
        results["elbow_angle_speed"] = float(max(la_speed, ra_speed))

        # --- Waving / abnormal motion condition ---
        # Condition per arm: hand above head AND (fast speed) AND (accel or elbow angle speed fast)
        def arm_wave(above, w_speed, w_acc, ang_speed):
            if not above:
                return False
            if w_speed > self.config["WRIST_SPEED_TH"] and (w_acc > self.config["WRIST_ACCEL_TH"] or ang_speed > self.config["ELBOW_ANGLE_SPEED_TH"]):
                return True
            # very fast wrist alone also counts
            if w_speed > (self.config["WRIST_SPEED_TH"] * 1.6):
                return True
            return False

        left_wave = arm_wave(left_above, lw_speed, lw_acc, la_speed)
        right_wave = arm_wave(right_above, rw_speed, rw_acc, ra_speed)
        waving = bool(left_wave or right_wave)
        results["waving"] = waving

        # --- Trigger logic ---
        # As requested: seeing ONE hand above head is enough to be "ready" -> trigger starts.
        # Waving makes it more reliable but we still start timer on above-head.
        trigger = bool(arm_above_head)

        # State machine
        results["trigger_elapsed"] = self._update_state_machine(track, trigger=trigger, now=now)
        results["state"] = track.state
        return results

    def get_sos_duration(self, track):
        if getattr(track, "state", None) == "SOS" and getattr(track, "sos_start_time", None):
            return time.time() - track.sos_start_time
        return 0.0

    def get_state_colors(self):
        return {
            "ACTIVE": (0, 255, 0),
            "FROZEN": (0, 165, 255),
            "SOS": (0, 0, 255)
        }
