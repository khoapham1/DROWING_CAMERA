
import time
import math
import threading
import numpy as np
import cv2
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import requests
from PID_controller import PIDController
from collections import deque
import mediapipe as mp
from person_detector import PersonDetector
from drowing_detector import DrowningDetector
import os

try:
    from drop import BallDropper
except Exception as e:
    BallDropper = None
    print(f"⚠️ BallDropper not available: {e}")


# ===================== CAMERA CALIBRATION =====================
try:
    CAMERA_MATRIX = np.load("camera_matrix_logitech.npy")
    DIST_COEFF = np.load("dist_coeff_logitech.npy")
    USE_UNDISTORT = True
    print("✅ Loaded camera calibration matrices")
except Exception as e:
    CAMERA_MATRIX = None
    DIST_COEFF = None
    USE_UNDISTORT = False
    print("⚠️ Camera calibration not loaded:", e)
_undistort_map1 = None
_undistort_map2 = None
# ===================== CAMERA SETUP =====================
horizontal_res = 640
vertical_res = 480

_latest_frame_lock = threading.Lock()
_latest_frame_jpeg = None
_usb_cam = None
_camera_thread = None
_camera_running = False
IMG_SIZE = 320
DROP_UART_PORT = "/dev/ttyUSB0"
DROP_TRIGGER_PERCENT = 90

def start_camera(camera_index=0):
    """Start USB camera"""
    global _usb_cam, _camera_running, _camera_thread

    if _camera_running:
        return

    try:
        _usb_cam = cv2.VideoCapture(camera_index)
        _usb_cam.set(cv2.CAP_PROP_FRAME_WIDTH, horizontal_res)
        _usb_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, vertical_res)
        _usb_cam.set(cv2.CAP_PROP_FPS, 30)

        if not _usb_cam.isOpened():
            raise RuntimeError("Cannot open USB camera")

        _camera_running = True
        _camera_thread = threading.Thread(
            target=_camera_loop,
            daemon=True
        )
        _camera_thread.start()
        print("✅ USB camera started")

    except Exception as e:
        print("❌ Failed to start USB camera:", e)

def _camera_loop():
    global _latest_frame_jpeg, _latest_frame_lock
    global _camera_running, _usb_cam
    global _undistort_map1, _undistort_map2

    while _camera_running and _usb_cam:
        try:
            ret, frame = _usb_cam.read()
            if not ret:
                time.sleep(0.05)
                continue

            # ---------- UNDISTORT ----------
            global _undistort_map1, _undistort_map2

            if USE_UNDISTORT and CAMERA_MATRIX is not None:
                if _undistort_map1 is None:
                    h, w = frame.shape[:2]
                    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
                        CAMERA_MATRIX,
                        DIST_COEFF,
                        (w, h),
                        alpha=0  # 0 = crop, 1 = keep full FOV
                    )

                    _undistort_map1, _undistort_map2 = cv2.initUndistortRectifyMap(
                        CAMERA_MATRIX,
                        DIST_COEFF,
                        None,
                        new_camera_mtx,
                        (w, h),
                        cv2.CV_16SC2
                    )
                    print("✅ Undistort map initialized")

                frame = cv2.remap(
                    frame,
                    _undistort_map1,
                    _undistort_map2,
                    interpolation=cv2.INTER_LINEAR
                )
            # Encode JPEG
            ret, jpeg = cv2.imencode(
                '.jpg',
                frame,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY), 85,
                    int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
                ]
            )

            if ret:
                with _latest_frame_lock:
                    _latest_frame_jpeg = jpeg.tobytes()

            time.sleep(0.033)  # ~30 FPS

        except Exception as e:
            print("Camera loop error:", e)
            time.sleep(0.1)

def stop_camera():
    """Stop USB camera"""
    global _usb_cam, _camera_running, _camera_thread

    _camera_running = False

    if _camera_thread:
        _camera_thread.join(timeout=2.0)
        _camera_thread = None

    if _usb_cam:
        try:
            _usb_cam.release()
            _usb_cam = None
        except Exception as e:
            print("Error releasing camera:", e)

    print("🛑 Camera stopped")

def get_lastest_frame():
    """Return latest JPEG bytes"""
    global _latest_frame_jpeg, _latest_frame_lock
    with _latest_frame_lock:
        return _latest_frame_jpeg

# ==================== MEDIA PIPE POSE ====================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

latest_pose_landmarks = None
latest_frame = None

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)



# ===== CONFIGURATION MEDIAPIPE ========
DROWNING_CONFIG = {
    "WRIST_SPEED_TH": 260.0,
    "WRIST_ACCEL_TH": 700.0,
    "ELBOW_ANGLE_SPEED_TH": 140.0,
    "T_FROZEN": 1.5,
    "T_SOS": 3.0,
    "RESET_GAP": 1.0,
    "HEAD_Y_MARGIN": 0.0
}

# ===================== DRONE CONTROLLER =====================
class DroneController:
    def __init__(self, connection_str='/dev/ttyACM0', takeoff_height=4):
        """Create DroneController and connect to vehicle"""
        self.connection_str = connection_str
        print(f"Connecting to vehicle on {connection_str}")

        try:
            self.vehicle = connect(
                connection_str,
                baud=115200,
                wait_ready=True,
                timeout=120
            )
            print("✅ Vehicle connected successfully")
        except Exception as e:
            print(f"❌ Failed to connect to vehicle: {e}")
            self.vehicle = None

        # Telemetry buffer
        self._telemetry_lock = threading.Lock()
        self.latest_telemetry = {
            'lat': None,
            'lon': None,
            'alt': None,
            'mode': None,
            'velocity': 0.0,
            'connected': bool(self.vehicle),
            'heading': None
        }

        if self.vehicle:
            try:
                # Setup listeners
                self.vehicle.add_attribute_listener(
                    'location.global_frame', self._location_listener
                )
                self.vehicle.add_attribute_listener(
                    'location.global_relative_frame', self._rel_location_listener
                )
                self.vehicle.add_attribute_listener(
                    'velocity', self._velocity_listener
                )
                self.vehicle.add_attribute_listener(
                    'mode', self._mode_listener
                )
                self.vehicle.add_attribute_listener(
                    'heading', self._heading_listener
                )
                
                # Landing parameters
                self.vehicle.parameters['PLND_ENABLED'] = 1
                self.vehicle.parameters['PLND_TYPE'] = 1
                self.vehicle.parameters['LAND_SPEED'] = 30
                
                print("✅ Listeners and parameters set")
            except Exception as e:
                print(f"Warning: Failed to set some listeners: {e}")

        self.takeoff_height = takeoff_height
        self.flown_path = []

        
        # Person detection
        self.person_detector = PersonDetector("yolov5n_quant.onnx", IMG_SIZE, 0.45, 0.3, detect_interval=1)
        self.person_thread = None
        self.person_running = False
        self.detected_persons = []
        self.last_detection_time = 0
        self.detection_interval = 0.05  # seconds between processing steps (loop pacing only)
        self.drowing_detector = DrowningDetector(DROWNING_CONFIG)

        class _PersonTrack:
            pass
        self._person_track = _PersonTrack()
        # Rate-limit server posting (do NOT block detection loop)
        self.server_post_interval = 2.0
        self._last_server_post = 0.0
        self.latest_pose_landmarks = None
        self.server_post_interval = 2.0 
        self._last_server_post = 0.0
        self._locked_bbox = None
        self._locked_bbox_last_seen = 0.0
        self._bbox_lock_ttl = 0.8
        self._bbox_smooth_alpha = 0.35
        self.detection_fps = 0.0
        self._detection_frame_count = 0
        self._detection_fps_last_time = time.time()
        self.PID_X = PIDController(
            0.0035, 0.0, 0.00000015,
            max_output=0.08,
            integral_limit=300,
            derivative_filter_tau=0.06,
            derivative_limit=1800
        )
        self.PID_Y = PIDController(
            0.004, 0.0, 0.00000015,
            max_output=0.1,
            integral_limit=300,
            derivative_filter_tau=0.06,
            derivative_limit=1800
        )
        self.visual_servo_enable = True
        self.visual_servo_deadband_px = 8.0
        self.visual_servo_target_ttl = 0.8
        self.visual_servo_no_person_timeout = float(os.getenv("VISUAL_SERVO_NO_PERSON_TIMEOUT", "5.0"))
        self.visual_servo_lock = threading.Lock()
        self.visual_servo_target = None
        self.visual_servo_last_command = {
            "active": False,
            "ex": 0.0,
            "ey": 0.0,
            "vx": 0.0,
            "vy": 0.0,
        }
        self._visual_servo_no_person_exit_printed = False
        self._last_pid_debug_print = 0.0
        self._pid_debug_interval = 0.25
        self._current_goto_target = None
        self._current_goto_speed = 0.7
        self._active_state_start = 0.0
        self._active_timeout_sec = float(os.getenv("ACTIVE_STATE_TIMEOUT", "5.0"))
        self._active_exit_cooldown_sec = float(os.getenv("ACTIVE_EXIT_COOLDOWN", "15.0"))
        self._active_timeout_exit_printed = False
        self._pause_block_until = 0.0
        self._drop_completed = False
        self._drop_in_progress = False
        self._last_drop_time = 0.0
        self._drop_rearm_delay = float(os.getenv("DROP_REARM_DELAY_SECONDS", "20"))
        self._drop_lock = threading.Lock()
        self._drop_center_required_px = 0.0
        self._drop_port = DROP_UART_PORT
        self._drop_baudrate = int(os.getenv("DROP_BAUDRATE", "9600"))
        self.drop_trigger_percent = float(DROP_TRIGGER_PERCENT)

        self.dropper = None
        if BallDropper and self._drop_port:
            try:
                self.dropper = BallDropper(self._drop_port, self._drop_baudrate, connect_on_init=False)
                print(f"✅ Ball dropper configured on {self._drop_port} @ {self._drop_baudrate}")
            except Exception as e:
                self.dropper = None
                print(f"⚠️ Ball dropper init failed: {e}")
        else:
            print("⚠️ Drop disabled: set DROP_PORT or BALL_DROPPER_PORT to enable BallDropper")

        self.drop_mode = os.getenv("DROP_MODE", "manual")  # 'auto' | 'manual'
        self.drop_center_aligned = False

        #SOS send control
        self._sos_active = False
        self._last_sos_post = 0.0
        self.sos_post_interval = 1
        
        # PAUSE / RESUME
        self._pause_lock = threading.Lock()
        self._pause_last_trigger = 0.0
        self._pause_hold_s = float(os.getenv("PERSON_HOLD_SECONDS", "10"))  # seconds to hold
        self._pause_cooldown_s = float(os.getenv("PERSON_HOLD_COOLDOWN_SECONDS", "15"))  # prevent re-trigger spam
        labels_env = os.getenv("PERSON_HOLD_LABELS", "person_in_water,drowning")
        self._pause_labels = set([s.strip().lower() for s in labels_env.split(",") if s.strip()])

            # -------- Mission pause / resume (person check) --------
        # When a person is detected: request pause (hover/hold). If SOS -> keep paused and report GPS.
        # If no SOS within a short inspection window -> auto-resume and continue to the current waypoint.
        self.pause_enable = True
        self.pause_min_hold_sec = 2.0          # minimum pause time before allowing auto-resume
        self.pause_max_hold_sec = 8.0          # maximum inspection time if no SOS
        self.pause_clear_no_person_sec = self.visual_servo_no_person_timeout   # resume if person disappears for this long
        self.pause_hold_hz = 5.0               # how often to send hold commands while paused
        self.pause_retrigger_cooldown_sec = 5.0   # avoid pause-resume oscillation

        self._pause_lock = threading.Lock()
        self._pause_requested = False
        self._pause_reason = None
        self._pause_started = None
        self._last_pause_cleared = 0.0
        self._last_person_seen = 0.0


# -------- Telemetry listeners --------
    def _location_listener(self, vehicle, attr_name, value):
        try:
            if not value:
                return
            lat = float(value.lat) if value.lat is not None else None
            lon = float(value.lon) if value.lon is not None else None

            with self._telemetry_lock:
                self.latest_telemetry['lat'] = lat
                self.latest_telemetry['lon'] = lon
                self.latest_telemetry['connected'] = True
        except Exception as e:
            print("Location listener error:", e)

    def _rel_location_listener(self, vehicle, attr_name, value):
        try:
            if not value:
                return
            alt = float(value.alt) if value.alt is not None else None

            with self._telemetry_lock:
                self.latest_telemetry['alt'] = alt
                self.latest_telemetry['connected'] = True
        except Exception as e:
            print("Relative location listener error:", e)

    def _velocity_listener(self, vehicle, attr_name, value):
        try:
            if not value:
                return
            vx, vy, vz = value
            speed = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

            with self._telemetry_lock:
                self.latest_telemetry['velocity'] = speed
                self.latest_telemetry['connected'] = True
        except Exception as e:
            print("Velocity listener error:", e)

    def _mode_listener(self, vehicle, attr_name, value):
        try:
            mode_name = value.name if value is not None else None
            with self._telemetry_lock:
                self.latest_telemetry['mode'] = mode_name
                self.latest_telemetry['connected'] = True
        except Exception as e:
            print("Mode listener error:", e)

    def _heading_listener(self, vehicle, attr_name, value):
        try:
            with self._telemetry_lock:
                self.latest_telemetry['heading'] = value
                self.latest_telemetry['connected'] = True
        except Exception as e:
            print("Heading listener error:", e)

    # -------- Camera control --------
    def start_image_stream(self):
        """Start camera stream"""
        try:
            start_camera()
            print("✅ Camera stream started")
        except Exception as e:
            print("❌ Failed to start camera:", e)

    def stop_image_stream(self):
        """Stop camera stream"""
        try:
            stop_camera()
            print("✅ Camera stream stopped")
        except Exception as e:
            print("❌ Failed to stop camera:", e)

    # -------- Person detection --------
    def start_person_detection(self):
        """Start person detection in separate thread"""
        if self.person_running:
            print("⚠️ Person detection already running")
            return

        self.person_running = True
        self.person_thread = threading.Thread(
            target=self._person_detection_loop,
            daemon=True
        )
        self.person_thread.start()
        print("✅ Person detection started")

    # ------- MEDIAPIPE --------
    
    def draw_results(self, frame, bbox, results):
        """Overlay drowning state (single person, no ID)."""
        x1, y1, x2, y2 = map(int, bbox)

        colors = {
            "ACTIVE": (0, 255, 0),
            "FROZEN": (0, 165, 255),
            "SOS": (0, 0, 255)
        }
        state = results.get("state", "ACTIVE")
        color = colors.get(state, (0, 255, 0))

        # bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # water line (optional)
        water_y = results.get("water_level", None)
        if water_y is not None:
            wy = int(water_y)
            cv2.line(frame, (x1, wy), (x2, wy), (255, 255, 0), 1, cv2.LINE_AA)

        # main state label
        cv2.putText(frame, f"{state}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # SOS duration
        if state == "SOS":
            sos_duration = results.get("sos_duration", 0.0)
            cv2.putText(frame, f"SOS: {sos_duration:.1f}s", (x1, y1 - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # lightweight debug (optional, only if enough space)
        h, w = frame.shape[:2]
        if (y2 + 70) < h:
            info_y = y2 + 20
            lines = [
                f"AboveHead: {results.get('arm_above_head', False)}",
                f"Waving: {results.get('waving', False)}",
                f"Vwrist: {results.get('wrist_speed', 0.0):.0f}px/s",
                f"A: {results.get('wrist_accel', 0.0):.0f}px/s2",
                f"Ang: {results.get('elbow_angle_speed', 0.0):.0f}deg/s",
                f"T: {results.get('trigger_elapsed', 0.0):.1f}s",
            ]
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (x1, info_y + i * 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        return color

    def draw_skeleton(self, frame, pose_landmarks, color):
        """Vẽ skeleton lên frame"""
        if pose_landmarks:
            mp_draw.draw_landmarks(
                frame, pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=color, thickness=2),
                mp_draw.DrawingSpec(color=(200, 200, 200), thickness=1)
            )
            
            # Vẽ điểm đầu (nose)
            h, w = frame.shape[:2]
            nose = pose_landmarks.landmark[0]
            nose_x, nose_y = int(nose.x * w), int(nose.y * h)
            cv2.circle(frame, (nose_x, nose_y), 3, (255, 0, 0), -1)


    def _bbox_from_pose_landmarks(self, pose_landmarks, w, h):
        """Build a loose bbox from MediaPipe Pose landmarks (single person).

        Returns [x1,y1,x2,y2] in image pixels or None if not enough valid points.
        """
        if pose_landmarks is None:
            return None

        # Use a subset of stable landmarks (head/shoulders/hips/knees/ankles/wrists)
        idxs = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

        xs, ys = [], []
        for i in idxs:
            try:
                p = pose_landmarks.landmark[i]
                # Prefer reasonably visible points (when available)
                if hasattr(p, 'visibility') and p.visibility is not None and float(p.visibility) < 0.35:
                    continue
                x = float(p.x) * float(w)
                y = float(p.y) * float(h)
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                xs.append(x)
                ys.append(y)
            except Exception:
                continue

        if len(xs) < 4 or len(ys) < 4:
            return None

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        # Pad so bbox covers full body even when some joints are missing
        pad_x = bw * 0.20 + 15.0
        pad_y = bh * 0.25 + 20.0

        x1 = int(max(0, x1 - pad_x))
        y1 = int(max(0, y1 - pad_y))
        x2 = int(min(w - 1, x2 + pad_x))
        y2 = int(min(h - 1, y2 + pad_y))

        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def _smooth_and_lock_bbox(self, bbox, now):
        """Keep a stable single-person bbox so the stream does not jump around."""
        if bbox is None:
            if self._locked_bbox is not None and (now - self._locked_bbox_last_seen) <= self._bbox_lock_ttl:
                return list(self._locked_bbox), "locked"
            self._locked_bbox = None
            return None, None

        bbox = [int(round(float(v))) for v in bbox]
        if self._locked_bbox is None:
            self._locked_bbox = bbox
        else:
            alpha = float(self._bbox_smooth_alpha)
            self._locked_bbox = [
                int(round((1.0 - alpha) * old + alpha * new))
                for old, new in zip(self._locked_bbox, bbox)
            ]
        self._locked_bbox_last_seen = now
        return list(self._locked_bbox), "mediapipe"

    def _update_detection_fps(self, now):
        self._detection_frame_count += 1
        elapsed = now - self._detection_fps_last_time
        if elapsed >= 1.0:
            self.detection_fps = self._detection_frame_count / elapsed
            self._detection_frame_count = 0
            self._detection_fps_last_time = now

    def _update_visual_servo_target(self, bbox, frame_w, frame_h, now):
        if bbox is None or frame_w <= 0 or frame_h <= 0:
            return

        x1, y1, x2, y2 = [float(v) for v in bbox]
        bbox_cx = (x1 + x2) * 0.5
        bbox_cy = (y1 + y2) * 0.5
        frame_cx = float(frame_w) * 0.5
        frame_cy = float(frame_h) * 0.5
        ex = bbox_cx - frame_cx
        ey = bbox_cy - frame_cy

        with self.visual_servo_lock:
            self.visual_servo_target = {
                "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                "frame_w": int(frame_w),
                "frame_h": int(frame_h),
                "ex": float(ex),
                "ey": float(ey),
                "updated_at": float(now),
            }

    def _clear_visual_servo_target_if_stale(self, now):
        was_active = bool(self.visual_servo_last_command.get("active", False))
        with self.visual_servo_lock:
            target = self.visual_servo_target
            if target and (now - float(target.get("updated_at", 0.0))) <= self.visual_servo_target_ttl:
                return

            self.visual_servo_target = None
            self.visual_servo_last_command = {
                "active": False,
                "ex": 0.0,
                "ey": 0.0,
                "vx": 0.0,
                "vy": 0.0,
            }
        self.PID_X.reset()
        self.PID_Y.reset()
        if was_active:
            self._stop_visual_servo_motion()

    def _stop_visual_servo_motion(self):
        try:
            if self.vehicle and self.vehicle.armed and self.vehicle.mode.name == "GUIDED":
                self.send_local_ned_velocity(0.0, 0.0, 0.0)
        except Exception:
            pass

    def _force_exit_visual_servo_to_mission(self, now, reason="no_person_timeout"):
        with self.visual_servo_lock:
            self.visual_servo_target = None
            self.visual_servo_last_command = {
                "active": False,
                "ex": 0.0,
                "ey": 0.0,
                "pid_ex": 0.0,
                "pid_ey": 0.0,
                "vx": 0.0,
                "vy": 0.0,
            }
        self.PID_X.reset()
        self.PID_Y.reset()
        self._stop_visual_servo_motion()
        self.clear_pause()
        if not self._visual_servo_no_person_exit_printed:
            print(
                f"[PID CENTER] exit -> GPS mission resume "
                f"reason={reason} no_person_timeout={self.visual_servo_no_person_timeout:.1f}s"
            )
            self._visual_servo_no_person_exit_printed = True

    def _exit_visual_servo_if_no_person_timeout(self, now, person_now):
        if person_now:
            self._visual_servo_no_person_exit_printed = False
            return

        last_seen = float(getattr(self, "_last_person_seen", 0.0) or 0.0)
        if last_seen <= 0.0:
            return

        no_person_gap = float(now - last_seen)
        if no_person_gap >= float(self.visual_servo_no_person_timeout):
            # Guard: only call once per person-absence event to prevent repeated
            # velocity(0,0,0) commands that would override simple_goto on mission resume.
            if not self._visual_servo_no_person_exit_printed:
                self._force_exit_visual_servo_to_mission(now)

    def _visual_servo_step(self, now=None):
        if not self.visual_servo_enable:
            return False
        if not self.vehicle:
            return False
        try:
            if not self.vehicle.armed or self.vehicle.mode.name != "GUIDED":
                return False
        except Exception:
            return False

        now = time.time() if now is None else float(now)
        with self.visual_servo_lock:
            target = dict(self.visual_servo_target) if self.visual_servo_target else None

        if not target or (now - float(target.get("updated_at", 0.0))) > self.visual_servo_target_ttl:
            self._clear_visual_servo_target_if_stale(now)
            return False

        ex = float(target.get("ex", 0.0))
        ey = float(target.get("ey", 0.0))
        pid_ex = 0.0 if abs(ex) <= self.visual_servo_deadband_px else self.PID_X.update(ex)
        pid_ey = 0.0 if abs(ey) <= self.visual_servo_deadband_px else self.PID_Y.update(ey)

        # Downward-facing camera mapping:
        # bbox below frame center => ey > 0 => move forward in body NED is negative vx.
        # bbox right of frame center => ex > 0 => move right is positive vy.
        vx = -pid_ey
        vy = pid_ex

        self.send_local_ned_velocity(vx, vy, 0.0)
        with self.visual_servo_lock:
            self.visual_servo_last_command = {
                "active": True,
                "ex": ex,
                "ey": ey,
                "pid_ex": float(pid_ex),
                "pid_ey": float(pid_ey),
                "vx": float(vx),
                "vy": float(vy),
            }
        if now - self._last_pid_debug_print >= self._pid_debug_interval:
            print(
                "[PID CENTER] "
                f"ex={ex:.1f}px ey={ey:.1f}px "
                f"pid_x={pid_ex:.4f} pid_y={pid_ey:.4f} "
                f"vx={vx:.4f} vy={vy:.4f}"
            )
            self._last_pid_debug_print = now
        return True

    def _is_frame_center_inside_bbox(self, bbox, frame_w, frame_h, margin_px=0.0):
        if bbox is None or frame_w <= 0 or frame_h <= 0:
            return False
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx = float(frame_w) * 0.5
        cy = float(frame_h) * 0.5
        return (
            (x1 - margin_px) <= cx <= (x2 + margin_px) and
            (y1 - margin_px) <= cy <= (y2 + margin_px)
        )

    def _drop_payload_async(self):
        with self._drop_lock:
            if self._drop_in_progress or self._drop_completed:
                return
            self._drop_in_progress = True

        def worker():
            ok = False
            try:
                if not self.dropper:
                    print("⚠️ DROP skipped: dropper is not configured")
                    return

                print("🎯 DROP condition met: SOS + frame center inside bbox")
                ok = bool(self.dropper.mo_tung_cai())
                print(f"{'✅' if ok else '❌'} DROP command result: {ok}")
            except Exception as e:
                print(f"❌ DROP failed: {e}")
            finally:
                with self._drop_lock:
                    self._drop_completed = bool(ok)
                    if ok:
                        self._last_drop_time = time.time()
                    self._drop_in_progress = False
                if ok:
                    self._stop_visual_servo_motion()
                    self.clear_pause()

        threading.Thread(target=worker, daemon=True).start()

    def _maybe_drop_for_sos(self, bbox, frame_w, frame_h, sos_now):
        center_aligned = bool(
            sos_now and bbox is not None and
            self._is_frame_center_inside_bbox(bbox, frame_w, frame_h, margin_px=self._drop_center_required_px)
        )
        self.drop_center_aligned = center_aligned
        if not sos_now:
            self.drop_center_aligned = False
            return
        with self._drop_lock:
            if self._drop_completed or self._drop_in_progress:
                return
        if self.drop_mode == 'auto' and center_aligned:
            self._drop_payload_async()

    def trigger_manual_drop(self) -> bool:
        """Trigger buoy drop from UI (bypasses center-alignment check)."""
        with self._drop_lock:
            if self._drop_completed or self._drop_in_progress:
                print("Drop already done or in progress")
                return False
        print("MANUAL DROP triggered from UI")
        self._drop_payload_async()
        return True

    def set_drop_mode(self, mode: str):
        """Switch drop mode: 'auto' drops when SOS+centered, 'manual' requires UI swipe."""
        if mode in ('auto', 'manual'):
            self.drop_mode = mode
            print(f"Drop mode -> {mode}")

    def _rearm_drop_if_clear(self, now, person_now):
        if person_now:
            return
        with self._drop_lock:
            if self._drop_completed and (now - self._last_drop_time) >= self._drop_rearm_delay:
                self._drop_completed = False
                print("✅ DROP gate re-armed for next target")


    def _person_detection_loop(self):
        """Person + Drowning detection loop"""
        while self.person_running:
            try:
                current_time = time.time()
                if current_time - self.last_detection_time < self.detection_interval:
                    time.sleep(0.01)
                    continue

                frame_jpeg = get_lastest_frame()
                if frame_jpeg is None:
                    time.sleep(0.01)
                    continue

                nparr = np.frombuffer(frame_jpeg, np.uint8)
                cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                h, w = cv_image.shape[:2]

                rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                pose_res = pose.process(rgb)
                if pose_res.pose_landmarks:
                    self.latest_pose_landmarks = pose_res.pose_landmarks
                
                if cv_image is None:
                    continue
                
                # Prefer MediaPipe Pose for the displayed/tracked bbox. It is much
                # cheaper and more stable after a person is present than repeatedly
                # switching to fresh YOLO/SORT boxes.
                pose_bbox = None
                if pose_res.pose_landmarks:
                    pose_bbox = self._bbox_from_pose_landmarks(pose_res.pose_landmarks, w, h)

                chosen_bbox, chosen_source = self._smooth_and_lock_bbox(pose_bbox, current_time)

                if chosen_bbox is None:
                    raw_dets = self.person_detector.detect(cv_image)
                    cur_fc = getattr(self.person_detector.tracker, "frame_count", None)

                    yolo_fresh = []
                    if cur_fc is not None:
                        for d in raw_dets:
                            t = d.get("track_obj", None)
                            if t is None:
                                continue
                            if int(getattr(t, "last_updated_frame", -1)) == int(cur_fc):
                                yolo_fresh.append(d)
                    else:
                        yolo_fresh = list(raw_dets or [])

                    if yolo_fresh:
                        chosen_bbox, _ = self._smooth_and_lock_bbox(yolo_fresh[0].get("bbox"), current_time)
                        chosen_source = "yolo_seed"

                detections = []
                if chosen_bbox is not None:
                    detections = [{
                        "id": 1,
                        "bbox": chosen_bbox,
                        "track_obj": self._person_track,
                        "source": chosen_source,
                    }]

                states = {"ACTIVE": 0, "FROZEN": 0, "SOS": 0}

                for det in detections:
                    track = self._person_track
                    bbox = det["bbox"]
                    self._update_visual_servo_target(bbox, w, h, current_time)
                    # Phát hiện đuối nước
                    results = self.drowing_detector.detect(
                        track, bbox, pose_res.pose_landmarks, (h, w)
                    )
                    
                    # Lấy thời gian SOS nếu có
                    if results["state"] == "SOS":
                        results["sos_duration"] = self.drowing_detector.get_sos_duration(track)
                    
                    # Thêm drowning_state vào detection
                    det["drowning_state"] = results
                    
                    # Cập nhật thống kê
                    states[results['state']] = states.get(results['state'], 0) + 1
                    
                    # Vẽ kết quả lên frame
                    color = self.draw_results(cv_image, bbox, results)
                    
                    # Vẽ skeleton
                    if pose_res.pose_landmarks:
                        self.draw_skeleton(cv_image, pose_res.pose_landmarks, color)
                
                # Hiển thị thống kê
                cv2.putText(cv_image,
                   f"Active: {states.get('ACTIVE',0)} | Frozen: {states.get('FROZEN',0)} | SOS: {states.get('SOS',0)}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                try:
                    cv2.imshow("DROWNING DETECTION", cv_image)
                    cv2.waitKey(1)
                except:
                    pass
                    
                self.detected_persons = detections[:1] if detections else []
                if detections and not self._drop_completed:
                    self._visual_servo_step(current_time)
                if not detections:
                    self._clear_visual_servo_target_if_stale(current_time)
                self.last_detection_time = current_time
                self._update_detection_fps(current_time)
                # ===== Mission pause trigger (person detected) =====
                person_now = len(detections) > 0
                self._rearm_drop_if_clear(current_time, person_now)
                if person_now and self.pause_enable and not self._drop_completed:
                    try:
                        if self.vehicle and self.vehicle.armed and self.vehicle.mode.name == "GUIDED":
                            self.request_pause("person_detected")
                    except Exception:
                        pass

        # ===== SOS only GPS report =====
                sos_dets = [d for d in detections if d.get("drowning_state", {}).get("state") == "SOS"]
                sos_now = len(sos_dets) > 0 
                sos_bbox = sos_dets[0].get("bbox") if sos_dets else None
                if sos_now and self.pause_enable and not self._drop_completed:
                    # keep paused while SOS
                    self.request_pause("SOS")

                self._maybe_drop_for_sos(sos_bbox, w, h, sos_now)

                # ACTIVE state timeout: non-distress person (ACTIVE, not SOS/FROZEN) for
                # too long → exit PID and continue mission with cooldown on re-pause.
                if person_now and not sos_now:
                    first_det_state = (detections[0].get("drowning_state", {}).get("state", "ACTIVE")
                                       if detections else "ACTIVE")
                    if first_det_state == "ACTIVE":
                        if self._active_state_start <= 0.0:
                            self._active_state_start = current_time
                        active_dur = current_time - self._active_state_start
                        if active_dur >= self._active_timeout_sec and not self._active_timeout_exit_printed:
                            print(f"[ACTIVE TIMEOUT] Person ACTIVE {active_dur:.1f}s → resume mission (non-distress)")
                            self._active_timeout_exit_printed = True
                            self._pause_block_until = current_time + self._active_exit_cooldown_sec
                            self._active_state_start = 0.0
                            self._force_exit_visual_servo_to_mission(current_time, reason="active_state_timeout")
                    else:
                        # FROZEN state: person may be in distress, reset active timer
                        self._active_state_start = 0.0
                        self._active_timeout_exit_printed = False
                else:
                    # No person detected or SOS active: reset active state timer
                    self._active_state_start = 0.0
                    if not person_now:
                        self._active_timeout_exit_printed = False

                # Auto-resume policy (only when NOT SOS)
                self._update_pause_state(person_now=person_now, sos_now=sos_now, now=current_time)
                self._exit_visual_servo_if_no_person_timeout(current_time, person_now)


                # Detect SOS
                just_entered_sos = (sos_now and not self._sos_active)
                if not sos_now:
                    self._sos_active = False

                if sos_now:
                    self._sos_active = True
                    allow_send = just_entered_sos or ((current_time - self._last_sos_post) >= self.sos_post_interval)
                    if allow_send and self.vehicle:
                        try:
                            gf = self.vehicle.location.global_frame
                            gr = self.vehicle.location.global_relative_frame

                            lat = gf.lat if (gf and gf.lat is not None) else None
                            lon = gf.lon if (gf and gf.lon is not None) else None
                            alt = gr.alt if (gr and gr.alt is not None) else (gf.alt if (gf and hasattr(gf, 'alt')) else 0.0)

                            if lat is not None and lon is not None:
                                self._last_sos_post = current_time
                                threading.Thread(
                                    target=self.send_person_detection_to_server,
                                    args=(float(lat), float(lon), float(alt), sos_dets),
                                    daemon=True
                                ).start()
                                print("🚨 SOS detected! Sending alert to server...")
                        except Exception as e:
                            print(f"Error getting GPS for SOS detection: {e}")
            except Exception as e:
                print(f"Person detection error: {e}")
                time.sleep(0.1)

    def _sanitize_detections(self, detections):
        """Make detections JSON-serializable (remove track_obj, cast types)."""
        out = []
        for d in (detections or []):
            if not isinstance(d, dict):
                continue
            dd = {k: v for k, v in d.items() if k not in ('track_obj','id')}
            if 'id' in dd:
                try:
                    dd['id'] = int(dd['id'])
                except Exception:
                    pass
            if 'bbox' in dd and isinstance(dd['bbox'], (list, tuple)) and len(dd['bbox']) == 4:
                try:
                    dd['bbox'] = [int(round(float(x))) for x in dd['bbox']]
                except Exception:
                    pass
            if 'confidence' in dd:
                try:
                    dd['confidence'] = float(dd['confidence'])
                except Exception:
                    pass
            out.append(dd)
        return out

    def send_person_detection_to_server(self, lat, lon, alt, detections):
        """Send person detection results to server"""
        try:
            person_data = {
                'event': 'SOS',
                'state': 'SOS',
                'lat': lat,
                'lon': lon,
                'alt': alt,
                'timestamp': time.time(),
                'detections': self._sanitize_detections(detections),
                'count': len(detections)
            }
            
            response = requests.post(
                'http://127.0.0.1:5000/update_person_detection',
                json=person_data,
                timeout=2
            )
            if response.status_code == 200:
                print(f"✅ Sent {len(detections)} person detections to server")
            else:
                print(f"❌ Failed to send detections: {response.status_code}")
                
        except Exception as e:
            print(f"Error sending person detection: {e}")

    def stop_person_detection(self):
        """Stop person detection"""
        self.person_running = False
        if self.person_thread:
            self.person_thread.join(timeout=2.0)
        print("Person detection stopped")

    # -------- Pause / Resume logic (mission) --------
    def request_pause(self, reason="person_detected"):
            if not self.vehicle:
                return
            now = time.time()
            with self._pause_lock:
                # Cooldown to avoid pause-resume oscillation (SOS bypasses cooldown)
                if str(reason).upper() != "SOS":
                    if now < float(getattr(self, "_pause_block_until", 0.0)):
                        return
                    if (now - float(getattr(self, "_last_pause_cleared", 0.0))) < float(getattr(self, "pause_retrigger_cooldown_sec", 0.0)):
                        return

                if not self._pause_requested:
                    self._pause_requested = True
                    self._pause_reason = str(reason) if reason is not None else None
                    self._pause_started = now
                    print(f"Mission PAUSE requested: {self._pause_reason}")
                else:
                    # Escalate reason to SOS if needed
                    if str(reason).upper() == "SOS" and (self._pause_reason != "SOS"):
                        self._pause_reason = "SOS"
                        print("Mission PAUSE escalated: SOS")
    def clear_pause(self):
        with self._pause_lock:
            if self._pause_requested:
                self._pause_requested = False
                self._pause_reason = None
                self._pause_started = None
                self._last_pause_cleared = time.time()
                print("Mission RESUME (no SOS)")
        # Re-issue the active goto target immediately so ArduPilot gets a position
        # command right away, before the goto loop's next 200 ms sleep cycle.
        try:
            target = self._current_goto_target
            speed = float(self._current_goto_speed or 0.7)
            if target is not None and self.vehicle and self.vehicle.armed:
                self.vehicle.simple_goto(target, groundspeed=speed)
        except Exception:
            pass

    def is_pause_requested(self):
        with self._pause_lock:
            return bool(self._pause_requested)
    
    def get_pause_info(self):
        with self._pause_lock:
            paused = bool(self._pause_requested)
            started = self._pause_started
            reason = self._pause_reason
        elapsed = (time.time() - started) if (paused and started) else 0.0
        return {
            "mission_paused": paused,
            "pause_reason": reason,
            "pause_elapsed": float(elapsed),
            "visual_servo": dict(self.visual_servo_last_command)
        }

    def _hold_position_step(self):
        if not self.vehicle:
            return
        try:
            loc = self.vehicle.location.global_relative_frame
            if loc and (loc.lat is not None) and (loc.lon is not None):
                alt = float(loc.alt) if (loc.alt is not None) else float(self.takeoff_height)
                hold = LocationGlobalRelative(float(loc.lat), float(loc.lon), alt)
                try:
                    self.vehicle.simple_goto(hold, groundspeed=0.0)
                except Exception:
                    pass
            try:
                if not self._visual_servo_step():
                    self.send_local_ned_velocity(0.0, 0.0, 0.0)
            except Exception:
                pass
        except Exception:
            pass

    def _update_pause_state(self, person_now: bool, sos_now: bool, now: float):
        """Auto-resume policy when paused and no SOS."""
        if not self.pause_enable:
            return

        if person_now:
            with self._pause_lock:
                self._last_person_seen = float(now)

        with self._pause_lock:
            if not self._pause_requested:
                return

            # If SOS, keep paused
            if sos_now:
                if not self._drop_completed:
                    self._pause_reason = "SOS"
                    return
                self._pause_requested = False
                self._pause_reason = None
                self._pause_started = None
                self._last_pause_cleared = float(now)
                print("Mission RESUME after payload drop")
                return

            started = float(self._pause_started or now)
            elapsed = float(now - started)

            if elapsed < float(self.pause_min_hold_sec):
                return

            last_seen = float(self._last_person_seen or 0.0)
            no_person_gap = float(now - last_seen) if last_seen > 0 else 1e9

            if no_person_gap >= float(self.pause_clear_no_person_sec) or elapsed >= float(self.pause_max_hold_sec):
                self._pause_requested = False
                self._pause_reason = None
                self._pause_started = None
                self._last_pause_cleared = float(now)
                print("Mission RESUME (auto)")

            # -------- MAVLink control --------
    def send_local_ned_velocity(self, vx, vy, vz):
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0,
            self.vehicle._master.target_system,
            self.vehicle._master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            1479,
            0, 0, 0,
            float(vx), float(vy), float(vz),
            0, 0, 0,
            0.0, 0,
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    def set_speed(self, speed):
        """Set vehicle speed"""
        if not self.vehicle:
            return
        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0,
            1,
            speed,
            -1, 0, 0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()
        print(f"✅ Speed set to {speed} m/s")

    def set_fixed_heading(self, heading_deg, yaw_rate=10, relative=False):
        """Set fixed compass heading"""
        if not self.vehicle:
            return

        current = getattr(self.vehicle, 'heading', None)
        if current is None:
            direction = 1
        else:
            diff = (heading_deg - current + 360.0) % 360.0
            direction = 1 if diff <= 180.0 else -1

        is_relative = 1 if relative else 0

        msg = self.vehicle.message_factory.command_long_encode(
            0, 0,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,
            float(heading_deg),
            float(yaw_rate),
            float(direction),
            float(is_relative),
            0, 0, 0
        )
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()
        print(f"✅ Heading set to {heading_deg}°")

    # -------- Mission control --------
    def arm_and_takeoff(self, targetHeight):
        """Arm and takeoff to specified altitude"""
        if not self.vehicle:
            return

        while not self.vehicle.is_armable:
            print('Waiting for vehicle to become armable')
            time.sleep(1)

        while self.vehicle.mode.name != 'GUIDED':
            # self.vehicle.mode = VehicleMode("GUIDED")
            print('Waiting for GUIDED mode...')
            time.sleep(1)

        self.vehicle.armed = True
        while not self.vehicle.armed:
            print('Arming...')
            time.sleep(1)

        self.vehicle.simple_takeoff(targetHeight)
        while True:
            alt = self.vehicle.location.global_relative_frame.alt
            print(f'📊 Altitude: {alt:.2f}' if alt else 'Altitude: 0.00')
            if alt and alt >= 0.95 * targetHeight:
                break
            time.sleep(1)

        print("✅ Reached takeoff altitude")

    def arm_drone(self):
        """Arm drone without takeoff"""
        if not self.vehicle:
            return False

        while self.vehicle.mode.name != 'GUIDED':
            print('Waiting for GUIDED mode...')
            self.vehicle.mode = VehicleMode("GUIDED")
            time.sleep(1)

        self.vehicle.armed = True
        while not self.vehicle.armed:
            print('Arming...')
            time.sleep(1)

        print("✅ Drone is armed and ready")
        return True
        # ---------------- GOTO / WAYPOINTS ----------------
    def get_distance_meters(self, targetLocation, currentLocation):
        dLat = targetLocation.lat - currentLocation.lat
        dLon = targetLocation.lon - currentLocation.lon
        return math.sqrt((dLon * dLon) + (dLat * dLat)) * 1.113195e5
    

    def goto(self, targetLocation, tolerance=0.7, timeout=60, speed=0.7):
        if speed < 0.1 or speed > 5.0:
            print(f"Toc do {speed} m/s khong hop ly, set lai 0.7 m/s")
            speed = 0.7
        if not self.vehicle:
            return False

        self._current_goto_target = targetLocation
        self._current_goto_speed = speed

        distanceToTargetLocation = self.get_distance_meters(
            targetLocation, self.vehicle.location.global_relative_frame
        )
        self.set_speed(speed)
        self.vehicle.simple_goto(targetLocation, groundspeed=speed)

        start_dist = distanceToTargetLocation
        start_time = time.time()

        # pause-aware timeout
        pause_accum = 0.0
        pause_start = None

        while self.vehicle.mode.name == "GUIDED" and self.vehicle.armed:
            now = time.time()

            elapsed = now - start_time - pause_accum
            if pause_start is not None:
                elapsed -= (now - pause_start)
            if elapsed > timeout:
                break

            # ===== pause handling =====
            if self.is_pause_requested():
                if pause_start is None:
                    pause_start = now
                    print(f"[PAUSE] Holding position (reason={self._pause_reason})")
                self._hold_position_step()
                time.sleep(1.0 / self.pause_hold_hz)
                continue
            else:
                if pause_start is not None:
                    pause_accum += (now - pause_start)
                    pause_start = None
                    print("[PAUSE] Resume mission")
                    try:
                        self.set_speed(speed)
                        self.vehicle.simple_goto(targetLocation, groundspeed=speed)
                    except Exception:
                        pass
            # normal navigation
            currentDistance = self.get_distance_meters(
                targetLocation, self.vehicle.location.global_relative_frame
            )

            # Record current position
            current_pos = self.vehicle.location.global_relative_frame
            if current_pos.lat and current_pos.lon:
                self.flown_path.append([current_pos.lat, current_pos.lon])

            if currentDistance < max(tolerance, start_dist * 0.01):
                print("Reached target waypoint")
                self._current_goto_target = None
                return True

            time.sleep(0.02)
        print("Timeout reaching waypoint, proceeding anyway")
        self._current_goto_target = None
        return False


    def land(self):
        """Land the drone"""
        if not self.vehicle:
            return
        self.vehicle.mode = VehicleMode("LAND")
        while self.vehicle.armed:
            print("Landing...")
            time.sleep(1)
        print("✅ Landed successfully")


    def fly_and_precision_land_with_waypoints(self, waypoints, takeoff_height=4):
        """
        Fly to waypoints while detecting ArUco markerss
        """
        if not self.vehicle:
            print(" No vehicle connected")
            return

        if not waypoints or len(waypoints) < 2:
            raise ValueError("Invalid waypoints")

        self.flown_path = []

        print("Arming and taking off")
        self.arm_and_takeoff(takeoff_height)
        time.sleep(1)

        self.start_person_detection()

        home_lat = self.vehicle.location.global_relative_frame.lat
        home_lon = self.vehicle.location.global_relative_frame.lon
        wp_home = LocationGlobalRelative(home_lat, home_lon, takeoff_height)
        print(f" Home recorded at lat={home_lat:.6f}, lon={home_lon:.6f}")

        # Fly middle waypoints
        for i, wp in enumerate(waypoints[1:-1]):
            speed = wp.get('speed', 0.7)
            wp_loc = LocationGlobalRelative(wp['lat'], wp['lon'], takeoff_height)
            print(
                f"Flying to waypoint {i + 1}: {wp['lat']}, {wp['lon']} "
                f"at speed {speed} m/s"
            )
            self.goto(wp_loc, speed=speed)

        # Final goal
        goal_wp = waypoints[-1]
        speed = goal_wp.get('speed', 0.7)
        wp_target = LocationGlobalRelative(
            goal_wp['lat'], goal_wp['lon'], takeoff_height
        )
        print(
            f"Flying to final target {goal_wp['lat']}, {goal_wp['lon']} "
            f"at speed {speed} m/s"
        )
        self.goto(wp_target, speed=speed)

        self.stop_person_detection()

        print("Starting landing phase...")
        self.land()

        while self.vehicle.armed:
            print("Waiting for disarming...")
            time.sleep(1)

        print("Mission complete")

# ===================== SINGLETON CONTROLLER =====================
_controller = None

def get_controller(connection_str='tcp:127.0.0.1:5760', takeoff_height=5):
    global _controller
    if _controller is None:
        _controller = DroneController(
            connection_str=connection_str,
            takeoff_height=takeoff_height
        )
    return _controller

# ===================== CLEANUP =====================
def cleanup():
    """Cleanup resources"""
    global _controller
    if _controller:
        _controller.stop_person_detection()
        _controller.stop_image_stream()
    stop_camera()

import atexit
atexit.register(cleanup)
