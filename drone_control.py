
import time
import math
import threading
import numpy as np
import cv2
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import requests
from collections import deque
import mediapipe as mp
from person_detector import PersonDetector
from drowing_detector import DrowningDetector
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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
_hailo_pose_runtime = None
IMG_SIZE = 320
DROP_UART_PORT = "/dev/ttyUSB0"
DROP_TRIGGER_PERCENT = 90

HAILO_POSE_ENABLED = os.getenv("HAILO_POSE_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
HAILO_POSE_INPUT = os.getenv("HAILO_POSE_INPUT", "usb")
HAILO_POSE_MAX_TRACK_AGE = float(os.getenv("HAILO_POSE_MAX_TRACK_AGE", "1.0"))

COCO_KEYPOINTS = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16,
}

HAILO_COCO_CONNECTIONS = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
)

COCO_TO_MEDIAPIPE = {
    0: 0,    # nose
    1: 1,    # left eye
    2: 2,    # right eye
    5: 11,   # left shoulder
    6: 12,   # right shoulder
    7: 13,   # left elbow
    8: 14,   # right elbow
    9: 15,   # left wrist
    10: 16,  # right wrist
    11: 23,  # left hip
    12: 24,  # right hip
    13: 25,  # left knee
    14: 26,  # right knee
    15: 27,  # left ankle
    16: 28,  # right ankle
}


class HailoPoseLandmark:
    """Small MediaPipe-compatible point used by DrowningDetector."""
    __slots__ = ("x", "y", "visibility")

    def __init__(self, x=0.0, y=0.0, visibility=0.0):
        self.x = float(x)
        self.y = float(y)
        self.visibility = float(visibility)


class HailoPoseLandmarks:
    """MediaPipe-like landmark container built from Hailo COCO keypoints."""
    def __init__(self, coco_points, frame_w, frame_h):
        self.landmark = [HailoPoseLandmark(0.5, 0.5, 0.0) for _ in range(33)]
        if not frame_w or not frame_h:
            return
        for coco_idx, mp_idx in COCO_TO_MEDIAPIPE.items():
            if coco_idx >= len(coco_points):
                continue
            p = coco_points[coco_idx]
            if p is None:
                continue
            x, y, conf = p
            self.landmark[mp_idx] = HailoPoseLandmark(
                float(x) / float(frame_w),
                float(y) / float(frame_h),
                conf if conf is not None else 1.0
            )


def _point_value(point, name, default=None):
    try:
        value = getattr(point, name)
        return value() if callable(value) else value
    except Exception:
        return default


def _make_hailo_pose_landmarks(coco_points, frame_w, frame_h):
    return HailoPoseLandmarks(coco_points, frame_w, frame_h)


def _draw_hailo_pose_skeleton(frame, coco_points, color):
    if not coco_points:
        return
    for a, b in HAILO_COCO_CONNECTIONS:
        if a >= len(coco_points) or b >= len(coco_points):
            continue
        pa, pb = coco_points[a], coco_points[b]
        if pa is None or pb is None:
            continue
        ax, ay, _ = pa
        bx, by, _ = pb
        cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), (200, 200, 200), 1, cv2.LINE_AA)
    for p in coco_points:
        if p is None:
            continue
        x, y, _ = p
        cv2.circle(frame, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)


class HailoPoseRuntime:
    """Runs the Hailo pose GStreamer pipeline and exposes latest frame/detections."""
    def __init__(self, input_source=HAILO_POSE_INPUT):
        self.input_source = input_source
        self.lock = threading.Lock()
        self.latest_frame_jpeg = None
        self.latest_detections = []
        self.latest_frame_shape = None
        self.latest_timestamp = 0.0
        self.running = False
        self.thread = None
        self.app = None
        self.user_data = None
        self._imports = None
        self.ready_event = threading.Event()
        self.failed_event = threading.Event()

    def _load_hailo(self):
        if self._imports is not None:
            return self._imports
        project_root = Path(__file__).resolve().parents[2]
        hailo_site = project_root / "venv_hailo_rpi_examples" / "lib" / "python3.11" / "site-packages"
        if hailo_site.exists() and str(hailo_site) not in sys.path:
            sys.path.insert(0, str(hailo_site))
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        import hailo
        from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
        from hailo_apps.hailo_app_python.core.common.core import get_default_parser
        from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
        from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp
        from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import (
            SOURCE_PIPELINE, INFERENCE_PIPELINE, INFERENCE_PIPELINE_WRAPPER,
            TRACKER_PIPELINE, USER_CALLBACK_PIPELINE, DISPLAY_PIPELINE
        )
        self._imports = SimpleNamespace(
            Gst=Gst,
            hailo=hailo,
            get_caps_from_pad=get_caps_from_pad,
            get_numpy_from_buffer=get_numpy_from_buffer,
            get_default_parser=get_default_parser,
            app_callback_class=app_callback_class,
            GStreamerPoseEstimationApp=GStreamerPoseEstimationApp,
            SOURCE_PIPELINE=SOURCE_PIPELINE,
            INFERENCE_PIPELINE=INFERENCE_PIPELINE,
            INFERENCE_PIPELINE_WRAPPER=INFERENCE_PIPELINE_WRAPPER,
            TRACKER_PIPELINE=TRACKER_PIPELINE,
            USER_CALLBACK_PIPELINE=USER_CALLBACK_PIPELINE,
            DISPLAY_PIPELINE=DISPLAY_PIPELINE,
        )
        return self._imports

    def start(self):
        if self.running:
            return True
        if not HAILO_POSE_ENABLED:
            return False
        self.running = True
        self.ready_event.clear()
        self.failed_event.clear()
        try:
            self._create_app()
        except Exception as e:
            self.running = False
            self.failed_event.set()
            print(f"⚠️ Hailo pose stream unavailable, falling back to OpenCV camera: {e}")
            return False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.failed_event.wait(timeout=0.75)
        if self.failed_event.is_set():
            return False
        return True

    def stop(self):
        self.running = False
        try:
            if self.app and self.app.loop:
                self.app.loop.quit()
        except Exception:
            pass
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None

    def is_active(self):
        return bool(self.running and self.thread and self.thread.is_alive())

    def get_latest(self, max_age=HAILO_POSE_MAX_TRACK_AGE):
        with self.lock:
            ts = float(self.latest_timestamp or 0.0)
            if ts <= 0.0 or (time.time() - ts) > float(max_age):
                return [], None, None, ts
            detections = [dict(d) for d in self.latest_detections]
            return detections, self.latest_frame_jpeg, self.latest_frame_shape, ts

    def _store_frame_and_detections(self, frame, detections, frame_shape):
        global _latest_frame_jpeg
        jpeg_bytes = None
        if frame is not None:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ok, jpeg = cv2.imencode('.jpg', bgr, [
                int(cv2.IMWRITE_JPEG_QUALITY), 85,
                int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
            ])
            if ok:
                jpeg_bytes = jpeg.tobytes()
        with self.lock:
            if jpeg_bytes is not None:
                self.latest_frame_jpeg = jpeg_bytes
                with _latest_frame_lock:
                    _latest_frame_jpeg = jpeg_bytes
            self.latest_detections = detections
            self.latest_frame_shape = frame_shape
            self.latest_timestamp = time.time()

    def _create_app(self):
        imports = self._load_hailo()
        project_root = Path(__file__).resolve().parents[2]
        env_file = project_root / ".env"
        if env_file.exists():
            os.environ["HAILO_ENV_FILE"] = str(env_file)

        class _UserData(imports.app_callback_class):
            def __init__(self):
                super().__init__()
                self.use_frame = True

        class _HeadlessPoseApp(imports.GStreamerPoseEstimationApp):
            def get_pipeline_string(app_self):
                source_pipeline = imports.SOURCE_PIPELINE(
                    video_source=app_self.video_source,
                    video_width=app_self.video_width,
                    video_height=app_self.video_height,
                    frame_rate=app_self.frame_rate,
                    sync=app_self.sync
                )
                infer_pipeline = imports.INFERENCE_PIPELINE(
                    hef_path=app_self.hef_path,
                    post_process_so=app_self.post_process_so,
                    post_function_name=app_self.post_process_function,
                    batch_size=app_self.batch_size
                )
                return (
                    f'{source_pipeline} ! '
                    f'{imports.INFERENCE_PIPELINE_WRAPPER(infer_pipeline)} ! '
                    f'{imports.TRACKER_PIPELINE(class_id=0)} ! '
                    f'{imports.USER_CALLBACK_PIPELINE()} ! '
                    f'{imports.DISPLAY_PIPELINE(video_sink="fakesink", sync=app_self.sync, show_fps=app_self.show_fps)}'
                )

        old_argv = sys.argv[:]
        sys.argv = [old_argv[0], "--input", self.input_source, "--disable-sync"]
        try:
            self.user_data = _UserData()
            self.app = _HeadlessPoseApp(self._callback, self.user_data, imports.get_default_parser())
        finally:
            sys.argv = old_argv
        print(f"✅ Hailo pose stream prepared from {self.input_source}")
        self.ready_event.set()

    def _callback(self, pad, info, user_data):
        imports = self._load_hailo()
        buffer = info.get_buffer()
        if buffer is None:
            return imports.Gst.PadProbeReturn.OK

        fmt, width, height = imports.get_caps_from_pad(pad)
        frame = None
        if fmt is not None and width is not None and height is not None:
            try:
                frame = imports.get_numpy_from_buffer(buffer, fmt, width, height)
            except Exception:
                frame = None

        detections = []
        try:
            roi = imports.hailo.get_roi_from_buffer(buffer)
            hailo_dets = roi.get_objects_typed(imports.hailo.HAILO_DETECTION)
            for det in hailo_dets:
                if det.get_label() != "person":
                    continue
                bbox = det.get_bbox()
                x1 = int(max(0, round(bbox.xmin() * width)))
                y1 = int(max(0, round(bbox.ymin() * height)))
                x2 = int(min(width - 1, round((bbox.xmin() + bbox.width()) * width)))
                y2 = int(min(height - 1, round((bbox.ymin() + bbox.height()) * height)))
                track_id = len(detections) + 1
                track = det.get_objects_typed(imports.hailo.HAILO_UNIQUE_ID)
                if len(track) == 1:
                    track_id = int(track[0].get_id())

                coco_points = [None] * 17
                landmarks = det.get_objects_typed(imports.hailo.HAILO_LANDMARKS)
                if landmarks:
                    for name, idx in COCO_KEYPOINTS.items():
                        point = landmarks[0].get_points()[idx]
                        px = (float(_point_value(point, "x", 0.0)) * bbox.width() + bbox.xmin()) * width
                        py = (float(_point_value(point, "y", 0.0)) * bbox.height() + bbox.ymin()) * height
                        conf = _point_value(point, "confidence", None)
                        coco_points[idx] = (float(px), float(py), float(conf if conf is not None else 1.0))

                pose_landmarks = _make_hailo_pose_landmarks(coco_points, width, height)
                detections.append({
                    "id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(det.get_confidence()),
                    "pose_landmarks": pose_landmarks,
                    "pose_keypoints": coco_points,
                    "source": "hailo_pose",
                })
        except Exception as e:
            print(f"Hailo pose callback error: {e}")

        self._store_frame_and_detections(frame, detections, (height, width))
        return imports.Gst.PadProbeReturn.OK

    def _run(self):
        try:
            print(f"✅ Hailo pose stream started from {self.input_source}")
            try:
                self.app.run()
            except SystemExit:
                pass
            finally:
                self.running = False
        except Exception as e:
            self.running = False
            self.failed_event.set()
            print(f"⚠️ Hailo pose stream unavailable, falling back to OpenCV camera: {e}")

def start_camera(camera_index=0):
    """Start USB camera"""
    global _usb_cam, _camera_running, _camera_thread, _hailo_pose_runtime

    if _camera_running:
        if _hailo_pose_runtime is not None and not _hailo_pose_runtime.is_active() and _usb_cam is None:
            _camera_running = False
        else:
            return

    if HAILO_POSE_ENABLED and threading.current_thread() is threading.main_thread():
        try:
            if _hailo_pose_runtime is None:
                _hailo_pose_runtime = HailoPoseRuntime(HAILO_POSE_INPUT)
            if _hailo_pose_runtime.start():
                _camera_running = True
                print("✅ Hailo pose camera stream requested")
                return
        except Exception as e:
            print(f"⚠️ Hailo pose camera start failed, using OpenCV camera: {e}")
    elif HAILO_POSE_ENABLED:
        print("⚠️ Hailo pose must be initialized from the main thread; using OpenCV camera")

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
    global _usb_cam, _camera_running, _camera_thread, _hailo_pose_runtime

    _camera_running = False

    if _hailo_pose_runtime:
        try:
            _hailo_pose_runtime.stop()
        except Exception as e:
            print("Error stopping Hailo pose stream:", e)

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
    global _latest_frame_jpeg, _latest_frame_lock, _hailo_pose_runtime
    if _hailo_pose_runtime:
        try:
            _, frame_jpeg, _, _ = _hailo_pose_runtime.get_latest(max_age=2.0)
            if frame_jpeg is not None:
                return frame_jpeg
        except Exception:
            pass
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
    "T_FROZEN": 0.5,
    "T_SOS": 2.0,
    # RESET_GAP lớn hơn để camera hướng xuống (pose chập chờn) không reset bộ đếm
    # distress liên tục -> FROZEN có cơ hội escalate lên SOS thay vì kẹt ở FROZEN.
    "RESET_GAP": 2.0,
    "HEAD_Y_MARGIN": 0.0
}

# =====================================================================
# DOWNWARD CAMERA  ->  SINGLE GPS SOS TARGET  (latch-once state machine)
# ---------------------------------------------------------------------
# Camera nằm dưới bụng drone, hướng thẳng xuống đất. Khi phát hiện người ở
# trạng thái FROZEN/SOS (đã xác nhận qua nhiều frame), ta ước lượng MỘT điểm
# GPS duy nhất của nạn nhân từ bbox, khoá (latch) điểm đó lại, cho drone bay
# tới đúng 1 lần, đến nơi thì giữ vị trí và thả phao đúng 1 lần.
# Tuyệt đối KHÔNG gửi GPS target / GOTO liên tục mỗi frame.
# =====================================================================

# ----- Camera intrinsics (ưu tiên ma trận calibration nếu có) -----
IMAGE_WIDTH = horizontal_res
IMAGE_HEIGHT = vertical_res
if CAMERA_MATRIX is not None:
    CAMERA_FX = float(CAMERA_MATRIX[0, 0])
    CAMERA_FY = float(CAMERA_MATRIX[1, 1])
    CAMERA_CX = float(CAMERA_MATRIX[0, 2])
    CAMERA_CY = float(CAMERA_MATRIX[1, 2])
else:
    CAMERA_FX = 600.0
    CAMERA_FY = 600.0
    CAMERA_CX = IMAGE_WIDTH / 2.0
    CAMERA_CY = IMAGE_HEIGHT / 2.0

# Hướng lắp camera. Nếu cạnh trên ảnh trùng mũi drone thì offset = 0.
CAMERA_MOUNT = "DOWNWARD"
CAMERA_TOP_ALIGNED_WITH_DRONE_NOSE = True
CAMERA_YAW_OFFSET_RAD = 0.0 if CAMERA_TOP_ALIGNED_WITH_DRONE_NOSE else math.radians(
    float(os.getenv("CAMERA_YAW_OFFSET_DEG", "0"))
)

# ----- Altitude AGL -----
USE_RANGEFINDER_ALTITUDE = True
DEFAULT_ALTITUDE_AGL = None   # None => không có giá trị fallback cứng
MIN_VALID_HEIGHT_M = 0.5      # dưới mức này coi như altitude không đáng tin

# ----- Xác nhận distress -----
# DISTRESS_STATES = các trạng thái được phép GOM frame xác nhận. FROZEN được gom
# để bắt đầu đếm thời gian, nhưng việc KHOÁ GPS target chỉ xảy ra khi đủ điều
# kiện SOS (xem SOS_LOCK_STATE + escalate bên dưới) -> yêu cầu #1/#10.
DISTRESS_STATES = ["FROZEN", "SOS"]
# Ngưỡng confidence của bbox người (Hailo thường ~0.5-0.7). Đặt 0.75 sẽ loại bỏ
# hầu hết detection thật -> SOS không bao giờ bắt đầu xác nhận. Hạ về 0.5.
DISTRESS_MIN_CONF = float(os.getenv("DISTRESS_MIN_CONF", "0.5"))
DISTRESS_CONFIRM_FRAMES = 3
DISTRESS_CONFIRM_TIME = 0.5
DISTRESS_LOST_RESET_SEC = 2.0   # mất detect trước khi confirm xong -> reset confirm

# ----- Điều kiện coi là "SOS" để KHOÁ GPS target (yêu cầu #1, #10) -----
# Chỉ khi AI báo đúng trạng thái SOS_LOCK_STATE, HOẶC trạng thái distress
# (FROZEN/SOS) được giữ liên tục đủ SOS_FROZEN_ESCALATE_SEC giây thì mới tạo &
# khoá MỘT GPS target. Detect người bình thường (ACTIVE) tuyệt đối không tạo GPS.
SOS_LOCK_STATE = "SOS"
ESCALATE_FROZEN_TO_SOS = os.getenv("ESCALATE_FROZEN_TO_SOS", "1").strip().lower() not in ("0", "false", "no", "off")
SOS_FROZEN_ESCALATE_SEC = float(os.getenv("SOS_FROZEN_ESCALATE_SEC", "2.0"))

# ----- Camera FOV (dùng cho hàm estimate_sos_gps_from_bbox theo yêu cầu #7) -----
# Ưu tiên ma trận calibration (fx/fy) nếu có; FOV chỉ là phương án dự phòng.
CAMERA_FOV_X_DEG = float(os.getenv("CAMERA_FOV_X_DEG", "70.0"))
CAMERA_FOV_Y_DEG = float(os.getenv("CAMERA_FOV_Y_DEG", "55.0"))

# ----- Lọc nhiễu bbox & giới hạn offset -----
TARGET_SMOOTHING_FRAMES = 5
MAX_VALID_OFFSET_M = 30.0
BBOX_EDGE_MARGIN_FRAC = 0.05    # center phải nằm trong vùng an toàn 5% viền ảnh

# ----- Vòng đời SOS target / bán kính -----
SOS_TARGET_TTL = 20.0
SOS_GOTO_RADIUS_M = 1.5
DROP_RADIUS_M = 1.2
DROP_HOLD_TIME = 1.0
SOS_GOTO_SPEED = float(os.getenv("SOS_GOTO_SPEED", "0.7"))
POST_DROP_COOLDOWN_SEC = float(os.getenv("POST_DROP_COOLDOWN_SEC", "30.0"))

# ----- Drop & resume -----
DROP_ONCE = True
RESUME_MISSION_AFTER_DROP = True
DROP_METHOD = os.getenv("DROP_METHOD", "SERVO").upper()   # "UART" | "SERVO"
DROP_UART_PORT = "/dev/ttyUSB0"
DROP_UART_BAUD = 9600
DROP_COMMAND = "DROP\n"

# ----- An toàn / test -----
ENABLE_SOS_GOTO = os.getenv("ENABLE_SOS_GOTO", "1").strip().lower() not in ("0", "false", "no", "off")
MOCK_MODE = os.getenv("MOCK_MODE", "0").strip().lower() in ("1", "true", "yes", "on")

# WGS84 mean Earth radius dùng cho đổi offset NED -> lat/lon.
EARTH_RADIUS_M = 6378137.0


# =====================================================================
# HÀM PHỤ: hình học camera hướng xuống  (pixel -> body -> NED -> GPS)
# =====================================================================
def is_valid_bbox(bbox, frame_w, frame_h):
    """Kiểm tra bbox hợp lệ và center không nằm sát mép ảnh quá nhiều."""
    if bbox is None or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return False
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    if not frame_w or not frame_h or frame_w <= 0 or frame_h <= 0:
        return False
    # bbox quá nhỏ -> bỏ
    if (x2 - x1) < 4.0 or (y2 - y1) < 4.0:
        return False
    # center phải nằm trong vùng an toàn (tránh nhiễu ở mép ảnh)
    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0
    mx = BBOX_EDGE_MARGIN_FRAC * frame_w
    my = BBOX_EDGE_MARGIN_FRAC * frame_h
    if u < mx or u > (frame_w - mx) or v < my or v > (frame_h - my):
        return False
    return True


def pixel_to_body_offset(u, v, altitude_agl, fx=CAMERA_FX, fy=CAMERA_FY,
                         cx=CAMERA_CX, cy=CAMERA_CY):
    """Đổi 1 điểm ảnh (u, v) sang offset mét trong BODY frame của drone.

    Camera hướng xuống, mặt đất phẳng ở độ cao altitude_agl bên dưới drone.
      x_norm = (u - cx) / fx ;  y_norm = (v - cy) / fy
      offset_right_m   = x_norm * altitude_agl   (u > cx -> nạn nhân bên phải)
      offset_forward_m = -y_norm * altitude_agl  (v < cy -> phía trước mũi drone)
    """
    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy
    offset_right_m = x_norm * altitude_agl
    offset_forward_m = -y_norm * altitude_agl
    # Nếu camera xoay lệch so với mũi drone, xoay offset trong body frame.
    if CAMERA_YAW_OFFSET_RAD != 0.0:
        c = math.cos(CAMERA_YAW_OFFSET_RAD)
        s = math.sin(CAMERA_YAW_OFFSET_RAD)
        f = offset_forward_m * c - offset_right_m * s
        r = offset_forward_m * s + offset_right_m * c
        offset_forward_m, offset_right_m = f, r
    return offset_forward_m, offset_right_m, x_norm, y_norm


def body_to_ned_offset(body_forward, body_right, yaw_rad):
    """Đổi offset BODY (forward/right) sang NED (north/east) theo yaw drone."""
    dN = body_forward * math.cos(yaw_rad) - body_right * math.sin(yaw_rad)
    dE = body_forward * math.sin(yaw_rad) + body_right * math.cos(yaw_rad)
    return dN, dE


def ned_offset_to_gps(current_lat, current_lon, dN, dE):
    """Đổi offset mét NED sang GPS lat/lon quanh vị trí hiện tại."""
    target_lat = current_lat + (dN / EARTH_RADIUS_M) * 180.0 / math.pi
    target_lon = current_lon + (dE / (EARTH_RADIUS_M * math.cos(current_lat * math.pi / 180.0))) * 180.0 / math.pi
    return target_lat, target_lon


def get_distance_metres(lat1, lon1, lat2, lon2):
    """Khoảng cách ngang (m) gần đúng giữa 2 điểm lat/lon."""
    dlat = (lat2 - lat1) * math.pi / 180.0 * EARTH_RADIUS_M
    dlon = (lon2 - lon1) * math.pi / 180.0 * EARTH_RADIUS_M * math.cos(lat1 * math.pi / 180.0)
    return math.sqrt(dlat * dlat + dlon * dlon)


def estimate_sos_gps_from_bbox(current_lat, current_lon, current_alt, bbox,
                               image_width, image_height,
                               camera_fov_x=None, camera_fov_y=None,
                               yaw_deg=0.0):
    """Ước lượng GPS ảo của người SOS từ bbox bằng FOV camera (yêu cầu #7).

    Đây là phương án dùng FOV (khi KHÔNG có ma trận intrinsics). Camera hướng
    thẳng xuống, mặt đất phẳng ở độ cao current_alt (AGL) bên dưới drone.

    Trả về (sos_lat, sos_lon). Hàm này CHỈ nên được gọi khi state == "SOS".
    """
    if bbox is None or len(bbox) != 4 or current_alt is None or current_alt <= 0:
        return None
    if camera_fov_x is None:
        camera_fov_x = math.radians(CAMERA_FOV_X_DEG)
    if camera_fov_y is None:
        camera_fov_y = math.radians(CAMERA_FOV_Y_DEG)

    x1, y1, x2, y2 = [float(v) for v in bbox]
    bbox_center_x = (x1 + x2) / 2.0
    bbox_center_y = (y1 + y2) / 2.0

    image_center_x = image_width / 2.0
    image_center_y = image_height / 2.0

    # Sai số ảnh (pixel) so với tâm ảnh.
    ex = bbox_center_x - image_center_x
    ey = bbox_center_y - image_center_y

    # Bề rộng/cao mặt đất mà khung hình bao phủ ở độ cao current_alt.
    ground_width_m = 2.0 * current_alt * math.tan(camera_fov_x / 2.0)
    ground_height_m = 2.0 * current_alt * math.tan(camera_fov_y / 2.0)

    # Offset mét trong frame camera (x: phải, y: xuống dưới ảnh).
    offset_x_m = (ex / image_width) * ground_width_m
    offset_y_m = (ey / image_height) * ground_height_m

    # Camera hướng xuống, cạnh trên ảnh = mũi drone:
    #   ảnh đi lên (ey < 0) => phía trước mũi => forward dương.
    body_forward = -offset_y_m
    body_right = offset_x_m

    # Xoay offset body -> NED theo yaw drone.
    yaw_rad = math.radians(yaw_deg)
    dN = body_forward * math.cos(yaw_rad) - body_right * math.sin(yaw_rad)
    dE = body_forward * math.sin(yaw_rad) + body_right * math.cos(yaw_rad)

    # NED (m) -> lat/lon.
    delta_lat = dN / 111320.0
    delta_lon = dE / (111320.0 * math.cos(math.radians(current_lat)))
    sos_lat = current_lat + delta_lat
    sos_lon = current_lon + delta_lon
    return sos_lat, sos_lon


def estimate_target_offset_from_camera(bbox, altitude_agl, camera_intrinsics, yaw_rad,
                                       current_lat=None, current_lon=None, current_alt=None,
                                       use_bottom_center=False):
    """Ước lượng offset + GPS của nạn nhân từ bbox với camera hướng xuống.

    camera_intrinsics = (fx, fy, cx, cy). Trả về dict hoặc None nếu thiếu dữ liệu.
    """
    if bbox is None or len(bbox) != 4 or altitude_agl is None or altitude_agl <= 0:
        return None
    fx, fy, cx, cy = camera_intrinsics
    x1, y1, x2, y2 = [float(v) for v in bbox]
    # B1: điểm đại diện của nạn nhân trong ảnh (mặc định bbox center).
    u = (x1 + x2) / 2.0
    v = y2 if use_bottom_center else (y1 + y2) / 2.0
    # B2: pixel -> offset BODY (mét).
    body_forward, body_right, x_norm, y_norm = pixel_to_body_offset(u, v, altitude_agl, fx, fy, cx, cy)
    # B3: BODY -> NED theo yaw.
    dN, dE = body_to_ned_offset(body_forward, body_right, yaw_rad)
    distance_offset_m = math.sqrt(dN * dN + dE * dE)
    out = {
        "u": u, "v": v, "x_norm": x_norm, "y_norm": y_norm,
        "dN": dN, "dE": dE, "distance_offset_m": distance_offset_m,
        "target_lat": None, "target_lon": None, "target_alt": current_alt,
    }
    # B4: NED -> GPS (nếu có vị trí hiện tại).
    if current_lat is not None and current_lon is not None:
        tlat, tlon = ned_offset_to_gps(current_lat, current_lon, dN, dE)
        out["target_lat"] = tlat
        out["target_lon"] = tlon
    return out


# =====================================================================
# SOS TARGET MANAGER  -  state machine khoá 1 GPS target & thả phao 1 lần
# =====================================================================
class SosTargetManager:
    """Quản lý toàn bộ vòng đời 1 SOS target theo state machine:

      MISSION_NORMAL -> DISTRESS_CONFIRMING -> TARGET_LOCKED
                     -> GOTO_SOS_TARGET -> DROP_PAYLOAD -> POST_DROP

    Mỗi lần phát hiện distress hợp lệ chỉ tạo ĐÚNG 1 GPS target và gửi GOTO 1 lần.
    `drone` là DroneController cung cấp GPS/yaw/altitude/goto/drop.
    """

    MISSION_NORMAL = "MISSION_NORMAL"
    DISTRESS_CONFIRMING = "DISTRESS_CONFIRMING"
    TARGET_LOCKED = "TARGET_LOCKED"
    GOTO_SOS_TARGET = "GOTO_SOS_TARGET"
    DROP_PAYLOAD = "DROP_PAYLOAD"
    POST_DROP = "POST_DROP"

    def __init__(self, drone):
        self.drone = drone
        self.lock = threading.Lock()
        self.state = self.MISSION_NORMAL

        # Buffer bbox center gần nhất (u, v, conf, ts) để lọc nhiễu.
        self.bbox_buffer = deque(maxlen=TARGET_SMOOTHING_FRAMES)
        self.frame_w = IMAGE_WIDTH
        self.frame_h = IMAGE_HEIGHT
        self.confirm_count = 0
        self.confirm_started = 0.0
        self.last_distress_seen = 0.0
        self.last_state_seen = None

        # Latched target (chỉ tạo 1 lần).
        self.sos_target_locked = False
        self.sos_goto_sent = False
        self.drop_done = False
        self.sos_target_lat = None
        self.sos_target_lon = None
        self.sos_target_alt = None
        self.sos_target_state = None
        self.sos_target_created_time = 0.0
        self.last_sos_target = None      # dict debug (dN, dE, ...)
        self.last_drop_time = None
        self.drop_hold_started = 0.0
        self.cooldown_until = 0.0

        self._uart = None

    # ------------------------------------------------------------------
    # API trạng thái (thread-safe đọc cho Flask/UI)
    # ------------------------------------------------------------------
    def get_status(self):
        with self.lock:
            return {
                "state": self.state,
                "locked": self.sos_target_locked,
                "goto_sent": self.sos_goto_sent,
                "drop_done": self.drop_done,
                "target_lat": self.sos_target_lat,
                "target_lon": self.sos_target_lon,
                "target_alt": self.sos_target_alt,
                "target_state": self.sos_target_state,
                "debug": dict(self.last_sos_target) if self.last_sos_target else None,
            }

    def is_executing(self):
        """True khi đã KHOÁ 1 GPS target và đang thực thi (bay tới / thả / hậu thả).

        Lúc này SosTargetManager đang chiếm quyền điều khiển vehicle, nên vòng lặp
        detection KHÔNG được giữ vị trí (pause/hold) để tránh đánh nhau với GOTO.
        DISTRESS_CONFIRMING không tính là executing: lúc đó drone vẫn nên giữ vị
        trí 2s trong GUIDED (yêu cầu #3).
        """
        with self.lock:
            return self.state in (
                self.TARGET_LOCKED, self.GOTO_SOS_TARGET,
                self.DROP_PAYLOAD, self.POST_DROP,
            )

    def reset(self, reason="manual_reset"):
        """Reset hoàn toàn về MISSION_NORMAL (dùng khi stop detection / reset thủ công)."""
        with self.lock:
            self.state = self.MISSION_NORMAL
            self.bbox_buffer.clear()
            self.confirm_count = 0
            self.confirm_started = 0.0
            self.last_distress_seen = 0.0
            self.last_state_seen = None
            self.sos_target_locked = False
            self.sos_goto_sent = False
            self.drop_done = False
            self.sos_target_lat = None
            self.sos_target_lon = None
            self.sos_target_alt = None
            self.sos_target_state = None
            self.sos_target_created_time = 0.0
            self.last_sos_target = None
            self.drop_hold_started = 0.0
        if reason:
            print(f"[SOS RESET] reason={reason}")

    # ------------------------------------------------------------------
    # 1) Nhận detection mỗi frame (gọi từ vision thread)
    # ------------------------------------------------------------------
    def update_detection(self, state, bbox, conf, frame_w, frame_h, now):
        """Đưa 1 detection distress vào buffer xác nhận. KHÔNG bay ở đây."""
        if not ENABLE_SOS_GOTO:
            return
        if state not in DISTRESS_STATES:
            return
        if conf is not None and conf < DISTRESS_MIN_CONF:
            return
        if not is_valid_bbox(bbox, frame_w, frame_h):
            return

        with self.lock:
            self.frame_w = int(frame_w)
            self.frame_h = int(frame_h)
            self.last_distress_seen = now
            self.last_state_seen = state
            print(f"[DETECT DISTRESS] state={state} conf={(conf if conf is not None else 1.0):.2f} "
                  f"bbox=({int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])})")

            # Đã khoá target hoặc đang trong cooldown -> không gom thêm.
            if self.sos_target_locked or self.state not in (self.MISSION_NORMAL, self.DISTRESS_CONFIRMING):
                return
            if now < self.cooldown_until:
                return

            # Bắt đầu xác nhận.
            if self.state == self.MISSION_NORMAL:
                self.state = self.DISTRESS_CONFIRMING
                self.bbox_buffer.clear()
                self.confirm_count = 0
                self.confirm_started = now

            u = (float(bbox[0]) + float(bbox[2])) / 2.0
            v = (float(bbox[1]) + float(bbox[3])) / 2.0
            self.bbox_buffer.append((u, v, float(conf if conf is not None else 1.0), now))
            self.confirm_count += 1
            print(f"[DISTRESS CONFIRMING] frames={self.confirm_count}/{DISTRESS_CONFIRM_FRAMES}")

    # ------------------------------------------------------------------
    # 2) Bộ điều khiển state machine (gọi mỗi vòng lặp, kể cả khi mất detect)
    # ------------------------------------------------------------------
    def tick(self, now):
        if not ENABLE_SOS_GOTO:
            return
        with self.lock:
            st = self.state
        if st == self.DISTRESS_CONFIRMING:
            self._tick_confirming(now)
        elif st == self.TARGET_LOCKED:
            self._tick_locked(now)
        elif st == self.GOTO_SOS_TARGET:
            self._tick_goto(now)
        elif st == self.DROP_PAYLOAD:
            self._tick_drop(now)
        elif st == self.POST_DROP:
            self._tick_post_drop(now)
        # TTL: target đã khoá nhưng quá lâu -> bỏ và resume mission.
        self._check_target_timeout(now)

    # ----- DISTRESS_CONFIRMING -----
    def _tick_confirming(self, now):
        with self.lock:
            # Mất detect trước khi confirm xong -> huỷ confirm.
            if (now - self.last_distress_seen) > DISTRESS_LOST_RESET_SEC:
                print("[DISTRESS CONFIRMING] lost detection -> back to MISSION_NORMAL")
                self.state = self.MISSION_NORMAL
                self.bbox_buffer.clear()
                self.confirm_count = 0
                return
            enough_frames = self.confirm_count >= DISTRESS_CONFIRM_FRAMES
            enough_time = (now - self.confirm_started) >= DISTRESS_CONFIRM_TIME and self.confirm_count >= 2
            if not (enough_frames or enough_time):
                return

            # ----- YÊU CẦU #1/#10: chỉ KHOÁ target khi đủ điều kiện SOS -----
            # 1) AI báo đúng trạng thái "SOS", HOẶC
            # 2) distress (FROZEN/SOS) đã được giữ liên tục đủ lâu để escalate
            #    thành SOS (giải quyết camera hướng xuống hiếm khi ra SOS thật).
            confirmed_state = self.last_state_seen
            distress_duration = now - self.confirm_started
            qualifies_sos = (confirmed_state == SOS_LOCK_STATE) or (
                ESCALATE_FROZEN_TO_SOS and distress_duration >= SOS_FROZEN_ESCALATE_SEC
            )

        if not qualifies_sos:
            # Đã xác nhận CÓ người distress nhưng CHƯA đủ điều kiện SOS -> chưa tạo
            # GPS target. Drone vẫn đang giữ vị trí trong GUIDED qua cơ chế pause.
            return

        # Đủ điều kiện SOS -> ước lượng + khoá MỘT GPS target duy nhất.
        target = self.estimate_gps_target(now)
        if target is None:
            # Chưa đủ dữ liệu drone (GPS/alt/yaw) -> tiếp tục gom thêm frame.
            return
        # Đánh dấu rõ đây là sự kiện SOS (kể cả khi escalate từ FROZEN) cho log/telemetry.
        target["state"] = "SOS"
        if confirmed_state != SOS_LOCK_STATE:
            print(f"[SOS ESCALATE] FROZEN giữ liên tục {distress_duration:.1f}s "
                  f">= {SOS_FROZEN_ESCALATE_SEC:.1f}s -> coi là SOS")
        self.lock_target(target, now)

    # ----- Ước lượng GPS từ bbox trung bình -----
    def estimate_gps_target(self, now):
        with self.lock:
            if not self.bbox_buffer:
                return None
            sw = sum(c for (_, _, c, _) in self.bbox_buffer)
            if sw <= 0:
                return None
            u_avg = sum(u * c for (u, _, c, _) in self.bbox_buffer) / sw
            v_avg = sum(v * c for (_, v, c, _) in self.bbox_buffer) / sw
            state = self.last_state_seen

        drone = self.drone
        # Điều kiện drone hợp lệ.
        if not drone.is_vehicle_ready_for_sos_goto(require_guided=False):
            return None
        alt_agl = drone.get_altitude_agl()
        yaw = drone.get_vehicle_yaw_rad()
        base = drone._get_current_latlon()
        if alt_agl is None or alt_agl <= MIN_VALID_HEIGHT_M or yaw is None or base is None:
            return None
        base_lat, base_lon, base_alt = base

        body_forward, body_right, x_norm, y_norm = pixel_to_body_offset(u_avg, v_avg, alt_agl)
        dN, dE = body_to_ned_offset(body_forward, body_right, yaw)
        distance_offset_m = math.sqrt(dN * dN + dE * dE)
        # Offset quá lớn -> nhiều khả năng sai, không khoá.
        if distance_offset_m > MAX_VALID_OFFSET_M:
            print(f"[GPS ESTIMATE] rejected offset={distance_offset_m:.1f}m > {MAX_VALID_OFFSET_M:.1f}m")
            return None
        tlat, tlon = ned_offset_to_gps(base_lat, base_lon, dN, dE)
        talt = base_alt if base_alt is not None else float(drone.takeoff_height)

        print(f"[GPS ESTIMATE] u={u_avg:.1f} v={v_avg:.1f} alt_agl={alt_agl:.2f} "
              f"x_norm={x_norm:.3f} y_norm={y_norm:.3f} dN={dN:.2f} dE={dE:.2f} "
              f"lat={tlat:.7f} lon={tlon:.7f}")
        return {
            "state": state, "lat": tlat, "lon": tlon, "alt": talt,
            "dN": dN, "dE": dE, "u": u_avg, "v": v_avg,
            "alt_agl": alt_agl, "distance_offset_m": distance_offset_m,
        }

    # ----- Khoá target (chỉ 1 lần) -----
    def lock_target(self, target, now):
        with self.lock:
            if self.sos_target_locked:
                return
            self.sos_target_locked = True
            self.sos_target_lat = target["lat"]
            self.sos_target_lon = target["lon"]
            self.sos_target_alt = target["alt"]
            self.sos_target_state = target["state"]
            self.sos_target_created_time = now
            self.last_sos_target = dict(target)
            self.state = self.TARGET_LOCKED
        print(f"[GPS TARGET LOCKED] state={target['state']} "
              f"px={target['u']:.1f} py={target['v']:.1f} "
              f"dN={target['dN']:.2f} dE={target['dE']:.2f} "
              f"lat={target['lat']:.7f} lon={target['lon']:.7f}")

    # ----- TARGET_LOCKED: gửi goto đúng 1 lần -----
    def _tick_locked(self, now):
        self.send_goto_once()
        with self.lock:
            if self.sos_goto_sent:
                self.state = self.GOTO_SOS_TARGET

    def send_goto_once(self):
        with self.lock:
            if self.sos_goto_sent:
                print("[SOS GOTO] skipped because target already locked/sent")
                return
            lat, lon, alt = self.sos_target_lat, self.sos_target_lon, self.sos_target_alt
            state = self.sos_target_state
        drone = self.drone
        if not drone.is_vehicle_ready_for_sos_goto(require_guided=False):
            print("[SOS GOTO] aborted reason=vehicle_not_ready")
            return
        if lat is None or lon is None or alt is None:
            print("[SOS GOTO] aborted reason=invalid_target")
            return
        # Lưu lại mission waypoint hiện tại trước khi rời đi để có thể resume.
        drone._begin_sos_divert()
        ok = drone.command_goto_gps(lat, lon, alt, SOS_GOTO_SPEED)
        if not ok:
            print("[SOS GOTO] aborted reason=command_failed")
            return
        with self.lock:
            self.sos_goto_sent = True
        print(f"[SOS GOTO] command sent reason={state} lat={lat:.7f} lon={lon:.7f} "
              f"alt={alt:.2f} speed={SOS_GOTO_SPEED:.2f}")

    # ----- GOTO_SOS_TARGET: chỉ monitor khoảng cách -----
    def _tick_goto(self, now):
        if self.check_arrival(now):
            with self.lock:
                self.state = self.DROP_PAYLOAD
                self.drop_hold_started = now

    def check_arrival(self, now):
        base = self.drone._get_current_latlon()
        with self.lock:
            lat, lon = self.sos_target_lat, self.sos_target_lon
        if base is None or lat is None or lon is None:
            return False
        dist = get_distance_metres(base[0], base[1], lat, lon)
        if dist <= DROP_RADIUS_M:
            print(f"[SOS ARRIVED] dist={dist:.2f}m")
            # Giữ vị trí: dừng tại điểm SOS.
            self.drone.hold_current_position()
            return True
        return False

    # ----- DROP_PAYLOAD: giữ vị trí, chờ rồi thả phao 1 lần -----
    def _tick_drop(self, now):
        with self.lock:
            started = self.drop_hold_started or now
        # Giữ vị trí trong lúc chờ ổn định trước khi thả.
        self.drone.hold_current_position()
        if (now - started) < DROP_HOLD_TIME:
            return
        self.drop_payload_once(now)
        with self.lock:
            self.state = self.POST_DROP

    def drop_payload_once(self, now=None):
        now = time.time() if now is None else now
        with self.lock:
            if self.drop_done and DROP_ONCE:
                return
            lat, lon, alt = self.sos_target_lat, self.sos_target_lon, self.sos_target_alt
        try:
            if MOCK_MODE:
                print("[DROP] (MOCK) command sent once")
            elif DROP_METHOD == "UART":
                self._drop_via_uart()
            else:  # SERVO
                self.drone._drop_payload_async()
            print("[DROP] command sent once")
        except Exception as e:
            print(f"[DROP] error: {e}")
        with self.lock:
            self.drop_done = True
            self.last_drop_time = now
        if lat is not None and lon is not None:
            print(f"[DROP] payload released at lat={lat:.7f} lon={lon:.7f} "
                  f"alt={(alt if alt is not None else 0.0):.2f}")
        else:
            print("[DROP] payload released")

    def _drop_via_uart(self):
        try:
            import serial
        except Exception as e:
            print(f"[DROP] UART unavailable ({e}); falling back to SERVO")
            self.drone._drop_payload_async()
            return
        try:
            if self._uart is None:
                self._uart = serial.Serial(DROP_UART_PORT, DROP_UART_BAUD, timeout=1)
            self._uart.write(DROP_COMMAND.encode())
            self._uart.flush()
        except Exception as e:
            print(f"[DROP] UART write failed ({e}); falling back to SERVO")
            self.drone._drop_payload_async()

    # ----- POST_DROP: resume mission hoặc hold + cooldown -----
    def _tick_post_drop(self, now):
        self.reset_after_drop(now)

    def reset_after_drop(self, now):
        if RESUME_MISSION_AFTER_DROP:
            self.drone._end_sos_divert_resume(resume=True)
            print("[MISSION RESUME] previous mission continued")
        else:
            # Giữ LOITER/GUIDED tại chỗ: không resume mission.
            self.drone.hold_current_position()
            print("[POST DROP] holding position (resume disabled)")
        with self.lock:
            # Đánh dấu target đã hoàn tất, đặt cooldown để không bắt lại ngay.
            self.cooldown_until = now + POST_DROP_COOLDOWN_SEC
            self.state = self.MISSION_NORMAL
            self.bbox_buffer.clear()
            self.confirm_count = 0
            self.sos_target_locked = False
            self.sos_goto_sent = False
            self.sos_target_lat = None
            self.sos_target_lon = None
            self.sos_target_alt = None
            self.sos_target_state = None
            self.last_state_seen = None
            # drop_done giữ True nếu DROP_ONCE để không thả lại trong cooldown.
            if not DROP_ONCE:
                self.drop_done = False

    # ----- TTL: target khoá quá lâu mà chưa drop -----
    def _check_target_timeout(self, now):
        with self.lock:
            if not self.sos_target_locked or self.drop_done:
                return
            if self.state in (self.DROP_PAYLOAD, self.POST_DROP):
                return
            if (now - self.sos_target_created_time) < SOS_TARGET_TTL:
                return
        print(f"[SOS TARGET] timeout after {SOS_TARGET_TTL:.0f}s -> abort & resume mission")
        if RESUME_MISSION_AFTER_DROP:
            self.drone._end_sos_divert_resume(resume=True)
            print("[MISSION RESUME] previous mission continued")
        else:
            self.drone._end_sos_divert_resume(resume=False)
        with self.lock:
            self.cooldown_until = now + POST_DROP_COOLDOWN_SEC
            self.state = self.MISSION_NORMAL
            self.bbox_buffer.clear()
            self.confirm_count = 0
            self.sos_target_locked = False
            self.sos_goto_sent = False
            self.sos_target_lat = None
            self.sos_target_lon = None
            self.sos_target_alt = None
            self.sos_target_state = None


# ===================== DRONE CONTROLLER =====================
class DroneController:
    def __init__(self, connection_str='/dev/ttyAMA0', takeoff_height=4):
        """Create DroneController and connect to vehicle"""
        self.connection_str = connection_str
        print(f"Connecting to vehicle on {connection_str}")

        try:
            self.vehicle = connect(
                connection_str,
                baud=115200,
                wait_ready=True,
                timeout=60
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

        
        # Person detection. Prefer Hailo pose on AI HAT; keep YOLO/MediaPipe as fallback.
        self.use_hailo_pose = bool(HAILO_POSE_ENABLED)
        self.person_detector = None
        self.person_thread = None
        self.person_running = False
        self._person_stop_event = threading.Event()
        self.detected_persons = []
        self.last_detection_time = 0
        self.detection_interval = 0.05  # seconds between processing steps (loop pacing only)
        self.drowing_detector = DrowningDetector(DROWNING_CONFIG)

        self._person_track = self._new_person_track()
        # Rate-limit server posting (do NOT block detection loop)
        self.server_post_interval = 2.0
        self._last_server_post = 0.0
        self.latest_pose_landmarks = None
        self.server_post_interval = 2.0 
        self._last_server_post = 0.0
        self.latest_pose_source = None
        self.latest_pose_keypoints = None
        self._hailo_tracks = {}
        self._locked_bbox = None
        self._locked_bbox_last_seen = 0.0
        self._bbox_lock_ttl = 0.8
        self._bbox_smooth_alpha = 0.35
        self.detection_fps = 0.0
        self._detection_frame_count = 0
        self._detection_fps_last_time = time.time()
        # ----- Camera intrinsics for downward-camera GPS estimation -----
        # Prefer the loaded calibration matrix; otherwise fall back to image
        # center + a nominal focal length.
        self.cam_fx = CAMERA_FX
        self.cam_fy = CAMERA_FY
        self.cam_cx = CAMERA_CX
        self.cam_cy = CAMERA_CY

        # ----- SOS target state machine (latch 1 GPS target, drop once) -----
        self.sos_manager = SosTargetManager(self)

        # Mission divert / resume bookkeeping (so the SOS detour never loses the
        # original mission waypoint).
        self._mission_resume_target = None
        self._mission_resume_speed = 0.7
        self._mission_resume_active = False
        self._mission_resume_lock = threading.Lock()
        self._sos_divert_active = False
        self._sos_divert_started = 0.0
        self._sos_divert_accum = 0.0

        # UI mirror of the latest GPS-target debug info.
        self.visual_servo_lock = threading.Lock()
        self.visual_servo_last_command = {
            "active": False,
            "ex": 0.0,
            "ey": 0.0,
            "vx": 0.0,
            "vy": 0.0,
        }
        self.camera_center_offset_x = float(os.getenv("CAMERA_CENTER_OFFSET_X", "0"))
        self.camera_center_offset_y = float(os.getenv("CAMERA_CENTER_OFFSET_Y", "200"))
        self.visual_servo_no_person_timeout = float(os.getenv("VISUAL_SERVO_NO_PERSON_TIMEOUT", "5.0"))
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

        self.drop_mode = os.getenv("DROP_MODE", "manual")  # 'auto' | 'manual'
        self.drop_center_aligned = False
        #setup SERVO
        self.servo = 1
        self.dropoff_pwm = 1900
        self.holdon_pwm = 1100
        # Trạng thái servo cho công tắc OPEN/CLOSE thủ công trên UI.
        self.servo_open = False
        print(f"✅ Servo drop configured: ch={self.servo}, drop_pwm={self.dropoff_pwm}, hold_pwm={self.holdon_pwm}")

        #SOS send control
        self._sos_active = False
        self._last_sos_post = 0.0
        self.sos_post_interval = 1
        # Báo SOS + GPS ảo về server đúng MỘT lần cho mỗi target được khoá
        # (yêu cầu #2: không gửi GPS/alert liên tục mỗi frame).
        self._sos_target_announced = False
        
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

    def _new_person_track(self):
        class _PersonTrack:
            pass
        return _PersonTrack()

    def _ensure_person_detector(self):
        if self.person_detector is None:
            self.person_detector = PersonDetector("yolov5n_quant.onnx", IMG_SIZE, 0.45, 0.3, detect_interval=1)
        return self.person_detector

    def _reset_person_detection_state(self):
        """Clear all detection state that can survive a stop/start cycle."""
        self.detected_persons = []
        self.latest_pose_landmarks = None
        self.latest_pose_source = None
        self.latest_pose_keypoints = None
        self.last_detection_time = 0.0
        self.detection_fps = 0.0
        self._detection_frame_count = 0
        self._detection_fps_last_time = time.time()
        self._locked_bbox = None
        self._locked_bbox_last_seen = 0.0
        self._person_track = self._new_person_track()
        self._hailo_tracks = {}
        self._active_state_start = 0.0
        self._active_timeout_exit_printed = False
        self._sos_active = False
        self._last_sos_post = 0.0
        self._sos_target_announced = False

        try:
            if self.person_detector is not None:
                self.person_detector.frame_id = 0
                self.person_detector.tracker.tracks = []
                self.person_detector.tracker.next_id = 1
                self.person_detector.tracker.frame_count = 0
        except Exception:
            pass

        with self.visual_servo_lock:
            self.visual_servo_last_command = {
                "active": False,
                "ex": 0.0,
                "ey": 0.0,
                "vx": 0.0,
                "vy": 0.0,
            }
        # Reset the SOS target state machine and mission-divert state.
        self.sos_manager.reset(reason="detection_restart")
        self._sos_divert_active = False
        self._sos_divert_started = 0.0
        self._sos_divert_accum = 0.0
        self._mission_resume_active = False
        self._mission_resume_target = None


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
        if self.person_thread and self.person_thread.is_alive() and not self._person_stop_event.is_set():
            print("⚠️ Person detection already running")
            return

        self._reset_person_detection_state()
        self._person_stop_event = threading.Event()
        self.person_running = True
        self.person_thread = threading.Thread(
            target=self._person_detection_loop,
            args=(self._person_stop_event,),
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

    def _draw_gps_target_debug(self, frame):
        """Overlay the SOS state machine + locked GPS target on the frame."""
        try:
            cam_cx, cam_cy = int(self.cam_cx), int(self.cam_cy)
            cv2.drawMarker(frame, (cam_cx, cam_cy), (255, 255, 255),
                           cv2.MARKER_CROSS, 18, 2)

            status = self.sos_manager.get_status()
            dbg = status.get("debug")
            lines = [f"SOS STATE: {status.get('state')}"]
            if dbg is not None:
                # Vẽ điểm ảnh đại diện của nạn nhân (bbox center trung bình).
                tcx, tcy = int(dbg.get("u", cam_cx)), int(dbg.get("v", cam_cy))
                cv2.circle(frame, (tcx, tcy), 6, (0, 0, 255), 2)
                cv2.arrowedLine(frame, (cam_cx, cam_cy), (tcx, tcy), (0, 0, 255),
                                1, cv2.LINE_AA, tipLength=0.2)
                lines += [
                    f"dN={dbg.get('dN', 0.0):.2f}m dE={dbg.get('dE', 0.0):.2f}m",
                    f"alt_agl={dbg.get('alt_agl', 0.0):.2f}m",
                ]
            if status.get("target_lat") is not None:
                lines += [
                    f"lat={status.get('target_lat'):.7f}",
                    f"lon={status.get('target_lon'):.7f}",
                ]
            lines += [
                f"locked={status.get('locked')} goto={status.get('goto_sent')} "
                f"drop={status.get('drop_done')}",
            ]
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (10, 90 + i * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        except Exception:
            pass


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

    # ===================================================================
    # ACCESSORS & GPS COMMANDS dùng cho SosTargetManager
    # ===================================================================
    def _get_current_latlon(self):
        """Vị trí hiện tại (lat, lon, alt) từ global_relative_frame, hoặc None."""
        if not self.vehicle:
            return None
        try:
            loc = self.vehicle.location.global_relative_frame
            if loc is None or loc.lat is None or loc.lon is None:
                return None
            alt = float(loc.alt) if loc.alt is not None else None
            return (float(loc.lat), float(loc.lon), alt)
        except Exception:
            return None

    def get_altitude_agl(self):
        """Độ cao AGL (m). Ưu tiên rangefinder/lidar; nếu không có dùng relative
        altitude (độ chính xác thấp hơn)."""
        if MOCK_MODE and DEFAULT_ALTITUDE_AGL is not None:
            return float(DEFAULT_ALTITUDE_AGL)
        if not self.vehicle:
            return DEFAULT_ALTITUDE_AGL
        if USE_RANGEFINDER_ALTITUDE:
            try:
                rng = getattr(self.vehicle, "rangefinder", None)
                if rng is not None and rng.distance is not None and rng.distance > 0:
                    return float(rng.distance)
            except Exception:
                pass
        # Fallback: relative altitude (ít chính xác hơn rangefinder).
        try:
            alt = self.vehicle.location.global_relative_frame.alt
            if alt is not None:
                return float(alt)
        except Exception:
            pass
        return DEFAULT_ALTITUDE_AGL

    def get_vehicle_yaw_rad(self):
        """Yaw drone theo radian (NED), hoặc None nếu không có."""
        if not self.vehicle:
            return None
        try:
            att = self.vehicle.attitude
            if att is None or att.yaw is None:
                return None
            return float(att.yaw)
        except Exception:
            return None

    def is_vehicle_ready_for_sos_goto(self, require_guided=True):
        """Drone có đủ điều kiện an toàn để bay tới SOS target không?

        - Không tự arm, không tự takeoff: chỉ can thiệp khi đang bay.
        - Cần GPS fix tốt, altitude hợp lệ, EKF ổn.
        """
        if MOCK_MODE:
            return True
        if not self.vehicle:
            return False
        try:
            if not self.vehicle.armed:
                return False
            # Phải đang bay (altitude hợp lệ) - không can thiệp khi còn dưới đất.
            alt = self.get_altitude_agl()
            if alt is None or alt <= MIN_VALID_HEIGHT_M:
                return False
            # GPS fix tốt (>= 3D fix).
            gps = getattr(self.vehicle, "gps_0", None)
            if gps is not None and gps.fix_type is not None and gps.fix_type < 3:
                return False
            # EKF / hệ thống an toàn.
            if hasattr(self.vehicle, "ekf_ok") and self.vehicle.ekf_ok is False:
                return False
            if require_guided and self.vehicle.mode.name != "GUIDED":
                return False
        except Exception:
            return False
        return True

    def command_goto_gps(self, lat, lon, alt, speed):
        """Chuyển GUIDED nếu cần rồi gửi simple_goto đúng 1 lần. Trả True nếu OK."""
        if MOCK_MODE:
            print(f"[MOCK] command_goto_gps lat={lat:.7f} lon={lon:.7f} alt={alt:.2f}")
            return True
        if not self.vehicle:
            return False
        try:
            if self.vehicle.mode.name != "GUIDED":
                self.vehicle.mode = VehicleMode("GUIDED")
            self.set_speed(speed)
            self.vehicle.simple_goto(
                LocationGlobalRelative(float(lat), float(lon), float(alt)),
                groundspeed=float(speed)
            )
            return True
        except Exception as e:
            print(f"command_goto_gps error: {e}")
            return False

    def hold_current_position(self):
        """Giữ vị trí hiện tại (dừng drone). Không spam: gửi 1 lệnh hold mỗi lần gọi."""
        if MOCK_MODE or not self.vehicle:
            return
        try:
            if self.vehicle.mode.name != "GUIDED" or not self.vehicle.armed:
                return
            loc = self.vehicle.location.global_relative_frame
            if loc and (loc.lat is not None) and (loc.lon is not None):
                alt = float(loc.alt) if (loc.alt is not None) else float(self.takeoff_height)
                hold = LocationGlobalRelative(float(loc.lat), float(loc.lon), alt)
                self.vehicle.simple_goto(hold, groundspeed=0.0)
        except Exception:
            pass

    # -------- Mission divert / resume (lưu waypoint mission khi đi cứu SOS) -----
    def _begin_sos_divert(self):
        """Lưu lại mission waypoint hiện tại trước khi rời sang SOS target."""
        with self._mission_resume_lock:
            if self._sos_divert_active:
                return
            if self._current_goto_target is not None:
                self._mission_resume_target = self._current_goto_target
                self._mission_resume_speed = float(self._current_goto_speed or 0.7)
                self._mission_resume_active = True
                try:
                    print(
                        f"[SOS DIVERT] mission waypoint saved "
                        f"lat={self._mission_resume_target.lat:.7f} "
                        f"lon={self._mission_resume_target.lon:.7f} "
                        f"alt={getattr(self._mission_resume_target, 'alt', 0.0):.2f} "
                        f"speed={self._mission_resume_speed:.2f}"
                    )
                except Exception:
                    pass
            self._sos_divert_active = True
            self._sos_divert_started = time.time()

    def _end_sos_divert_resume(self, resume=True):
        """Kết thúc divert. Nếu resume=True thì re-issue lại mission waypoint cũ."""
        self._sos_divert_active = False
        if self._sos_divert_started > 0:
            self._sos_divert_accum += time.time() - self._sos_divert_started
            self._sos_divert_started = 0.0
        if not resume:
            return
        with self._mission_resume_lock:
            resume_target = self._mission_resume_target if self._mission_resume_active else None
            resume_speed = float(self._mission_resume_speed or 0.7)
        if resume_target is None:
            print("[MISSION RESUME] skipped reason=no_saved_mission_target")
            return
        if MOCK_MODE:
            return
        try:
            armed = bool(self.vehicle and self.vehicle.armed)
            mode = self.vehicle.mode.name if self.vehicle else ""
        except Exception:
            armed, mode = False, ""
        if not (armed and mode == "GUIDED"):
            print("[MISSION RESUME] skipped reason=not_guided_or_not_armed")
            return
        try:
            self.set_speed(resume_speed)
            self.vehicle.simple_goto(resume_target, groundspeed=resume_speed)
        except Exception as e:
            print(f"[MISSION RESUME] error: {e}")

    def _force_exit_visual_servo_to_mission(self, now, reason="no_person_timeout"):
        """Non-distress exit: clear any pause so the GPS mission resumes."""
        self.clear_pause()

    def _direction_to_vxvy(self, direction: str, speed: float):
        d = (direction or "").lower()
        if d == "forward":
            return float(speed), 0.0        
        if d == "backward":
            return -float(speed), 0.0
        if d == "left":
            return 0.0, -float(speed)
        if d == "right":
            return 0.0, float(speed)
        return 0.0, 0.0

    def _is_frame_center_inside_bbox(self, bbox, frame_w, frame_h, margin_px=0.0):
        if bbox is None or frame_w <= 0 or frame_h <= 0:
            return False
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx = float(frame_w) * 0.5 + self.camera_center_offset_x
        cy = float(frame_h) * 0.5 + self.camera_center_offset_y
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
                # if not self.vehicle or not self.vehicle.armed:
                #     print("⚠️ DROP skipped: vehicle not available or not armed")
                #     return

                print("🎯 DROP condition met: SOS + frame center inside bbox")
                self.controlServo(self.servo, self.dropoff_pwm)
                time.sleep(1.5)
                self.controlServo(self.servo, self.holdon_pwm)
                ok = True
                print("✅ DROP complete: payload released via servo")
            except Exception as e:
                print(f"❌ DROP failed: {e}")
            finally:
                with self._drop_lock:
                    self._drop_completed = bool(ok)
                    if ok:
                        self._last_drop_time = time.time()
                    self._drop_in_progress = False
                if ok:
                    # Resume (if any) is owned by the SosTargetManager / goto loop,
                    # so just drop the pause flag without re-issuing a waypoint here.
                    self.clear_pause(resume=False)

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

    def _announce_sos_target_once(self, now):
        """Gửi SOS + GPS ảo về server đúng MỘT lần khi target vừa được khoá.

        Latch theo trạng thái `locked` của SosTargetManager nên không bao giờ spam
        mỗi frame (yêu cầu #2). Tự reset khi target được clear (sau drop/timeout).
        """
        status = self.sos_manager.get_status()
        if not status.get("locked"):
            # Không còn target SOS nào đang active -> reset cờ báo & cờ telemetry.
            self._sos_target_announced = False
            self._sos_active = False
            return
        # Đang có 1 SOS target active (telemetry sos_active = True).
        self._sos_active = True
        if self._sos_target_announced:
            return
        self._sos_target_announced = True
        if not self.vehicle:
            return
        try:
            gf = self.vehicle.location.global_frame
            gr = self.vehicle.location.global_relative_frame
            lat = gf.lat if (gf and gf.lat is not None) else None
            lon = gf.lon if (gf and gf.lon is not None) else None
            alt = gr.alt if (gr and gr.alt is not None) else (gf.alt if (gf and hasattr(gf, 'alt')) else 0.0)
            if lat is not None and lon is not None:
                self._last_sos_post = now
                threading.Thread(
                    target=self.send_person_detection_to_server,
                    args=(float(lat), float(lon), float(alt), list(self.detected_persons or [])),
                    daemon=True
                ).start()
                print("🚨 SOS CONFIRMED! Sending alert + virtual GPS target to server...")
        except Exception as e:
            print(f"Error sending SOS alert: {e}")

    def _handle_person_hold_and_sos(self, person_now, now):
        """Điều phối hold-2s (yêu cầu #3) và nhường quyền cho SOS divert (yêu cầu #4).

        - Có người mà SosTargetManager CHƯA khoá target (gồm cả lúc đang xác nhận
          distress): giữ vị trí trong GUIDED (không đổi sang POSHOLD thật). Cơ chế
          pause dùng pause_min_hold_sec = 2.0s đúng yêu cầu "sau 2 giây đứng yên".
        - Khi target đã khoá & đang bay tới SOS: clear pause để GOTO không bị
          lệnh hold đánh nhau.
        - Báo SOS đúng 1 lần khi target được khoá.
        """
        manager_owns = self.sos_manager.is_executing()
        if person_now and not manager_owns and self.pause_enable and not self._drop_completed:
            try:
                if self.vehicle and self.vehicle.armed and self.vehicle.mode.name == "GUIDED":
                    # Giữ vị trí trong GUIDED (poshold-like). Việc "đứng yên sau 2s"
                    # do goto()/_update_pause_state điều khiển qua pause_min_hold_sec.
                    self.request_pause("person_detected")
            except Exception:
                pass
        elif manager_owns and self.is_pause_requested():
            # SOS target đã khoá -> nhường quyền cho SosTargetManager bay tới GPS ảo.
            self.clear_pause(resume=False)

        self._announce_sos_target_once(now)

    def _get_hailo_track(self, track_id, now):
        try:
            track_id = int(track_id)
        except Exception:
            track_id = 1
        track = self._hailo_tracks.get(track_id)
        if track is None:
            track = self._new_person_track()
            track.id = track_id
            self._hailo_tracks[track_id] = track
        track._last_seen = float(now)
        return track

    def _prune_hailo_tracks(self, now):
        stale_ids = [
            tid for tid, track in self._hailo_tracks.items()
            if (now - float(getattr(track, "_last_seen", 0.0) or 0.0)) > 2.0
        ]
        for tid in stale_ids:
            self._hailo_tracks.pop(tid, None)

    def _person_detection_loop_hailo(self, stop_event=None):
        """Hailo AI HAT pose loop; applies existing drowning/mission logic."""
        global _hailo_pose_runtime
        stop_event = stop_event or self._person_stop_event
        if _hailo_pose_runtime is None:
            start_camera()

        while not stop_event.is_set():
            try:
                current_time = time.time()
                if current_time - self.last_detection_time < self.detection_interval:
                    time.sleep(0.01)
                    continue

                if _hailo_pose_runtime is None or not _hailo_pose_runtime.is_active():
                    print("⚠️ Hailo pose runtime is not active; switching to MediaPipe fallback")
                    self.use_hailo_pose = False
                    stop_camera()
                    start_camera()
                    return self._person_detection_loop(stop_event)

                raw_dets, _, frame_shape, _ = _hailo_pose_runtime.get_latest()
                if frame_shape is None:
                    time.sleep(0.01)
                    continue

                h, w = int(frame_shape[0]), int(frame_shape[1])
                detections = []
                states = {"ACTIVE": 0, "FROZEN": 0, "SOS": 0}

                for raw in raw_dets:
                    bbox = raw.get("bbox")
                    pose_landmarks = raw.get("pose_landmarks")
                    if bbox is None or pose_landmarks is None:
                        continue

                    track = self._get_hailo_track(raw.get("id", 1), current_time)
                    results = self.drowing_detector.detect(track, bbox, pose_landmarks, (h, w))
                    if results["state"] == "SOS":
                        results["sos_duration"] = self.drowing_detector.get_sos_duration(track)

                    det = {
                        "id": int(getattr(track, "id", raw.get("id", 1))),
                        "bbox": [int(v) for v in bbox],
                        "confidence": float(raw.get("confidence", 0.0)),
                        "track_obj": track,
                        "source": "hailo_pose",
                        "pose_landmarks": pose_landmarks,
                        "pose_keypoints": raw.get("pose_keypoints"),
                        "drowning_state": results,
                    }
                    detections.append(det)
                    states[results["state"]] = states.get(results["state"], 0) + 1

                detections.sort(key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
                if detections:
                    first = detections[0]
                    self.latest_pose_landmarks = first.get("pose_landmarks")
                    self.latest_pose_keypoints = first.get("pose_keypoints")
                    self.latest_pose_source = "hailo"
                    # FROZEN/SOS -> đưa detection vào state machine SosTargetManager
                    # (xác nhận nhiều frame -> khoá 1 GPS target -> bay -> drop).
                    distress_det = next(
                        (d for d in detections
                         if d.get("drowning_state", {}).get("state") in DISTRESS_STATES),
                        None
                    )
                    if distress_det is not None:
                        dstate = distress_det["drowning_state"]["state"]
                        self.sos_manager.update_detection(
                            dstate, distress_det["bbox"],
                            float(distress_det.get("confidence", 1.0)),
                            w, h, current_time
                        )
                else:
                    self.latest_pose_landmarks = None
                    self.latest_pose_keypoints = None
                    self.latest_pose_source = None

                # Chạy state machine SOS mỗi vòng lặp (kể cả khi mất detect).
                self.sos_manager.tick(current_time)

                self._prune_hailo_tracks(current_time)
                self.detected_persons = detections
                self.last_detection_time = current_time
                self._update_detection_fps(current_time)

                person_now = len(detections) > 0
                self._rearm_drop_if_clear(current_time, person_now)
                # Bất kỳ người nào (ACTIVE/FROZEN) đều giữ vị trí 2s trong GUIDED;
                # chỉ khi SosTargetManager đã KHOÁ 1 SOS target mới nhường quyền cho
                # nó bay tới GPS ảo. Chỉ SOS mới tạo GPS (yêu cầu #1/#3/#4).
                self._handle_person_hold_and_sos(person_now, current_time)

                sos_dets = [d for d in detections if d.get("drowning_state", {}).get("state") == "SOS"]
                sos_now = len(sos_dets) > 0

                if person_now and not sos_now:
                    first_det_state = detections[0].get("drowning_state", {}).get("state", "ACTIVE")
                    if first_det_state == "ACTIVE":
                        if self._active_state_start <= 0.0:
                            self._active_state_start = current_time
                        active_dur = current_time - self._active_state_start
                        if active_dur >= self._active_timeout_sec and not self._active_timeout_exit_printed:
                            print(f"[ACTIVE TIMEOUT] Person ACTIVE {active_dur:.1f}s -> resume mission (non-distress)")
                            self._active_timeout_exit_printed = True
                            self._pause_block_until = current_time + self._active_exit_cooldown_sec
                            self._active_state_start = 0.0
                            self._force_exit_visual_servo_to_mission(current_time, reason="active_state_timeout")
                    else:
                        self._active_state_start = 0.0
                        self._active_timeout_exit_printed = False
                else:
                    self._active_state_start = 0.0
                    if not person_now:
                        self._active_timeout_exit_printed = False

                self._update_pause_state(person_now=person_now, sos_now=sos_now, now=current_time)
                # Báo SOS + GPS ảo về server đã được xử lý đúng-1-lần trong
                # _handle_person_hold_and_sos() -> _announce_sos_target_once().
            except Exception as e:
                print(f"Hailo person detection error: {e}")
                time.sleep(0.1)


    def _person_detection_loop(self, stop_event=None):
        """Person + Drowning detection loop"""
        if self.use_hailo_pose:
            return self._person_detection_loop_hailo(stop_event)

        stop_event = stop_event or self._person_stop_event
        while not stop_event.is_set():
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
                if cv_image is None:
                    time.sleep(0.01)
                    continue

                h, w = cv_image.shape[:2]

                rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                pose_res = pose.process(rgb)
                if stop_event.is_set():
                    break
                if pose_res.pose_landmarks:
                    self.latest_pose_landmarks = pose_res.pose_landmarks
                    self.latest_pose_source = "mediapipe"
                    self.latest_pose_keypoints = None
                else:
                    self.latest_pose_landmarks = None
                    self.latest_pose_source = None
                    self.latest_pose_keypoints = None
                
                # Prefer MediaPipe Pose for the displayed/tracked bbox. It is much
                # cheaper and more stable after a person is present than repeatedly
                # switching to fresh YOLO/SORT boxes.
                pose_bbox = None
                if pose_res.pose_landmarks:
                    pose_bbox = self._bbox_from_pose_landmarks(pose_res.pose_landmarks, w, h)

                chosen_bbox, chosen_source = self._smooth_and_lock_bbox(pose_bbox, current_time)

                if chosen_bbox is None:
                    self._ensure_person_detector()
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
                    # Phát hiện đuối nước
                    results = self.drowing_detector.detect(
                        track, bbox, pose_res.pose_landmarks, (h, w)
                    )

                    # Lấy thời gian SOS nếu có
                    if results["state"] == "SOS":
                        results["sos_duration"] = self.drowing_detector.get_sos_duration(track)

                    # FROZEN/SOS -> đưa vào state machine SosTargetManager
                    # (xác nhận nhiều frame -> khoá 1 GPS target -> bay -> drop).
                    if results["state"] in DISTRESS_STATES:
                        self.sos_manager.update_detection(
                            results["state"], bbox,
                            float(det.get("confidence", 1.0)),
                            w, h, current_time
                        )

                    # Thêm drowning_state vào detection
                    det["drowning_state"] = results

                    # Cập nhật thống kê
                    states[results['state']] = states.get(results['state'], 0) + 1

                    # Vẽ kết quả lên frame
                    color = self.draw_results(cv_image, bbox, results)

                    # Vẽ skeleton
                    if pose_res.pose_landmarks:
                        self.draw_skeleton(cv_image, pose_res.pose_landmarks, color)

                # Chạy state machine SOS mỗi vòng lặp (kể cả khi mất detect).
                self.sos_manager.tick(current_time)

                # Hiển thị thống kê
                cv2.putText(cv_image,
                   f"Active: {states.get('ACTIVE',0)} | Frozen: {states.get('FROZEN',0)} | SOS: {states.get('SOS',0)}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # GPS-target debug overlay
                self._draw_gps_target_debug(cv_image)

                try:
                    cv2.imshow("DROWNING DETECTION", cv_image)
                    cv2.waitKey(1)
                except:
                    pass

                self.detected_persons = detections[:1] if detections else []
                self.last_detection_time = current_time
                self._update_detection_fps(current_time)
                # ===== Mission pause trigger (person detected) =====
                person_now = len(detections) > 0
                self._rearm_drop_if_clear(current_time, person_now)
                # Bất kỳ người nào (ACTIVE/FROZEN) đều giữ vị trí 2s trong GUIDED;
                # chỉ khi SosTargetManager đã KHOÁ 1 SOS target mới nhường quyền cho
                # nó bay tới GPS ảo. Chỉ SOS mới tạo GPS (yêu cầu #1/#3/#4).
                self._handle_person_hold_and_sos(person_now, current_time)

                sos_dets = [d for d in detections if d.get("drowning_state", {}).get("state") == "SOS"]
                sos_now = len(sos_dets) > 0

                # ACTIVE state timeout: non-distress person (ACTIVE, not SOS/FROZEN) for
                # too long → resume mission with cooldown on re-pause.
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
                # Báo SOS + GPS ảo về server đã được xử lý đúng-1-lần trong
                # _handle_person_hold_and_sos() -> _announce_sos_target_once().
            except Exception as e:
                print(f"Person detection error: {e}")
                time.sleep(0.1)

    def _sanitize_detections(self, detections):
        """Make detections JSON-serializable (remove track_obj, cast types)."""
        out = []
        for d in (detections or []):
            if not isinstance(d, dict):
                continue
            dd = {k: v for k, v in d.items() if k not in ('track_obj', 'pose_landmarks')}
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

            # Đính kèm GPS target đã khoá của nạn nhân SOS/FROZEN (nếu có).
            status = self.sos_manager.get_status()
            if status.get('target_lat') is not None:
                dbg = status.get('debug') or {}
                person_data.update({
                    'target_lat': status.get('target_lat'),
                    'target_lon': status.get('target_lon'),
                    'target_alt': status.get('target_alt'),
                    'target_offset_north_m': dbg.get('dN'),
                    'target_offset_east_m': dbg.get('dE'),
                    'target_height_m': dbg.get('alt_agl'),
                    'target_state': status.get('target_state'),
                })

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
        self._person_stop_event.set()
        if self.person_thread:
            self.person_thread.join(timeout=2.0)
            if self.person_thread.is_alive():
                print("⚠️ Person detection thread did not stop within timeout")
            else:
                self.person_thread = None
        self._reset_person_detection_state()
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
    def clear_pause(self, resume=True):
        with self._pause_lock:
            if self._pause_requested:
                self._pause_requested = False
                self._pause_reason = None
                self._pause_started = None
                self._last_pause_cleared = time.time()
                print("Mission RESUME (no SOS)")
        # When resume=False (e.g. clearing a hold to fly to an SOS GPS target),
        # do NOT re-issue the mission waypoint here. self._current_goto_target is
        # always the original mission waypoint, never the SOS target.
        if not resume:
            return
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
    
    def reset_sos_target(self, reason="manual_reset"):
        """Reset thủ công SOS target (cho UI/Flask) và resume mission nếu đang divert."""
        if self._sos_divert_active:
            self._end_sos_divert_resume(resume=RESUME_MISSION_AFTER_DROP)
        self.sos_manager.reset(reason=reason)

    def get_sos_status(self):
        """Trạng thái state machine SOS cho UI/Flask."""
        return self.sos_manager.get_status()

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
            "visual_servo": dict(self.visual_servo_last_command),
            "sos_target": self.sos_manager.get_status(),
        }

    def _hold_position_step(self):
        if not self.vehicle:
            return
        # Never fight the SOS goto: while diverted to an SOS GPS target the
        # SosTargetManager owns the vehicle, so don't send a hold command here.
        if self._sos_divert_active:
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

    # -------- State accessors for GPS target estimation --------
    def _get_height_agl(self):
        """Height above ground in meters. Prefer rangefinder, fall back to alt."""
        if not self.vehicle:
            return None
        try:
            rng = getattr(self.vehicle, "rangefinder", None)
            if rng is not None and rng.distance is not None and rng.distance > 0:
                return float(rng.distance)
        except Exception:
            pass
        try:
            alt = self.vehicle.location.global_relative_frame.alt
            return float(alt) if alt is not None else None
        except Exception:
            return None

    def _get_current_ned(self):
        """Local position [north, east, down] in meters (relative to EKF origin)."""
        if not self.vehicle:
            return None
        try:
            lf = self.vehicle.location.local_frame
            if lf is None or lf.north is None or lf.east is None or lf.down is None:
                return None
            return [float(lf.north), float(lf.east), float(lf.down)]
        except Exception:
            return None

    def _get_current_vel_ned(self):
        """Local velocity [vn, ve, vd] in m/s."""
        if not self.vehicle:
            return None
        try:
            v = self.vehicle.velocity
            if v is None or any(c is None for c in v):
                return None
            return [float(v[0]), float(v[1]), float(v[2])]
        except Exception:
            return None

    def _get_yaw(self):
        """Vehicle yaw in radians (NED), or None if unavailable."""
        if not self.vehicle:
            return None
        try:
            att = self.vehicle.attitude
            if att is None or att.yaw is None:
                return None
            return float(att.yaw)
        except Exception:
            return None

    def _control_ready(self):
        """True when it is safe to stream pos+vel setpoints."""
        if not self.vehicle:
            return False
        try:
            if not self.vehicle.armed:
                return False
            if self.vehicle.mode.name != "GUIDED":
                return False
        except Exception:
            return False
        if self._get_current_ned() is None:
            return False
        if self._get_yaw() is None:
            return False
        return True

    def controlServo(self, servo_number,pwm_value):
        msg = self.vehicle.message_factory.command_long_encode(
            0,
            0,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0,
            servo_number,
            pwm_value,
            0,
            0,
            0,
            0,
            0)
        self.vehicle.send_mavlink(msg)

    def set_servo_position(self, open_servo: bool):
        """Công tắc OPEN/CLOSE servo thủ công từ UI.

        open_servo=True  -> đẩy servo về dropoff_pwm (mở/nhả phao).
        open_servo=False -> đẩy servo về holdon_pwm (giữ/đóng phao).
        Đây là điều khiển TAY, độc lập với logic thả tự động theo SOS.
        """
        if not self.vehicle:
            return False
        pwm = self.dropoff_pwm if open_servo else self.holdon_pwm
        try:
            self.controlServo(self.servo, pwm)
            self.vehicle.flush()
            self.servo_open = bool(open_servo)
            print(f"🔧 Manual servo {'OPEN' if open_servo else 'CLOSE'} (ch={self.servo}, pwm={pwm})")
            return True
        except Exception as e:
            print(f"Servo control error: {e}")
            return False

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

        # self.vehicle.armed = True
        while not self.vehicle.armed:
            print('Arming...')
            time.sleep(3)
            
        while self.vehicle.mode.name != 'GUIDED':
            # self.vehicle.mode = VehicleMode("GUIDED")
            print('Waiting for GUIDED mode...')
            time.sleep(1)

        

        self.vehicle.simple_takeoff(targetHeight)
        while True:
            alt = self.vehicle.location.global_relative_frame.alt
            print(f'📊 Altitude: {alt:.2f}' if alt else 'Altitude: 0.00')
            if alt and alt >= 0.85 * targetHeight:
                break
            time.sleep(0.2)

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
        # Baseline of the SOS-divert accumulator so time spent diverted to an
        # SOS GPS target is excluded from this waypoint's mission timeout.
        divert_accum_start = self._sos_divert_accum

        while self.vehicle.mode.name == "GUIDED" and self.vehicle.armed:
            now = time.time()

            # ===== SOS divert handling =====
            # While diverted to a distress GPS target, the mission is paused:
            # do not time out and do not test the waypoint distance. When the
            # SosTargetManager finishes (drop/timeout), _end_sos_divert_resume()
            # re-issues simple_goto(targetLocation) and normal navigation resumes.
            if self._sos_divert_active:
                time.sleep(0.1)
                continue

            elapsed = now - start_time - pause_accum
            elapsed -= (self._sos_divert_accum - divert_accum_start)
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


    def fly_and_precision_land_with_waypoints(self, waypoints, takeoff_height=5):
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
