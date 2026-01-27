
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
    "SOS_FRAMES": 40,
    "WARNING_FRAMES": 20,
    "ARM_SPEED_TH": 8,
    "BODY_MOVE_TH": 4,
    "HEAD_SHOULDER_RATIO_TH": 0.9,
    "FACE_WATER_TH": 0.15,
    "VERTICAL_ANGLE_TH": 30
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
        self.person_detector = PersonDetector("yolov5n_quant.onnx", IMG_SIZE, 0.4, 0.45)
        self.person_thread = None
        self.person_running = False
        self.detected_persons = []
        self.last_detection_time = 0
        self.detection_interval = 0.1  # 2 FPS for detection
        self.drowing_detector = DrowningDetector(DROWNING_CONFIG)
        self.latest_pose_landmarks = None
        

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
    def draw_results(self, frame, bbox, results, track_id):
        """Vẽ kết quả phát hiện lên frame"""
        x1, y1, x2, y2 = bbox
        
        # Lấy màu theo trạng thái
        colors = {
            "ACTIVE": (0, 255, 0),
            "WARNING": (0, 165, 255),
            "SOS": (0, 0, 255)
        }
        color = colors.get(results["state"], (0, 255, 0))
        
        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Vẽ đường mực nước
        water_y = int(results["water_level"])
        cv2.line(frame, (x1, water_y), (x2, water_y), 
                (255, 255, 0), 1, cv2.LINE_AA)
        
        # Vẽ thông tin trạng thái
        cv2.putText(frame, f"ID {track_id} | {results['state']}", 
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Hiển thị thời gian SOS nếu có
        if results["state"] == "SOS":
            sos_duration = results.get("sos_duration", 0)
            cv2.putText(frame, f"SOS: {sos_duration:.1f}s", 
                    (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, color, 2)
        
        # Thông tin chi tiết (chỉ hiển thị nếu có đủ không gian)
        h, w = frame.shape[:2]
        if (y2 + 100) < h:
            info_y = y2 + 20
            info_lines = [
                f"ArmAbove: {results['arm_above_head']}",
                f"ArmSpeed: {results['arm_speed']:.1f}",
                f"Frozen: {results['body_frozen']}",
                f"HeadRatio: {results['head_shoulder_ratio']:.2f}",
                f"Tilt: {results['body_tilt']:.1f}°",
                f"FaceWater: {results['face_in_water']}"
            ]
            
            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (x1, info_y + i*15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        
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
                # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                h, w = cv_image.shape[:2]

                pose_res = pose.process(cv_image)
                if pose_res.pose_landmarks:
                    self.latest_pose_landmarks = pose_res.pose_landmarks
                
                if cv_image is None:
                    continue
                
                # Phát hiện người bằng YOLO
                detections = self.person_detector.detect(cv_image)
                states = {"ACTIVE": 0, "WARNING": 0, "SOS": 0}

                for det in detections:
                    track = det["track_obj"]
                    bbox = det["bbox"]
                    track_id = det["id"]
                    
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
                    states[results["state"]] += 1
                    
                    # Vẽ kết quả lên frame
                    color = self.draw_results(cv_image, bbox, results, track_id)
                    
                    # Vẽ skeleton
                    if pose_res.pose_landmarks:
                        self.draw_skeleton(cv_image, pose_res.pose_landmarks, color)
                
                # Hiển thị thống kê
                cv2.putText(cv_image, 
                   f"Active: {states['ACTIVE']} | Warn: {states['WARNING']} | SOS: {states['SOS']}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                try:
                    cv2.imshow("DROWNING DETECTION", cv_image)
                    cv2.waitKey(1)
                except:
                    pass
                    
                self.detected_persons = detections
                self.last_detection_time = current_time

                if detections:
                    # Get current GPS position
                    if self.vehicle:
                        try:
                            loc = self.vehicle.location.global_frame
                            if loc is None or loc.lat is None or loc.lon is None:
                                loc = self.vehicle.location.global_relative_frame

                            if loc and loc.lat and loc.lon:
                                # Send to server
                                self.send_person_detection_to_server(
                                    lat=float(loc.lat),
                                    lon=float(loc.lon),
                                    alt=float(loc.alt) if loc.alt else 0.0,
                                    detections=detections
                                )
                            time.sleep(2)
                        except Exception as e:
                            print(f"Error getting GPS for detection: {e}")

            except Exception as e:
                print(f"Person detection error: {e}")
                time.sleep(0.1)

    def send_person_detection_to_server(self, lat, lon, alt, detections):
        """Send person detection results to server"""
        try:
            person_data = {
                'lat': lat,
                'lon': lon,
                'alt': alt,
                'timestamp': time.time(),
                'detections': detections,
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
        print("✅ Person detection stopped")

    # -------- MAVLink control --------
    def send_local_ned_velocity(self, vx, vy, vz):
        """Send velocity in BODY_OFFSET_NED frame"""
        if not self.vehicle:
            return
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
            0, 0, 0,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            0b0000111111000111,  # enable vx, vy, vz
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, 0
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
    
    def goto(self, targetLocation, tolerance=0.6, timeout=60, speed=0.7):
        """
        simple_goto với tolerance & timeout nới lỏng, có record vị trí.
        """
        if speed < 0.1 or speed > 5.0:
            print(f"Tốc độ {speed} m/s không hợp lệ, đặt về 0.7 m/s")
            speed = 0.7

        if not self.vehicle:
            return False

        distanceToTargetLocation = self.get_distance_meters(
            targetLocation,
            self.vehicle.location.global_relative_frame
        )
        self.set_speed(speed)
        self.vehicle.simple_goto(targetLocation, groundspeed=speed)

        start_dist = distanceToTargetLocation
        start_time = time.time()

        while self.vehicle.mode.name == "GUIDED" and time.time() - start_time < timeout:
            currentDistance = self.get_distance_meters(
                targetLocation,
                self.vehicle.location.global_relative_frame
            )
            if currentDistance < max(tolerance, start_dist * 0.01):
                print("Reached target waypoint")
                return True
            time.sleep(0.02)

        print("Timeout reaching waypoint, proceeding anyway")
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

def get_controller(connection_str='/dev/ttyACM0', takeoff_height=5):
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