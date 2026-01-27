import math
import numpy as np
from collections import deque
import time

class DrowningDetector:
    def __init__(self, config=None):
        # Cấu hình mặc định
        default_config = {
            "SOS_FRAMES": 40,
            "WARNING_FRAMES": 20,
            "ARM_SPEED_TH": 8,
            "BODY_MOVE_TH": 4,
            "HEAD_SHOULDER_RATIO_TH": 0.9,
            "FACE_WATER_TH": 0.15,
            "VERTICAL_ANGLE_TH": 30
        }
        
        self.config = {**default_config, **(config or {})}
        
    def init_track(self, track):
        """Khởi tạo các biến theo dõi cho một track mới"""
        if hasattr(track, "inited"):
            return
        track.inited = True
        track.left_wrist = deque(maxlen=30)
        track.right_wrist = deque(maxlen=30)
        track.center_hist = deque(maxlen=30)
        track.head_shoulder_ratio_hist = deque(maxlen=30)
        track.body_tilt_hist = deque(maxlen=30)
        track.face_water_hist = deque(maxlen=15)
        track.sos_cnt = 0
        track.warning_cnt = 0
        track.state = "ACTIVE"
        track.sos_start_time = None
    
    def calculate_angle(self, a, b, c):
        """Tính góc giữa 3 điểm"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle
    
    def calculate_body_tilt(self, shoulder_left, shoulder_right, hip_left, hip_right):
        """Tính độ nghiêng cơ thể"""
        shoulder_center = ((shoulder_left[0] + shoulder_right[0]) // 2,
                          (shoulder_left[1] + shoulder_right[1]) // 2)
        hip_center = ((hip_left[0] + hip_right[0]) // 2,
                     (hip_left[1] + hip_right[1]) // 2)
        
        dx = shoulder_center[0] - hip_center[0]
        dy = shoulder_center[1] - hip_center[1]
        
        if dy == 0:
            return 90
        
        angle = math.degrees(math.atan(abs(dx) / abs(dy)))
        return angle
    
    def calculate_arm_speed(self, hist):
        """Tính tốc độ di chuyển của tay"""
        if len(hist) < 2:
            return 0
        speeds = []
        for i in range(1, len(hist)):
            d = math.hypot(hist[i][0] - hist[i-1][0], hist[i][1] - hist[i-1][1])
            speeds.append(d)
        return np.mean(speeds) if speeds else 0
    
    def detect(self, track, bbox, pose_landmarks, frame_shape):
        """
        Phát hiện các dấu hiệu đuối nước cho một track
        
        Args:
            track: Track object từ SORT tracker
            bbox: bounding box [x1, y1, x2, y2]
            pose_landmarks: MediaPipe pose landmarks
            frame_shape: (height, width) của frame
            
        Returns:
            dict: Kết quả phát hiện và các thông số
        """
        # Khởi tạo track nếu chưa
        self.init_track(track)
        
        # Cập nhật bbox
        track.bbox = bbox
        
        # Lấy kích thước frame
        h, w = frame_shape[:2]
        
        # Lấy tọa độ center
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        track.center_hist.append((cx, cy))
        
        # Khởi tạo kết quả mặc định
        results = {
            "state": track.state,
            "sos_cnt": track.sos_cnt,
            "arm_above_head": False,
            "arm_speed": 0,
            "body_frozen": False,
            "head_shoulder_ratio": 0,
            "body_tilt": 0,
            "face_in_water": False,
            "primary_condition": False,
            "secondary_condition": False,
            "tertiary_condition": False,
            "water_level": y1 + (y2 - y1) * 2/3
        }
        
        # Nếu không có pose landmarks, trả về kết quả mặc định
        if not pose_landmarks:
            return results
        
        lm = pose_landmarks.landmark
        
        # Chuyển đổi landmarks sang tọa độ pixel
        def lm_xy(lm_point):
            return int(lm_point.x * w), int(lm_point.y * h)
        
        # Lấy các điểm quan trọng
        nose = lm_xy(lm[0])
        left_eye = lm_xy(lm[1])
        right_eye = lm_xy(lm[2])
        left_shoulder = lm_xy(lm[11])
        right_shoulder = lm_xy(lm[12])
        left_hip = lm_xy(lm[23])
        right_hip = lm_xy(lm[24])
        left_wrist = lm_xy(lm[15])
        right_wrist = lm_xy(lm[16])
        
        # Cập nhật lịch sử cổ tay
        track.left_wrist.append(left_wrist)
        track.right_wrist.append(right_wrist)
        
        # ================= CÁC ĐIỀU KIỆN PHÁT HIỆN =================
        
        # 1. TAY TRÊN ĐẦU
        head_center_y = (nose[1] + left_eye[1] + right_eye[1]) / 3
        arm_above_head = (left_wrist[1] < head_center_y) or (right_wrist[1] < head_center_y)
        results["arm_above_head"] = arm_above_head
        
        # 2. TỐC ĐỘ TAY
        left_speed = self.calculate_arm_speed(track.left_wrist)
        right_speed = self.calculate_arm_speed(track.right_wrist)
        arm_speed = (left_speed + right_speed) / 2
        results["arm_speed"] = arm_speed
        arm_active = arm_speed > self.config["ARM_SPEED_TH"]
        
        # 3. CƠ THỂ BẤT ĐỘNG
        if len(track.center_hist) > 5:
            center_list = list(track.center_hist)
            start_idx = max(0, len(center_list) - 10)
            recent_centers = center_list[start_idx:]
            
            xs = [p[0] for p in recent_centers]
            ys = [p[1] for p in recent_centers]
            body_move = np.std(xs) + np.std(ys) if len(xs) > 1 else 999
        else:
            body_move = 999
        
        body_frozen = body_move < self.config["BODY_MOVE_TH"]
        results["body_frozen"] = body_frozen
        
        # 4. TỶ LỆ ĐẦU-VAI (dấu hiệu đầu chìm)
        shoulder_width = math.hypot(left_shoulder[0] - right_shoulder[0], 
                                   left_shoulder[1] - right_shoulder[1])
        head_height = abs(nose[1] - (left_shoulder[1] + right_shoulder[1]) / 2)
        
        if shoulder_width > 0:
            head_shoulder_ratio = head_height / shoulder_width
            results["head_shoulder_ratio"] = head_shoulder_ratio
            track.head_shoulder_ratio_hist.append(head_shoulder_ratio)
            
            if len(track.head_shoulder_ratio_hist) >= 15:
                ratio_list = list(track.head_shoulder_ratio_hist)
                low_ratio_frames = sum(1 for r in ratio_list 
                                     if r < self.config["HEAD_SHOULDER_RATIO_TH"])
                head_submerged = low_ratio_frames >= 10
            else:
                head_submerged = False
        else:
            head_submerged = False
        
        # 5. GÓC NGHIÊNG CƠ THỂ
        body_tilt = self.calculate_body_tilt(left_shoulder, right_shoulder, 
                                            left_hip, right_hip)
        results["body_tilt"] = body_tilt
        track.body_tilt_hist.append(body_tilt)
        
        if len(track.body_tilt_hist) >= 10:
            tilt_list = list(track.body_tilt_hist)
            start_idx = max(0, len(tilt_list) - 10)
            recent_tilts = tilt_list[start_idx:]
            avg_tilt = np.mean(recent_tilts)
            body_unstable = avg_tilt > self.config["VERTICAL_ANGLE_TH"]
        else:
            body_unstable = False
        
        # 6. MẶT Ở DƯỚI NƯỚC
        water_level = y1 + (y2 - y1) * 2/3
        face_in_water = nose[1] > water_level
        results["face_in_water"] = face_in_water
        results["water_level"] = water_level
        
        track.face_water_hist.append(face_in_water)
        if len(track.face_water_hist) >= 10:
            water_list = list(track.face_water_hist)
            face_water_frames = sum(1 for f in water_list if f)
            prolonged_face_water = face_water_frames >= 8
        else:
            prolonged_face_water = False
        
        # ================= QUYẾT ĐỊNH ĐA TIÊU CHÍ =================
        primary_condition = arm_above_head and arm_active and body_frozen
        secondary_condition = head_submerged and body_unstable
        tertiary_condition = prolonged_face_water and body_frozen
        
        results["primary_condition"] = primary_condition
        results["secondary_condition"] = secondary_condition
        results["tertiary_condition"] = tertiary_condition
        
        # Cập nhật SOS counter
        if primary_condition:
            track.sos_cnt += 2  # Tăng nhanh hơn cho điều kiện chính
        elif secondary_condition or tertiary_condition:
            track.sos_cnt += 1
        else:
            track.sos_cnt = max(0, track.sos_cnt - 1)
        
        # Phân loại trạng thái
        if track.sos_cnt > self.config["SOS_FRAMES"]:
            track.state = "SOS"
            if track.sos_start_time is None:
                track.sos_start_time = time.time()
        elif track.sos_cnt > self.config["WARNING_FRAMES"]:
            track.state = "WARNING"
            track.sos_start_time = None
        else:
            track.state = "ACTIVE"
            track.sos_start_time = None
        
        results["state"] = track.state
        results["sos_cnt"] = track.sos_cnt
        
        # Cập nhật landmarks cho track
        track.pose_landmarks = pose_landmarks
        
        return results
    
    def get_state_colors(self):
        """Trả về màu sắc cho từng trạng thái"""
        return {
            "ACTIVE": (0, 255, 0),      # Xanh lá
            "WARNING": (0, 165, 255),   # Cam
            "SOS": (0, 0, 255)          # Đỏ
        }
    
    def get_sos_duration(self, track):
        """Lấy thời gian đã ở trạng thái SOS"""
        if track.state == "SOS" and track.sos_start_time:
            return time.time() - track.sos_start_time
        return 0

        