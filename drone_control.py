
import time
import math
import threading
import numpy as np
import cv2
import csv
import datetime
import json
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import requests
from PID_controller import PIDController
from collections import deque
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
PERSON_REPORT_DIR = os.getenv("PERSON_REPORT_DIR", "person_detection_reports")
PERSON_EVAL_GT_PATH = os.getenv("PERSON_EVAL_GT_PATH", "").strip()
PERSON_EVAL_IOU_TH = float(os.getenv("PERSON_EVAL_IOU_TH", "0.5"))
PERSON_EVAL_CONF_TH = float(os.getenv("PERSON_EVAL_CONF_TH", "0.0"))

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


class PersonDetectionReport:
    """Collect runtime detections/PID samples and export one report folder."""

    def __init__(self, root_dir, gt_path="", iou_threshold=0.5, conf_threshold=0.0):
        self.root_dir = Path(root_dir or "person_detection_reports")
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.root_dir / f"session_{self.session_id}"
        self.start_time = time.time()
        self.start_iso = datetime.datetime.now().isoformat(timespec="seconds")
        self.gt_path = str(gt_path or "").strip()
        self.iou_threshold = float(iou_threshold)
        self.conf_threshold = float(conf_threshold)
        self.frame_index = 0
        self.frame_rows = []
        self.detection_rows = []
        self.pid_rows = []
        self.gt_boxes = {}
        self.gt_has_person = {}
        self.gt_load_error = None
        self._lock = threading.Lock()
        self._exported = False
        if self.gt_path:
            self._load_ground_truth(self.gt_path)

    @property
    def exported(self):
        return self._exported

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _safe_int(value, default=0):
        try:
            if value is None or value == "":
                return default
            return int(float(value))
        except Exception:
            return default

    @staticmethod
    def _safe_bool(value):
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "y", "person", "present"):
            return True
        if text in ("0", "false", "no", "n", "none", "absent"):
            return False
        return None

    @staticmethod
    def _timestamp_iso(ts):
        try:
            return datetime.datetime.fromtimestamp(float(ts)).isoformat(timespec="milliseconds")
        except Exception:
            return ""

    @classmethod
    def _box_from_values(cls, values, bbox_format=None):
        if values is None:
            return None
        try:
            vals = [float(v) for v in list(values)[:4]]
        except Exception:
            return None
        if len(vals) < 4:
            return None

        fmt = str(bbox_format or "").strip().lower()
        if fmt in ("xywh", "coco", "x_y_width_height"):
            x1, y1, bw, bh = vals
            x2 = x1 + bw
            y2 = y1 + bh
        else:
            x1, y1, x2, y2 = vals
            if x2 <= x1 or y2 <= y1:
                x2 = x1 + vals[2]
                y2 = y1 + vals[3]

        if x2 <= x1 or y2 <= y1:
            return None
        return [float(x1), float(y1), float(x2), float(y2)]

    @classmethod
    def _box_from_entry(cls, entry, bbox_format=None):
        if isinstance(entry, (list, tuple)):
            return cls._box_from_values(entry, bbox_format)
        if not isinstance(entry, dict):
            return None

        label = entry.get("label", entry.get("class", entry.get("name", None)))
        class_id = entry.get("class_id", entry.get("category_id", None))
        if label is not None and str(label).strip().lower() not in ("person", "0"):
            if class_id not in (0, "0"):
                return None
        if class_id not in (0, "0", None) and label is None:
            return None

        fmt = entry.get("bbox_format", bbox_format)
        if "bbox" in entry:
            return cls._box_from_values(entry.get("bbox"), fmt)
        if "box" in entry:
            return cls._box_from_values(entry.get("box"), fmt)
        if all(k in entry for k in ("x1", "y1", "x2", "y2")):
            return cls._box_from_values([entry["x1"], entry["y1"], entry["x2"], entry["y2"]], "xyxy")
        if all(k in entry for k in ("xmin", "ymin", "xmax", "ymax")):
            return cls._box_from_values([entry["xmin"], entry["ymin"], entry["xmax"], entry["ymax"]], "xyxy")
        if all(k in entry for k in ("left", "top", "right", "bottom")):
            return cls._box_from_values([entry["left"], entry["top"], entry["right"], entry["bottom"]], "xyxy")
        if all(k in entry for k in ("x", "y", "w", "h")):
            return cls._box_from_values([entry["x"], entry["y"], entry["w"], entry["h"]], "xywh")
        if all(k in entry for k in ("x", "y", "width", "height")):
            return cls._box_from_values([entry["x"], entry["y"], entry["width"], entry["height"]], "xywh")
        return None

    @classmethod
    def _boxes_from_frame_entry(cls, entry):
        boxes = []
        if isinstance(entry, list):
            candidates = entry
            fmt = None
        elif isinstance(entry, dict):
            fmt = entry.get("bbox_format")
            candidates = []
            for key in ("boxes", "bboxes", "annotations", "objects", "detections"):
                value = entry.get(key)
                if isinstance(value, dict):
                    nested = value.get("boxes", value.get("annotations", []))
                    if isinstance(nested, list):
                        candidates.extend(nested)
                elif isinstance(value, list):
                    candidates.extend(value)
            if not candidates:
                single_box = cls._box_from_entry(entry, fmt)
                return [single_box] if single_box else []
        else:
            return []

        for item in candidates:
            box = cls._box_from_entry(item, fmt)
            if box:
                boxes.append(box)
        return boxes

    def _store_gt_frame(self, frame_id, boxes, has_person=None):
        if frame_id is None:
            return
        fid = self._safe_int(frame_id, default=None)
        if fid is None:
            return
        if boxes:
            self.gt_boxes.setdefault(fid, []).extend(boxes)
            self.gt_has_person[fid] = True
            return
        if has_person is not None:
            self.gt_has_person[fid] = bool(has_person)

    def _load_ground_truth(self, gt_path):
        try:
            path = Path(gt_path)
            if not path.exists():
                self.gt_load_error = f"ground truth file not found: {gt_path}"
                return
            if path.suffix.lower() == ".csv":
                self._load_ground_truth_csv(path)
            else:
                self._load_ground_truth_json(path)
        except Exception as e:
            self.gt_load_error = str(e)

    def _load_ground_truth_json(self, path):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and isinstance(data.get("frames"), list):
            frames = data["frames"]
        elif isinstance(data, dict):
            frames = []
            for key, value in data.items():
                if key in ("meta", "metadata", "classes", "labels"):
                    continue
                if isinstance(value, dict):
                    item = dict(value)
                    item.setdefault("frame_id", key)
                else:
                    item = {"frame_id": key, "boxes": value}
                frames.append(item)
        elif isinstance(data, list):
            frames = data
        else:
            frames = []

        for idx, entry in enumerate(frames, start=1):
            if not isinstance(entry, dict):
                entry = {"frame_id": idx, "boxes": entry}
            frame_id = entry.get("frame_id", entry.get("frame", entry.get("image_id", idx)))
            boxes = self._boxes_from_frame_entry(entry)
            has_person = self._safe_bool(
                entry.get("has_person", entry.get("person_present", entry.get("person")))
            )
            if has_person is None and "person_count" in entry:
                has_person = self._safe_float(entry.get("person_count"), 0.0) > 0
            if has_person is None and any(k in entry for k in ("boxes", "bboxes", "annotations", "objects", "detections")):
                has_person = len(boxes) > 0
            self._store_gt_frame(frame_id, boxes, has_person)

    def _load_ground_truth_csv(self, path):
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                frame_id = row.get("frame_id", row.get("frame", row.get("image_id", idx)))
                box = self._box_from_entry(row, row.get("bbox_format"))
                has_person = self._safe_bool(
                    row.get("has_person", row.get("person_present", row.get("person")))
                )
                if has_person is None and "person_count" in row:
                    has_person = self._safe_float(row.get("person_count"), 0.0) > 0
                self._store_gt_frame(frame_id, [box] if box else [], has_person)

    def log_detection_frame(self, timestamp, frame_shape, detections):
        detections = detections or []
        h = self._safe_int(frame_shape[0], 0) if frame_shape else 0
        w = self._safe_int(frame_shape[1], 0) if frame_shape and len(frame_shape) > 1 else 0
        state_counts = {"ACTIVE": 0, "FROZEN": 0, "SOS": 0}
        top_conf = 0.0
        top_bbox = ["", "", "", ""]
        if detections:
            first = detections[0]
            top_conf = self._safe_float(first.get("confidence"), 0.0)
            raw_bbox = first.get("bbox", ["", "", "", ""])
            if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
                top_bbox = [self._safe_int(v, "") for v in raw_bbox]

        with self._lock:
            self.frame_index += 1
            frame_id = self.frame_index
            for det in detections:
                state = det.get("drowning_state", {}).get("state", "ACTIVE")
                state_counts[state] = state_counts.get(state, 0) + 1

            self.frame_rows.append({
                "frame_id": frame_id,
                "timestamp": float(timestamp),
                "timestamp_iso": self._timestamp_iso(timestamp),
                "t_rel_s": float(timestamp - self.start_time),
                "frame_w": int(w),
                "frame_h": int(h),
                "person_count": int(len(detections)),
                "active_count": int(state_counts.get("ACTIVE", 0)),
                "frozen_count": int(state_counts.get("FROZEN", 0)),
                "sos_count": int(state_counts.get("SOS", 0)),
                "top_confidence": float(top_conf),
                "top_x1": top_bbox[0],
                "top_y1": top_bbox[1],
                "top_x2": top_bbox[2],
                "top_y2": top_bbox[3],
            })

            for det_index, det in enumerate(detections):
                bbox = det.get("bbox", ["", "", "", ""])
                if len(bbox) != 4:
                    bbox = ["", "", "", ""]
                state = det.get("drowning_state", {}).get("state", "ACTIVE")
                self.detection_rows.append({
                    "frame_id": frame_id,
                    "timestamp": float(timestamp),
                    "timestamp_iso": self._timestamp_iso(timestamp),
                    "t_rel_s": float(timestamp - self.start_time),
                    "det_index": int(det_index),
                    "track_id": self._safe_int(det.get("id"), det_index + 1),
                    "label": "person",
                    "confidence": self._safe_float(det.get("confidence"), 0.0),
                    "x1": self._safe_int(bbox[0], 0),
                    "y1": self._safe_int(bbox[1], 0),
                    "x2": self._safe_int(bbox[2], 0),
                    "y2": self._safe_int(bbox[3], 0),
                    "source": det.get("source", "hailo_pose"),
                    "drowning_state": state,
                })
            return frame_id

    def log_pid_command(self, timestamp, command):
        command = command or {}
        with self._lock:
            self.pid_rows.append({
                "timestamp": float(timestamp),
                "timestamp_iso": self._timestamp_iso(timestamp),
                "t_rel_s": float(timestamp - self.start_time),
                "frame_id": int(self.frame_index),
                "active": int(bool(command.get("active", False))),
                "phase": str(command.get("phase", "")),
                "ex_px": self._safe_float(command.get("ex"), 0.0),
                "ey_px": self._safe_float(command.get("ey"), 0.0),
                "pid_ex": self._safe_float(command.get("pid_ex"), 0.0),
                "pid_ey": self._safe_float(command.get("pid_ey"), 0.0),
                "vx": self._safe_float(command.get("vx"), 0.0),
                "vy": self._safe_float(command.get("vy"), 0.0),
            })

    @staticmethod
    def _safe_div(num, den):
        return None if den == 0 else float(num) / float(den)

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return 0.0 if union <= 0.0 else inter / union

    def _metrics_from_counts(self, tp, fp, fn, tn=None, mode="object_detection"):
        precision = self._safe_div(tp, tp + fp)
        recall = self._safe_div(tp, tp + fn)
        f1 = None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        accuracy = None
        if tn is not None:
            accuracy = self._safe_div(tp + tn, tp + fp + fn + tn)
        return {
            "available": True,
            "mode": mode,
            "iou_threshold": self.iou_threshold,
            "confidence_threshold": self.conf_threshold,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": None if tn is None else int(tn),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
        }

    def _evaluate_object_detection(self, preds_by_frame):
        frame_ids = sorted(set(self.gt_boxes.keys()) | set(self.gt_has_person.keys()))
        tp = fp = fn = 0
        for frame_id in frame_ids:
            gt_boxes = list(self.gt_boxes.get(frame_id, []))
            pred_boxes = list(preds_by_frame.get(frame_id, []))
            matched_gt = set()
            matched_pred = set()

            pairs = []
            for pi, pbox in enumerate(pred_boxes):
                for gi, gbox in enumerate(gt_boxes):
                    pairs.append((self._iou(pbox, gbox), pi, gi))
            pairs.sort(reverse=True)

            for iou, pi, gi in pairs:
                if iou < self.iou_threshold:
                    break
                if pi in matched_pred or gi in matched_gt:
                    continue
                matched_pred.add(pi)
                matched_gt.add(gi)

            tp += len(matched_pred)
            fp += max(0, len(pred_boxes) - len(matched_pred))
            fn += max(0, len(gt_boxes) - len(matched_gt))

        metrics = self._metrics_from_counts(tp, fp, fn, tn=None, mode="object_detection")
        metrics["evaluated_frames"] = len(frame_ids)
        metrics["note"] = "TN is not defined for object detection unless negative regions are explicitly labelled."
        return metrics

    def _evaluate_frame_presence(self, preds_by_frame):
        tp = fp = fn = tn = 0
        for frame_id, gt_present in self.gt_has_person.items():
            pred_present = len(preds_by_frame.get(frame_id, [])) > 0
            if gt_present and pred_present:
                tp += 1
            elif (not gt_present) and pred_present:
                fp += 1
            elif gt_present and not pred_present:
                fn += 1
            else:
                tn += 1
        metrics = self._metrics_from_counts(tp, fp, fn, tn=tn, mode="frame_presence")
        metrics["evaluated_frames"] = len(self.gt_has_person)
        return metrics

    def _evaluate(self, detection_rows):
        if not self.gt_boxes and not self.gt_has_person:
            return {
                "available": False,
                "mode": "not_evaluated",
                "reason": "No ground-truth labels configured. Set PERSON_EVAL_GT_PATH to a JSON/CSV annotation file.",
                "gt_path": self.gt_path,
                "gt_load_error": self.gt_load_error,
            }

        preds_by_frame = {}
        for row in detection_rows:
            conf = self._safe_float(row.get("confidence"), 0.0)
            if conf < self.conf_threshold:
                continue
            frame_id = self._safe_int(row.get("frame_id"), 0)
            box = self._box_from_values([row.get("x1"), row.get("y1"), row.get("x2"), row.get("y2")], "xyxy")
            if box:
                preds_by_frame.setdefault(frame_id, []).append(box)

        if self.gt_boxes:
            return self._evaluate_object_detection(preds_by_frame)
        return self._evaluate_frame_presence(preds_by_frame)

    @staticmethod
    def _write_csv(path, rows, fieldnames):
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})

    @staticmethod
    def _metric_text(value):
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.4f}"
        except Exception:
            return str(value)

    def _write_confusion_matrix_csv(self, path, metrics):
        rows = []
        if metrics.get("available"):
            rows = [
                {"actual": "person", "pred_person": metrics.get("tp", 0), "pred_background": metrics.get("fn", 0)},
                {"actual": "background", "pred_person": metrics.get("fp", 0), "pred_background": metrics.get("tn", "N/A")},
            ]
        self._write_csv(path, rows, ["actual", "pred_person", "pred_background"])

    def _write_placeholder_png(self, path, title, message):
        img = np.full((420, 760, 3), 255, dtype=np.uint8)
        cv2.putText(img, title, (32, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
        y = 135
        words = str(message).split()
        line = ""
        for word in words:
            if len(line + " " + word) > 58:
                cv2.putText(img, line, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1, cv2.LINE_AA)
                y += 30
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            cv2.putText(img, line, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1, cv2.LINE_AA)
        cv2.imwrite(str(path), img)

    def _write_confusion_matrix_png(self, path, metrics):
        if not metrics.get("available"):
            self._write_placeholder_png(
                path,
                "Confusion Matrix",
                metrics.get("reason", "No ground truth labels available.")
            )
            return

        img = np.full((520, 760, 3), 255, dtype=np.uint8)
        cv2.putText(img, "Person Detection Confusion Matrix", (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(img, f"Precision: {self._metric_text(metrics.get('precision'))}", (40, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(img, f"Recall: {self._metric_text(metrics.get('recall'))}", (260, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(img, f"F1-score: {self._metric_text(metrics.get('f1_score'))}", (440, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)

        x0, y0 = 220, 150
        cell_w, cell_h = 210, 115
        labels = [
            ("TP", metrics.get("tp", 0), (214, 245, 214)),
            ("FN", metrics.get("fn", 0), (218, 232, 252)),
            ("FP", metrics.get("fp", 0), (248, 224, 224)),
            ("TN", metrics.get("tn", "N/A"), (230, 230, 230)),
        ]
        cv2.putText(img, "Pred: person", (x0 + 25, y0 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(img, "Pred: background", (x0 + cell_w + 10, y0 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(img, "Actual: person", (40, y0 + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(img, "Actual: background", (40, y0 + cell_h + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)

        for idx, (name, value, color) in enumerate(labels):
            row = idx // 2
            col = idx % 2
            x1 = x0 + col * cell_w
            y1 = y0 + row * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (80, 80, 80), 1)
            cv2.putText(img, name, (x1 + 22, y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(img, str(value), (x1 + 22, y1 + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 2, cv2.LINE_AA)

        cv2.imwrite(str(path), img)

    def _write_line_chart(self, path, rows, series, title, y_label):
        if not rows:
            self._write_placeholder_png(path, title, "No samples were logged for this chart.")
            return

        points = []
        for row in rows:
            x = self._safe_float(row.get("t_rel_s"), None)
            if x is None:
                continue
            for key, _, _ in series:
                y = self._safe_float(row.get(key), None)
                if y is not None:
                    points.append((x, y))
        if not points:
            self._write_placeholder_png(path, title, "No numeric samples were logged for this chart.")
            return

        img = np.full((520, 960, 3), 255, dtype=np.uint8)
        left, right, top, bottom = 80, 900, 70, 440
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        if abs(x_max - x_min) < 1e-9:
            x_max = x_min + 1.0
        if abs(y_max - y_min) < 1e-9:
            y_min -= 1.0
            y_max += 1.0
        y_pad = (y_max - y_min) * 0.08
        y_min -= y_pad
        y_max += y_pad

        cv2.putText(img, title, (40, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.line(img, (left, bottom), (right, bottom), (60, 60, 60), 1, cv2.LINE_AA)
        cv2.line(img, (left, top), (left, bottom), (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(img, "time (s)", (430, 492), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(img, y_label, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(img, f"{y_max:.2f}", (10, top + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(img, f"{y_min:.2f}", (10, bottom + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(img, f"{x_min:.1f}", (left - 8, bottom + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(img, f"{x_max:.1f}", (right - 35, bottom + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)

        def xy_to_px(x, y):
            px = int(left + (x - x_min) / (x_max - x_min) * (right - left))
            py = int(bottom - (y - y_min) / (y_max - y_min) * (bottom - top))
            return px, py

        for key, label, color in series:
            poly = []
            for row in rows:
                x = self._safe_float(row.get("t_rel_s"), None)
                y = self._safe_float(row.get(key), None)
                if x is not None and y is not None:
                    poly.append(xy_to_px(x, y))
            if len(poly) >= 2:
                cv2.polylines(img, [np.array(poly, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
            elif len(poly) == 1:
                cv2.circle(img, poly[0], 3, color, -1, cv2.LINE_AA)

        legend_x = 690
        legend_y = 30
        for idx, (_, label, color) in enumerate(series):
            y = legend_y + idx * 24
            cv2.line(img, (legend_x, y), (legend_x + 28, y), color, 3, cv2.LINE_AA)
            cv2.putText(img, label, (legend_x + 36, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (40, 40, 40), 1, cv2.LINE_AA)

        cv2.imwrite(str(path), img)

    def _write_readme(self, path, metrics):
        lines = [
            "Person detection report",
            "",
            "Files:",
            "- frames.csv: one row per processed Hailo detection loop frame.",
            "- detections.csv: one row per predicted person bbox.",
            "- ex_y_error.csv / ex_y_error.png: bbox-center pixel error ex, ey.",
            "- pid_response.csv / pid_response.png: pid_ex, pid_ey, vx, vy over time.",
            "- confusion_matrix.csv / confusion_matrix.png: TP/FP/FN/TN table when ground truth is available.",
            "- metrics.json: Precision, Recall, F1-score and report metadata.",
            "",
            "Important:",
            "Precision, Recall, F1-score and confusion matrix need ground-truth labels.",
            "Set PERSON_EVAL_GT_PATH to a JSON/CSV annotation file before running detection.",
            "For bbox evaluation, frame_id in the annotation must match frames.csv frame_id.",
            "",
            "Supported JSON examples:",
            '{"frames":[{"frame_id":1,"boxes":[[x1,y1,x2,y2]]}]}',
            '{"1":[[x1,y1,x2,y2]],"2":[]}',
            "",
            f"Evaluation available: {metrics.get('available')}",
            f"Mode: {metrics.get('mode')}",
            f"Reason/note: {metrics.get('reason', metrics.get('note', ''))}",
        ]
        with path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def export(self, reason="stop"):
        with self._lock:
            if self._exported:
                return str(self.output_dir)
            self._exported = True
            frame_rows = list(self.frame_rows)
            detection_rows = list(self.detection_rows)
            pid_rows = list(self.pid_rows)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        end_time = time.time()
        metrics = self._evaluate(detection_rows)
        metrics["gt_path"] = self.gt_path
        metrics["gt_load_error"] = self.gt_load_error

        frame_fields = [
            "frame_id", "timestamp", "timestamp_iso", "t_rel_s", "frame_w", "frame_h",
            "person_count", "active_count", "frozen_count", "sos_count",
            "top_confidence", "top_x1", "top_y1", "top_x2", "top_y2",
        ]
        detection_fields = [
            "frame_id", "timestamp", "timestamp_iso", "t_rel_s", "det_index", "track_id",
            "label", "confidence", "x1", "y1", "x2", "y2", "source", "drowning_state",
        ]
        ex_fields = ["timestamp", "timestamp_iso", "t_rel_s", "frame_id", "phase", "ex_px", "ey_px"]
        pid_fields = [
            "timestamp", "timestamp_iso", "t_rel_s", "frame_id", "active", "phase",
            "ex_px", "ey_px", "pid_ex", "pid_ey", "vx", "vy",
        ]

        self._write_csv(self.output_dir / "frames.csv", frame_rows, frame_fields)
        self._write_csv(self.output_dir / "detections.csv", detection_rows, detection_fields)
        self._write_csv(self.output_dir / "ex_y_error.csv", pid_rows, ex_fields)
        self._write_csv(self.output_dir / "pid_response.csv", pid_rows, pid_fields)
        self._write_confusion_matrix_csv(self.output_dir / "confusion_matrix.csv", metrics)
        self._write_confusion_matrix_png(self.output_dir / "confusion_matrix.png", metrics)
        self._write_line_chart(
            self.output_dir / "ex_y_error.png",
            pid_rows,
            [
                ("ex_px", "ex px", (255, 70, 70)),
                ("ey_px", "ey px", (70, 70, 255)),
            ],
            "Pixel Error ex/ey",
            "pixel error",
        )
        self._write_line_chart(
            self.output_dir / "pid_response.png",
            pid_rows,
            [
                ("pid_ex", "pid_ex", (255, 70, 70)),
                ("pid_ey", "pid_ey", (70, 70, 255)),
                ("vx", "vx", (60, 160, 60)),
                ("vy", "vy", (180, 110, 40)),
            ],
            "PID Response",
            "output",
        )

        summary = {
            "session_id": self.session_id,
            "reason": reason,
            "started_at": self.start_iso,
            "ended_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "duration_s": float(end_time - self.start_time),
            "frames_logged": len(frame_rows),
            "detections_logged": len(detection_rows),
            "pid_samples_logged": len(pid_rows),
            "report_dir": str(self.output_dir),
            "metrics": metrics,
        }
        with (self.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self._write_readme(self.output_dir / "README.txt", metrics)
        return str(self.output_dir)


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
            print(f"⚠️ Hailo pose stream unavailable: {e}")
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
            print(f"⚠️ Hailo pose stream unavailable: {e}")

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
            print(f"⚠️ Hailo pose camera start failed: {e}")
        print("⚠️ Hailo-only mode: non-Hailo camera fallback is disabled")
        return
    elif HAILO_POSE_ENABLED:
        print("⚠️ Hailo pose must be initialized from the main thread; non-Hailo camera fallback is disabled")
        return
    else:
        print("⚠️ Hailo pose is disabled; non-Hailo camera fallback is disabled")
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

# ===== DROWNING DETECTION CONFIGURATION ========
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

                # Never auto-yaw during GUIDED navigation. Keep this available
                # if waypoint navigation causes yaw oscillation during mission.
                # self.vehicle.parameters['WP_YAW_BEHAVIOR'] = 0

                print("✅ Listeners and parameters set")
            except Exception as e:
                print(f"Warning: Failed to set some listeners: {e}")

        self.takeoff_height = takeoff_height
        self.flown_path = []

        
        # Person detection uses only the Hailo pose pipeline on the AI HAT.
        self.use_hailo_pose = bool(HAILO_POSE_ENABLED)
        self.person_thread = None
        self.person_running = False
        self._person_stop_event = threading.Event()
        self.detected_persons = []
        self.last_detection_time = 0
        self.detection_interval = 0.05  # seconds between processing steps (loop pacing only)
        self.drowing_detector = DrowningDetector(DROWNING_CONFIG)
        self.person_report = None
        self.person_report_dir = PERSON_REPORT_DIR
        self.person_eval_gt_path = PERSON_EVAL_GT_PATH
        self.person_eval_iou_threshold = PERSON_EVAL_IOU_TH
        self.person_eval_conf_threshold = PERSON_EVAL_CONF_TH

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
        self.PID_X = PIDController(
            0.0028, 0.0, 0.0003,
            max_output=0.2,
            integral_limit=300,
            derivative_filter_tau=0.06,
            derivative_limit=1800
        )
        
        self.PID_Y = PIDController(
            0.0032, 0.0, 0.00035,
            max_output=0.2,
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
            "pid_ex": 0.0,
            "pid_ey": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "phase": "idle",
        }
        self._visual_servo_no_person_exit_printed = False
        self._last_pid_debug_print = 0.0
        self._pid_debug_interval = 0.25
        # After a person is detected, brake immediately, then wait this long
        # before engaging PID centering on bbox ex/ey.
        self.visual_servo_start_delay_sec = float(os.getenv("VISUAL_SERVO_START_DELAY_SECONDS", "0.5"))
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
        self.dropoff_pwm = 1800
        self.holdon_pwm = 1100
        self.servo_state = "closed"
        self.servo_pwm = self.holdon_pwm
        print(f"✅ Servo drop configured: ch={self.servo}, drop_pwm={self.dropoff_pwm}, hold_pwm={self.holdon_pwm}")

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
        # Person pause braking is velocity-based: stop first, then engage
        # visual-servo PID after the short settle delay.

    def _new_person_track(self):
        class _PersonTrack:
            pass
        return _PersonTrack()

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
                "phase": "idle",
            }
        self.PID_X.reset()
        self.PID_Y.reset()

    def _start_person_report(self):
        self.person_report = PersonDetectionReport(
            self.person_report_dir,
            gt_path=self.person_eval_gt_path,
            iou_threshold=self.person_eval_iou_threshold,
            conf_threshold=self.person_eval_conf_threshold,
        )
        print(
            "[REPORT] Person detection report session started "
            f"id={self.person_report.session_id}"
        )

    def _export_person_report(self, reason="stop"):
        report = getattr(self, "person_report", None)
        if not report or report.exported:
            self.person_report = None
            return None
        try:
            report_dir = report.export(reason=reason)
            print(f"[REPORT] Person detection report exported: {report_dir}")
            return report_dir
        except Exception as e:
            print(f"[REPORT] Failed to export person detection report: {e}")
            return None
        finally:
            self.person_report = None

    def _log_person_detection_report(self, timestamp, frame_shape, detections):
        report = getattr(self, "person_report", None)
        if not report:
            return
        try:
            report.log_detection_frame(timestamp, frame_shape, detections)
        except Exception as e:
            print(f"[REPORT] Detection log error: {e}")

    def _log_person_pid_command(self, timestamp, command):
        report = getattr(self, "person_report", None)
        if not report:
            return
        try:
            report.log_pid_command(timestamp, command)
        except Exception as e:
            print(f"[REPORT] PID log error: {e}")


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

        if self.person_report and not self.person_report.exported:
            self._export_person_report(reason="restart_person_detection")
        self._reset_person_detection_state()
        self._start_person_report()
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
                "pid_ex": 0.0,
                "pid_ey": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "phase": "idle",
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
                "phase": "idle",
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

    def _direction_to_vxvy(self, direction: str, speed: float):
        d = (direction or "").lower()
        if d == "forward":
            return float(speed), 0.0        
        if d == "backward":
            return -float(speed), 0.0
        if d == "left":
            return 0.0, -float(speed)
        if d == "rigth":
            return 0.0, float(speed)
        return 0.0, 0.0
    
    
    
        
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

        # ex/ey are pixel errors from the detected bbox center to the frame center.
        # ex > 0: bbox is right of frame center. ey > 0: bbox is below frame center.
        ex = float(target.get("ex", 0.0))
        ey = float(target.get("ey", 0.0))

        error_x = 0.0 if abs(ex) <= self.visual_servo_deadband_px else ex
        error_y = 0.0 if abs(ey) <= self.visual_servo_deadband_px else ey

        if error_x == 0.0:
            self.PID_X.reset()
        if error_y == 0.0:
            self.PID_Y.reset()

        correction_active = (error_x != 0.0 or error_y != 0.0)
        pid_ex = 0.0
        pid_ey = 0.0
        if correction_active:
            pid_ex = 0.0 if error_x == 0.0 else self.PID_X.update(-error_x)
            pid_ey = 0.0 if error_y == 0.0 else self.PID_Y.update(error_y)

        # Body-NED mapping from the bbox center error:
        # bbox below center -> -ey PID -> negative vx; bbox right -> -ex PID -> negative vy.
        vx = pid_ey
        vy = pid_ex

        self.send_local_ned_velocity(vx, vy, 0.0)
        command = {
            "active": bool(correction_active),
            "ex": ex,
            "ey": ey,
            "pid_ex": float(pid_ex),
            "pid_ey": float(pid_ey),
            "vx": float(vx),
            "vy": float(vy),
            "phase": "pid" if correction_active else "centered",
        }
        with self.visual_servo_lock:
            self.visual_servo_last_command = command
        self._log_person_pid_command(now, command)
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
                # if not self.vehicle or not self.vehicle.armed:
                #     print("⚠️ DROP skipped: vehicle not available or not armed")
                #     return

                print("🎯 DROP condition met: SOS + frame center inside bbox")
                self.set_payload_servo_state("open")
                time.sleep(1.5)
                self.set_payload_servo_state("closed")
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

    def set_payload_servo_state(self, state: str):
        """Set payload servo directly from UI. state: 'open' or 'closed'."""
        if state not in ("open", "closed"):
            raise ValueError("state must be 'open' or 'closed'")

        pwm_value = self.dropoff_pwm if state == "open" else self.holdon_pwm
        self.controlServo(self.servo, pwm_value)
        self.servo_state = state
        self.servo_pwm = pwm_value
        print(f"Servo payload -> {state} (ch={self.servo}, pwm={pwm_value})")
        return {
            "servo": self.servo,
            "servo_state": self.servo_state,
            "servo_pwm": self.servo_pwm,
        }

    def _rearm_drop_if_clear(self, now, person_now):
        if person_now:
            return
        with self._drop_lock:
            if self._drop_completed and (now - self._last_drop_time) >= self._drop_rearm_delay:
                self._drop_completed = False
                print("✅ DROP gate re-armed for next target")

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
                    print("⚠️ Hailo pose runtime is not active; waiting for Hailo pose")
                    start_camera()
                    time.sleep(0.5)
                    continue

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
                    self._update_visual_servo_target(first["bbox"], w, h, current_time)
                else:
                    self.latest_pose_landmarks = None
                    self.latest_pose_keypoints = None
                    self.latest_pose_source = None

                self._prune_hailo_tracks(current_time)
                self.detected_persons = detections
                person_now = len(detections) > 0
                self._log_person_detection_report(current_time, (h, w), detections)
                # Keep the visual-servo target fresh for UI/debug. The pause
                # loop now brakes with zero BODY_NED velocity after confirmation.
                if not detections:
                    self._clear_visual_servo_target_if_stale(current_time)
                self.last_detection_time = current_time
                self._update_detection_fps(current_time)

                self._rearm_drop_if_clear(current_time, person_now)
                if person_now and self.pause_enable and not self._drop_completed:
                    try:
                        if self.vehicle and self.vehicle.armed and self.vehicle.mode.name == "GUIDED":
                            self.request_pause("person_detected")
                    except Exception:
                        pass

                sos_dets = [d for d in detections if d.get("drowning_state", {}).get("state") == "SOS"]
                sos_now = len(sos_dets) > 0
                sos_bbox = sos_dets[0].get("bbox") if sos_dets else None
                if sos_now and self.pause_enable and not self._drop_completed:
                    self.request_pause("SOS")

                self._maybe_drop_for_sos(sos_bbox, w, h, sos_now)

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
                self._exit_visual_servo_if_no_person_timeout(current_time, person_now)

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
                print(f"Hailo person detection error: {e}")
                time.sleep(0.1)


    def _person_detection_loop(self, stop_event=None):
        """Person + drowning detection loop backed only by Hailo pose."""
        return self._person_detection_loop_hailo(stop_event)

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
        self._export_person_report(reason="stop_person_detection")
        self._reset_person_detection_state()
        print("Person detection stopped")

    # -------- Pause / Resume logic (mission) --------
    def request_pause(self, reason="person_detected"):
        if not self.vehicle:
            return
        now = time.time()
        new_pause = False
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
                new_pause = True
                print(f"Mission PAUSE requested: {self._pause_reason}")
            else:
                # Escalate reason to SOS if needed
                if str(reason).upper() == "SOS" and (self._pause_reason != "SOS"):
                    self._pause_reason = "SOS"
                    print("Mission PAUSE escalated: SOS")
        if new_pause:
            self._hold_zero_velocity_with_visual_state("braking")

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

    def _hold_zero_velocity_with_visual_state(self, phase="holding"):
        with self.visual_servo_lock:
            target = dict(self.visual_servo_target) if self.visual_servo_target else None

        ex = float(target.get("ex", 0.0)) if target else 0.0
        ey = float(target.get("ey", 0.0)) if target else 0.0
        try:
            self.send_local_ned_velocity(0.0, 0.0, 0.0)
            command = {
                "active": False,
                "ex": ex,
                "ey": ey,
                "pid_ex": 0.0,
                "pid_ey": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "phase": str(phase),
            }
            with self.visual_servo_lock:
                self.visual_servo_last_command = command
            self._log_person_pid_command(time.time(), command)
            self.PID_X.reset()
            self.PID_Y.reset()
        except Exception:
            pass

    def _hold_position_step(self):
        if not self.vehicle:
            return
        now = time.time()
        with self._pause_lock:
            started = self._pause_started
        started = float(started) if started is not None else now
        settle_elapsed = now - started

        # Person detected: brake first, let the aircraft settle, then engage
        # PID_x/PID_y from bbox center error ex/ey.
        if settle_elapsed < float(self.visual_servo_start_delay_sec):
            self._hold_zero_velocity_with_visual_state("settling")
            return

        if not self._visual_servo_step(now):
            self._hold_zero_velocity_with_visual_state("holding")

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
            time.sleep(2)
            
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
