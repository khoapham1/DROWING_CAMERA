import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2


class Track:
    def __init__(self, bbox, track_id=1):
        self.bbox = bbox  # [x1,y1,x2,y2]
        self.id = track_id
        self.pose_landmarks = None  # full-frame normalized landmarks
        self.miss_pose = 0
        self.last_source = "none"   # "yolo" | "mp"


class PersonDetector:
    """
    Person detector that uses:
      - YOLOv5 ONNX: to (re)acquire person bbox (every detect_interval frames or when pose lost)
      - MediaPipe Pose: to track & update bbox every frame (realtime), no "2 previous frames" cache

    Returns list of dicts:
      [{"id": int, "bbox": [x1,y1,x2,y2], "track_obj": Track, "pose_landmarks": NormalizedLandmarkList}]
    """

    def __init__(
        self,
        model_path="yolov5n_quant.onnx",
        img_size=416,
        conf_thres=0.4,
        iou_thres=0.5,
        detect_interval=1,              # YOLO re-acquire interval
        max_pose_miss=12,               # drop track if pose missing too long
        force_yolo_after_pose_miss=3,   # after N missed pose frames -> run YOLO immediately
        roi_pad=0.25,                   # expand ROI around bbox for pose
        mp_model_complexity=0,
        mp_min_det=0.4,
        mp_min_track=0.4,
        **kwargs
    ):
        self.img_size = int(img_size)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)

        self.detect_interval = max(1, int(detect_interval))
        self.max_pose_miss = int(max_pose_miss)
        self.force_yolo_after_pose_miss = int(force_yolo_after_pose_miss)
        self.roi_pad = float(roi_pad)

        self.frame_id = 0

        # ---------- YOLO ----------
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # ---------- MediaPipe Pose ----------
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=int(mp_model_complexity),
            smooth_landmarks=True,
            min_detection_confidence=float(mp_min_det),
            min_tracking_confidence=float(mp_min_track),
        )

        # ---------- Single active track (your pipeline uses detections[:1]) ----------
        self.track = Track(bbox=[0, 0, 0, 0], track_id=1)
        self.has_track = False

        print("✅ YOLOv5 ONNX + MediaPipe Pose tracking loaded")

    # ---------------- YOLO helpers ----------------
    @staticmethod
    def _iou_xyxy(a, b):
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
        areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
        return inter / (areaA + areaB - inter + 1e-6)

    def _nms_xyxy(self, dets):
        # dets: [{"bbox":[x1,y1,x2,y2], "score":float}, ...]
        if not dets:
            return []
        dets = sorted(dets, key=lambda d: d["score"], reverse=True)
        keep = []
        while dets:
            best = dets.pop(0)
            keep.append(best)
            dets = [d for d in dets if self._iou_xyxy(best["bbox"], d["bbox"]) < self.iou_thres]
        return keep

    def _letterbox(self, img, size):
        h, w = img.shape[:2]
        r = min(size / h, size / w)
        nh, nw = int(h * r), int(w * r)
        resized = cv2.resize(img, (nw, nh))
        top = (size - nh) // 2
        left = (size - nw) // 2
        out = np.full((size, size, 3), 114, dtype=np.uint8)
        out[top:top+nh, left:left+nw] = resized
        return out, r, left, top

    def _yolo_detect_person(self, frame):
        img, r, dx, dy = self._letterbox(frame, self.img_size)
        blob = img.transpose(2, 0, 1)
        blob = np.expand_dims(blob, 0).astype(np.float32) / 255.0

        pred = self.session.run([self.output_name], {self.input_name: blob})[0]
        if pred is None:
            return []
        pred = np.array(pred)
        if pred.ndim == 3:
            pred = pred[0]  # (N, 85)

        if pred.size == 0:
            return []

        # YOLOv5: [cx,cy,w,h,obj, cls...]
        obj = pred[:, 4]
        cls_prob = pred[:, 5:]
        cls_id = np.argmax(cls_prob, axis=1)
        cls_score = cls_prob[np.arange(pred.shape[0]), cls_id]
        score = obj * cls_score

        # keep class=person (0)
        keep = (cls_id == 0) & (score >= self.conf_thres)
        pred = pred[keep]
        score = score[keep]
        if pred.shape[0] == 0:
            return []

        dets = []
        H, W = frame.shape[:2]
        for i in range(pred.shape[0]):
            cx, cy, bw, bh = pred[i, 0:4]
            x1 = (cx - bw/2 - dx) / r
            y1 = (cy - bh/2 - dy) / r
            x2 = (cx + bw/2 - dx) / r
            y2 = (cy + bh/2 - dy) / r

            x1 = int(max(0, min(W - 1, x1)))
            y1 = int(max(0, min(H - 1, y1)))
            x2 = int(max(0, min(W - 1, x2)))
            y2 = int(max(0, min(H - 1, y2)))

            if x2 > x1 and y2 > y1:
                dets.append({"bbox": [x1, y1, x2, y2], "score": float(score[i])})

        dets = self._nms_xyxy(dets)
        return dets

    # ---------------- MediaPipe helpers ----------------
    @staticmethod
    def _expand_bbox(bbox, W, H, pad=0.25):
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        nw = bw * (1.0 + 2.0 * pad)
        nh = bh * (1.0 + 2.0 * pad)
        nx1 = int(max(0, min(W - 1, cx - nw / 2)))
        ny1 = int(max(0, min(H - 1, cy - nh / 2)))
        nx2 = int(max(0, min(W - 1, cx + nw / 2)))
        ny2 = int(max(0, min(H - 1, cy + nh / 2)))
        if nx2 <= nx1 or ny2 <= ny1:
            return [0, 0, W - 1, H - 1]
        return [nx1, ny1, nx2, ny2]

    @staticmethod
    def _pose_bbox_from_landmarks(lm_list, W, H, vis_th=0.35, pad=0.08):
        xs, ys = [], []
        for lm in lm_list.landmark:
            if hasattr(lm, "visibility") and lm.visibility < vis_th:
                continue
            xs.append(float(lm.x))
            ys.append(float(lm.y))
        if not xs:
            return None
        x1 = min(xs) * W
        y1 = min(ys) * H
        x2 = max(xs) * W
        y2 = max(ys) * H
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        x1 -= bw * pad
        y1 -= bh * pad
        x2 += bw * pad
        y2 += bh * pad
        x1 = int(max(0, min(W - 1, x1)))
        y1 = int(max(0, min(H - 1, y1)))
        x2 = int(max(0, min(W - 1, x2)))
        y2 = int(max(0, min(H - 1, y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def _to_fullframe_landmarks(pose_lms_roi, roi_xyxy, W, H):
        """Convert ROI-normalized landmarks -> full-frame normalized landmarks."""
        x1, y1, x2, y2 = roi_xyxy
        rw = max(1, x2 - x1)
        rh = max(1, y2 - y1)

        out = landmark_pb2.NormalizedLandmarkList()
        for lm in pose_lms_roi.landmark:
            fx = (lm.x * rw + x1) / float(W)
            fy = (lm.y * rh + y1) / float(H)
            nlm = landmark_pb2.NormalizedLandmark(
                x=float(np.clip(fx, 0.0, 1.0)),
                y=float(np.clip(fy, 0.0, 1.0)),
                z=float(lm.z),
                visibility=float(getattr(lm, "visibility", 0.0)),
                presence=float(getattr(lm, "presence", 0.0)) if hasattr(lm, "presence") else 0.0
            )
            out.landmark.append(nlm)
        return out

    # ---------------- main API ----------------
    def detect(self, frame):
        """
        Realtime:
          - Pose runs every frame to update bbox.
          - YOLO runs only when needed.
        """
        self.frame_id += 1
        H, W = frame.shape[:2]

        # Decide if we must run YOLO
        need_yolo = (not self.has_track) or (self.frame_id % self.detect_interval == 0) \
                    or (self.track.miss_pose >= self.force_yolo_after_pose_miss)

        if need_yolo:
            yolo_dets = self._yolo_detect_person(frame)
            if yolo_dets:
                # choose best by score, tie-break by area
                best = max(
                    yolo_dets,
                    key=lambda d: (d["score"], (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]))
                )
                self.track.bbox = best["bbox"]
                self.track.last_source = "yolo"
                self.track.miss_pose = 0
                self.has_track = True

        if not self.has_track:
            return []

        # Run MediaPipe Pose on ROI around current bbox (faster + stable)
        roi = self._expand_bbox(self.track.bbox, W, H, pad=self.roi_pad)
        rx1, ry1, rx2, ry2 = roi
        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return []

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        if res.pose_landmarks:
            full_lms = self._to_fullframe_landmarks(res.pose_landmarks, roi, W, H)
            self.track.pose_landmarks = full_lms

            # Update bbox from pose landmarks (full frame)
            bbox_mp = self._pose_bbox_from_landmarks(full_lms, W, H)
            if bbox_mp is not None:
                self.track.bbox = bbox_mp

            self.track.last_source = "mp"
            self.track.miss_pose = 0
        else:
            # Pose missing: do not "use 2 previous frames" logic.
            # Only keep track alive briefly; YOLO will reacquire quickly.
            self.track.miss_pose += 1
            if self.track.miss_pose >= self.max_pose_miss:
                self.has_track = False
                self.track.pose_landmarks = None
                return []

        return [{
            "id": self.track.id,
            "bbox": list(map(int, self.track.bbox)),
            "track_obj": self.track,
            "pose_landmarks": self.track.pose_landmarks
        }]

