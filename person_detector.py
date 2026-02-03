import cv2
import numpy as np
import onnxruntime as ort

# ================= TRACK =================
class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox
        self.id = track_id
        self.miss = 0
        self.pose_landmarks = None
        self.pose_history = []
        self.last_updated_frame = 0


class SortTracker:
    def __init__(self, iou_thres=0.5, max_miss=30):
        self.iou_thres = iou_thres
        self.max_miss = max_miss
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0

    def iou(self, a, b):
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (a[2]-a[0]) * (a[3]-a[1])
        areaB = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (areaA + areaB - inter + 1e-6)

    def update(self, detections):
        self.frame_count += 1
        
        for d in detections:
            d["used"] = False

        updated_tracks = []

        for t in self.tracks:
            best_iou = 0
            best_det = None

            for d in detections:
                if d["used"]:
                    continue
                i = self.iou(t.bbox, d["bbox"])
                if i > best_iou:
                    best_iou = i
                    best_det = d

            if best_iou > self.iou_thres:
                t.bbox = best_det["bbox"]
                t.miss = 0
                t.last_updated_frame = self.frame_count
                best_det["used"] = True
            else:
                t.miss += 1

            # Keep tracks only up to max_miss (no history-based "person present" logic).
            # Any higher-level persistence (e.g., via MediaPipe Pose) should be done outside.
            if t.miss <= self.max_miss:
                updated_tracks.append(t)

        for d in detections:
            if d["used"]:
                continue

            overlap = False
            for t in updated_tracks:
                if self.iou(t.bbox, d["bbox"]) > 0.4:
                    overlap = True
                    break

            if not overlap:
                new_track = Track(d["bbox"], self.next_id)
                new_track.last_updated_frame = self.frame_count
                updated_tracks.append(new_track)
                self.next_id += 1

        self.tracks = updated_tracks
        return self.tracks


# ================= PERSON DETECTOR =================
class PersonDetector:
    def __init__(
        self,
        model_path="yolov5n_quant.onnx",
        img_size=416,
        conf_thres=0.4,
        iou_thres=0.5,
        detect_interval=5,
        max_miss=0
    ):
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.detect_interval = detect_interval
        self.frame_id = 0

        # Default max_miss=0 so we do NOT return stale tracks when YOLO misses.
        self.tracker = SortTracker(iou_thres=0.4, max_miss=max_miss)

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("✅ YOLOv5n + SORT loaded")

    def letterbox(self, img, size):
        h, w = img.shape[:2]
        r = min(size / h, size / w)
        nh, nw = int(h * r), int(w * r)
        img = cv2.resize(img, (nw, nh))
        top = (size - nh) // 2
        left = (size - nw) // 2
        out = np.full((size, size, 3), 114, dtype=np.uint8)
        out[top:top+nh, left:left+nw] = img
        return out, r, left, top

    def nms(self, detections):
        if len(detections) == 0:
            return []

        boxes = [d["bbox"] for d in detections]
        scores = [d["confidence"] for d in detections]

        idxs = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.conf_thres,
            self.iou_thres
        )

        if len(idxs) == 0:
            return []

        return [detections[i] for i in idxs.flatten()]

    def detect(self, frame):
        self.frame_id += 1

        if self.frame_id % self.detect_interval != 0 and self.frame_id > 1:
            return [
                {"id": t.id, "bbox": t.bbox, "track_obj": t}
                for t in self.tracker.tracks
            ]

        img, r, dx, dy = self.letterbox(frame, self.img_size)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0).astype(np.float32) / 255.0

        pred = self.session.run(
            [self.output_name],
            {self.input_name: img}
        )[0]

        detections = []

        for x in pred:
            x = x[x[:, 4] > self.conf_thres]
            if len(x) == 0:
                continue

            conf = np.maximum(x[:, 4], x[:, 5:].max(1))
            cls = x[:, 5:].argmax(1)

            keep = cls == 0
            x = x[keep]
            conf = conf[keep]

            for i in range(len(x)):
                cx, cy, w, h = x[i][:4]

                x1 = int((cx - w/2 - dx) / r)
                y1 = int((cy - h/2 - dy) / r)
                x2 = int((cx + w/2 - dx) / r)
                y2 = int((cy + h/2 - dy) / r)

                x1 = max(0, min(x1, frame.shape[1]-1))
                y1 = max(0, min(y1, frame.shape[0]-1))
                x2 = max(0, min(x2, frame.shape[1]-1))
                y2 = max(0, min(y2, frame.shape[0]-1))

                if x2 > x1 and y2 > y1:
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(conf[i])
                    })

        detections = self.nms(detections)

        tracks = self.tracker.update(detections)

        return [
            {"id": t.id, "bbox": t.bbox, "track_obj": t}
            for t in tracks
        ]