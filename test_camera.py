import cv2
import numpy as np
import onnxruntime as ort
import time

# ===================== CONFIG =====================
MODEL_PATH = "yolov5n_quant.onnx"
IMG_SIZE = 320
CONF_THRES = 0.8
IOU_THRES = 0.45
MAX_DET = 5
CLASSES = [0]  # person
# ==================================================

# ---------------- Letterbox ----------------
def letterbox(img, new_shape=(320, 320), color=(114, 114, 114)):
    shape = img.shape[:2]  # h, w
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(
        img,
        int(round(dh - 0.1)), int(round(dh + 0.1)),
        int(round(dw - 0.1)), int(round(dw + 0.1)),
        cv2.BORDER_CONSTANT,
        value=color
    )

    return img, r, dw, dh


# ---------------- NMS ----------------
def non_max_suppression(pred):
    detections = []

    for x in pred:
        x = x[x[:, 4] > CONF_THRES]
        if not len(x):
            continue

        # obj_conf * cls_conf
        x[:, 5:] *= x[:, 4:5]

        # xywh → xyxy
        boxes = np.zeros((x.shape[0], 4))
        boxes[:, 0] = x[:, 0] - x[:, 2] / 2
        boxes[:, 1] = x[:, 1] - x[:, 3] / 2
        boxes[:, 2] = x[:, 0] + x[:, 2] / 2
        boxes[:, 3] = x[:, 1] + x[:, 3] / 2

        conf = x[:, 5:].max(1)
        cls = x[:, 5:].argmax(1)

        if CLASSES is not None:
            keep = np.isin(cls, CLASSES)
            boxes, conf, cls = boxes[keep], conf[keep], cls[keep]

        idxs = np.argsort(-conf)[:MAX_DET]
        boxes, conf, cls = boxes[idxs], conf[idxs], cls[idxs]

        xywh = np.column_stack((
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1]
        ))

        indices = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            conf.tolist(),
            CONF_THRES,
            IOU_THRES
        )

        if len(indices) > 0:
            for i in indices.flatten():
                detections.append(
                    np.array([*boxes[i], conf[i], cls[i]])
                )

    return np.array(detections)


# ---------------- Load ONNX ----------------
sess_opt = ort.SessionOptions()
sess_opt.intra_op_num_threads = 4
session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=sess_opt,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("✅ ONNX model loaded")

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ Camera opened")

prev_time = time.time()

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img, r, dw, dh = letterbox(frame, (IMG_SIZE, IMG_SIZE))

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0).astype(np.float32) / 255.0

    # Inference
    pred = session.run([output_name], {input_name: img})[0]

    # NMS
    detections = non_max_suppression(pred)

    # Draw
    for det in detections:
        x1, y1, x2, y2, conf, cls = det

        x1 = int((x1 - dw) / r)
        y1 = int((y1 - dh) / r)
        x2 = int((x2 - dw) / r)
        y2 = int((y2 - dh) / r)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"PERSON {conf:.2f}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # FPS
    curr = time.time()
    fps = 1.0 / (curr - prev_time)
    prev_time = curr

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("YOLOv5n ONNX Person", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ===================== CLEANUP =====================
cap.release()
cv2.destroyAllWindows()
