# test_person_detector.py
import cv2
import time
from person_detector import PersonDetector

def main():
    # ========== CONFIG ==========
    MODEL_PATH = "yolov5n_quant.onnx"  # chỉnh nếu cần
    CAMERA_ID = 0                      # a0 = webcam / USB cam
    IMG_SIZE = 320

    # ========== INIT ==========
    detector = PersonDetector(
        model_path=MODEL_PATH,
        img_size=IMG_SIZE,
        conf_thres=0.5,
        iou_thres=0.45
    )

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("❌ Không mở được camera")
        return

    prev_time = time.time()

    print("▶ Nhấn Q để thoát")

    # ========== LOOP ==========
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không đọc được frame")
            break

        detections = detector.detect(frame)

        # ---- VẼ KẾT QUẢ ----
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]

            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )
            cv2.putText(
                frame,
                f"Person {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # ---- FPS ----
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(
            frame,
            f"Persons: {len(detections)} | FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow("Person Detector Test", frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
