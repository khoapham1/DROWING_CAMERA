# test_drowning_system.py
import cv2
from drowing_risk import DrowningRiskDetector

def test_with_video(video_path=0):  # 0 cho webcam
    detector = DrowningRiskDetector()
    
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = detector.process_frame(frame)
        frame = detector.draw_results(frame, results)
        
        cv2.imshow("Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_with_video()  # Hoặc test_with_video("test_video.mp4")
    test_with_video()  # Hoặc test_with_video("test_video.mp4")