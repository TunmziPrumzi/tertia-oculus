import cv2
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Classes to keep: person, backpack, handbag, suitcase
TARGET_CLASSES = [0, 24, 26, 28, 43]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

WINDOW_NAME = "Tertia Oculus - Live Detection"
cv2.namedWindow(WINDOW_NAME)

prev_time = time.time()
fps_smooth = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        results = model.track(frame, classes=TARGET_CLASSES, persist=True, verbose=False)
        annotated_frame = results[0].plot()

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        fps_smooth = fps_smooth * 0.9 + fps * 0.1

        cv2.putText(
            annotated_frame,
            f"FPS: {fps_smooth:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow(WINDOW_NAME, annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        # Also exit if the window was closed via the X button
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()