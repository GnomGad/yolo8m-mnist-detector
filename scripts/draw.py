import cv2
import numpy as np
import time
from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.pt")

canvas_size = 256
img = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

drawing = False
erasing = False
last_prediction_time = 0
prediction_interval = 0.5  # секунд

def draw(event, x, y, flags, param):
    global drawing, erasing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        erasing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img, (x, y), 8, 255, -1)
        elif erasing:
            cv2.circle(img, (x, y), 10, 0, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_RBUTTONUP:
        erasing = False

window_name = "Draw Digit"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, draw)

current_prediction = "?"

while True:
    display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imshow(window_name, display)
    cv2.setWindowTitle(window_name, f"{current_prediction}")

    key = cv2.waitKey(1)

    if key == ord("c"):
        img[:] = 0
    elif key == 27:  # Esc
        break

    if time.time() - last_prediction_time >= prediction_interval:
        last_prediction_time = time.time()

        digit_img = cv2.resize(img, (128, 128))
        digit_img = cv2.cvtColor(digit_img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite("input_digit.png", digit_img)

        results = model.predict(source="input_digit.png", conf=0.35, save=False, device="cuda")

        current_prediction = "?"
        for r in results:
            if r.boxes:
                best = r.boxes[0]
                cls_id = int(best.cls[0])
                conf = float(best.conf[0])
                current_prediction = f"{model.names[cls_id]} ({conf:.2f})"

cv2.destroyAllWindows()
