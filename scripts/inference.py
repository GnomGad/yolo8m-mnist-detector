from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.pt")
results = model("input_digit.png")
results[0].show()
