from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8m.pt")
    model.train(
    data="mnist.yaml",
    epochs=10,
    imgsz=128,
    batch=256,
    workers=4,
    mosaic=0.0,
    mixup=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    flipud=0.0,
    fliplr=0.0,
    translate=0.0,
    shear=0.0,
    perspective=0.0,
    erasing=0.0,
    auto_augment=None
    )

