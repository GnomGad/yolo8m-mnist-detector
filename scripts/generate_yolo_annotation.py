import os

def generate_yolo_annotation(image_dir, label_dir, box_size=0.9):
    os.makedirs(label_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if not filename.endswith(".png"):
            continue

        class_id = int(filename.split("_")[0])

        x_center = 0.5
        y_center = 0.5
        width = box_size
        height = box_size

        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_filename)

        with open(label_path, "w") as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        print(f"add: {label_filename}")


if __name__ == "__main__":
    generate_yolo_annotation("mnist/images/train", "mnist/labels/train")
    generate_yolo_annotation("mnist/images/test", "mnist/labels/test")
