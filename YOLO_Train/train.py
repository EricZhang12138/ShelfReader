from ultralytics import YOLO #type: ignore

model = YOLO('yolov8m.pt')  # Start from pretrained

model.train(
    data='path/to/data.yaml',  # Your dataset config
    epochs=100,
    imgsz=640,
    batch=16
)