from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='dataset_fruits/data.yaml', 
    epochs=100,
    patience=20,
    imgsz=640,
    batch=8,
    cache=False,
    device=0,
    workers=0,
    deterministic=True
)
