from ultralytics import YOLO

# Load mô hình gốc
model = YOLO('yolov8n.pt')  # nhẹ, nhanh

# Huấn luyện với dataset trái cây
model.train(
    data='dataset_traicay/data.yaml', 
    epochs=100,
    patience=20,
    imgsz=640,
    batch=8,
    cache=False,
    device=0,
    workers=0,
    deterministic=True
)
