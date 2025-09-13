import cv2
import os
import glob
from ultralytics import YOLO

# === CẤU HÌNH ===
WEIGHTS = "runs/detect/train/weights/best.pt"   # sửa đúng đường dẫn
INPUT   = "img_input"               # file.jpg hoặc folder chứa ảnh
SAVE_DIR = "img_out"                                   # nơi lưu ảnh đã vẽ
CONF = 0.4
IOU  = 0.5
IMGSZ = 640
DEVICE = 0   # 0 = GPU, -1 = CPU

model = YOLO(WEIGHTS)

def infer_and_draw(img_bgr):
    # dự đoán 1 ảnh
    res = model(img_bgr, conf=CONF, iou=IOU, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
    # vẽ nhanh bằng hàm có sẵn
    annotated = res.plot()
    return annotated, res

def process_file(img_path, save=True, show=True):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Không đọc được ảnh: {img_path}")
        return
    annotated, _ = infer_and_draw(img)
    if save:
        os.makedirs(SAVE_DIR, exist_ok=True)
        out_path = os.path.join(SAVE_DIR, os.path.basename(img_path))
        cv2.imwrite(out_path, annotated)
        print(f"✔ Saved: {out_path}")
    if show:
        cv2.imshow("Phan loai trai cay", annotated)
        cv2.waitKey(0)

def process_dir(dir_path, save=True, show=False):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(dir_path, ext))
    if not files:
        print("[INFO] Thư mục không có ảnh hợp lệ.")
        return
    for f in files:
        process_file(f, save=save, show=show)
    if show:
        cv2.destroyAllWindows()

# === CHẠY ===
if os.path.isfile(INPUT):
    process_file(INPUT, save=True, show=True)
else:
    process_dir(INPUT, save=True, show=False)  # show=False để chạy hàng loạt nhanh