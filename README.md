# Fruit Detection Project

## 📌 Giới thiệu

Dự án này triển khai mô hình **YOLOv8** để phát hiện và phân loại các loại trái cây từ ảnh. Dự án bao gồm:

- **Huấn luyện mô hình** trên tập dữ liệu trái cây thu thập từ Roboflow.
- **Chạy inference** trực tiếp qua `app.py` (Gradio UI hoặc script).
- **Đánh giá mô hình** trên tập test để đo lường độ chính xác.

## 🗂 Cấu trúc thư mục

```bash
project/
├── program_folder/
│   ├── app.py                # Entry point để chạy ứng dụng
│   └── requirements.txt      # Danh sách thư viện Python
│
├── train_folder/
│   ├── dataset.py            # Xử lý dataset, chuẩn bị dữ liệu huấn luyện
│   ├── evaluate_test.py      # Script đánh giá mô hình
│   ├── final_cam.py          # Chạy camera để detect realtime
│   ├── final_img.py          # Chạy detect trên ảnh
│   ├── yolov8n.pt            # Trọng số mô hình YOLOv8
│   └── dataset_traicay/      # Dữ liệu huấn luyện và kiểm thử
│       ├── data.yaml         # Cấu hình dataset cho YOLO
│       ├── train/            # Ảnh huấn luyện
│       └── test/             # Ảnh kiểm thử
```

## ⚙️ Cài đặt

### Cài đặt trong môi trường ảo (tuỳ chọn)

1. **Tạo môi trường ảo (khuyến nghị)**

```bash
python -m venv venv
source venv/bin/activate  # Trên Linux/Mac
venv\Scripts\activate     # Trên Windows
```

2. **Cài đặt thư viện cần thiết**

```bash
pip install -r program_folder/requirements.txt
```

### Cài đặt và chạy trên Spyder

1. Mở **Anaconda Navigator** → cài đặt hoặc mở **Spyder IDE**.
2. Chọn kernel / environment mà bạn muốn sử dụng.
3. Đảm bảo cài đủ thư viện trong environment hiện tại:

```bash
pip install -r program_folder/requirements.txt
```

4. Mở file `app.py` trong Spyder và nhấn **Run** để khởi động giao diện.
5. Có thể mở `final_img.py`, `final_cam.py` hoặc `evaluate_test.py` và chạy trực tiếp trên Spyder để test ảnh, realtime hoặc đánh giá mô hình.

## 🚀 Cách chạy nhanh

### Chạy ứng dụng giao diện (Gradio UI)

```bash
python program_folder/app.py
```

Ứng dụng sẽ mở trên trình duyệt.

### Chạy detect trên ảnh

```bash
python train_folder/final_img.py --source path/to/image.jpg
```

### Chạy detect realtime bằng webcam

```bash
python train_folder/final_cam.py
```

### Đánh giá mô hình

```bash
python train_folder/evaluate_test.py
```

## 🧠 Mô hình

- Sử dụng **YOLOv8n** (phiên bản nhẹ, tối ưu cho tốc độ).
- Dữ liệu được cấu hình theo chuẩn YOLO trong `data.yaml`.
- Có thể huấn luyện lại bằng cách chạy `dataset.py` để chuẩn bị dữ liệu và `yolo train` để huấn luyện.

## 📊 Kết quả mong đợi

- Mô hình có thể phát hiện nhiều loại trái cây trong cùng một ảnh.
- Độ chính xác phụ thuộc chất lượng tập dữ liệu.

## 📄 Ghi chú

- Nếu muốn huấn luyện lại mô hình, đảm bảo cài **ultralytics**:

```bash
pip install ultralytics
```

- Có thể chỉnh tham số trong `data.yaml` hoặc script huấn luyện để tăng/giảm epoch.

## 👤 Tác giả

- **Nguyễn Minh Quân**
- **Hoàng Quốc Khánh**
- **Lê Hoàng Lan**
- **Triệu Yến Vi**

