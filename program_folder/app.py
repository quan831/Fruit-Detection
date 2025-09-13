import os
os.environ["GRADIO_USE_BROTLI"] = "false"
import cv2
import numpy as np
import gradio as gr
import torch

# Ultralytics YOLO
try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Ultralytics is required. Try: pip install ultralytics") from e

# ---- Model loading ----
# Prefer the trained model in your project if it exists; otherwise fallback to a generic YOLOv8n.
DEFAULT_WEIGHTS = r"..\train_folder\runs\detect\train\weights\best.pt"
FALLBACK_WEIGHTS = r"..\train_folder\yolov8n.pt"

MODEL_PATH = DEFAULT_WEIGHTS if os.path.exists(DEFAULT_WEIGHTS) else FALLBACK_WEIGHTS

model = YOLO(MODEL_PATH)

# Global default params (can be overridden by UI controls)
DEFAULT_CONF = 0.35
DEFAULT_IOU  = 0.5
DEFAULT_IMGSZ = 960
DEFAULT_DEVICE_UI = 0 if torch.cuda.is_available() else -1 # 0 = GPU, -1 = CPU

def _to_device(d):
    # Radio trả về -1/0 -> chuyển sang 'cpu' hoặc 0
    try:
        d = int(d)
    except Exception:
        return "cpu"
    return "cpu" if d == -1 else d

def _predict_and_draw(frame_bgr, conf=DEFAULT_CONF, iou=DEFAULT_IOU, imgsz=DEFAULT_IMGSZ, device=DEFAULT_DEVICE_UI):
    """
    Run YOLO on a single BGR frame (numpy array) and return the annotated BGR image.
    """
    res = model(frame_bgr, conf=conf, iou=iou, imgsz=imgsz, device=_to_device(device), verbose=False)[0]
    annotated = res.plot()  # returns BGR
    return annotated, res

def detect_on_image(image, conf, iou, imgsz, device, return_json):
    """
    image: PIL.Image or numpy array (RGB) from Gradio
    Returns: annotated image (RGB) and optionally a JSON of detections.
    """
    if image is None:
        return None, None

    # Ensure numpy array in BGR for OpenCV/Ultralytics
    if isinstance(image, np.ndarray):
        img_rgb = image
    else:
        img_rgb = np.array(image)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    annotated_bgr, res = _predict_and_draw(img_bgr, conf, iou, imgsz, device)

    # Convert back to RGB for display
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Prepare lightweight JSON of boxes
    det_json = None
    if return_json:
        dets = []
        names = res.names if hasattr(res, "names") else getattr(model, "names", {})
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            cls_id = int(b.cls[0])
            conf_v = float(b.conf[0])
            dets.append({
                "bbox_xyxy": [x1, y1, x2, y2],
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "confidence": round(conf_v, 4),
            })
        det_json = {"model": os.path.basename(MODEL_PATH), "num_detections": len(dets), "detections": dets}

    return annotated_rgb, det_json

def detect_on_camera(frame, conf, iou, imgsz, device):
    """
    frame: single RGB frame from webcam (numpy array).
    Returns annotated RGB frame.
    """
    if frame is None:
        return None
    # Convert RGB -> BGR, run, then back to RGB
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    annotated_bgr, _ = _predict_and_draw(frame_bgr, conf, iou, imgsz, device)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb

with gr.Blocks(theme=gr.themes.Soft(), css="footer, .svelte-1ipelgc {display:none !important;}") as demo:
    gr.Markdown(
        """
        # 🍊 Nhận Diện Trái Cây — YOLOv8
        Giao diện đẹp, nhanh, hỗ trợ **ảnh** và **camera**. Bạn có thể điều chỉnh ngưỡng tự tin (confidence), IOU và kích thước ảnh suy luận.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            conf = gr.Slider(0.0, 1.0, value=DEFAULT_CONF, step=0.05, label="Confidence")
            iou  = gr.Slider(0.0, 1.0, value=DEFAULT_IOU, step=0.05, label="IoU")
            imgsz = gr.Slider(320, 1280, value=DEFAULT_IMGSZ, step=32, label="Kích thước suy luận (imgsz)")
            device = gr.Radio(choices=[-1, 0], value=DEFAULT_DEVICE_UI, label="Thiết bị (CPU=-1, GPU=0)")
        with gr.Column(scale=1):
            model_info = gr.Json(
                value={
                    "using_model": MODEL_PATH,
                    "classes": list(getattr(model, "names", {}).values())
                },
                label="Thông tin model"
            )

    with gr.Tabs():
        with gr.Tab("🖼️  Nhận diện bằng Ảnh"):
            with gr.Row():
                in_img = gr.Image(type="numpy", label="Tải ảnh vào đây", sources=["upload", "clipboard"])

            btn_run = gr.Button("Phát hiện")
            
            with gr.Row():
                out_img = gr.Image(label="Kết quả", interactive=False)
                out_json = gr.Json(label="Kết quả (JSON)", visible=False)
            return_json = gr.Checkbox(value=False, label="Xuất JSON kết quả")

            btn_run.click(
                fn=detect_on_image,
                inputs=[in_img, conf, iou, imgsz, device, return_json],
                outputs=[out_img, out_json],
            )

            # Toggle JSON visibility
            def _toggle_json(show): 
                return gr.update(visible=show)
            return_json.change(_toggle_json, inputs=return_json, outputs=out_json)

        with gr.Tab("📷  Nhận diện bằng Camera (Real-time)"):
            # Fallback tương thích nhiều phiên bản Gradio
            cam = gr.Image(
                sources="webcam",   # lấy từ webcam
                streaming=True,     # stream liên tục
                type="numpy",       # trả về numpy RGB, đúng với detect_on_camera
                label="Camera"
            )
            cam_out = gr.Image(label="Kết quả real-time", streaming=True)
        
            # Gradio v4: dùng .stream (không truyền concurrency_count/stream_every)
            cam.stream(
                fn=detect_on_camera,
                inputs=[cam, conf, iou, imgsz, device],
                outputs=cam_out
            )

    gr.Markdown(
        "⚙️ <small>Model: <code>{}</code>. Bạn có thể thay bằng file khác nếu muốn.</small>".format(MODEL_PATH)
    )

if __name__ == "__main__":
    # Launch with sharing disabled by default; set share=True if you need a public URL (careful with credentials).
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)
