import os
os.environ["GRADIO_USE_BROTLI"] = "false"
import cv2
import numpy as np
import gradio as gr
import torch

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Ultralytics is required. Try: pip install ultralytics") from e

DEFAULT_WEIGHTS = r"..\train_folder\runs\detect\train\weights\best.pt"
FALLBACK_WEIGHTS = r"..\train_folder\yolov8n.pt"

MODEL_PATH = DEFAULT_WEIGHTS if os.path.exists(DEFAULT_WEIGHTS) else FALLBACK_WEIGHTS

model = YOLO(MODEL_PATH)

DEFAULT_CONF = 0.35
DEFAULT_IOU  = 0.5
DEFAULT_IMGSZ = 960
DEFAULT_DEVICE_UI = 0 if torch.cuda.is_available() else -1

def _to_device(d):
    try:
        d = int(d)
    except Exception:
        return "cpu"
    return "cpu" if d == -1 else d

def _predict_and_draw(frame_bgr, conf=DEFAULT_CONF, iou=DEFAULT_IOU, imgsz=DEFAULT_IMGSZ, device=DEFAULT_DEVICE_UI):
    res = model(frame_bgr, conf=conf, iou=iou, imgsz=imgsz, device=_to_device(device), verbose=False)[0]
    annotated = res.plot()
    return annotated, res

def detect_on_image(image, conf, iou, imgsz, device, return_json):
    if image is None:
        return None, None
    if isinstance(image, np.ndarray):
        img_rgb = image
    else:
        img_rgb = np.array(image)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    annotated_bgr, res = _predict_and_draw(img_bgr, conf, iou, imgsz, device)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

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
    if frame is None:
        return None
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    annotated_bgr, _ = _predict_and_draw(frame_bgr, conf, iou, imgsz, device)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb

with gr.Blocks(theme=gr.themes.Soft(), css="footer, .svelte-1ipelgc {display:none !important;}") as demo:
    gr.Markdown(
        """
        # 🍊 Fruit Detection — YOLOv8
        A clean, fast interface supporting **image** and **camera** input. You can adjust confidence, IoU, and inference image size.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            conf = gr.Slider(0.0, 1.0, value=DEFAULT_CONF, step=0.05, label="Confidence")
            iou  = gr.Slider(0.0, 1.0, value=DEFAULT_IOU, step=0.05, label="IoU")
            imgsz = gr.Slider(320, 1280, value=DEFAULT_IMGSZ, step=32, label="Image size (imgsz)")
            device = gr.Radio(choices=[-1, 0], value=DEFAULT_DEVICE_UI, label="Device (CPU=-1, GPU=0)")
        with gr.Column(scale=1):
            model_info = gr.Json(
                value={
                    "using_model": MODEL_PATH,
                    "classes": list(getattr(model, "names", {}).values())
                },
                label="Model Information",
            )

    with gr.Tabs():
        with gr.Tab("🖼️  Image Detection"):
            with gr.Row():
                in_img = gr.Image(type="numpy", label="Upload Image Here", sources=["upload", "clipboard"])

            btn_run = gr.Button("Detect")

            with gr.Row():
                out_img = gr.Image(label="Result", interactive=False)
                out_json = gr.Json(label="Result (JSON)", visible=False)
            return_json = gr.Checkbox(value=False, label="Return JSON Result")

            btn_run.click(
                fn=detect_on_image,
                inputs=[in_img, conf, iou, imgsz, device, return_json],
                outputs=[out_img, out_json],
            )

            def _toggle_json(show): 
                return gr.update(visible=show)
            return_json.change(_toggle_json, inputs=return_json, outputs=out_json)

        with gr.Tab("📷  Real-time Camera Detection"):
            cam = gr.Image(
                sources="webcam",
                streaming=True,
                type="numpy",
                label="Camera"
            )
            cam_out = gr.Image(label="Real-time Result", streaming=True)

            cam.stream(
                fn=detect_on_camera,
                inputs=[cam, conf, iou, imgsz, device],
                outputs=cam_out
            )

    gr.Markdown(
        "⚙️ <small>Model: <code>{}</code>. You can replace it with another file if you want.</small>".format(MODEL_PATH)
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)
