import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=str, required=True, help="Path to '.../project/train_folder'")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save-samples", action="store_true", help="Also save predicted images for the test set")
    args = ap.parse_args()

    from ultralytics import YOLO

    root = Path(args.project_root).expanduser().resolve()
    weights = root / "runs" / "detect" / "train" / "weights" / "best.pt"
    data_yaml = root / "dataset_traicay" / "data.yaml"
    test_images = root / "dataset_traicay" / "test" / "images"

    assert weights.exists(), f"Missing weights: {weights}"
    assert data_yaml.exists(), f"Missing data yaml: {data_yaml}"
    assert test_images.exists(), f"Missing test images dir: {test_images}"

    model = YOLO(str(weights))

    # 1) Evaluate on test split -> will write plots (PR, F1, Confusion Matrix) under runs/detect/val*
    metrics = model.val(data=str(data_yaml), split='test', imgsz=args.imgsz, save=True, plots=True, conf=0.001, iou=0.6)

    # 2) Summarize key metrics to JSON
    summary = {
        "mAP50-95": metrics.box.map,
        "mAP50": metrics.box.map50,
        "mAP75": metrics.box.map75,
        "precision_mean": metrics.box.mp,
        "recall_mean": metrics.box.mr,
        "per_class_mAP50-95": list(metrics.box.maps) if hasattr(metrics.box, "maps") else None,
    }
    out_json = root / "runs" / "detect" / "val_summary_test.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Saved summary metrics -> {out_json}")

    # 3) Optionally run predictions on test images for visualizations
    if args.save_samples:
        preds = model.predict(source=str(test_images), imgsz=args.imgsz, conf=args.conf, save=True)
        print("[OK] Saved predicted images to:", preds[0].save_dir if preds else "<unknown>")

    # Print a neat summary
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
