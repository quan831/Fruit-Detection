# Fruit Detection Project

## 📌 Overview

This project implements a **YOLOv8** model to detect and classify fruits from images or camera. It provides a full pipeline including:

- **Model Training** - Train YOLOv8 on a fruit dataset collected from Roboflow.
- **Inference** - Run predictions via `app.py` (using Gradio UI or CLI).
- **Model Evaluation** - Test and measure accuracy on the provided test set.

## 🗂 Project Structure

```bash
Fruits-Detection/
├── program_folder/
│   ├── app.py                   # Entry point for running the application
│   └── requirements.txt         # Python dependencies
│
├── train_folder/
│   ├── dataset.py               # Dataset preprocessing and preparation
│   ├── evaluate_test.py         # Model evaluation script
│   ├── final_cam.py             # Real-time detection using webcam
│   ├── final_img.py             # Run detection on images
│   └── dataset_fruits/          # Training & test dataset
│       ├── data.yaml            # YOLO dataset configuration file
│       ├── train/               # Training images
│       ├── valid/               # Validating images
│       ├── test/                # Testing images
│       ├── README.dataset.md    # Documentation for the modified dataset (license, structure, attribution)
│       └── README.roboflow.txt  # Original Roboflow export metadata (source information)
```

## ⚙️ Installation

### Install in Virtual Environment (Optional)

1. **Create a Virtual Environment (Recommend)**

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows
```

2. **Install Dependencies**

```bash
pip install -r program_folder/requirements.txt
```

### Installation and Running on Spyder

1. Open **Anaconda Navigator** → install or launch **Spyder IDE**.
2. Select the kernel/environment you want to use.
3. Make sure all dependencies are installed in the current environment:

```bash
pip install -r program_folder/requirements.txt
```

4. Open the `app.py` file in Spyder and click **Run** to start the UI.
5. You can also open `final_img.py`, `final_cam.py` or `evaluate_test.py` and run them directly in Spyder to test images, run real-time detection, or evaluate the model.

## 🚀 Quick Start

### Run the Application (Gradio UI)

```bash
python program_folder/app.py
```

The application will open in your browser.

### Run Detection on an Image

```bash
python train_folder/final_img.py --source path/to/image.jpg
```

### Run Real-time Detection with Webcam

```bash
python train_folder/final_cam.py
```

### Evaluate the Model

```bash
python train_folder/evaluate_test.py
```

## 🧠 Model

- Uses **YOLOv8n** (lightweight version optimized for speed).
- Dataset is configured in YOLO format via `data.yaml`.
- You can retrain the model by running `dataset.py` to prepare data and using `yolo train` to start training.

## 📊 Expected Results

- The model can detect multiple types of fruits in a single image.
- Accuracy depends on the quality of the dataset.

## 📄 Notes

- To retrain the model, ensure **ultralytics** is installed:

```bash
pip install ultralytics
```

- You can adjust parameters in `data.yaml` or the training script to increase/decrease the number of epochs.

## 👤 Authors

- **Nguyễn Minh Quân**
- **Hoàng Quốc Khánh**
- **Lê Hoàng Lan**
- **Triệu Yến Vi**