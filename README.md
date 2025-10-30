# 🍎 Fruit Detection (YOLOv8)

> Real-time fruit detection powered by **YOLOv8**, using a custom fruit dataset split into train/val/test.  
> Built for quick deployment and clean reproducibility.

---

## 🚀 Features
- 🧠 **YOLOv8n** model trained on a custom fruit dataset from **Roboflow**  
- 🧩 **Custom Split:** dataset manually divided into **train**, **val**, and **test** sets  
- 🎯 **Pretrained Weights:** runs directly using `best.pt` without retraining  
- 💻 **Simple Interface:** just run one Python file - no complex setup needed  

---

## 🍉 Dataset Overview

The dataset contains **6 fruit classes** used for object detection:

| Class | Description |
|:------|:-------------|
| 🍍 **Pineapple** | Tropical fruit with spiky skin and sweet yellow flesh. |
| 🍒 **Cherry** | Small red fruit often appearing in pairs. |
| 🥭 **Mango** | Yellow-orange fruit with smooth skin and sweet aroma. |
| 🍑 **Plum** | Round fruit with smooth skin, purple or red when ripe. |
| 🍅 **Tomato** | Red juicy fruit often mistaken for a vegetable. |
| 🍉 **Watermelon** | Large green fruit with red interior and black seeds. |

---

## 🗂 Project Structure
```bash
Fruits-Detection/
├── program.py               # Main entry point for running detection
├── requirements.txt         # Python dependencies
├── LICENSE
├── SECURITY.md
│
├── weights/
│   ├── best.pt              # Trained YOLOv8 model weights
│   └── last.pt
│
└── dataset_fruits/
    ├── data.yaml            # Dataset configuration for YOLOv8
    ├── README.dataset.md    # Info on dataset source & how val set was created
    ├── README.roboflow.txt  # Original Roboflow export metadata (source information)
    ├── train/               # Training images & labels
    ├── valid/               # Validation images & labels
    └── test/                # Test images & labels
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/quan831/Fruit-Detection.git
cd Fruit-Detection
```

### 2️⃣ (Optional) Create virtual environment
```bash
python -m venv venv
# Activate:
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run Detection

### 🧩 Option 1 — Detect via Script
```bash
python program.py
```

Make sure your working directory includes:
- `weights/best.pt`
- `dataset_fruits/data.yaml`

The program loads the YOLOv8 model and runs inference directly.

### 🧠 Option 2 — Run in Spyder (Recommended for GUI)
1. Open **Anaconda Navigator** → Launch **Spyder**  
2. Open `program.py`  
3. Hit **Run (F5)** to start detection  
4. Check outputs and logs inside the console or generated output folder (if any)

---

## 🧠 Model Details
- **Model:** `best.pt` (trained YOLOv8n)  
- **Framework:** Ultralytics YOLOv8 (Python)  
- **Dataset:** Custom split version of Roboflow fruit dataset  
- **Train/Val/Test Ratio:** defined manually in `README.dataset.md`  

---

## 🧾 License
This project is licensed under the [MIT License](./LICENSE).

---

## 🛡 Security
See [SECURITY.md](./SECURITY.md) for details on responsible disclosure.

---

## 💖 Credits
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- [Roboflow](https://roboflow.com) for dataset hosting  
- **Original dataset:** [nhận diện trái cây v2 Computer Vision Dataset](https://universe.roboflow.com/hcmus-sbpod/nhan-dien-trai-cay-v2)
- Custom dataset split and model tuning by **Quan (James)**

---

## 🌟 Show Your Support
If this repo helps you, please give it a ⭐ on GitHub - it really motivates me!

---

## 👤 Authors

- **Nguyễn Minh Quân (Leader)**
- **Hoàng Quốc Khánh**
- **Lê Hoàng Lan**
- **Triệu Yến Vi**