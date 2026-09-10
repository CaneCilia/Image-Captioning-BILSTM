# 📸 Image Captioning System using YOLOv5 & BiLSTM

An end-to-end deep learning framework that combines computer vision object detection with natural language sequence generation. The system leverages **YOLOv5** (and **ResNet50**) for spatial feature extraction and a **Bidirectional LSTM (BiLSTM)** for generating context-aware image captions. It features a complete **FastAPI backend**, an interactive **Web Frontend**, real-time **webcam object detection**, and model training scripts.

---

## 📌 Table of Contents
- [Architecture Overview](#-architecture-overview)
- [Features](#-features)
- [Project Directory Structure](#-project-directory-structure)
- [Prerequisites & Installation](#-prerequisites--installation)
- [How to Work with the Project](#-how-to-work-with-the-project)
  - [1. Run the FastAPI Backend Server](#1-run-the-fastapi-backend-server)
  - [2. Launch the Web Frontend](#2-launch-the-web-frontend)
  - [3. Train the BiLSTM Model](#3-train-the-bilstm-model)
  - [4. Run CLI Inference](#4-run-cli-inference)
  - [5. Run Live Webcam Object Detection](#5-run-live-webcam-object-detection)
  - [6. Jupyter Notebook & Feature Extraction](#6-jupyter-notebook--feature-extraction)
- [Credits & External Acknowledgments](#-credits--external-acknowledgments)
- [License](#-license)

---

## 🏗 Architecture Overview

```
 ┌────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
 │   Input Image  │ ───► │ YOLOv5 / ResNet50 Feature│ ───► │ Projection & Embedding  │
 └────────────────┘      │        Extractor        │      └────────────┬────────────┘
                         └─────────────────────────┘                   │
                                                                       ▼
 ┌────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
 │ Generated Text │ ◄─── │  Linear Output Layer    │ ◄─── │  BiLSTM Caption Decoder │
 │    Caption     │      └─────────────────────────┘      └─────────────────────────┘
 └────────────────┘
```

1. **Object Detection & Feature Extraction:**
   - **YOLOv5 (`yolov5su.pt`):** Identifies objects, bounding boxes, and spatial coordinates from the image.
   - **ResNet50 Backbone:** Generates dense visual feature representations (2048-dimensional embeddings).
2. **Feature Projection & Sequence Modeling:**
   - Visual features are projected into an embedding space (`feature_projection`) and prepended to text embeddings.
   - A **Bidirectional LSTM (BiLSTM)** processes both forward and backward contextual information to generate meaningful descriptions.
3. **Application & API Layers:**
   - **FastAPI REST API:** Handles image uploads and serves predictions.
   - **Web UI:** Responsive client interface for uploading images and rendering generated captions.

---

## ✨ Features

- 🎯 **Real-time Object Detection:** Live webcam detection powered by YOLOv5 and OpenCV.
- 🔤 **Bidirectional Caption Generation:** BiLSTM decoder for rich context understanding.
- 🚀 **RESTful API Backend:** Fast asynchronous API built with FastAPI and Uvicorn.
- 🖥️ **Interactive Web Interface:** Clean UI built with HTML5, CSS3, and JavaScript.
- 🧪 **Notebook Development:** Jupyter notebook (`cnn_nlp_model_dev.ipynb`) for CNN-NLP visual-semantic alignment experiments.

---

## 📁 Project Directory Structure

```
Image-Captioning-BILSTM/
│
├── backend/                  # FastAPI REST API backend
│   └── main.py               # API routes (/upload-image/), CORS setup & inference runner
│
├── src/                      # Core Deep Learning source code
│   ├── bilstm_model.py       # PyTorch BiLSTMCaptioner neural network model
│   ├── dataset.py            # Dataset loader and mock dataset class
│   ├── inference.py          # Model inference script for caption generation
│   ├── live_yolo.py          # Live webcam object detection with YOLOv5 & OpenCV
│   ├── train.py              # Model training loop & checkpoint saver
│   └── yolo_feature_ext.py   # YOLOv5 feature extraction module
│
├── frontend/                 # Web User Interface
│   ├── index.html            # Webpage layout & file upload form
│   ├── styles.css            # Styling and visual design
│   └── app.js                # Client JS for calling FastAPI endpoints
│
├── models/                   # Model weight checkpoints directory (.pth)
├── data/                     # Dataset storage (images & annotations)
├── uploads/                  # Upload directory for user-submitted images
├── cnn_nlp_model_dev.ipynb   # Jupyter Notebook for CNN feature extraction & NLP
├── create_notebook.py        # Helper script to generate the development notebook
├── sample_image.jpg          # Sample test image
├── requirements.txt          # Python dependencies list
├── CREDITS.md                # Credits for external models, libraries, and developers
└── README.md                 # Project documentation
```

---

## ⚙️ Prerequisites & Installation

### 1. Requirements
- **Python:** Version `3.8` or higher installed.
- **Git** (optional, for cloning).

### 2. Environment Setup

Clone the repository (or navigate to the workspace directory):
```bash
git clone https://github.com/CaneCilia/Image-Captioning-BILSTM.git
cd Image-Captioning-BILSTM
```

Create and activate a virtual environment:

- **On Windows (PowerShell / Command Prompt):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **On Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies

Install all required packages via `requirements.txt`:
```bash
pip install -r requirements.txt
```

*(Optional)* Install `uvicorn` if not already installed for running the FastAPI web server:
```bash
pip install uvicorn
```

---

## 🚀 How to Work with the Project

### 1. Run the FastAPI Backend Server

Start the REST API backend using Uvicorn:

```bash
uvicorn backend.main:app --reload --port 8000
```
Or execute directly with Python:
```bash
python backend/main.py
```

- **API Server Address:** `http://127.0.0.1:8000`
- **Interactive Swagger Documentation:** Open `http://127.0.0.1:8000/docs` in your browser to test endpoints interactively.

---

### 2. Launch the Web Frontend

1. Ensure the FastAPI backend server is running on port 8000.
2. Open [`frontend/index.html`](file:///D:/GitDesk/Image-Captioning-BILSTM/frontend/index.html) directly in your web browser, or serve it using Python's HTTP server:
   ```bash
   python -m http.server 3000 --directory frontend
   ```
3. Visit `http://127.0.0.1:3000` in your browser.
4. Click **Select Image**, choose any image file (`.jpg`, `.jpeg`, `.png`), and click **Generate Caption**.

---

### 3. Train the BiLSTM Model

To train the captioning model using PyTorch:

```bash
python src/train.py
```

- **Configuration:** You can adjust hyperparameters (batch size, learning rate, embedding size, epochs) inside [`src/train.py`](file:///D:/GitDesk/Image-Captioning-BILSTM/src/train.py#L9-L17).
- **Model Checkpoints:** Saved automatically to `models/bilstm_captioner.pth`.

---

### 4. Run CLI Inference

To test caption generation directly from the command line:

```bash
python src/inference.py
```

This will run the [`generate_caption`](file:///D:/GitDesk/Image-Captioning-BILSTM/src/inference.py#L7) pipeline with YOLO feature extraction and the pre-trained BiLSTM weights.

---

### 5. Run Live Webcam Object Detection

To test real-time YOLO object detection with your computer's camera:

```bash
python src/live_yolo.py
```

- Press **`q`** in the video display window to exit the stream.

---

### 6. Jupyter Notebook & Feature Extraction

For interactive model exploration and CNN-NLP alignment experiments:

1. Open [`cnn_nlp_model_dev.ipynb`](file:///D:/GitDesk/Image-Captioning-BILSTM/cnn_nlp_model_dev.ipynb) in Jupyter Notebook or VS Code / Antigravity IDE.
2. If you need to regenerate the notebook script:
   ```bash
   python create_notebook.py
   ```

---

## 📜 Credits & External Acknowledgments

This project incorporates and credits open-source deep learning architectures, research papers, and open-source software libraries:

- **YOLOv5 (Ultralytics):** Real-time object detection architecture by Glenn Jocher et al.
- **ResNet-50:** Deep Residual Learning by Kaiming He et al. (Microsoft Research).
- **BiLSTM Architecture:** Bidirectional Recurrent Neural Networks by Alex Graves et al. & Hochreiter & Schmidhuber.
- **PyTorch & torchvision:** Core deep learning framework by Meta AI and contributors.
- **FastAPI & Uvicorn:** Web API framework by Sebastián Ramírez (`tiangolo`).
- **OpenCV & NLTK:** Computer vision & Natural Language Processing suites.

For complete attribution details, links, licenses, and references, please view the dedicated [`CREDITS.md`](file:///D:/GitDesk/Image-Captioning-BILSTM/CREDITS.md) file.

---

## 📄 License

This project is open-source under the MIT License. See individual library licenses in [`CREDITS.md`](file:///D:/GitDesk/Image-Captioning-BILSTM/CREDITS.md) for third-party component terms.
