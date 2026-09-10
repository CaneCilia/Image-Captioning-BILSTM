# Credits & Acknowledgments

This project, **Image-Captioning-BILSTM**, relies on and builds upon several incredible open-source projects, machine learning models, research contributions, and software libraries. We gratefully acknowledge the creators, developers, and researchers behind these technologies.

---

## 🤖 Deep Learning Models & Architectures

### 1. YOLOv5 / YOLO Series
* **Creators / Maintainers:** [Ultralytics](https://github.com/ultralytics/ultralytics) (Glenn Jocher et al.) & Joseph Redmon et al.
* **Usage in Project:** Real-time object detection and spatial feature extraction (`yolov5su.pt` / `src/yolo_feature_ext.py` & `src/live_yolo.py`).
* **Paper / Reference:** *Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection"* & Ultralytics YOLOv5 Architecture.

### 2. ResNet-50 (Convolutional Neural Network)
* **Creators:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research).
* **Usage in Project:** Deep CNN feature extraction backbone for visual features (`torchvision.models.resnet50` in `cnn_nlp_model_dev.ipynb`).
* **Paper:** *"Deep Residual Learning for Image Recognition"* (CVPR 2016).

### 3. Bidirectional Long Short-Term Memory (BiLSTM)
* **Creators:** Sepp Hochreiter, Jürgen Schmidhuber, and Alex Graves et al.
* **Usage in Project:** Sequence-to-sequence caption generation using forward and backward context (`src/bilstm_model.py`).
* **Paper:** *"Framewise Phoneme Classification with Bidirectional LSTM and Other Neural Network Architectures"* (2005).

---

## 🛠️ Open-Source Frameworks & Libraries

| Framework / Library | Primary Role | License / Source |
| :--- | :--- | :--- |
| **PyTorch** (`torch`, `torch.nn`) | Tensor computations, deep learning training loop, and neural network layers | BSD License / Meta AI ([pytorch.org](https://pytorch.org/)) |
| **torchvision** | Pre-trained vision backbones and image transformations | BSD License / PyTorch Team |
| **Ultralytics** | YOLO model loading, inference, and visualization utilities | AGPL-3.0 / Enterprise ([ultralytics.com](https://ultralytics.com)) |
| **FastAPI** | High-performance Python backend REST API server | MIT License / Sebastián Ramírez ([fastapi.tiangolo.com](https://fastapi.tiangolo.com/)) |
| **Uvicorn** | Lightning-fast ASGI server implementation | BSD License |
| **OpenCV** (`cv2`) | Real-time webcam video stream processing and window display | Apache 2.0 / OpenCV team ([opencv.org](https://opencv.org/)) |
| **NLTK** | Natural Language Processing tools (tokenizer, POS tagger) | Apache 2.0 / NLTK Team ([nltk.org](https://www.nltk.org/)) |
| **Pillow (PIL)** | Image loading and preprocessing | HPND License |
| **NumPy & Pandas** | Array manipulations and data structure management | BSD License |
| **Matplotlib** | Data visualization and image plotting | BSD License |

---

## 🌐 Frontend & User Interface

* **HTML5 / CSS3 / JavaScript (ES6+):** Responsive web interface constructed with pure vanilla web technologies for image upload and live caption rendering.
* **Fetch API:** Asynchronous client-backend HTTP communication.

---

## 📊 Datasets & Reference Corpora

* **COCO Dataset (Common Objects in Context):** Benchmark dataset for object detection and image captioning.
* **Flickr8k / Flickr30k:** Reference image captioning datasets used for vocabulary and caption training setup.

---

## 👥 Project Team & Developers

* **Repository Maintainers & Developers:** `CaneCilia / Image-Captioning-BILSTM` team.
* **Special Thanks:** Open-source AI/ML community and all contributors who maintain the tools used in this ecosystem.
