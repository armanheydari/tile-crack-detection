# Tile Crack Detection

> A computer vision research project exploring multiple approaches — Siamese networks, transfer learning, U-Net segmentation, and YOLOv5 — for detecting surface cracks in ceramic tile images using a pattern-matching preprocessing pipeline.

[![Python Version](https://img.shields.io/badge/python-3.x%2B-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

## 📌 Overview

Tile crack detection is a non-trivial visual inspection problem: a cracked tile pixel looks nearly identical to a tile texture pixel unless you know what the intact tile *should* look like. This project tackles that challenge by building a preprocessing pipeline that extracts the tile region from its background, histogram-matches it against a reference pattern image, and aligns the pattern to the correct rotation — before any model sees the data.

Six notebooks document the evolution across four distinct modeling strategies, each building on lessons from the previous. The preprocessing pipeline is shared across all approaches; what differs is how tile patches and their reference patterns are finally classified. A structured CV report (`CV_report.pdf`) documents results and analysis.

## 🧪 Approaches

All notebooks share the same image preprocessing pipeline:
1. **Otsu thresholding** → binary mask of the tile region
2. **Morphological open/close** (35×35 kernel) → noise removal
3. **Contour-based crop** → isolates the tile from the background
4. **Histogram matching** (`skimage.exposure.match_histograms`) → normalizes the tile image to the reference pattern's color distribution
5. **Rotation alignment** (via `imagehash.average_hash`) → finds the best of 4 rotations (0°, 90°, 180°, 270°) for the pattern
6. **Sliding window** (120×120 patches) → generates tile/pattern patch pairs with Shapely-based polygon intersection for crack labels

---

### `Detect_Cracks.ipynb` — Siamese Network (baseline)

The first approach. A lightweight custom CNN encoder (Conv2D → AveragePooling2D stacks with `tanh` activation) is used as the twin branch in a Siamese network. The two branches process a tile patch and its corresponding pattern patch independently; Euclidean distance between embeddings is the similarity score. Training uses **contrastive loss** (margin=1) with RMSprop, early stopping on val F1, and class weighting proportional to the crack/non-crack imbalance.

---

### `Detect_CracksV1.ipynb` — Siamese + Saved Dataset

Extends the baseline by adding a `save_image()` function that persists the preprocessed tile/pattern pairs to a `Customized_Dataset/` directory. This separates preprocessing from training and enables reuse across runs. The Siamese architecture and contrastive loss remain the same; the V1 also explores adding a ResNet50 (ImageNet-pretrained) backbone as an alternative encoder within the same notebook.

---

### `Detect_CracksV2.ipynb` — Custom Histogram Equalization

Replaces `skimage`'s histogram matching with a **custom-implemented histogram equalization** function (manual CDF computation, pixel remapping). The equalization is applied per-channel before matching. The rest of the pipeline — Siamese architecture, contrastive loss, sliding window — is unchanged. This was an experiment to see whether a more controlled normalization would improve patch alignment.

---

### `Detect_Cracks_PreTreined.ipynb` — Siamese + ResNet50 Backbone (Google Colab)

The transfer learning approach. The Siamese encoder is replaced with **ResNet50** (ImageNet weights, `include_top=False`) followed by Dense(512) → BatchNorm → Dense(256) → BatchNorm → Dense(256). The dataset is loaded from Google Drive via `gdown`. Class weight for positive (crack) samples is set to a fixed 50 to aggressively counteract class imbalance. Otherwise the training loop (contrastive loss, RMSprop, early stopping on val F1) is identical to prior notebooks.

---

### `Detect_Cracks-Unet.ipynb` — U-Net Segmentation

Pivots from patch classification to pixel-level segmentation. Preprocessing here uses **adaptive thresholding** (`cv.adaptiveThreshold` with `ADAPTIVE_THRESH_MEAN_C`) and custom histogram equalization before feeding images to a U-Net architecture. The model is evaluated with the same precision/recall/F1 custom metrics. This approach treats crack detection as a semantic segmentation task rather than a binary classification of sliding-window patches.

---

### `Yolo.ipynb` — YOLOv5 Label Conversion

A utility notebook that converts the annotated dataset into YOLO format. It reads the Shapely polygon annotations from JSON files, computes axis-aligned bounding boxes (center_x, center_y, width, height — normalized by image dimensions), and writes `.txt` label files compatible with YOLOv5 training. The converted dataset is written to a `Customized_Dataset/yolo/` directory for use with an external YOLOv5 training run.

---

## 🛠️ Tech Stack

* **Core Language:** Python 3
* **Deep Learning:** TensorFlow / Keras
* **Pretrained Models:** ResNet50 (ImageNet weights via `tensorflow.keras.applications.resnet`)
* **Computer Vision:** OpenCV (`cv2`), scikit-image (`skimage.exposure`)
* **Image Hashing:** `imagehash` (perceptual hash for rotation alignment)
* **Geometry:** `shapely` (polygon intersection for sliding-window labeling)
* **Object Detection:** YOLOv5 (external training; `Yolo.ipynb` handles label conversion)
* **Environment:** Jupyter Notebook; `Detect_Cracks_PreTreined.ipynb` targets Google Colab

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow opencv-python scikit-image imagehash shapely Pillow matplotlib
```

For the Colab notebook, dataset files are downloaded automatically via `gdown`.

### Installation

```bash
git clone https://github.com/armanheydari/Detect-Tile-Cracks.git
cd Detect-Tile-Cracks
```

Place your tile images in a `Dataset/` folder and corresponding pattern reference images in `Patterns/`. Each image should have a matching `.json` annotation file (LabelMe format) specifying the crack polygon and the reference pattern filename.

## 💻 Usage

Open any notebook in Jupyter and run all cells:

```bash
jupyter notebook Detect_Cracks_PreTreined.ipynb  # Transfer learning approach (recommended)
jupyter notebook Detect_Cracks-Unet.ipynb         # Segmentation approach
```

All notebooks produce:
- Trained Siamese/segmentation model
- Training curves (accuracy, contrastive loss) plotted via `plt_metric()`
- Patch-level predictions visualized with `visualize()`

## 📁 Project Structure

```
Detect-Tile-Cracks/
├── Detect_Cracks.ipynb            # Approach 1: Siamese network (custom CNN encoder)
├── Detect_CracksV1.ipynb          # Approach 2: Siamese + saved preprocessed dataset
├── Detect_CracksV2.ipynb          # Approach 3: Siamese + custom histogram equalization
├── Detect_Cracks_PreTreined.ipynb # Approach 4: Siamese + ResNet50 backbone (Colab)
├── Detect_Cracks-Unet.ipynb       # Approach 5: U-Net pixel-level segmentation
├── Yolo.ipynb                     # Approach 6: YOLOv5 label format conversion
└── CV_report.pdf                  # Full project report with results and analysis
```
