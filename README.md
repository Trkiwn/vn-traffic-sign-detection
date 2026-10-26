# Vietnamese Traffic Sign Detection

Final project by Group 5 for the Introduction to Computer Science course (CSE2025, G1) at Vietnamese-German University (VGU).

## Members:
Nguyen Trung Kien - 10425148

Nguyen Khoa Lam - 10425102

Tran Van Gia Khang - 10425146


## Motivation and goal

Traffic signs communicate important rules, warnings, and directions. We are interested in finding which object-detection models work best on Vietnamese traffic signs, with the longer-term aim of applying a suitable model in a real-time system.

To compare detection performance, we plan to train YOLO11n, RT-DETR, and Faster R-CNN on the same dataset and evaluate them using the same metrics. This comparison will help us understand the trade-offs between detection quality and suitability for real-time use. The project is exploratory and is not intended for safety-critical or real-world driving use.

## Dataset

The project uses the [Vietnamese Traffic Signs dataset on Kaggle](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs). It contains 52 sign categories. The preprocessing pipeline creates a reproducible validation split from the training data while retaining the provided test split.

The prepared dataset includes:

- Images divided into training, validation, and test splits.
- COCO annotations in `dataset/model_ready/annotations/`.
- YOLO-format labels in `dataset/model_ready/labels/` and a configuration file at `dataset/model_ready/data.yaml`.

## Current project status

- Dataset download and preparation scripts are in place.
- The planned model comparison includes YOLO11n, RT-DETR, and Faster R-CNN. The model, training, and prediction files exist under `DETR/scripts/models/`, but are currently empty; training and inference commands and model results are not yet available.

## Setup

Use Python 3.11.17, then install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

For an NVIDIA GPU using CUDA 13.2, install the matching PyTorch wheels separately:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

## Prepare the dataset

Run these commands from the repository root:

```bash
python DETR/scripts/data/import_dataset.py
python DETR/scripts/preprocess/preprocessing.py
```

The first command downloads the Kaggle dataset into `dataset/`. The second validates the source data and creates `dataset/model_ready/` with COCO annotations and YOLO labels.

Preprocessing stops rather than overwriting an existing `dataset/model_ready/` directory. Move or rename that directory before running preprocessing again.

## Repository layout

```text
DETR/
  scripts/
    data/import_dataset.py       # Download and copy the Kaggle dataset
    preprocess/preprocessing.py  # Validate and prepare COCO and YOLO data
    models/                      # Model, training, and prediction entry points
dataset/
  archive/                       # Downloaded source dataset
  model_ready/                   # Prepared images, annotations, labels, and data.yaml
requirements.txt
```

## Dependencies

See [`requirements.txt`](./requirements.txt) for the Python dependencies. GPU-specific PyTorch installation may vary by CUDA version and platform.
