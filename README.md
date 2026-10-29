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
- Hugging Face RT-DETR fine-tuning and COCO evaluation are complete.
- YOLO11n and Faster R-CNN training pipelines and comparative results are pending.

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

## Fine-tune RT-DETR

After preparing the dataset and installing dependencies, run:

```bash
python DETR/scripts/models/train.py
```

The Hugging Face training pipeline uses a reproducible 50-epoch, 640-pixel configuration and reads the COCO JSON annotations. It selects the best checkpoint by validation loss and saves the selected model to `outputs/rtdetr_r50vd_comparison_50ep/`. For a clean comparison run, do not set `RESUME_CHECKPOINT`; set it explicitly only when continuing a run created with the same training arguments. On Kaggle with two GPUs, launch the same script with:

```bash
torchrun --standalone --nproc_per_node=2 DETR/scripts/models/train.py
```

The `OUTPUT_DIR` and `PROJECT_ROOT` environment variables can override the default paths. The evaluator accepts either the training output directory or one checkpoint directory. When given the training output directory, it automatically selects the checkpoint with the lowest recorded validation loss. To evaluate the selected checkpoint on validation, run:

```bash
python DETR/scripts/evaluate/evaluation.py --split validation
```

After selecting the model using validation results, evaluate it on the held-out test split:

```bash
python DETR/scripts/evaluate/evaluation.py --split test
```

The evaluator reports COCO bbox mAP@[.50:.95], mAP@.50, mAP@.75, and average recall. Use the same selected checkpoint for both commands; reserve the test split for final reporting. Metrics and raw predictions are saved under the selected checkpoint's `evaluation/` directory. To evaluate a specific checkpoint, pass it explicitly:

```bash
python DETR/scripts/evaluate/evaluation.py \
  --checkpoint outputs/rtdetr_r50vd_comparison_50ep/checkpoint-4320 \
  --split test
```

For Kaggle, set `PROJECT_ROOT` to the repository path before running the commands. Use the same evaluator and COCO annotation files for YOLO11n and Faster R-CNN after converting their predictions to the evaluator's COCO detection format.

## Evaluation methodology

All models are intended to use the same:

- Training, validation, and test splits.
- 640-pixel input resolution.
- 50-epoch training budget and fixed random seed where supported.
- COCO-format annotations and bounding-box evaluation.
- Validation split for model/checkpoint selection.
- Held-out test split for the final comparison only.

The evaluator uses `pycocotools.COCOeval` with `iouType="bbox"` and a low score threshold of `0.001`, which preserves low-confidence detections required for COCO mAP calculation. It reports:

- `mAP@[.50:.95]`: the primary metric, averaged over IoU thresholds from 0.50 to 0.95.
- `mAP@.50`: detection precision at the more permissive IoU threshold of 0.50.
- `mAP@.75`: stricter localization quality.
- `mAP` by object size: small, medium, and large signs.
- `AR@1`, `AR@10`, and `AR@100`: average recall with the specified maximum detections per image.

For RT-DETR, the selected checkpoint was `checkpoint-4320`, chosen using the lowest validation loss from the 50-epoch run. The validation and test results are:

| Model | Split | mAP@[.50:.95] | mAP@.50 | mAP@.75 | AP small | AP medium | AP large | AR@100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR R50vd | Validation | 0.711 | 0.942 | 0.844 | 0.644 | 0.787 | 0.814 | 0.778 |
| RT-DETR R50vd | Test | 0.757 | 0.973 | 0.917 | 0.695 | 0.801 | 0.891 | 0.816 |
| YOLO11n | Test | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Faster R-CNN | Test | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

The RT-DETR test result indicates strong overall detection and localization. Its `mAP@.50` of 0.973 shows high precision at the standard IoU threshold, while `mAP@.75` of 0.917 indicates accurate box placement. `AR@100` of 0.816 indicates strong recall. Small signs are the most difficult group (`AP small` of 0.695), but performance remains substantially strong. The test results are higher than validation results, so these results show no evident overfitting; the test split may simply be easier. The test set must not be used to tune the models.

The final comparison table will be completed after YOLO11n and Faster R-CNN are trained and evaluated with the same protocol. In addition to accuracy metrics, the project should report parameter count, model size, and inference latency or FPS because real-time suitability is part of the project goal.

## Repository layout

```text
DETR/
  scripts/
    data/import_dataset.py       # Download and copy the Kaggle dataset
    preprocess/preprocessing.py  # Validate and prepare COCO and YOLO data
    models/                      # RT-DETR model and training entry points
    evaluate/evaluation.py       # COCO detection metrics for a fine-tuned checkpoint
dataset/
  archive/                       # Downloaded source dataset
  model_ready/                   # Prepared images, annotations, labels, and data.yaml
requirements.txt
```

## Dependencies

See [`requirements.txt`](./requirements.txt) for the Python dependencies. GPU-specific PyTorch installation may vary by CUDA version and platform.
