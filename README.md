# Vietnamese Traffic Sign Detection

Final project by Group 5 for the Introduction to Computer Science course
(CSE2025, G1) at Vietnamese-German University (VGU).

## Members

- Nguyen Trung Kien - 10425148
- Nguyen Khoa Lam - 10425102
- Tran Van Gia Khang - 10425146

## Project objective

This project compares object-detection models on Vietnamese traffic-sign
images. The completed pipelines currently cover RT-DETR and Faster R-CNN;
YOLO11s is planned but has not yet been trained and evaluated in this
repository.

The comparison is designed to answer two questions:

1. How accurately can each model locate and classify the annotated signs?
2. What trade-offs exist between detection quality and eventual real-time use?

To compare detection performance, we plan to train YOLO11s, RT-DETR, and Faster R-CNN on the same dataset and evaluate them using the same metrics. This comparison will help us understand the trade-offs between detection quality and suitability for real-time use. However, the second question cannot be answered from accuracy alone. Parameter count, model size, and latency or FPS still need to be measured under the same hardware and inference settings. This project is exploratory and is not intended for safety-critical or real-world driving use.

## Dataset

The project uses the
[Vietnamese Traffic Signs dataset on Kaggle](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs).
The prepared data contains 52 categories and COCO-format bounding-box
annotations.

The current prepared split statistics are:

| Split | Images | Annotations | Categories |
|---|---:|---:|---:|
| Train | 2,297 | 5,638 | 52 |
| Validation | 255 | 1,051 | 52 |
| Test | 639 | 1,645 | 52 |

The preprocessing pipeline creates the validation split from the training
data, while retaining the provided test split. The prepared files are:

- Images under `dataset/model_ready/images/`
- COCO annotations under `dataset/model_ready/annotations/`
- YOLO-format labels under `dataset/model_ready/labels/`
- YOLO configuration at `dataset/model_ready/data.yaml`

### Annotation coverage limitation

The original dataset annotations supplied by the dataset author do not
necessarily label every visually noticeable sign in an image. For example,
`1671.jpg` contains several signs in the background, but its source COCO
annotation contains only one labelled object. This is a limitation of the
source dataset, not an error introduced by this project's preprocessing.

Consequently, a qualitative image may show visible signs that are absent from
the ground truth. Those signs cannot be counted as missed detections by COCO
evaluation. Qualitative predictions must therefore be compared with the
provided annotations, while model quality should be judged primarily using
metrics computed across the complete split.

## Current status

- Dataset import and preprocessing: complete.
- RT-DETR training and COCO evaluation: complete.
- Faster R-CNN training and COCO evaluation: complete.
- Shared training/validation loss plotting: complete.
- Single-image prediction and visualization for RT-DETR: complete.
- Single-image prediction and visualization for Faster R-CNN: complete.
- YOLO11s training and matching evaluation: pending.
- Parameter, model-size, and latency/FPS comparison: pending.

## Installation

Use the Python version and dependency versions appropriate for the target
environment, then install the repository dependencies:

```bash
python -m pip install -r requirements.txt
```

The exact PyTorch wheel depends on the operating system and CUDA version. For
the CUDA 13.2 wheel index used in the local setup:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

The dependency file documents the verified Kaggle environment separately from
the platform-specific PyTorch installation.

## Prepare the dataset

Run these commands from the repository root:

```bash
python scripts/data/import_dataset.py
python scripts/preprocessing/preprocessing.py
```

The first command downloads and copies the source dataset into `dataset/`.
The second validates the source data and creates `dataset/model_ready/`.
Preprocessing stops rather than overwriting an existing prepared directory;
move or rename that directory before preparing the data again.

## Common experimental protocol

The completed RT-DETR and Faster R-CNN runs were intended to use the same:

- Prepared train, validation, and test splits
- 640 x 640 image processing
- 50-epoch training budget
- Seed 42 where supported
- COCO bounding-box annotations
- Validation data for checkpoint selection
- Held-out test data for final reporting

The test split must not be used to select checkpoints or tune thresholds.

Both evaluators use `pycocotools.COCOeval` with `iouType="bbox"`. Evaluation
uses a score threshold of `0.001` so low-confidence detections remain
available to the COCO precision-recall calculation. This is different from
the `0.5` default used for readable demo visualizations.

Reported metrics are:

- `mAP@[.50:.95]`: primary COCO average precision across IoU thresholds
  0.50 through 0.95.
- `mAP@.50`: average precision at IoU 0.50.
- `mAP@.75`: average precision at the stricter IoU 0.75.
- `AP small`, `AP medium`, and `AP large`: AP grouped by object area.
- `AR@1`, `AR@10`, and `AR@100`: average recall with the corresponding
  maximum number of detections per image.

## RT-DETR

### Training

Run on one GPU:

```bash
python scripts/DETR/models/train.py
```

On Kaggle with two GPUs:

```bash
torchrun --standalone --nproc_per_node=2 scripts/DETR/models/train.py
```

The default run uses 50 epochs, 640-pixel processing, per-device batch size
2, gradient accumulation 2, learning rate `5e-5`, weight decay `1e-4`, and
FP16. The effective batch size with two devices is 8.

The default output directory is
`outputs/rtdetr_r50vd_comparison_50ep/`. `PROJECT_ROOT` and `OUTPUT_DIR`
environment variables can override the default paths. Do not set
`RESUME_CHECKPOINT` for a new comparison run; use it only to continue a run
with compatible training arguments.

### Evaluation

When an output directory is supplied, the evaluator selects the checkpoint
with the lowest recorded validation loss:

```bash
python scripts/DETR/evaluate/evaluation.py --split validation
python scripts/DETR/evaluate/evaluation.py --split test
```

For the reported run, the selected checkpoint was
`checkpoint-4320` (approximately epoch 15):

```bash
python scripts/DETR/evaluate/evaluation.py \
  --checkpoint outputs/rtdetr_r50vd_comparison_50ep/checkpoint-4320 \
  --split test
```

Predictions and metrics are written below the evaluated checkpoint's
`evaluation/` directory.

### Single-image visualization

Print detections as JSON:

```bash
python scripts/DETR/models/predict.py \
  --image path/to/image.jpg \
  --checkpoint outputs/rtdetr_r50vd_comparison_50ep/checkpoint-4320 \
  --threshold 0.5
```

Create an annotated image:

```bash
python scripts/DETR/visualisation/visualise.py \
  --image path/to/image.jpg \
  --checkpoint outputs/rtdetr_r50vd_comparison_50ep/checkpoint-4320 \
  --threshold 0.5 \
  --output outputs/predictions/rtdetr/image_detected.png
```

The threshold changes which detections are displayed in the demo; it does not
change the trained model or the COCO evaluation protocol.

## Faster R-CNN

### Training

Faster R-CNN uses a pretrained torchvision
`fasterrcnn_resnet50_fpn_v2` backbone and replaces the COCO prediction head
with 52 traffic-sign classes plus the background class.

Run on one GPU:

```bash
python "scripts/Faster R-CNN/models/train_rcnn.py"
```

On Kaggle with two GPUs:

```bash
torchrun --standalone --nproc_per_node=2 \
  "scripts/Faster R-CNN/models/train_rcnn.py"
```

The default output directory is
`outputs/faster_rcnn_comparison_50ep/`. Training resizes images and boxes to
640 x 640, uses per-device batch size 2, gradient accumulation 2, SGD with
learning rate `0.005`, momentum `0.9`, weight decay `0.0005`, and learning
rate milestones at epochs 30 and 40.

Validation loss is computed after every epoch and the lowest-loss model is
saved as `best.pt`. To resume a compatible run, set `RESUME_CHECKPOINT`; do
not resume after changing the model, optimizer, batch configuration, or
training budget.

### Evaluation

Evaluate a selected checkpoint on validation and test data:

```bash
python "scripts/Faster R-CNN/evaluate/evaluate_rcnn.py" \
  --checkpoint outputs/faster_rcnn_comparison_50ep/best.pt \
  --split validation

python "scripts/Faster R-CNN/evaluate/evaluate_rcnn.py" \
  --checkpoint outputs/faster_rcnn_comparison_50ep/best.pt \
  --split test
```

The evaluator exports COCO-format predictions and applies the same
`pycocotools.COCOeval` metrics used by RT-DETR.

### Single-image visualization

```bash
python "scripts/Faster R-CNN/visualisation/visualise_rcnn.py" \
  --image path/to/image.jpg \
  --checkpoint outputs/faster_rcnn_comparison_50ep/best.pt \
  --threshold 0.5 \
  --output outputs/predictions/faster_rcnn/image_detected.png
```

If `--output` is omitted, the image is saved under
`outputs/predictions/faster_rcnn/`.

## Loss visualization and checkpoint interpretation

RT-DETR and Faster R-CNN use the same plotting script:

```bash
python scripts/visual_plot/plot_losses.py \
  --output-dir outputs/rtdetr_r50vd_comparison_50ep \
  --model-name rtdetr \
  --output outputs/plots/rtdetr_full_training_loss.png

python scripts/visual_plot/plot_losses.py \
  --output-dir outputs/faster_rcnn_comparison_50ep \
  --model-name faster_rcnn \
  --output outputs/plots/faster_rcnn_full_training_loss.png
```

The script reads RT-DETR `trainer_state.json` or Faster R-CNN
`training_history.json`, plots training and validation loss, and marks the
lowest validation-loss epoch.

For RT-DETR, validation loss reached its recorded minimum at epoch 15 and
then generally increased while training loss continued to fall. For Faster
R-CNN, validation loss reached `0.2639` at epoch 22, then mostly plateaued
around `0.27` while training loss continued to decrease. These patterns are
evidence of loss-based overfitting or convergence after the selected point,
not proof that every later prediction is incorrect.

The curves are useful for checkpoint selection, but loss is not the final
detection metric. Faster R-CNN's validation loss is a sum of model
classification/regression losses, while COCO mAP evaluates post-processed
detections at multiple IoU thresholds. They measure related but different
properties. The overfitting signal is therefore reported as a limitation,
while the validation-selected checkpoint remains suitable for comparison
because it avoids choosing the final model solely because its training loss is
lower.

## Results

The table reports held-out test-set COCO results only. Validation data was
used for checkpoint selection, but validation performance is omitted from the
reported comparison.

<<<<<<< HEAD
| Model | mAP@[.50:.95] | mAP@.50 | mAP@.75 | AP small | AP medium | AP large | AR@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR R50vd | 0.757 | 0.973 | 0.917 | 0.695 | 0.801 | 0.891 | 0.816 |
| Faster R-CNN | 0.681 | 0.936 | 0.845 | 0.667 | 0.724 | 0.812 | 0.731 |
| YOLO11s | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
=======
| Model | Split | mAP@[.50:.95] | mAP@.50 | mAP@.75 | AP small | AP medium | AP large | AR@100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR R50vd | Test | 0.757 | 0.973 | 0.917 | 0.695 | 0.801 | 0.891 | 0.816 |
| Faster R-CNN | Test | 0.681 | 0.936 | 0.845 | 0.667 | 0.724 | 0.812 | 0.731 |
| YOLO11s | Test | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
>>>>>>> 453f1d4702bb64e6620b07bfa93ce188fd9d27e1

### Evidence-based interpretation

RT-DETR has the higher recorded test `mAP@[.50:.95]` than Faster R-CNN
(`0.757` versus `0.681`), a difference of `0.076`. It also has higher test
`AR@100` (`0.816` versus `0.731`), a difference of `0.085`. These metrics
support the limited conclusion that RT-DETR was the stronger accuracy model
among the two completed runs under this protocol.

Faster R-CNN's test `mAP@[.50:.95]` was `0.681`, `mAP@.50` was `0.936`, and
`AR@100` was `0.731`. Its loss curve supports choosing the validation-selected
`best.pt` checkpoint rather than the final checkpoint; it does not by itself
mean training failed. No claim about real-time suitability is made until
latency, FPS, parameter count, and model size are measured.

The test results are descriptive of this split and should not be used to tune
the models. The source annotation limitation also means that COCO metrics
measure performance on labelled objects, not every sign visible to a human
viewer.

## Repository layout

```text
scripts/
  DETR/
    models/                       # RT-DETR training and prediction
    evaluate/                     # RT-DETR COCO evaluation
    visualisation/                # RT-DETR image visualization
  Faster R-CNN/
    models/                       # Faster R-CNN training and prediction
    evaluate/                     # Faster R-CNN COCO evaluation
    visualisation/                # Faster R-CNN image visualization
  preprocessing/                  # Prepare COCO and YOLO dataset files
  visual_plot/                    # Shared training/validation loss plots
  data/                           # Dataset download/import
dataset/
  archive/                        # Downloaded source dataset
  model_ready/                    # Prepared images and annotations
outputs/                          # Training, evaluation, prediction, and plot outputs
requirements.txt
```

## Dependencies

See [requirements.txt](./requirements.txt) for the project dependencies.
PyTorch and torchvision wheels remain platform- and CUDA-specific.
