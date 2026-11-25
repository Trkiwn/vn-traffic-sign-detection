# Vietnamese Traffic Sign Detection

Final project by Group 5 for the Introduction to Computer Science course
(CSE2025, G1) at Vietnamese-German University (VGU).

## At a glance

| | |
|---|---|
| Task | Object detection of 52 Vietnamese traffic-sign classes (COCO bounding boxes) |
| Models | RT-DETR R50vd, Faster R-CNN (ResNet-50 FPN v2) and YOLO11s are trained and evaluated |
| Shared protocol | Same train/validation/test split, 640 x 640 stretched input, no augmentation, 50 epochs, COCO-pretrained weights, checkpoint selection by validation mAP, `pycocotools` COCO evaluation (YOLO11s differences are listed in Section 4) |
| Main result | On the test split, RT-DETR reaches mAP@[.50:.95] 0.759, versus 0.728 for Faster R-CNN and 0.703 for YOLO11s |
| Speed | On one Tesla T4 at batch size 1, end to end: YOLO11s 19.6 ms (FP32) / 23.2 ms (FP16), RT-DETR 72.4 / 63.4 ms, Faster R-CNN 116.6 / 55.9 ms |
| Outputs | `outputs/` is not tracked by git; its contents are in the Kaggle dataset [trkiwn-working-repo](https://www.kaggle.com/datasets/kiennt25/trkiwn-working-repo/) |
| How to read it | Scores measure performance on held-out frames from the collected scenes (see Section 6); removing near-duplicate test frames changes them by about 0.01 mAP at most (Section 5) |

## 1. Introduction

This project compares object-detection models on Vietnamese traffic-sign
images under a shared dataset, training/validation protocol, and COCO
evaluation. All three pipelines (RT-DETR, Faster R-CNN and YOLO11s) are
complete. A benchmark script (`scripts/benchmark/`) compares inference speed
and model size (latency, FPS, parameters, FLOPs, weight size) for all three
models on one GPU, so that accuracy and cost can be discussed together.
Edge-device deployment is outside the scope of this project and is listed as
possible future work.

The project is an exploratory course study of detector behaviour on a public
dataset. It is a controlled comparison of three standard detectors on one
dataset, not a new method, and its findings are intended for model comparison;
they are not a validated driving system.

### Members

- Nguyen Trung Kien - 10425148
- Nguyen Khoa Lam - 10425102
- Tran Van Gia Khang - 10425146

## 2. Dataset

The project uses the
[Vietnam Road Signs Dataset (VNTS) on Kaggle](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs).
The archive (Kaggle version 8) contains 3,216 JPEG images (960 x 540: 2,364;
1,920 x 1,080: 813; 540 x 960: 39) with 8,334 bounding boxes in 52 classes.
Class names are read from the archive's `classes_en.txt`. The prepared dataset
provides the same 52 categories in COCO format.

The Vietnam Road Signs Dataset (VNTS) is used under the
[Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)
license. It was created and collected by the Data Palpating Team.
Source: [Kaggle VNTS dataset](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs).

The prepared split statistics are:

| Split | Images | Annotations | Boxes per image | Categories |
|---|---:|---:|---:|---:|
| Train | 2,297 | 5,638 | 2.45 | 52 |
| Validation | 255 | 1,051 | 4.12 | 52 |
| Test | 639 | 1,645 | 2.57 | 52 |

The archive provides train and test image lists. The project derives
validation from the training list and retains the provided test membership.
The three splits contain 3,191 images in total. The remaining 25 of the 3,216
archive images appear in neither provided list and have empty label files;
they are not used.

The validation images are chosen by `scripts/preprocessing/preprocessing.py`.
It takes 10% of the provided training list (255 of 2,552 images, seed 42) and
selects them greedily so that each class contributes about 10% of its images
(at least one) to validation. Images that contain still-under-represented
classes are preferred, and every class keeps at least one training image. The
selection is class-aware but not scene-aware: neighbouring frames of a
validation image can remain in the training set. This is adequate for choosing
checkpoints. Validation images have a different mix from the test set (denser
and enriched in rare classes), and neighbouring frames can remain in training.

Dataset properties that matter for interpreting the results:

- **Class imbalance.** Classes range from 3 to 1,071 boxes. Aggregate mAP is
  dominated by the common classes, so per-class results are reported
  separately (Section 6.3).
- **Small objects.** In the dataset audit, 46.7% of boxes are COCO "small" at
  original resolution, and 41.5% are smaller than 16 px when the longer image
  side is scaled to 640 px (18.6% at 960 px). The stretched 640 x 640 input used
  here enlarges boxes somewhat relative to that scaling, so 41.5% is an upper
  bound for the models' input. `AP small` is therefore an informative metric
  for this dataset.
- **Validation density.** Validation images contain more boxes per image
  (4.12) than train (2.45) or test (2.57), which is consistent with the
  preference for images containing several under-represented classes
  described above. Validation is used only for checkpoint selection.

## 3. Dataset preprocessing

### Prepare the dataset

Run these commands from the repository root:

```bash
python scripts/data/import_dataset.py
python scripts/preprocessing/preprocessing.py
```

The first command downloads and copies the source dataset into `dataset/`.
The second validates the source data and creates `dataset/model_ready/`.
Preprocessing stops rather than overwriting an existing prepared directory;
move or rename that directory before preparing the data again.

The prepared files are:

- Images under `dataset/model_ready/images/`
- COCO annotations under `dataset/model_ready/annotations/`
- YOLO-format labels under `dataset/model_ready/labels/`
- YOLO configuration at `dataset/model_ready/data.yaml`

The generated YOLO configuration omits a machine-specific dataset root.
Ultralytics resolves its relative split paths from the configuration file's
directory, so the file can be used after moving `model_ready/` to another
machine or runtime. (The file is written in JSON syntax, which is valid YAML.)

The preprocessing script checks every label line (class range, finite and
positive box sizes), clips boxes to the image bounds and reports how many were
clipped, and writes no output if it finds any dataset issue. It builds the
prepared directory in a temporary location and moves it into place only on
success. COCO category IDs are 1-based (1-52), YOLO class IDs stay 0-based, and
image IDs restart at 1 within each split.

### Installation

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

The RT-DETR and Faster R-CNN runs used Kaggle with two NVIDIA Tesla T4 GPUs.
RT-DETR training installed `transformers==5.20.2`, `accelerate==1.15.0`,
`torchmetrics==1.9.0`, `trackio==0.43.0` and `pycocotools==2.0.11` with pip at
the start of the notebook, and the checkpoint metadata records Transformers
5.20.2. An environment check on 12 November 2026 reported PyTorch 2.11.0+cu128,
torchvision 0.26.0+cu128, CUDA 12.8, NumPy 2.1.3 and Pillow 13.0.0. The YOLO11s
run used a single Colab GPU; its exact model is recorded in the
`visible_cuda_devices` field of the saved YOLO11s JSON files. The reported
results correspond to repository commit `41a3983`.

## 4. Methodology

### Common experimental protocol

All three runs (RT-DETR, Faster R-CNN and YOLO11s) share:

- Prepared train, validation, and test splits (the same annotation files)
- 640 x 640 image processing (images are resized to a square, not letterboxed)
- A 50-epoch training budget
- COCO-pretrained starting weights with a new 52-class prediction head
- Seed 42 (see the reproducibility notes below)
- COCO bounding-box annotations
- No data augmentation
- Checkpoint selection by validation mAP among the saved checkpoints (the candidate sets differ by model, see below)
- The provided test images for final reporting only

Test images are not used to select checkpoints or tune thresholds. RT-DETR
training and validation do not access the test split; the explicit COCO
evaluator is run on the test split only after the checkpoint is selected.
Because the test set is defined by image membership in the source archive
rather than by recording, it measures performance on held-out frames from the
collected scenes (Section 6.1). The rule used to choose checkpoints has one qualification, described
under Checkpoint selection below.

All evaluators use `pycocotools.COCOeval` with `iouType="bbox"` and a score
threshold of `0.001`, so low-confidence detections remain available to the COCO
precision-recall calculation (this differs from the `0.5` default used for
readable demo visualizations).

For RT-DETR, the evaluator rescales boxes to the original image size, clips
them to the image, converts them to COCO `[x, y, width, height]`, and maps
labels to COCO category IDs 1-52 (label + 1). The model returns up to 300
detections per image, and `COCOeval` uses its default maximum of 100 per image
for AP and `AR@100`. All evaluators write metrics under the same names.
YOLO11s predictions are generated in the Colab notebook, mapped back to
original image coordinates, and scored with the same `COCOeval` settings;
`replay_coco.py` re-scores the archived predictions.

For Faster R-CNN, the exporter maps boxes from the 640 x 640 network input
back to original image coordinates and uses labels 1-52 directly as COCO
category IDs; torchvision clips boxes to the image. The exporter sets the
detector's internal score threshold (`model.roi_heads.score_thresh`, default
0.05) to the requested `0.001`, so the models are scored at the same
threshold. Non-maximum suppression at IoU 0.5 and a maximum of 100 detections
per image remain at torchvision defaults. An earlier evaluation without this
setting was effectively scored at 0.05 and gave a lower test mAP@[.50:.95]
(0.681 for the epoch-22 checkpoint, against 0.695 after the fix), which is why the
setting is documented.

Reported metrics are:

- `mAP@[.50:.95]`: primary COCO average precision across IoU thresholds
  0.50 through 0.95.
- `mAP@.50`: average precision at IoU 0.50.
- `mAP@.75`: average precision at the stricter IoU 0.75.
- `AP small`, `AP medium`, and `AP large`: AP grouped by object area.
- `AR@1`, `AR@10`, and `AR@100`: average recall with the corresponding
  maximum number of detections per image.

### Checkpoint selection

All three models train for the same 50 epochs, and the reported checkpoint is
chosen afterwards, using validation data only. The rule is the highest validation
mAP@[.50:.95] among the checkpoints saved during training. The candidates are
RT-DETR epochs 15, 49 and 50 (only three are kept), Faster R-CNN epochs 20 to 50
in steps of five plus epoch 22 (`best.pt`), and YOLO11s all 50 epochs (selected
by validation COCO AP). The selected epochs are RT-DETR 50, Faster R-CNN 40 and
YOLO11s 36.

**Why the rule was changed.** The first rule was the lowest validation loss.
Loss and mAP can disagree, and the three models use different loss definitions
that cannot be compared with each other. A validation check showed that Faster
R-CNN's validation mAP kept rising by about 0.04 after its loss minimum, so the
loss rule picked a checkpoint that understated the model. Choosing by validation
mAP uses the metric that is reported, computed by the same scorer for every
model.

**What difference it makes.** Test mAP@[.50:.95] changed by +0.002 for RT-DETR,
+0.033 for Faster R-CNN and +0.0001 for YOLO11s, so the rule matters mainly for
Faster R-CNN. It helps the project in three ways: the three models are compared
under one metric-based rule instead of three incompatible loss definitions;
Faster R-CNN is no longer understated; and YOLO11s's pilot, which applied the
rule from the start, is its reported result, which removes the mismatch between
its selection and that of the other two models.

**Disclosure.** The rule was changed after the test scores of the loss-selected
checkpoints had been seen, so the RT-DETR and Faster R-CNN results are not
prospective with respect to the new rule, although the new choices use validation
data only. The candidates are limited to the saved checkpoints and differ in
number between models.

### Training configurations

| | RT-DETR R50vd | Faster R-CNN | YOLO11s |
|---|---|---|---|
| Library | Hugging Face Transformers | torchvision | Ultralytics 8.4.189 |
| Starting weights | `PekingU/rtdetr_r50vd` (COCO) | `FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT` (COCO) | COCO-pretrained `yolo11s.pt` |
| Prediction head | Re-initialised for 52 labels (0-51, no background) | Replaced: 53 outputs (background + 52) | Rebuilt for 52 classes (the initializer has 80) |
| Input | Resize to 640 x 640 (`RTDetrImageProcessor`, bilinear by default) | Resize to 640 x 640 (PIL, bicubic); boxes scaled to match | Images pre-resized to 640 x 640 (bicubic, stretched); `rect=False`, no padding |
| Augmentation | None | None | None (mosaic, mixup, flips, HSV and geometric augmentation all set to 0) |
| Optimizer | Trainer default (AdamW), lr `5e-5`, weight decay `1e-4` | SGD, lr `0.005`, momentum `0.9`, weight decay `5e-4` | AdamW, lr `0.000179` (the value Ultralytics' `optimizer=auto` computes for 52 classes), weight decay `5e-4` |
| LR schedule | Trainer default linear decay, 100 warm-up steps | Step x0.1 at epochs 30 and 40, no warm-up | Ultralytics defaults (3 warm-up epochs, linear decay to 1% of the initial rate) |
| Batch | 2 per device x accumulation 2 (8 on two GPUs) | 2 per device x accumulation 2 (8 on two GPUs) | 8 on one GPU |
| Precision | FP16 | Mixed precision (AMP, FP16) | AMP |
| Epochs | 50 | 50 | 50 |
| Training platform | Kaggle (2 x T4) | Kaggle (2 x T4) | Google Colab |
| Checkpoint rule | Highest validation mAP among epochs 15, 49 and 50: epoch 50 | Highest validation mAP among epochs 20 to 50: epoch 40 | Highest validation COCO AP over the 50 saved checkpoints: epoch 36 |

Each model was trained with one configuration; no per-model hyperparameter
search is reported.

### Implementation notes

- **Class indices.** RT-DETR uses labels 0-51 (COCO category ID minus one,
  no background class). Faster R-CNN uses labels 1-52 with 0 reserved for
  background (53 outputs). Evaluators convert both back to COCO category IDs
  1-52.
- **GPU count and effective batch size.** The effective batch size is
  per-device batch x accumulation x number of GPUs: 8 on two GPUs and 4 on
  one, with the same learning rate. The reported RT-DETR checkpoint step is
  consistent with two-GPU training (4,320 steps = 15 epochs x 288 optimizer
  steps per epoch). Use two GPUs to reproduce the reported runs.
- **Seeds and determinism.** RT-DETR sets `seed` and `data_seed` to 42.
  Faster R-CNN seeds Python, NumPy and PyTorch with 42 plus the process rank.
  cuDNN determinism is not enforced, so repeated runs can differ slightly.
- **Loss definitions.** RT-DETR's validation loss is Hugging Face's
  Hungarian-matching objective (classification, L1 and GIoU box terms with
  weights 1, 5 and 2, plus auxiliary losses from the decoder layers),
  Faster R-CNN's is the sum of its four torchvision loss terms, and YOLO11s's is
  the sum of its gain-weighted box, classification and DFL terms. They are used
  for monitoring and for the earlier checkpoint rule only; their values are not comparable
  between models.
- **Normalization and versions.** RT-DETR freezes its backbone batch-norm
  layers (`freeze_backbone_batch_norms`), whereas torchvision's
  `fasterrcnn_resnet50_fpn_v2` trains its batch-norm layers, here with 2 images
  per GPU. This recipe difference may affect Faster R-CNN's results and is not
  controlled in this comparison. The saved RT-DETR checkpoint metadata records
  Transformers 5.20.2; other versions are in `requirements.txt`.
- **Faster R-CNN validation mode.** torchvision detectors return losses only
  in training mode, so validation loss is computed under `model.train()` with
  gradients disabled. This also leaves normalization layers in training
  mode: batch statistics are used and running statistics are updated on
  validation images. The effect is expected to be small, and the reported run
  keeps this behaviour. Setting these layers to evaluation mode during
  validation is a listed refinement (Section 6.3).
- **Checkpoint retention.** RT-DETR keeps the best checkpoint plus the most
  recent ones (`save_total_limit=3`). Faster R-CNN saves `best.pt` and a
  checkpoint every five epochs.

### YOLO11s run and differences from the other two models

YOLO11s was trained with Ultralytics 8.4.189 on the same prepared split
(identical image bytes and annotations) with the settings in the table above.
The images were resized once to 640 x 640 (bicubic) and used as the YOLO inputs
for training, loss computation and inference, so the geometry matches Faster
R-CNN's stretched resize. Horizontal flipping and all other augmentation were
disabled. The class list contains mirror-image pairs (for example No Left Turn
and No Right Turn, see Appendix A), so mirroring would change the label of a
sign.

The YOLO11s run differs from the other two in these ways:

- **Checkpoint selection.** Validation was switched off during training
  (`val=False`) and all 50 epoch checkpoints were saved. The reported checkpoint
  is the one with the highest validation COCO AP: epoch 36 (`epoch35.pt` in
  Ultralytics' zero-based file naming). The lowest-validation-loss alternative
  (epoch 37, `epoch36.pt`, computed offline in evaluation mode) is archived in
  `min_val_loss/` and used for the speed benchmark. The three models' loss
  definitions are not comparable.
- **Rule history.** The validation-AP checkpoint comes from the first (pilot)
  run, which applied that rule from the start. The loss-based run was added
  afterwards to match the other two models, and is now an archived reference.
- **Inference settings.** Predictions use score threshold 0.001, NMS at IoU 0.7
  and at most 100 detections per image. Ultralytics' prediction mode keeps one
  class per candidate box, whereas the other two models can return several
  class hypotheses for the same box, and YOLO11s returned 2,461 test detections
  (about 3.9 per image; 638 of 639 images have at least one) against 20,126 for Faster R-CNN. This may lower
  YOLO11s's recall-oriented metrics, such as mAP@.50. A re-run with multi-label
  NMS is listed in Section 6.3.
- **Platform.** YOLO11s was trained on Google Colab with one GPU (effective
  batch size 8), and the other two on Kaggle.

### RT-DETR

#### Training

Run on one GPU (effective batch size 4):

```bash
python scripts/DETR/models/train.py
```

On Kaggle with two GPUs (effective batch size 8, as in the reported run):

```bash
torchrun --standalone --nproc_per_node=2 scripts/DETR/models/train.py
```

The default run uses 50 epochs, 640-pixel processing, per-device batch size
2, gradient accumulation 2, learning rate `5e-5`, weight decay `1e-4`, 100
warm-up steps, and FP16.

The default output directory is
`outputs/rtdetr_r50vd_comparison_50ep/`. `PROJECT_ROOT` and `OUTPUT_DIR`
environment variables can override the default paths. Do not set
`RESUME_CHECKPOINT` for a new comparison run; use it only to continue a run
with compatible training arguments.

#### Evaluation

When a training output directory is supplied, the evaluator uses the model saved
there at the end of training, which is the lowest-validation-loss checkpoint of
the earlier rule (epoch 15). If no saved
model is present, it scans the checkpoint directories for the lowest recorded
validation loss:

```bash
python scripts/DETR/evaluate/evaluation.py --split validation
python scripts/DETR/evaluate/evaluation.py --split test
```

The reported checkpoint, chosen by validation mAP, is `checkpoint-14400`
(epoch 50); `checkpoint-4320` (epoch 15) is the earlier rule's checkpoint:

```bash
python scripts/DETR/evaluate/evaluation.py \
  --checkpoint outputs/rtdetr_r50vd_comparison_50ep/checkpoint-14400 \
  --split test
```

Predictions and metrics are written below the evaluated checkpoint's
`evaluation/` directory.

#### Single-image visualization

Write detections as JSON (by default to `outputs/predictions/rtdetr/<image>.json`; override with `--output`). Labels are 0-based class indices, so the COCO category ID is the label plus one:

```bash
python scripts/DETR/models/predict.py \
  --image path/to/image.jpg \
  --checkpoint outputs/rtdetr_r50vd_comparison_50ep/checkpoint-14400 \
  --threshold 0.5
```

Create an annotated image:

```bash
python scripts/DETR/visualisation/visualise.py \
  --image path/to/image.jpg \
  --checkpoint outputs/rtdetr_r50vd_comparison_50ep/checkpoint-14400 \
  --threshold 0.5 \
  --output outputs/predictions/rtdetr/image_detected.png
```

The threshold changes which detections are displayed in the demo; it does not
change the trained model or the COCO evaluation protocol.

### Faster R-CNN

#### Training

Faster R-CNN uses the COCO-pretrained torchvision
`fasterrcnn_resnet50_fpn_v2` detector and replaces its prediction head with
52 traffic-sign classes plus the background class.

Run on one GPU (effective batch size 4):

```bash
python "scripts/Faster R-CNN/models/train_rcnn.py"
```

On Kaggle with two GPUs (effective batch size 8, as in the reported run):

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
saved as `best.pt`; the reported checkpoint is chosen afterwards by validation
mAP (see Checkpoint selection). To resume a compatible run, set `RESUME_CHECKPOINT` to a
checkpoint in the same `OUTPUT_DIR`; training history and best-checkpoint
selection are restored from that checkpoint. The script refuses to resume
when the output history is newer than the chosen checkpoint. Do not resume
after changing the model, optimizer, batch configuration, or training budget.

The reported Faster R-CNN run was trained before the final partial
gradient-accumulation group was scaled correctly; the scripts in this
repository contain the correction. The affected update is the last one of an
epoch when the number of batches is odd (about 1 of 288 updates per epoch on
two GPUs, roughly 0.35%), and its gradient was scaled by half. This is not
expected to change the comparison, and the run was not repeated: the reported
numbers come from the original run.

#### Evaluation

Evaluate the reported checkpoint (`checkpoint-40.pt`; `best.pt` is the earlier
lowest-loss checkpoint) on validation and test data:

```bash
python "scripts/Faster R-CNN/evaluate/evaluate_rcnn.py" \
  --checkpoint outputs/faster_rcnn_comparison_50ep/checkpoint-40.pt \
  --split validation

python "scripts/Faster R-CNN/evaluate/evaluate_rcnn.py" \
  --checkpoint outputs/faster_rcnn_comparison_50ep/checkpoint-40.pt \
  --split test
```

The evaluator exports COCO-format predictions and applies the same
`pycocotools.COCOeval` metrics used by RT-DETR. Predictions come from
`predict_rcnn.py` (in the training-script folder), and predictions and metrics
are written next to the checkpoint as `<split>_predictions.json` and
`<split>_metrics.json`; each run overwrites these files, so copy any result you
want to keep.

#### Single-image visualization

```bash
python "scripts/Faster R-CNN/visualisation/visualise_rcnn.py" \
  --image path/to/image.jpg \
  --checkpoint outputs/faster_rcnn_comparison_50ep/checkpoint-40.pt \
  --threshold 0.5 \
  --output outputs/predictions/faster_rcnn/image_detected.png
```

If `--output` is omitted, the image is saved under
`outputs/predictions/faster_rcnn/`.

### YOLO11s

#### Training and checkpoint selection

YOLO11s is trained and selected in the Colab notebook
[`scripts/YOLO11s/models/VGU_YOLO11s_MinValLoss_GitHub_Ready.ipynb`](scripts/YOLO11s/models/VGU_YOLO11s_MinValLoss_GitHub_Ready.ipynb);
there is no standalone `train.py`. Open it in Google Colab with a GPU runtime and
set its first code cell. The notebook has three modes: `review` (default;
displays the archived results, no GPU or Drive needed), `select` (re-verifies
the saved checkpoints and recomputes the validation-loss selection without
training) and `reproduce` (trains if checkpoints are missing; needs a GPU and the
prepared dataset). For an entirely new run, set `MODE = "reproduce"`, a new
`RUN_NAME` (do not overwrite the archived pilot), `DATASET` (the prepared
`model_ready` directory) and `WEIGHTS` (the official `yolo11s.pt`). Do not
retrain the historical experiment just to evaluate it.

The notebook pins `ultralytics==8.4.189` and `pycocotools==2.0.11`.
`requirements.txt` documents the Kaggle environment and is not a bit-for-bit
record of the Colab runtime.

The notebook implements the lowest-validation-loss rule. The reported YOLO11s
checkpoint, chosen by validation COCO AP, comes from the earlier pilot archived in
`outputs/yolo11s/max_val_ap_pilot/`.

The saved JSON evidence is under `outputs/yolo11s/`. The lowest-validation-loss
weights, used for the speed benchmark and the demos, are expected at
`outputs/yolo11s/epoch36.pt`, with SHA-256
`05cf921fffd4c46df7b18ea8958c84e29bc228207a65590958c9d16cee42544c`; the
50-epoch checkpoint series is kept on the run owner's Google Drive
(`VGU_TrafficSign_Project/yolo11s_colab_runs/yolo11s_colab_pilot_01/`), where
the file is `training/weights/epoch36.pt`; the reported validation-AP checkpoint is
`training/weights/epoch35.pt` in the same series. The methodology, results and
limitations are written up in
[outputs/yolo11s/docs/YOLO11s_EXPERIMENT_REPORT.md](outputs/yolo11s/docs/YOLO11s_EXPERIMENT_REPORT.md).

#### Evaluation

The archived test predictions can be re-scored on any machine, without a GPU or
the weights:

```bash
python -m pip install pycocotools==2.0.11
python scripts/YOLO11s/evaluate/replay_coco.py --split test
python scripts/YOLO11s/evaluate/replay_coco.py --split test --track max_val_ap_pilot
```

By default the script recomputes the COCO metrics of the lowest-validation-loss
run from `outputs/yolo11s/min_val_loss/test_predictions.json` and checks them
against the recorded `test_metrics.json` (tolerance 1e-8). `--track
max_val_ap_pilot` replays the reported validation-AP checkpoint instead. The
metrics of the two tracks must not be mixed up.

#### Loss plot and single-image visualization

```bash
python scripts/YOLO11s/visualisation/plot_val_loss.py

python scripts/YOLO11s/visualisation/visualise_yolo11s.py \
  --image path/to/image.jpg \
  --checkpoint outputs/yolo11s/epoch36.pt \
  --threshold 0.5 \
  --output outputs/predictions/yolo11s/image_detected.png
```

The YOLO11s plot shows validation loss only, because validation was switched off
during training and the loss history was computed afterwards. The visualization
script reproduces the 640 x 640 stretched input and is a demo, not an evaluator.
It needs the `epoch36.pt` weights (default location `outputs/yolo11s/epoch36.pt`);
without `--output` it only prints the detections as JSON.

### Inference speed and model size benchmark

All three models are timed through one harness,
`scripts/benchmark/benchmark_inference.py`, in a single session on one GPU:

```bash
python scripts/benchmark/benchmark_inference.py \
  --rtdetr-checkpoint outputs/rtdetr_r50vd_comparison_50ep/checkpoint-4320 \
  --rcnn-checkpoint outputs/faster_rcnn_comparison_50ep/best.pt \
  --yolo-checkpoint outputs/yolo11s/epoch36.pt \
  --device cuda:0 --precision both

python scripts/benchmark/plot_accuracy_latency.py
```

Add `--num-images 5 --warmup 2 --runs 10 --repeats 1` for a quick smoke test
first. Results are written to `outputs/benchmark/` (`benchmark_results.json` and
`benchmark_table.md`), and the plot to `outputs/plot/accuracy_vs_latency.png`.
On a machine with two GPUs, only the device given by `--device` is used.

Protocol:

- **Inputs and batch size.** Batch size 1, the same 640 x 640 stretched
  preprocessing as in evaluation, and 100 test images (fixed seed) decoded once
  and cycled through, so that postprocessing sees realistic detection counts.
- **Timing.** 50 warm-up and 300 timed iterations per configuration, repeated 3
  times with the model order rotated, with `torch.cuda.synchronize()` around
  every timed stage. The median and 95th percentile are reported over all timed
  runs, together with how far the repeat medians differ.
- **What is timed.** End-to-end latency covers preprocessing, the forward pass,
  postprocessing at score threshold 0.001, and the copy of the detections to the
  host in original-image coordinates. Preprocessing, inference and
  postprocessing are also timed separately. torchvision's Faster R-CNN performs
  NMS inside its forward pass, so its inference time already includes
  postprocessing.
- **Precision.** FP32 and FP16 through `torch.autocast`, the mechanism used in
  training.
- **Model-specific settings.** YOLO11s uses Ultralytics' fused inference graph
  with the NMS settings of the reported evaluation (IoU 0.7, at most 100
  detections, one class per box; `--yolo-multi-label` changes this).
  RT-DETR is NMS-free. Faster R-CNN keeps torchvision's NMS at IoU 0.5 and at
  most 100 detections.
- **Size and cost.** Parameters and FP32 weight size of the model as trained,
  GFLOPs from `torch.utils.flop_counter` (convolution and matrix-multiply
  operations only, so operations such as grid sampling and RoIAlign are not
  counted and the values are approximate; they are measured on a blank 640 x 640
  input, so Faster R-CNN's value depends on the number of region proposals), and
  peak GPU memory allocated by PyTorch (excluding the CUDA context).

The benchmark measures inference cost under PyTorch eager mode on the GPU used.
It does not cover export formats (ONNX, TensorRT), edge devices or power.

### Clean-test and block-bootstrap analysis

`scripts/analysis/clean_test_and_bootstrap.py` works directly on the saved test
prediction files of the three reported checkpoints, so it needs no retraining and
no GPU:

```bash
python scripts/analysis/clean_test_and_bootstrap.py \
  --pairs path/to/near_duplicate_pairs.csv
```

- **Clean test subsets.** `--pairs` is the near-duplicate pair table written by
  `scripts/preprocessing/dataset_audit.ipynb`. The script keeps the test images
  whose nearest provided-training image is farther than each distance in
  `--deltas` (default 4 and 6) and re-scores every model on those images.
  `--clean-csv` accepts the audit's `split_v1.csv` instead (distance greater
  than 4).
- **Block bootstrap.** Consecutive test images (ordered by file number) are
  resampled in blocks of 10, 1,000 times, with the same resamples for all models,
  so that neighbouring frames of one recording stay together and differences
  between models are paired. The 95% intervals are percentile intervals
  re-centred on the full-sample value, because resampled AP is biased upwards by
  about 0.01; differences between models are almost unaffected.
- **Self-check.** Per-image matching is done once with pycocotools, and the
  bootstrap re-accumulates AP from those results with numpy. At start-up the
  script checks that this matches pycocotools on the full test set, and
  `test_bootstrap_core.py` checks it on random subsets.
- **Options and output.** `--include-earlier-rule` adds the lowest-validation-loss
  checkpoints. Results are written to `outputs/analysis/clean_test_bootstrap.md`
  and `.json`.

### Loss visualization and checkpoint selection

RT-DETR and Faster R-CNN use the same plotting script:

```bash
python scripts/visual_plot/plot_losses.py \
  --output-dir outputs/rtdetr_r50vd_comparison_50ep \
  --model-name rtdetr \
  --output outputs/plot/rtdetr_full_training_loss.png

python scripts/visual_plot/plot_losses.py \
  --output-dir outputs/faster_rcnn_comparison_50ep \
  --model-name faster_rcnn \
  --output outputs/plot/faster_rcnn_full_training_loss.png
```

The script reads RT-DETR `trainer_state.json` or Faster R-CNN
`training_history.json`, plots training and validation loss, and marks the
lowest validation-loss epoch.

RT-DETR's validation loss reached its recorded minimum at epoch 15 and
generally increased afterwards while training loss continued to fall. Faster
R-CNN's validation loss reached `0.2639` at epoch 22 and then mostly plateaued
around `0.27` while training loss continued to decrease.

Loss was the first checkpoint rule, but loss and mAP are related, not
interchangeable. Faster R-CNN's mAP kept improving after its loss minimum,
whereas for RT-DETR the two rules give nearly the same result. The reported checkpoints
are therefore chosen by validation mAP (Section 4). The loss curves remain useful
for monitoring training: they show that RT-DETR's final checkpoint is more
overfit in loss terms.

### Project structure and dependencies

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
  YOLO11s/
    models/                       # Colab notebook: training, selection and test evaluation
    evaluate/                     # Replay and verify archived COCO evaluation
    visualisation/                # Validation-loss plot and image visualization
    tests/                        # YOLO11s artifact, repository-structure and benchmark tests
  preprocessing/                  # Prepare COCO and YOLO dataset files; dataset_audit.ipynb
  benchmark/                      # Inference speed and model-size benchmark, accuracy-latency plot
  analysis/                       # Clean-test re-scoring and block-bootstrap intervals
  visual_plot/                    # Shared training/validation loss plots
  data/                           # Dataset download/import
dataset/
  archive/                        # Downloaded source dataset
  model_ready/                    # Prepared images and annotations
outputs/
  rtdetr_r50vd_comparison_50ep/   # RT-DETR training, checkpoints and evaluation
  faster_rcnn_comparison_50ep/    # Faster R-CNN training, checkpoints and evaluation
  yolo11s/
    min_val_loss/                 # Lowest-validation-loss run: predictions, metrics, loss history
    max_val_ap_pilot/             # Reported run (checkpoint chosen by validation AP)
    docs/                         # YOLO11s_EXPERIMENT_REPORT.md (methodology, results, limitations)
    summary.json                  # Selection and result summary for the YOLO11s run
    epoch36.pt                    # Lowest-validation-loss weights (speed benchmark, demos)
  benchmark/                      # Speed and model-size benchmark results
  analysis/                       # Clean-test and bootstrap results
  plot/                           # Loss and accuracy-latency plots
  predictions/                    # Single-image demo outputs
requirements.txt
```

The `outputs/` folder is not tracked by git because it holds checkpoints, predictions and
other large files. To see the outputs, use the Kaggle dataset [trkiwn-working-repo](https://www.kaggle.com/datasets/kiennt25/trkiwn-working-repo/),
which serves as the online working repository alongside GitHub.

See [requirements.txt](./requirements.txt) for project dependencies. PyTorch
and torchvision wheels remain platform- and CUDA-specific.

### Tests

The `scripts/YOLO11s/tests/` folder checks the archived YOLO11s evidence and the
repository layout. From the repository root:

```bash
python -m unittest discover -s scripts/YOLO11s/tests -v
```

The tests use only the standard library and need no GPU or weights. The
artifact tests read the archived YOLO11s files in `outputs/yolo11s/`, so download
`outputs/` from the Kaggle dataset first.
`test_yolo11s_artifacts.py` verifies the provenance of the YOLO11s result files:
50 consecutive epochs, 255 validation images in 32 batches per epoch,
consistent checkpoint hashes, the minimum validation loss at epoch 37, 639
evaluated test images, 2,461 well-formed predictions (638 images have at least
one; one test image has none above the score floor), and that the earlier pilot
is preserved. `test_yolo11s_structure.py` checks that the YOLO11s entry points
exist, that `replay_coco.py` resolves the repository root, and that this README
names the correct paths. `test_benchmark_core.py` checks the benchmark's
statistics, timing loop and output files with fake models, and
`test_bootstrap_core.py` checks the bootstrap's COCO computation against pycocotools.

## 5. Results and evaluation

The table reports test-set COCO results. Validation data was used for
checkpoint selection only and is not part of the reported comparison.

| Model | mAP@[.50:.95] | mAP@.50 | mAP@.75 | AP small | AP medium | AP large | AR@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR R50vd | 0.7591 | 0.9758 | 0.9065 | 0.7015 | 0.8086 | 0.8844 | 0.8169 |
| Faster R-CNN | 0.7280 | 0.9637 | 0.8994 | 0.6968 | 0.7752 | 0.8623 | 0.7808 |
| YOLO11s | 0.7033 | 0.9242 | 0.8437 | 0.6875 | 0.7523 | 0.8406 | 0.7486 |

All rows use the checkpoint with the highest validation mAP (RT-DETR epoch 50,
Faster R-CNN epoch 40, YOLO11s epoch 36) and are scored at score threshold 0.001
(Section 4). The selection and inference settings of YOLO11s differ from the
other two models (Section 4).

The checkpoints are chosen by validation mAP rather than validation loss (Section
4, Checkpoint selection); the choice matters mainly for Faster R-CNN.

Under this protocol, RT-DETR R50vd reached the highest test mAP@[.50:.95]
(0.759), ahead of Faster R-CNN (0.728, -0.031) and YOLO11s (0.703, -0.056), and
the highest AR@100 (0.817, versus 0.781 and 0.749). Against Faster R-CNN,
RT-DETR's gap is small at IoU 0.50 (+0.012) and at IoU 0.75 (+0.007). By object
size it is largest for medium (+0.033) and large (+0.022) signs and small for
small signs (+0.005). Against YOLO11s, the gap is larger at strict IoU (+0.063 at
IoU 0.75, against +0.052 at 0.50), which is consistent with more precisely placed
boxes; the proposed per-class and error analysis (Section 6.3) would test that
explanation.

Faster R-CNN is ahead of YOLO11s on every metric: mAP@[.50:.95] (+0.025),
mAP@.50 (+0.040), mAP@.75 (+0.056), AR@100 (+0.032) and all three size groups.
Part of YOLO11s's lower mAP@.50 may come from its single-class-per-box inference
(Section 4), which has not yet been tested. Because the selection rule alone
moved Faster R-CNN's score by 0.033, differences of that size between models
should be read with caution, and no repeated runs exist to quantify them. AR@1
and AR@10 are available in the evaluation outputs.

### Clean-test and block-bootstrap results

Test mAP@[.50:.95] with 95% block-bootstrap intervals (Section 4), on the full
test split and on two clean subsets: the test images whose nearest provided
training image is farther than hash distance 4 or 6.

| Model | Full test (639 images) | Clean, distance > 4 (407) | Clean, distance > 6 (290) |
|---|---|---|---|
| RT-DETR R50vd | 0.7591 [0.7284, 0.7843] | 0.7551 [0.7314, 0.7763] | 0.7579 [0.7324, 0.7820] |
| Faster R-CNN | 0.7280 [0.6982, 0.7541] | 0.7240 [0.7021, 0.7448] | 0.7202 [0.6926, 0.7462] |
| YOLO11s | 0.7033 [0.6688, 0.7345] | 0.6921 [0.6647, 0.7186] | 0.6941 [0.6593, 0.7305] |

Paired differences in mAP with 95% intervals (the same resamples for every model):

| Difference | Full test | Clean, distance > 4 | Clean, distance > 6 |
|---|---|---|---|
| RT-DETR - Faster R-CNN | +0.0311 [+0.0139, +0.0464] | +0.0310 [+0.0148, +0.0488] | +0.0377 [+0.0197, +0.0549] |
| RT-DETR - YOLO11s | +0.0558 [+0.0278, +0.0817] | +0.0630 [+0.0383, +0.0854] | +0.0638 [+0.0300, +0.0974] |
| Faster R-CNN - YOLO11s | +0.0246 [-0.0003, +0.0480] | +0.0320 [+0.0111, +0.0513] | +0.0261 [-0.0085, +0.0606] |

- **RT-DETR's lead holds.** Its intervals against both other models exclude 0 on
  all three subsets: +0.031 to +0.038 over Faster R-CNN and +0.056 to +0.064 over
  YOLO11s.
- **Faster R-CNN against YOLO11s is not established.** The interval includes 0 on
  the full test split (barely) and at distance 6, and excludes it only at
  distance 4. The two should be treated as close.
- **Near-duplicate frames do not drive the scores.** Removing them changes
  RT-DETR by -0.004 and -0.001, Faster R-CNN by -0.004 and -0.008, and YOLO11s by
  -0.011 and -0.009, all well within the intervals. The ranking RT-DETR, Faster
  R-CNN, YOLO11s is the same on every subset. The checkpoint choice also did not
  change RT-DETR or YOLO11s by more than 0.005 on any subset.
- **Resampling noise is about +/-0.03 per model.** The intervals on a single
  model's mAP are about 0.05 wide, so absolute scores are uncertain by about that
  much from the finite test set alone.

These checks show that the results are not an artefact of near-duplicate frames
within this test split. The clean subsets still come from the same recordings and
differ from the full split in image mix, so they are not evidence of performance
on new recordings, and the intervals do not include training variance (one run per
model).

### Inference speed and model size

Speed and size are measured separately from accuracy, with the benchmark
described in Section 4, on one Tesla T4 GPU (Kaggle) at batch size 1.

| Model | Precision | Params (M) | Weights (MB) | GFLOPs | Peak GPU mem (MB) | Preprocess (ms) | Inference (ms) | Postprocess (ms) | End-to-end p50 / p95 (ms) | FPS (1/p50) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR R50vd | FP32 | 42.8 | 172 | 137.7 | 335 | 6.16 | 65.58 | 0.90 | 72.36 / 77.02 | 13.8 |
| Faster R-CNN | FP32 | 43.5 | 174 | 451.3 | 697 | 12.67 | 101.23† | included | 116.61 / 125.97 | 8.6 |
| YOLO11s | FP32 | 9.4 | 38 | 21.5 | 114 | 10.39 | 7.96 | 1.22 | 19.60 / 36.13 | 51.0 |
| RT-DETR R50vd | FP16 | 42.8 | 172 | 137.7 | 315 | 5.56 | 56.41 | 0.89 | 63.37 / 70.60 | 15.8 |
| Faster R-CNN | FP16 | 43.5 | 174 | 451.3 | 692 | 12.52 | 42.30† | included | 55.86 / 69.21 | 17.9 |
| YOLO11s | FP16 | 9.4 | 38 | 21.5 | 96 | 10.41 | 11.47 | 1.25 | 23.21 / 39.73 | 43.1 |

Times are medians (p50) and 95th percentiles (p95) over all timed runs; stage
times are measured with a synchronization after every stage, so they do not add
up exactly to the end-to-end time. † Faster R-CNN's NMS is inside its forward
pass, so its postprocessing is marked `included` in the inference column; only
coordinate rescaling and the device-to-host copy (about 1 to 3 ms) happen
afterwards, and they are counted in the end-to-end latency only. FP16 is
`torch.autocast`.

Observations (this GPU, PyTorch eager mode, batch size 1):

- YOLO11s is the fastest and smallest model: 19.6 ms (51 FPS) end to end in
  FP32 and 23.2 ms in FP16, with 9.4 M parameters and 21.5 GFLOPs, against 72.4
  and 63.4 ms for RT-DETR and 116.6 and 55.9 ms for Faster R-CNN.
- RT-DETR has the highest test mAP@[.50:.95] (0.759), at about 3.7 times
  YOLO11s's end-to-end latency in FP32 and 2.7 times in FP16.
- Faster R-CNN reaches 0.728, which is 0.025 above YOLO11s, at about 6 times
  YOLO11s's latency in FP32 and 2.4 times in FP16. Compared with RT-DETR, it is
  about 12% faster in FP16 (55.9 versus 63.4 ms) at 0.031 lower mAP, and slower
  in FP32 (116.6 versus 72.4 ms).
- FP16 through autocast helped the two larger models (RT-DETR 72.4 to 63.4 ms,
  Faster R-CNN 116.6 to 55.9 ms) but not YOLO11s (19.6 to 23.2 ms), where the
  per-layer casts add overhead to a small network. The YOLO11s FP16 figures are
  therefore conservative; converting the weights to half precision would
  probably be faster.
- CPU-side preprocessing (resizing full-resolution frames with PIL) is roughly
  half of YOLO11s's end-to-end time (about 10.4 ms) and would shrink with
  GPU-side resizing. The p95 latencies likely reflect the mix of image sizes in
  the 100-image sample.
- The model scales differ: YOLO11s has 9.4 M parameters against 42.8 M and 43.5 M
  for RT-DETR and Faster R-CNN, so speed and accuracy differences mix architecture
  with capacity. No scale-matched YOLO11 variant was trained (Section 6.3).

The benchmark was run with the lowest-validation-loss checkpoints (RT-DETR epoch
15, Faster R-CNN `best.pt`, YOLO11s epoch 37). Inference speed depends on the
architecture, not the weights, apart from small differences in the number of
detections passed to postprocessing.

These numbers describe inference cost on one GPU under PyTorch. They do not by
themselves establish suitability for real-time or in-vehicle use, and the
ranking could change with export formats or other hardware.

## 6. Scope of the results and proposed solutions

Each limitation below is paired with a proposed solution. The solutions are
proposals, not completed work, and are ordered by cost within each subsection.

### 6.1 What the current evaluation measures

The archive contains sequentially numbered frames extracted from recordings.
It does not provide recording IDs or reliable timestamps. The provided train
and test lists share no filenames, but frames from the same scenes appear on
both sides. In the dataset audit (`scripts/preprocessing/dataset_audit.ipynb`), the share of
test images with a near neighbour in the training set (perceptual-hash Hamming
distance of at most 6) was 54.6%, close to the 56.2% share of training images
with a near neighbour elsewhere in the training set (57.2% when all training
images are used as the reference). Of the 349 linked test images, 339 have
consecutive-looking file names (same prefix, number gap of at most 5). A manual
review of 60 sampled train/test pairs (20 each at distances 0-2, 3-4 and 5-6)
judged all of them to be the same scene, and 232 of the 491 near-duplicate
groups contain both train and test images.

The reported scores therefore describe performance on held-out frames from the
collected scenes. Every model is trained and tested on the same split, so the
comparison between models is made under identical conditions. Absolute scores
are likely to be higher than performance on entirely new recordings, and the
ranking of models on new recordings is not guaranteed to match the ranking
here. The proposed solutions below measure how well these results carry over.

**Completed checks** (no retraining; results in Section 5, method in Section 4):
re-scoring the saved test predictions on the test images that have no
near-duplicate in the training set (407 of the 639 at distance greater than 4, as
in the audit's `split_v1.csv`; 290 at the stricter distance greater than 6,
measured against all 2,552 provided training images), and block-bootstrap
confidence intervals from blocks of consecutive frames, with the same resamples for
all models. The ranking of RT-DETR above the other two models persists on the
clean subsets, so the comparison is not an artefact of the frame-level leakage.
The scripts are `scripts/analysis/clean_test_and_bootstrap.py` and
`scripts/preprocessing/dataset_audit.ipynb`.

**Proposed solutions, ordered by cost:**

1. **Scene-group split.** Cluster images into approximate scene groups using
   perceptual hashes or image embeddings together with filename adjacency.
   Exact recording IDs are not required for this: merging too many images into
   one group only makes the split more conservative, so approximate grouping
   is a safe way to prevent leakage. If time allows, use the groups for a new
   train/validation/test split. A first version exists in the audit notebook
   (`split_v1.csv`: near-duplicate groups within the training list at distance 4
   and a group-aware validation cut of 426 images, leaving 2,126 for training);
   the provided test set is unchanged there.
2. **Independent test set.** Photograph or film a small set of new street
   scenes (about 100-200 images) and annotate only the 52 classes. This gives
   a test set that is independent by construction.

### 6.2 Source annotations

The source annotations do not label every visible sign. For example,
`1671.jpg` contains several signs in the background but has one labelled
object. This is a property of the source dataset, not of the preprocessing.
COCO evaluation measures performance on labelled objects, so a correct
detection of an unlabelled sign is counted as a false positive and measured
precision can be understated, while an unlabelled sign cannot be counted as a
miss.

**Proposed solution:** review a sample of high-confidence false positives from all
three models, report how many correspond to genuine unlabelled signs, and report an
annotation-adjusted precision next to the standard metrics. Qualitative
predictions should be read alongside the provided annotations.

### 6.3 Other limitations and proposed solutions

| Limitation | Effect on the conclusions | Proposed solution |
|---|---|---|
| One training run per model | The bootstrap intervals (Section 5) capture test-set sampling noise but not training variance. The selection rule alone changed Faster R-CNN's score by 0.033, comparable to the 0.025 gap between Faster R-CNN and YOLO11s, which the intervals also leave unresolved | Repeat one model's training with 2-3 seeds to measure variance directly; a Faster R-CNN repeat can also include the accumulation fix and normalization layers in evaluation mode during validation |
| Training recipes differ between the models (optimizer and schedule, batch-norm handling, resize interpolation, platform) | Part of an accuracy difference may come from the recipe, not the architecture | The differences are disclosed (Section 4). Test the most influential one with a controlled variant, for example Faster R-CNN with frozen backbone batch norm as in RT-DETR; in the longer term, train all three through one shared harness with matched settings |
| The checkpoint rule was changed after test scores were seen (RT-DETR, Faster R-CNN), and YOLO11s is scored with one class per box | The test scores of the loss-selected checkpoints had been viewed before the validation-mAP rule was adopted, and YOLO11s's mAP@.50 may be understated | Re-score YOLO11s with multi-label NMS (inference only, for example through Ultralytics' validation mode with `conf=0.001`, saving COCO-format predictions). Treat the independent test set (Section 6.1) as the first evaluation not seen during selection |
| Checkpoints are selected by validation mAP from candidate sets of different sizes | RT-DETR has 3 candidates, Faster R-CNN about 8 and YOLO11s 50, so a model with more candidates has more chances to catch a lucky peak. The validation set is small (255 images) and not scene-aware, so the choice can favour later, more memorized checkpoints | Score every fifth epoch for all three models (a future RT-DETR run should save every epoch) |
| Class imbalance: some classes have very few test boxes | Aggregate mAP is dominated by common classes, and AP for rare classes is unreliable | Per-class AP with class-frequency tiers (the audit's A/B/C split by box count), a confusion matrix covering mirror-image pairs and similar-sign groups (speed limits, vehicle restrictions), and a flag on classes with fewer than 10 test boxes. For training, collect more images of rare classes and consider class-balanced sampling |
| The validation split is class-aware but not scene-aware | Validation scores are optimistic and favour checkpoints that fit the training scenes, which makes checkpoint selection less reliable | Use the group-aware validation cut from the audit (Section 6.1, item 2) |
| Speed is measured on one GPU under PyTorch, with FP16 through autocast and CPU-side preprocessing | Rankings may change with other precision settings, export formats or hardware, and the YOLO11s FP16 figure is conservative | Benchmark true half-precision weights (`model.half()`) and GPU-side resizing, then export to ONNX or TensorRT on the same GPU |
| Model scales are not matched (YOLO11s has 9.4 M parameters, the other two about 43 M) | Speed and accuracy differences mix architecture with capacity | Train scale-matched variants (for example YOLO11m or YOLO11l) under the same protocol and compare at similar parameter counts; benchmark exported runtimes (ONNX, TensorRT) with true half-precision weights |
| One dataset and three detectors | Findings may not transfer to other datasets or to newer detectors | Evaluate on other, non-leaked Vietnamese traffic-sign datasets and add newer real-time detectors (for example RT-DETRv2, D-FINE, YOLOv10) |
| Edge-device deployment is not covered | No conclusion about embedded or in-vehicle use | Possible future work: export, quantization, embedded hardware and power measurement. Outside the scope of this project |

## 7. Conclusion

This project provides a reproducible pipeline for training and evaluating
traffic-sign detectors under one shared protocol. On the current test split,
RT-DETR R50vd reached the highest mAP@[.50:.95] (0.759), ahead of Faster R-CNN
(0.728) and YOLO11s (0.703). RT-DETR's advantage over Faster R-CNN is concentrated
in medium and large signs and in recall, and its advantage over YOLO11s is
largest at strict IoU thresholds. The results are reported together with their
scope: evaluation is at frame level; the checkpoint rule (highest validation mAP)
was adopted after the test scores of the loss-selected checkpoints had been seen,
and it changed Faster R-CNN's score by +0.033; YOLO11s uses a different inference
setup; the source annotations are incomplete for some images; and the speed
measurements describe one GPU under PyTorch only. On that GPU, YOLO11s is the
fastest model (19.6 ms end to end in FP32), RT-DETR the most accurate at about 3.7
times YOLO11s's latency, and Faster R-CNN sits between them in accuracy with the
longest FP32 latency. The clean-test and bootstrap checks show that RT-DETR's lead survives removing
near-duplicate frames and resampling noise, whereas the order of Faster R-CNN and
YOLO11s is not established. The proposed scene-group and independent-test
evaluations would show how well the findings carry over to unseen recordings.

## 8. References

- Dataset: Vietnam Road Signs Dataset (VNTS), Data Palpating Team, Kaggle
  (`maitam/vietnamese-traffic-signs`), CC-BY-SA 4.0.
- RT-DETR: Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection",
  CVPR 2024.
- Faster R-CNN: Ren et al., "Faster R-CNN: Towards Real-Time Object Detection
  with Region Proposal Networks", NeurIPS 2015; torchvision
  `fasterrcnn_resnet50_fpn_v2`.
- YOLO11: Ultralytics (Jocher and Qiu), 2024.
- COCO evaluation: Lin et al., "Microsoft COCO: Common Objects in Context",
  ECCV 2014, and `pycocotools`.

## Appendix A. Class list

Names come from the archive's `classes_en.txt`. The COCO category ID is
the RT-DETR label plus one (Faster R-CNN uses the COCO ID directly as its
label).

| COCO ID | Class |
|---:|---|
| 1 | Pedestrian Crossing |
| 2 | Equal-level Intersection |
| 3 | No Entry |
| 4 | Right Turn Only |
| 5 | Intersection |
| 6 | Intersection with a non-priority road |
| 7 | Danger zone on the left |
| 8 | No Left Turn |
| 9 | Bus Stop |
| 10 | Roundabout |
| 11 | No Stopping and No Parking |
| 12 | U-Turn Allowed |
| 13 | Lane Allocation |
| 14 | Slow Down |
| 15 | No Trucks Allowed |
| 16 | Narrow Road on the Right |
| 17 | Height Limit |
| 18 | No U-Turn |
| 19 | No Passenger Cars and Trucks |
| 20 | No U-Turn and No Right Turn |
| 21 | No Cars Allowed |
| 22 | Narrow Road on the Left |
| 23 | Uneven Road |
| 24 | No Two or Three-wheeled Vehicles |
| 25 | Customs Checkpoint |
| 26 | Motorcycles Only |
| 27 | Obstacle on the Road |
| 28 | Children Present |
| 29 | Trucks and Containers |
| 30 | No Motorcycles Allowed |
| 31 | Trucks Only |
| 32 | Road with Surveillance Camera |
| 33 | No Right Turn |
| 34 | Double curve first to right |
| 35 | No Containers Allowed |
| 36 | No Left or Right Turn |
| 37 | No Straight and Right Turn |
| 38 | Intersection with T-Junction |
| 39 | Speed limit (50km/h) |
| 40 | Speed limit (60km/h) |
| 41 | Speed limit (80km/h) |
| 42 | Speed limit (40km/h) |
| 43 | Left Turn |
| 44 | Low Clearance |
| 45 | Other Danger |
| 46 | One-way street |
| 47 | No Parking |
| 48 | No U-Turn for Cars |
| 49 | Level Crossing with Barriers |
| 50 | No U-Turn and No Left Turn |
| 51 | Danger zone on the right |
| 52 | Warning: Obstacle ahead – pass on the right |

Several classes form mirror-image pairs (for example No Left Turn / No Right
Turn, Danger zone on the left / on the right, Narrow Road on the Left / on the
Right, and No U-Turn and No Left Turn / No U-Turn and No Right Turn), and
others are visually similar (the four speed limits, and the vehicle-restriction
signs). These groups are the first places to look in the confusion matrix.