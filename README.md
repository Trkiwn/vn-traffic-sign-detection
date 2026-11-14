# Vietnamese Traffic Sign Detection

Final project by Group 5 for the Introduction to Computer Science course
(CSE2025, G1) at Vietnamese-German University (VGU).

## At a glance

| | |
|---|---|
| Task | Object detection of 52 Vietnamese traffic-sign classes (COCO bounding boxes) |
| Models | RT-DETR R50vd, Faster R-CNN (ResNet-50 FPN v2) and YOLO11s are trained and evaluated |
| Shared protocol | Same train/validation/test split, 640 x 640 stretched input, no augmentation, 50 epochs, COCO-pretrained weights, minimum validation-loss checkpoint selection, `pycocotools` COCO evaluation (YOLO11s differences are listed in Section 4) |
| Main result | On the test split, RT-DETR reaches mAP@[.50:.95] 0.757, versus 0.703 for YOLO11s and 0.695 for Faster R-CNN |
| How to read it | Scores measure performance on held-out frames from the collected scenes (see Section 6) |

## 1. Introduction

This project compares object-detection models on Vietnamese traffic-sign
images under a shared dataset, training/validation protocol, and COCO
evaluation. All three pipelines (RT-DETR, Faster R-CNN and YOLO11s) are
complete. Deployment-oriented measurements (latency, FPS,
parameter count, model size) are planned for a follow-up stage under
controlled hardware settings, so that accuracy and speed can be discussed
together.

The project is an exploratory course study of detector behaviour on a public
dataset. Its findings are intended for model comparison and are not a
validated driving system.

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
- Minimum validation loss for checkpoint selection (the loss definitions differ by model)
- The provided test images for final reporting only

Test images are not used to select checkpoints or tune thresholds. RT-DETR
training and validation do not access the test split; the explicit COCO
evaluator is run on the test split only after the checkpoint is selected.
Because the test set is defined by image membership in the source archive
rather than by recording, it measures performance on held-out frames from the
collected scenes (Section 6.1). The YOLO11s result has one qualification, described
in the YOLO11s section below.

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
(0.681), which is why the setting is documented.

Reported metrics are:

- `mAP@[.50:.95]`: primary COCO average precision across IoU thresholds
  0.50 through 0.95.
- `mAP@.50`: average precision at IoU 0.50.
- `mAP@.75`: average precision at the stricter IoU 0.75.
- `AP small`, `AP medium`, and `AP large`: AP grouped by object area.
- `AR@1`, `AR@10`, and `AR@100`: average recall with the corresponding
  maximum number of detections per image.

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
| Checkpoint rule | Lowest validation loss, evaluated every epoch during training | Lowest validation loss, evaluated every epoch during training | Lowest validation loss, computed offline over the 50 saved checkpoints (epoch 37) |

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
  for within-model checkpoint selection only; their values are not comparable
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
  (`val=False`) and all 50 epoch checkpoints were saved. The validation loss
  (box, classification and DFL terms, with the gains stored in each checkpoint)
  was then computed offline for every checkpoint on the 255 validation images,
  with the model in evaluation mode (32 batches of 8). The minimum, at epoch 37
  (`epoch36.pt` in Ultralytics' zero-based file naming), was 2.315651791 and was
  selected, with the earlier epoch winning ties. The other two models record validation loss during training,
  and the three loss definitions are not comparable with each other.
- **Retrospective test result.** The YOLO11s test set was scored once before
  this final rule was fixed. An earlier pilot selected a checkpoint by
  validation COCO AP (epoch 36, `epoch35.pt`) and scored mAP@[.50:.95] 0.7033
  on the test set; the checkpoint was then re-selected by minimum validation
  loss, to match the other two models, and scored 0.7032. The final selection
  uses validation data only, and the two scores differ by 0.0002, so the choice
  of rule has no practical effect here. The YOLO11s test result is still
  retrospective, because the test set had been viewed before the final rule was
  applied. Both runs are archived separately.
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

When a training output directory is supplied, the evaluator uses the best
model saved there at the end of training (lowest validation loss). If no saved
model is present, it scans the checkpoint directories for the lowest recorded
validation loss:

```bash
python scripts/DETR/evaluate/evaluation.py --split validation
python scripts/DETR/evaluate/evaluation.py --split test
```

For the reported run, the selected checkpoint was
`checkpoint-4320` (epoch 15):

```bash
python scripts/DETR/evaluate/evaluation.py \
  --checkpoint outputs/rtdetr_r50vd_comparison_50ep/checkpoint-4320 \
  --split test
```

Predictions and metrics are written below the evaluated checkpoint's
`evaluation/` directory.

#### Single-image visualization

Write detections as JSON (by default to `outputs/predictions/rtdetr/<image>.json`; override with `--output`). Labels are 0-based class indices, so the COCO category ID is the label plus one:

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
saved as `best.pt`. To resume a compatible run, set `RESUME_CHECKPOINT` to a
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
`pycocotools.COCOeval` metrics used by RT-DETR. Predictions come from
`predict_rcnn.py` (in the training-script folder), and predictions and metrics
are written next to the checkpoint as `<split>_predictions.json` and
`<split>_metrics.json`.

#### Single-image visualization

```bash
python "scripts/Faster R-CNN/visualisation/visualise_rcnn.py" \
  --image path/to/image.jpg \
  --checkpoint outputs/faster_rcnn_comparison_50ep/best.pt \
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

The repository holds the saved JSON evidence, not the model weights. The
50-epoch checkpoint series is kept on the run owner's Google Drive
(`VGU_TrafficSign_Project/yolo11s_colab_runs/yolo11s_colab_pilot_01/`). The
selected checkpoint is `training/weights/epoch36.pt`, with SHA-256
`05cf921fffd4c46df7b18ea8958c84e29bc228207a65590958c9d16cee42544c`. The
methodology, results and limitations are written up in
[docs/YOLO11s_EXPERIMENT_REPORT.md](docs/YOLO11s_EXPERIMENT_REPORT.md).

#### Evaluation

The archived test predictions can be re-scored on any machine, without a GPU or
the weights:

```bash
python -m pip install pycocotools==2.0.11
python scripts/YOLO11s/evaluate/replay_coco.py --split test
```

The script recomputes the COCO metrics from
`results/YOLO11s/min_val_loss/test_predictions.json` and checks them against the
recorded `test_metrics.json` (tolerance 1e-8). Adding `--track max_val_ap_pilot`
replays the earlier pilot instead; the pilot metrics must not be substituted for
the minimum-validation-loss results.

#### Loss plot and single-image visualization

```bash
python scripts/YOLO11s/visualisation/plot_val_loss.py

python scripts/YOLO11s/visualisation/visualise_yolo11s.py \
  --image path/to/image.jpg \
  --checkpoint /path/to/epoch36.pt \
  --threshold 0.5 \
  --output outputs/predictions/yolo11s/image_detected.png
```

The YOLO11s plot shows validation loss only, because validation was switched off
during training and the loss history was computed afterwards. The visualization
script reproduces the 640 x 640 stretched input and is a demo, not an evaluator.
It needs the `epoch36.pt` weights, which are not in the repository; without
`--output` it only prints the detections as JSON.

### Loss visualization and checkpoint selection

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

RT-DETR's validation loss reached its recorded minimum at epoch 15 and
generally increased afterwards while training loss continued to fall.
Faster R-CNN's validation loss reached `0.2639` at epoch 22 and then mostly
plateaued around `0.27` while training loss continued to decrease. Both
patterns indicate that the models stopped improving on held-out loss after
the selected epoch, which is why the validation-selected checkpoints are used
for reporting. Later checkpoints are not necessarily worse on detection
metrics; Section 6.3 describes selecting checkpoints by validation mAP as a
refinement.

Loss is used for checkpoint selection, and COCO mAP is the reporting metric.
mAP evaluates post-processed detections at multiple IoU thresholds, so the two
are related but not interchangeable.

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
  preprocessing/                  # Prepare COCO and YOLO dataset files; dataset_audit.ipynb
  visual_plot/                    # Shared training/validation loss plots
  data/                           # Dataset download/import
dataset/
  archive/                        # Downloaded source dataset
  model_ready/                    # Prepared images and annotations
outputs/                          # Local generated outputs; excluded from Git
results/
  YOLO11s/
    min_val_loss/                 # Reported run: predictions, metrics, loss history
    max_val_ap_pilot/             # Earlier pilot (checkpoint chosen by validation AP)
    summary.json                  # Selection and result summary for the YOLO11s run
docs/
  YOLO11s_EXPERIMENT_REPORT.md    # YOLO11s methodology, results and limitations
tests/                            # YOLO11s artifact and repository-structure tests
requirements.txt
```

See [requirements.txt](./requirements.txt) for project dependencies. PyTorch
and torchvision wheels remain platform- and CUDA-specific.

The root `.gitignore` excludes `outputs/`, so files generated there remain
local and are not included in GitHub commits. This directory currently holds
large checkpoints and other reproducible outputs (about 5.3 GB in the inspected
workspace). Committed experiment reports and selected result artifacts are
stored under `results/` and `docs/` instead.

### Tests

The `tests/` folder checks the archived YOLO11s evidence and the repository
layout. From the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests use only the standard library and need no GPU or weights.
`test_yolo11s_artifacts.py` verifies the provenance of the YOLO11s result files:
50 consecutive epochs, 255 validation images in 32 batches per epoch,
consistent checkpoint hashes, the minimum validation loss at epoch 37, 639
evaluated test images, 2,461 well-formed predictions (638 images have at least
one; one test image has none above the score floor), and that the earlier pilot
is preserved. `test_yolo11s_structure.py` checks that the YOLO11s entry points
exist, that `replay_coco.py` resolves the repository root, and that this README
names the correct paths.

## 5. Results and evaluation

The table reports test-set COCO results. Validation data was used for
checkpoint selection only and is not part of the reported comparison.

| Model | mAP@[.50:.95] | mAP@.50 | mAP@.75 | AP small | AP medium | AP large | AR@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR R50vd | 0.757 | 0.973 | 0.917 | 0.695 | 0.801 | 0.891 | 0.816 |
| Faster R-CNN | 0.695 | 0.961 | 0.849 | 0.671 | 0.736 | 0.834 | 0.755 |
| YOLO11s | 0.703 | 0.924 | 0.840 | 0.687 | 0.751 | 0.843 | 0.749 |

All rows are scored at score threshold 0.001 with validation-selected
checkpoints (Section 4). The YOLO11s row is a retrospective result, and its
selection and inference settings differ from the other two models (Section 4).

Under this protocol, RT-DETR R50vd reached the highest test mAP@[.50:.95]
(0.757), ahead of YOLO11s (0.703, -0.054) and Faster R-CNN (0.695, -0.062), and
the highest AR@100 (0.816, versus 0.749 and 0.755). Against Faster R-CNN,
RT-DETR's gap is +0.012 at IoU 0.50 and +0.068 at IoU 0.75, so the stricter
thresholds are where they separate. By object size the gap to Faster R-CNN is
+0.024 for small signs, +0.065 for medium and +0.057 for large signs. This
pattern is consistent with RT-DETR producing more precisely localized boxes; the
planned per-class and error analysis (Section 6.3) will test that explanation
directly.

YOLO11s and Faster R-CNN are close, and their order depends on the metric.
YOLO11s is higher on mAP@[.50:.95] (+0.008) and in all three size buckets,
while Faster R-CNN is higher on mAP@.50 (0.961 versus 0.924), mAP@.75 and
AR@100. Because mAP@[.50:.95] averages ten IoU thresholds, YOLO11s's boxes
appear tighter at the highest IoUs while it finds fewer objects at IoU 0.50.
Part of the lower mAP@.50 may come from its single-class-per-box inference
(Section 4). Without repeated runs, this README makes no claim about the order of
these two models. AR@1 and AR@10 are available in the evaluation outputs.

The Faster R-CNN checkpoint used is the validation-selected `best.pt` (epoch
22). Its loss curve supports choosing this checkpoint over the final one.

Accuracy is one half of a deployment comparison. Statements about real-time
suitability will be made after latency, FPS, parameter count, and model size
are measured under the controlled protocol in Section 6.3.

## 6. Scope of the results and future work

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
here. The planned checks below measure how well these results carry over.

**Planned checks, ordered by cost:**

1. **Clean-test evaluation (no retraining).** Re-score the saved test predictions of the
   three models on the subset of test images that have no near-duplicate in the training
   set (407 of the 639 at distance greater than 4, as in the audit's `split_v1.csv`; 290 at the stricter distance greater than 6, measured against all 2,552 provided training images). If
   the ranking and the gap persist on this subset, the comparison is robust to
   the frame-level leakage. The subset is smaller and may be harder, so it is
   reported next to the full test set, not instead of it.
2. **Scene-group split.** Cluster images into approximate scene groups using
   perceptual hashes or image embeddings together with filename adjacency.
   Exact recording IDs are not required for this: merging too many images into
   one group only makes the split more conservative, so approximate grouping
   is a safe way to prevent leakage. If time allows, use the groups for a new
   train/validation/test split. A first version exists in the audit notebook
   (`split_v1.csv`: near-duplicate groups within the training list at distance 4
   and a group-aware validation cut of 426 images, leaving 2,126 for training);
   the provided test set is unchanged there.
3. **Independent test set.** Photograph or film a small set of new street
   scenes (about 100-200 images) and annotate only the 52 classes. This gives
   a test set that is independent by construction.
4. **Uncertainty estimates.** Report bootstrap confidence intervals, resampling
   blocks of consecutive frames and using the same resamples for all models,
   so that correlated neighbouring frames do not make the intervals too narrow.

### 6.2 Source annotations

The source annotations do not label every visible sign. For example,
`1671.jpg` contains several signs in the background but has one labelled
object. This is a property of the source dataset, not of the preprocessing.
COCO evaluation measures performance on labelled objects, so a correct
detection of an unlabelled sign is counted as a false positive and measured
precision can be understated, while an unlabelled sign cannot be counted as a
miss.

**Planned check:** review a sample of high-confidence false positives from all
three models, report how many correspond to genuine unlabelled signs, and report an
annotation-adjusted precision next to the standard metrics. Qualitative
predictions should be read alongside the provided annotations.

## 7. Conclusion

This project provides a reproducible pipeline for training and evaluating
traffic-sign detectors under one shared protocol. On the current test split,
RT-DETR R50vd reached the highest mAP@[.50:.95] (0.757), ahead of YOLO11s (0.703)
and Faster R-CNN (0.695), whose order depends on the metric; RT-DETR's advantage
is largest at stricter IoU thresholds. The results are reported together with
their scope: evaluation is at frame level, the YOLO11s result is retrospective
and uses a different checkpoint-selection and inference setup, the source
annotations are incomplete for some images, and the efficiency measurements are
still to come. The planned clean-test, scene-group, and independent-test
evaluations will show how well the findings carry over to unseen recordings.

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