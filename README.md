# Vietnamese Traffic Sign Detection

Final project by Group 5 for the Introduction to Computer Science course
(CSE2025, G1) at Vietnamese-German University (VGU).

## At a glance

| | |
|---|---|
| Task | Object detection of 52 Vietnamese traffic-sign classes (COCO bounding boxes) |
| Models | RT-DETR R50vd and Faster R-CNN (ResNet-50 FPN v2) are trained and evaluated; YOLO11s is in progress |
| Shared protocol | Same train/validation/test split, 640 x 640 input, 50 epochs, COCO-pretrained weights, `pycocotools` COCO evaluation |
| Main result | On the test split, RT-DETR reaches mAP@[.50:.95] 0.757 versus 0.681 for Faster R-CNN |
| How to read it | Scores measure performance on held-out frames from the collected scenes (see Section 6) |

## 1. Introduction

This project compares object-detection models on Vietnamese traffic-sign
images under a shared dataset, training/validation protocol, and COCO
evaluation. The RT-DETR and Faster R-CNN pipelines are complete. YOLO11s is
the next model to be added. Deployment-oriented measurements (latency, FPS,
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
[Vietnamese Traffic Signs dataset on Kaggle](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs).
The archive contains 3,216 images with 8,334 bounding boxes in 52 classes,
mostly identified by QCVN sign codes. The prepared dataset provides the same
52 categories in COCO format.

<!-- TODO(confirm): dataset licence and attribution requirements from the Kaggle page. -->

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

<!-- TODO(confirm): how the validation images were selected from the training list (random, stratified, or grouped by scene), and whether val/train class distributions were compared. -->

Dataset properties that matter for interpreting the results:

- **Class imbalance.** Classes range from 3 to 1,071 boxes. Aggregate mAP is
  dominated by the common classes, so per-class results are reported
  separately (Section 6.3).
- **Small objects.** In the dataset audit, 41.5% of boxes are smaller than
  16 px at 640 px input (18.6% at 960 px). `AP small` is therefore an
  informative metric for this dataset.
- **Validation density.** Validation images contain more boxes per image
  (4.12) than train (2.45) or test (2.57). Validation is used only for
  checkpoint selection.

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
machine or runtime.

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

<!-- TODO(confirm): GPU model and count of the Kaggle runs, and the commit hash used for the reported results. -->

## 4. Methodology

### Common experimental protocol

The completed RT-DETR and Faster R-CNN runs share:

- Prepared train, validation, and test splits (the same annotation files)
- 640 x 640 image processing (images are resized to a square, not letterboxed)
- A 50-epoch training budget
- COCO-pretrained starting weights with a new 52-class prediction head
- Seed 42 (see the reproducibility notes below)
- COCO bounding-box annotations
- Validation loss for checkpoint selection
- The provided test images for final reporting only

Test images are not used to select checkpoints or tune thresholds. RT-DETR
training and validation do not access the test split; the explicit COCO
evaluator is run on the test split only after the checkpoint is selected.
Because the test set is defined by image membership in the source archive
rather than by recording, it measures performance on held-out frames from the
collected scenes (Section 6.1).

Both evaluators use `pycocotools.COCOeval` with `iouType="bbox"`. Evaluation
uses a score threshold of `0.001` so low-confidence detections remain
available to the COCO precision-recall calculation. This differs from the
`0.5` default used for readable demo visualizations.

<!-- TODO(confirm, from the evaluator scripts): predictions are mapped back to original image coordinates before COCOeval for both models; category IDs are mapped to COCO IDs 1-52 for both models; maxDets setting. -->

Reported metrics are:

- `mAP@[.50:.95]`: primary COCO average precision across IoU thresholds
  0.50 through 0.95.
- `mAP@.50`: average precision at IoU 0.50.
- `mAP@.75`: average precision at the stricter IoU 0.75.
- `AP small`, `AP medium`, and `AP large`: AP grouped by object area.
- `AR@1`, `AR@10`, and `AR@100`: average recall with the corresponding
  maximum number of detections per image.

### Training configurations

| | RT-DETR R50vd | Faster R-CNN | YOLO11s (planned) |
|---|---|---|---|
| Library | Hugging Face Transformers | torchvision | Ultralytics |
| Starting weights | `PekingU/rtdetr_r50vd` (COCO) | `FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT` (COCO) | COCO-pretrained `yolo11s.pt` |
| Prediction head | Re-initialised for 52 labels (0-51, no background) | Replaced: 53 outputs (background + 52) | Built by Ultralytics for 52 classes |
| Input | Resize to 640 x 640 (`RTDetrImageProcessor`) | Resize to 640 x 640; boxes scaled to match | Letterbox to 640 (Ultralytics default) |
| Augmentation | None in the training script | None in the training script | To be set explicitly (see below) |
| Optimizer | Trainer default (AdamW), lr `5e-5`, weight decay `1e-4` | SGD, lr `0.005`, momentum `0.9`, weight decay `5e-4` | To be documented |
| LR schedule | Trainer default linear decay, 100 warm-up steps | Step x0.1 at epochs 30 and 40, no warm-up | To be documented |
| Batch | 2 per device x accumulation 2 (8 on two GPUs) | 2 per device x accumulation 2 (8 on two GPUs) | To be documented |
| Precision | FP16 | Mixed precision (AMP, FP16) | AMP |
| Epochs | 50 | 50 | 50 |
| Checkpoint rule | Lowest validation loss, evaluated every epoch | Lowest validation loss, evaluated every epoch | To be documented |

Each model was trained with one configuration; no per-model hyperparameter
search is reported.

<!-- TODO(confirm): that no hyperparameter search was run for either model. -->

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
- **Loss definitions.** RT-DETR's validation loss comes from Hugging Face's
  set-matching objective, and Faster R-CNN's is the sum of its four torchvision
  loss terms. Both are used for within-model checkpoint selection only; their
  values are not comparable between models.
- **Faster R-CNN validation mode.** torchvision detectors return losses only
  in training mode, so validation loss is computed under `model.train()` with
  gradients disabled. This also leaves normalization layers in training
  mode: batch statistics are used and running statistics are updated on
  validation images. The effect is expected to be small. The planned
  retraining (Section 6.3) switches these layers to evaluation mode during
  validation.
- **Checkpoint retention.** RT-DETR keeps the best checkpoint plus the most
  recent ones (`save_total_limit=3`). Faster R-CNN saves `best.pt` and a
  checkpoint every five epochs.

### YOLO11s protocol (planned)

To keep YOLO11s comparable with the other two models, the plan is to:

- Use the same prepared split through `data.yaml`, the same 640-pixel input
  size, the same 50-epoch budget, seed 42, and COCO-pretrained weights.
- Set augmentation explicitly and record it in the table above. Ultralytics
  enables mosaic, colour jitter and horizontal flipping by default. Mirroring
  can change the meaning of direction-dependent signs, so horizontal flipping
  will be disabled.
- Export predictions at score threshold `0.001` in COCO format and score them
  with the same `COCOeval` script, because Ultralytics' built-in mAP is
  computed slightly differently.
- Report the checkpoint-selection rule used (Ultralytics selects `best.pt` by
  a validation mAP-based fitness score, whereas the other two use validation
  loss).

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

When an output directory is supplied, the evaluator selects the checkpoint
with the lowest recorded validation loss:

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

The recorded Faster R-CNN results were produced before the final partial
gradient-accumulation group was scaled correctly. The correction affects one
optimizer update per epoch when the number of batches is odd. The planned
retraining (Section 6.3) regenerates the results with the corrected
implementation.

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
`pycocotools.COCOeval` metrics used by RT-DETR.

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
  preprocessing/                  # Prepare COCO and YOLO dataset files
  visual_plot/                    # Shared training/validation loss plots
  data/                           # Dataset download/import
dataset/
  archive/                        # Downloaded source dataset
  model_ready/                    # Prepared images and annotations
outputs/                          # Training, evaluation, prediction, and plot outputs
requirements.txt
```

See [requirements.txt](./requirements.txt) for project dependencies. PyTorch
and torchvision wheels remain platform- and CUDA-specific.

## 5. Results and evaluation

The table reports test-set COCO results. Validation data was used for
checkpoint selection. Validation-set metrics will be added alongside the test
metrics in a later update.

| Model | mAP@[.50:.95] | mAP@.50 | mAP@.75 | AP small | AP medium | AP large | AR@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR R50vd | 0.757 | 0.973 | 0.917 | 0.695 | 0.801 | 0.891 | 0.816 |
| Faster R-CNN | 0.681 | 0.936 | 0.845 | 0.667 | 0.724 | 0.812 | 0.731 |
| YOLO11s | In progress | - | - | - | - | - | - |

Under this protocol, RT-DETR R50vd reached a higher test mAP@[.50:.95]
(0.757 versus 0.681, +0.076) and AR@100 (0.816 versus 0.731, +0.085) than
Faster R-CNN. Both models exceed 0.93 at IoU 0.50, so the stricter IoU
thresholds separate them more clearly: the gap is +0.037 at mAP@.50 and +0.072
at mAP@.75. The gap is also larger for medium and large signs (+0.077 and
+0.079) than for small signs (+0.028). This pattern is consistent with
RT-DETR producing more precisely localized boxes; the planned per-class and
error analysis (Section 6.3) will test that explanation directly. AR@1 and
AR@10 are available in the evaluation outputs.

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
both sides. In the dataset audit, the share of test images with a near
neighbour in the training set (perceptual-hash Hamming distance of at most 6)
was 54.6%, close to the 56.2% share of training images with a near neighbour
elsewhere in the training set. A manual review of 60 sampled train/test pairs
at that distance judged all of them to be the same scene, and 232 near-duplicate
groups contain both train and test images.

<!-- TODO(confirm): link the audit notebook / document that produced these figures. -->

The reported scores therefore describe performance on held-out frames from the
collected scenes. Every model is trained and tested on the same split, so the
comparison between models is made under identical conditions. Absolute scores
are likely to be higher than performance on entirely new recordings, and the
ranking of models on new recordings is not guaranteed to match the ranking
here. The planned checks below measure how well these results carry over.

**Planned checks, ordered by cost:**

1. **Clean-test evaluation (no retraining).** Re-evaluate the completed models
   on the subset of test images that have no near-duplicate in the training
   set (290 of the 639 images at distance 6 in the YOLO data preparation). If
   the ranking and the gap persist on this subset, the comparison is robust to
   the frame-level leakage. The subset is smaller and may be harder, so it is
   reported next to the full test set, not instead of it.
2. **Scene-group split.** Cluster images into approximate scene groups using
   perceptual hashes or image embeddings together with filename adjacency.
   Exact recording IDs are not required for this: merging too many images into
   one group only makes the split more conservative, so approximate grouping
   is a safe way to prevent leakage. If time allows, use the groups for a new
   train/validation/test split.
3. **Independent test set.** Photograph or film a small set of new street
   scenes (about 100-200 images) and annotate only the 52 classes. This gives
   a test set that is independent by construction.
4. **Uncertainty estimates.** Report bootstrap confidence intervals, resampling
   blocks of consecutive frames and using the same resamples for both models,
   so that correlated neighbouring frames do not make the intervals too narrow.

### 6.2 Source annotations

The source annotations do not label every visible sign. For example,
`1671.jpg` contains several signs in the background but has one labelled
object. This is a property of the source dataset, not of the preprocessing.
COCO evaluation measures performance on labelled objects, so a correct
detection of an unlabelled sign is counted as a false positive and measured
precision can be understated, while an unlabelled sign cannot be counted as a
miss.

**Planned check:** review a sample of high-confidence false positives from both
models, report how many correspond to genuine unlabelled signs, and report an
annotation-adjusted precision next to the standard metrics. Qualitative
predictions should be read alongside the provided annotations.

## 7. Conclusion

This project provides a reproducible pipeline for training and evaluating
traffic-sign detectors under one shared protocol. On the current test split,
RT-DETR R50vd reached higher mAP@[.50:.95] and AR@100 than Faster R-CNN (0.757
versus 0.681 and 0.816 versus 0.731), with the largest advantage at stricter
IoU thresholds. The results are reported together with their scope:
evaluation is at frame level, the source annotations are incomplete for some
images, and YOLO11s and the efficiency measurements are still to come. The
planned clean-test, scene-group, and independent-test evaluations will show
how well the findings carry over to unseen recordings.

## 8. References

- Dataset: Vietnamese Traffic Signs, Kaggle (`maitam/vietnamese-traffic-signs`).
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