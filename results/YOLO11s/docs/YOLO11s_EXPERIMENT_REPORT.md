# YOLO11s — Retrospective Minimum-Validation-Loss Experiment Report

**Vietnamese Traffic Sign Detection · VGU Group 5 · November 11, 2026**

## Main finding

The recorded 50-epoch YOLO11s training run was **not retrained**. Its stored checkpoints were retrospectively rescored on the validation set using Ultralytics 8.4.189 detection loss. The lowest validation-loss checkpoint was **epoch 37/50 (`epoch36.pt`)**. That checkpoint was subsequently evaluated on the historical test split.

| Item | Verified result |
|:--|--:|
| Train/validation/test images | 2,297 / 255 / 639 |
| Classes | 52 |
| Training epochs and checkpoint hashes | 50 |
| Minimum total validation loss | 2.315651791 |
| Box / classification / DFL loss | 0.837826 / 0.571851 / 0.905976 |
| Selected checkpoint | epoch 37/50 (`epoch36.pt`) |
| Selected checkpoint SHA-256 | `05cf921fffd4c46df7b18ea8958c84e29bc228207a65590958c9d16cee42544c` |
| Test predictions | 2,461 |
| COCO mAP@[.50:.95] | 0.703166680 |
| AP50 | 0.924056834 |
| AP75 | 0.839992754 |
| AP small / medium / large | 0.687156 / 0.751272 / 0.842518 |
| AR@1 / AR@10 / AR@100 | 0.705292 / 0.749060 / 0.749060 |

## Historical descriptive comparison

| Model | Selected by | Test COCO mAP@[.50:.95] | AP50 | AP75 | AR@100 |
|:--|:--|--:|--:|--:|--:|
| RT-DETR R50vd | Minimum validation loss | 0.757 | 0.973 | 0.917 | 0.816 |
| **YOLO11s** | **Minimum validation loss (retrospective)** | **0.703167** | **0.924057** | **0.839993** | **0.749060** |
| Faster R-CNN ResNet50-FPN-v2 | Minimum validation loss | 0.681 | 0.936 | 0.845 | 0.731 |

Numbers for RT-DETR and Faster R-CNN are as reported by the existing repository README, not independently recomputed in the package-build step. Their native training loss functions are **architecture-specific** and cannot be compared numerically to YOLO's gain-weighted sum. The shared *selection rule* increases protocol consistency; it does not remove differences in preprocessing/training setup.

## Checkpoint selection methods

- **Original (historical) pilot:** choose the maximum validation COCO AP across 50 checkpoints; selected epoch **36** (`epoch35.pt`); recorded test COCO mAP@[.50:.95] **0.703349221**.
- **Revised (current, retrospective):** calculate mean of per-batch gain-weighted `box_loss`, `cls_loss` and `dfl_loss` across **255 validation images / 32 batches**; choose their smallest sum, breaking ties toward the earliest epoch. Selected **epoch 37**, with loss **2.315651791**. The logged loss is a **batch-averaged** convention, not necessarily an equally image-weighted average because the last batch is smaller.
- The min-loss-selected test COCO mAP is **0.703166680**, lower by **0.018254 percentage points** from the prior pilot, indicating checkpoint policy consistency rather than a tuned accuracy gain.

**Evaluation:** COCOeval `bbox`; confidence threshold 0.001; NMS IoU 0.7; 100 max detections per image; `maxDets=[1,10,100]`. Preprocessing is RGB bicubic stretch to 640×640, without letterbox padding. Inference maps boxes back to original image coordinates.

## Evidence and reproducibility

`results/YOLO11s/min_val_loss/` contains `val_loss_history.json` (50 epochs), `selected_min_val_loss.json`, the 50-checkpoint SHA manifest, `test_protocol.json`, and the saved 2,461 COCO predictions/metrics. The original pilot has not been deleted; it is isolated under `results/YOLO11s/max_val_ap_pilot/`. It is important not to mix its checkpoint and outputs with the current run.

The packaging checks verified the 50 history epochs, 255 images / 32 batches per epoch, matching checkpoint SHA metadata, metric/prediction counts, 52 per-class records, all prediction IDs and ranges, the 100-detection limit, and the minimum-loss epoch. They **do not** constitute an independent COCOeval recomputation. To replay COCO metrics from the saved predictions and repo test annotations, run:

```bash
python -m pip install pycocotools==2.0.11
python scripts/YOLO11s/evaluate/replay_coco.py --split test
```

The GitHub notebook defaults to `MODE="review"` (read-only archived displays). `MODE="select"` validates the existing Drive checkpoints / reruns selection if needed, **without training**; `MODE="reproduce"` may train and needs a clean GPU environment and copied prepared data. The checkpoint weights are not bundled; they remain in the owner's Drive.

## Scientific limitations

1. **Scene-level leakage:** The existing historical split has unresolved near-frame/train-test scene overlaps. Prior reviews flagged at least 55/639 test images; therefore this is **not** a leakage-free benchmark. SHA-256 exact hash audit cannot rule out scene overlap.
2. **Test already inspected:** The minimum validation-loss rule was applied after observing results from the earlier max-AP pilot. Even though the revised epoch was chosen from validation data, the new test is **retrospective**, not an independent prospective hold-out.
3. **Model-specific objectives:** Minimum loss is the same decision rule, but numeric RT-DETR / Faster R-CNN / YOLO11s losses have different meaning and scales.
4. **Preprocessing/hardware differences:** The original YOLO11s Colab environment differs from the repository's Kaggle pinned environment; end-to-end runtime replication and identical speed benchmarking are still outstanding.
5. **No independent COCOeval replay yet:** This package's validation is a JSON/provenance check, not a second pycocotools execution on the predictions. Run the replay command above before merging.
6. **FPS/latency not benchmarked:** No real-time safety or driving-performance inference is warranted.

## Status

Historical retrospective experiment: **complete**. Controlled, scene-disjoint three-architecture benchmark: **not yet complete**.
