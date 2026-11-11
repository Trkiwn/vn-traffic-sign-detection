# YOLO11s — Minimum validation loss and auditable historical pilot

**Main result (retrospective):** epoch 37/50 (`epoch36.pt`), validation loss 2.315651791, COCO test mAP@[.50:.95] 0.7031666802.

- Notebook: `models/VGU_YOLO11s_MinValLoss_GitHub_Ready.ipynb`
- Main metrics + predictions: `../../results/YOLO11s/min_val_loss/`
- Original max-AP pilot: `../../results/YOLO11s/max_val_ap_pilot/` (do not merge results).
- Report: `../../docs/YOLO11s_EXPERIMENT_REPORT.md`

**Offline evaluation replay** from project root (needs `pycocotools`, but no GPU or weights):

```bash
python -m pip install pycocotools==2.0.11
python scripts/YOLO11s/evaluate/replay_coco.py --split test
python scripts/YOLO11s/evaluate/replay_coco.py --split test --track max_val_ap_pilot
python scripts/YOLO11s/evaluate/replay_coco.py --split validation --track max_val_ap_pilot
```

The replay script uses `dataset/model_ready/annotations/instances_{split}.json`; the prepared split must match the saved experiment. Min-loss selection means **minimum sum of box/cls/DFL validation loss** on all 50 frozen checkpoints, earliest epoch to break ties. This was retrospectively applied after pilot test results were known. Historical possible scene leakage remains. Full training/inference reproduction requires checkpoint weights and a GPU.

## Directory organization

- `models/`: canonical Colab notebook for training and retrospective checkpoint selection; there is **no** standalone `train.py` in this historical run.
- `evaluate/`: independent replay of archived COCO predictions on local test annotations.
- `visualisation/`: per-image rendering and historical validation-loss curve.

For comparison, this follows the same three-directory naming convention as `scripts/DETR/` and `scripts/Faster R-CNN/`, without implying that the training runtimes or implementations were identical.
