#!/usr/bin/env python3
"""Replay saved YOLO11s COCO bbox evaluation without a GPU or YOLO weights.

Usage (repository root):
    python -m pip install pycocotools==2.0.11
    python scripts/YOLO11s/evaluate/replay_coco.py --split test
    python scripts/YOLO11s/evaluate/replay_coco.py --split test --track max_val_ap_pilot
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
import math
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "results" / "YOLO11s" / "min_val_loss"
ANN = ROOT / "dataset" / "model_ready" / "annotations"
METRIC_NAMES = (
    "map", "map_50", "map_75", "map_small", "map_medium", "map_large",
    "mar_1", "mar_10", "mar_100", "mar_small", "mar_medium", "mar_large"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("test", "validation"), default="test")
    ap.add_argument("--data-root", type=Path, default=ROOT / "dataset" / "model_ready")
    ap.add_argument("--results", type=Path, default=ARTIFACTS)
    ap.add_argument("--track", choices=("min_val_loss", "max_val_ap_pilot"), default="min_val_loss")
    ap.add_argument("--tolerance", type=float, default=1e-8)
    args = ap.parse_args()
    if args.results == ARTIFACTS:
        args.results = ARTIFACTS.parent / args.track
    if args.track == "min_val_loss" and args.split == "validation":
        ap.error("min_val_loss has validation-loss history, not validation COCO predictions; use --track max_val_ap_pilot --split validation")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    suffix = "test" if args.split == "test" else "validation"
    source = (args.data_root / "annotations" / f"instances_{suffix}.json")
    pred = (args.results / ("test_predictions.json" if args.split == "test" else "validation_predictions.json"))
    expected = args.results / ("test_metrics.json" if args.split == "test" else "validation_metrics.json")
    if not source.is_file() or not pred.is_file() or not expected.is_file():
        raise FileNotFoundError("Missing COCO annotations or saved predictions/metrics")
    gt = COCO(str(source))
    predictions = json.loads(pred.read_text(encoding="utf-8"))
    recorded = json.loads(expected.read_text(encoding="utf-8"))
    per_image = Counter()
    for row in predictions:
        image_id, category_id, score, bbox = (row["image_id"], row["category_id"],
                                              row["score"], row["bbox"])
        if image_id not in gt.imgs or category_id not in gt.cats:
            raise ValueError("Prediction references an unknown image or class")
        if not (0.001 <= score <= 1.0) or not all(math.isfinite(x) for x in bbox):
            raise ValueError("Non-finite bounding box or invalid confidence score")
        if len(bbox) != 4 or min(bbox[2:]) <= 0:
            raise ValueError("Invalid bounding box extent")
        per_image[image_id] += 1
    if per_image and max(per_image.values()) > 100:
        raise ValueError("More than 100 detections for one image")
    if len(predictions) != recorded["predictions"] or len(gt.imgs) != recorded["evaluated_images"]:
        raise RuntimeError("Saved metrics and input sizes disagree")
    det = gt.loadRes(predictions) if predictions else COCO()
    if not predictions:
        det.dataset={"images": gt.dataset["images"], "categories": gt.dataset["categories"], "annotations": []}
        det.createIndex()
    e = COCOeval(gt, det, "bbox")
    e.params.maxDets=[1,10,100]
    e.evaluate(); e.accumulate(); e.summarize()
    got=dict(zip(METRIC_NAMES,map(float,e.stats)))
    problems={k:(got[k],recorded[k]) for k in got
              if not math.isclose(got[k],recorded[k],rel_tol=0,abs_tol=args.tolerance)}
    for k,v in got.items():
        print(f"{k:15s} {v:.9f}   recorded={recorded[k]:.9f}")
    if problems:
        raise SystemExit(f"COCOeval mismatch: {problems}")
    print(f"PASS: {args.split} COCOeval matches archived metrics within ±{args.tolerance}")


if __name__=="__main__":
    main()
