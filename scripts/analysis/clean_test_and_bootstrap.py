#!/usr/bin/env python3
"""Clean-test re-scoring and block-bootstrap confidence intervals (no retraining).

Works directly on the saved COCO prediction files of the three models:

1. Re-scores each model on the test images that have no near-duplicate in the
   training set (the "clean" test subsets from the dataset audit).
2. Computes block-bootstrap 95% confidence intervals for each model and for the
   differences between models. Consecutive test images (by file number) are
   resampled in blocks, so neighbouring frames of one recording stay together
   and the intervals are not too narrow. All models use the same resamples, so
   differences are paired.

Per-image matching is done once with pycocotools; every bootstrap replicate
then re-accumulates AP/AR from those cached results with numpy. At start-up the
fast computation is checked against pycocotools on the full test set.

Usage (repository root):
    python scripts/analysis/clean_test_and_bootstrap.py --pairs path/to/near_duplicate_pairs.csv
    python scripts/analysis/clean_test_and_bootstrap.py --clean-csv path/to/split_v1.csv
    python scripts/analysis/clean_test_and_bootstrap.py            # full-test bootstrap only

Inputs from the audit notebook (scripts/preprocessing/dataset_audit.ipynb):
    near_duplicate_pairs.csv  columns stem_a, split_a, stem_b, split_b, distance
    split_v1.csv              columns stem, provided, role, clean_test (distance > 4)
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RTDETR = ROOT / "outputs" / "rtdetr_r50vd_comparison_50ep"
RCNN = ROOT / "outputs" / "faster_rcnn_comparison_50ep"
YOLO = ROOT / "outputs" / "yolo11s"
REPORTED = {
    "RT-DETR R50vd": RTDETR / "checkpoint-14400" / "evaluation" / "test_predictions.json",
    "Faster R-CNN": RCNN / "test_predictions_ep40.json",
    "YOLO11s": YOLO / "max_val_ap_pilot" / "test_predictions.json",
}
EARLIER_RULE = {
    "RT-DETR R50vd (loss rule)": RTDETR / "checkpoint-4320" / "evaluation" / "test_predictions.json",
    "Faster R-CNN (loss rule)": RCNN / "test_predictions_best_ep22.json",
    "YOLO11s (loss rule)": YOLO / "min_val_loss" / "test_predictions.json",
}
METRICS = ("map", "map_50", "map_75", "mar_100")


# --------------------------------------------------------------------------
# Fast COCO accumulation over any multiset of images
# --------------------------------------------------------------------------
class FastCOCO:
    """Recompute COCO AP/AR (area 'all', maxDets 100) for any image multiset.

    Built from one pycocotools `evaluate()` call; `metrics(counts)` then needs
    only numpy. `counts[i]` is how many times image position `i` is included
    (0 excludes it, values above 1 are bootstrap repeats).
    """

    def __init__(self, evaluator):
        p = evaluator.params
        self.n_images = len(p.imgIds)
        self.rec_thrs = np.asarray(p.recThrs)
        self.n_iou = len(p.iouThrs)
        n_area = len(p.areaRng)
        self.cats = []
        for k in range(len(p.catIds)):
            scores, d_img, d_match, d_ign, g_ign, g_img = [], [], [], [], [], []
            for i in range(self.n_images):
                e = evaluator.evalImgs[k * n_area * self.n_images + i]  # area 'all' comes first
                if e is None:
                    continue
                n_det = len(e["dtScores"])
                if n_det:
                    scores.append(np.asarray(e["dtScores"], dtype=float))
                    d_img.append(np.full(n_det, i))
                    d_match.append(np.asarray(e["dtMatches"]) != 0)
                    d_ign.append(np.asarray(e["dtIgnore"]).astype(bool))
                gt = np.asarray(e["gtIgnore"])
                if gt.size:
                    g_ign.append(gt)
                    g_img.append(np.full(gt.size, i))
            self.cats.append({
                "score": np.concatenate(scores) if scores else np.zeros(0),
                "det_img": np.concatenate(d_img) if d_img else np.zeros(0, dtype=np.int64),
                "match": np.concatenate(d_match, axis=1) if d_match else np.zeros((self.n_iou, 0), dtype=bool),
                "ign": np.concatenate(d_ign, axis=1) if d_ign else np.zeros((self.n_iou, 0), dtype=bool),
                "gt_valid": (np.concatenate(g_ign) == 0).astype(np.int64) if g_ign else np.zeros(0, dtype=np.int64),
                "gt_img": np.concatenate(g_img) if g_img else np.zeros(0, dtype=np.int64),
            })

    def metrics(self, counts):
        counts = np.asarray(counts, dtype=np.int64)
        ap_rows, recall_rows = [], []
        for cat in self.cats:
            n_gt = int(np.dot(counts[cat["gt_img"]], cat["gt_valid"])) if cat["gt_img"].size else 0
            if n_gt == 0:
                continue  # pycocotools skips categories without ground truth
            weights = counts[cat["det_img"]] if cat["det_img"].size else np.zeros(0, dtype=np.int64)
            idx = np.repeat(np.arange(weights.size), weights)
            n_det = idx.size
            if n_det == 0:
                ap_rows.append(np.zeros(self.n_iou))
                recall_rows.append(np.zeros(self.n_iou))
                continue
            idx = idx[np.argsort(-cat["score"][idx], kind="mergesort")]
            match, ign = cat["match"][:, idx], cat["ign"][:, idx]
            tp = np.cumsum(match & ~ign, axis=1, dtype=np.float64)
            fp = np.cumsum(~match & ~ign, axis=1, dtype=np.float64)
            rc = tp / n_gt
            pr = tp / (fp + tp + np.spacing(1))
            pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
            pos = np.stack([np.searchsorted(rc[t], self.rec_thrs, side="left") for t in range(self.n_iou)])
            q = np.where(pos < n_det, np.take_along_axis(pr, np.minimum(pos, n_det - 1), axis=1), 0.0)
            ap_rows.append(q.mean(axis=1))
            recall_rows.append(rc[:, -1])
        if not ap_rows:
            return {m: float("nan") for m in METRICS}
        ap, recall = np.stack(ap_rows), np.stack(recall_rows)
        return {"map": float(ap.mean()), "map_50": float(ap[:, 0].mean()),
                "map_75": float(ap[:, 5].mean()), "mar_100": float(recall.mean())}


# --------------------------------------------------------------------------
# Subsets, blocks and bootstrap
# --------------------------------------------------------------------------
def stem_key(stem):
    return (0, int(stem)) if re.fullmatch(r"\d+", stem) else (1, stem)


def clean_stems_from_pairs(pairs_csv, test_stems, deltas):
    """Test stems whose nearest provided-train image is farther than each delta."""
    nearest = {}
    with open(pairs_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sa, sb = row["split_a"], row["split_b"]
            if {sa, sb} != {"train", "test"}:
                continue
            test_stem = row["stem_a"] if sa == "test" else row["stem_b"]
            distance = float(row["distance"])
            nearest[test_stem] = min(distance, nearest.get(test_stem, math.inf))
    return {d: {s for s in test_stems if nearest.get(s, math.inf) > d} for d in deltas}


def clean_stems_from_csv(split_csv):
    with open(split_csv, newline="", encoding="utf-8") as handle:
        return {row["stem"] for row in csv.DictReader(handle)
                if row["role"] == "test" and row["clean_test"].strip().lower() == "true"}


def make_blocks(positions, block_size):
    return [np.asarray(positions[i:i + block_size]) for i in range(0, len(positions), block_size)]


def bootstrap(models, blocks, n_images, replicates, rng):
    """Paired block bootstrap: the same resampled image counts for every model."""
    out = {name: {m: np.zeros(replicates) for m in METRICS} for name in models}
    for r in range(replicates):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        counts = np.bincount(np.concatenate([blocks[b] for b in chosen]), minlength=n_images)
        for name, model in models.items():
            for metric, value in model.metrics(counts).items():
                out[name][metric][r] = value
    return out


def interval(values, point):
    """95% percentile interval, re-centred on the full-sample value.

    AP computed on bootstrap resamples is biased upwards (about +0.008 mAP in
    tests, because resampling repeats images). The bias is removed by shifting
    the replicate distribution so that its mean equals the point estimate.
    Differences between models are almost unaffected (bias close to 0).
    """
    lo, hi = np.percentile(values, [2.5, 97.5])
    shift = point - float(np.mean(values))
    return float(lo + shift), float(hi + shift)


def analyse_subset(models, positions, n_images, block_size, replicates, seed):
    indicator = np.zeros(n_images, dtype=np.int64)
    indicator[positions] = 1
    blocks = make_blocks(positions, block_size)
    reps = bootstrap(models, blocks, n_images, replicates, np.random.default_rng(seed))
    rows = {}
    for name, model in models.items():
        point = model.metrics(indicator)
        rows[name] = {"point": point, "map_ci": interval(reps[name]["map"], point["map"]),
                      "bootstrap_bias": float(np.mean(reps[name]["map"]) - point["map"])}
    names = list(models)
    diffs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            delta = reps[a]["map"] - reps[b]["map"]
            point_diff = rows[a]["point"]["map"] - rows[b]["point"]["map"]
            diffs[f"{a} - {b}"] = {"point": point_diff, "ci": interval(delta, point_diff),
                                   "share_positive": float(np.mean(delta > 0))}
    return {"images": int(len(positions)), "blocks": len(blocks), "models": rows, "differences": diffs}


def format_markdown(results, block_size, replicates):
    lines = []
    for label, res in results.items():
        lines += [f"### {label} ({res['images']} images)", "",
                  "| Model | mAP@[.50:.95] [95% CI] | mAP@.50 | mAP@.75 | AR@100 |", "|---|---|---:|---:|---:|"]
        for name, row in res["models"].items():
            p, (lo, hi) = row["point"], row["map_ci"]
            lines.append(f"| {name} | {p['map']:.4f} [{lo:.4f}, {hi:.4f}] | {p['map_50']:.4f} | "
                         f"{p['map_75']:.4f} | {p['mar_100']:.4f} |")
        lines += ["", "| Difference in mAP | Point | 95% CI | Share of resamples > 0 |", "|---|---:|---|---:|"]
        for name, d in res["differences"].items():
            lines.append(f"| {name} | {d['point']:+.4f} | [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] | {d['share_positive']:.2f} |")
        lines.append("")
    lines.append(f"Block bootstrap: blocks of {block_size} consecutive test images (ordered by file number), "
                 f"{replicates} resamples, the same resamples for every model. Intervals are 95% percentile "
                 "intervals re-centred on the full-sample value (resampled AP is biased upwards by about 0.01). "
                 "A difference whose interval excludes 0 is distinguishable from resampling noise; it is not a "
                 "claim about new recordings.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def prepare_model(gt, predictions_path, check):
    from pycocotools.cocoeval import COCOeval
    detections = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    evaluator = COCOeval(gt, gt.loadRes(detections), "bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
    model = FastCOCO(evaluator)
    if check:
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.accumulate()
            evaluator.summarize()
        reference = dict(zip(METRICS, (evaluator.stats[0], evaluator.stats[1], evaluator.stats[2], evaluator.stats[8])))
        fast = model.metrics(np.ones(model.n_images, dtype=np.int64))
        worst = max(abs(fast[m] - reference[m]) for m in METRICS)
        if worst > 1e-6:
            raise RuntimeError(f"Fast computation disagrees with pycocotools by {worst:.2e}")
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", type=Path, default=ROOT / "dataset" / "model_ready" / "annotations" / "instances_test.json")
    parser.add_argument("--pairs", type=Path, help="near_duplicate_pairs.csv from the audit notebook.")
    parser.add_argument("--clean-csv", type=Path, help="split_v1.csv from the audit notebook (clean_test at distance > 4).")
    parser.add_argument("--deltas", type=int, nargs="+", default=[4, 6], help="Distances for --pairs (clean means farther than delta).")
    parser.add_argument("--include-earlier-rule", action="store_true", help="Also analyse the lowest-validation-loss checkpoints.")
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "analysis")
    args = parser.parse_args(argv)

    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(args.gt))
    image_ids = sorted(gt.getImgIds())
    stems = [Path(gt.imgs[i]["file_name"]).stem for i in image_ids]
    order = sorted(range(len(image_ids)), key=lambda i: stem_key(stems[i]))

    sources = dict(REPORTED)
    if args.include_earlier_rule:
        sources.update(EARLIER_RULE)
    models = {}
    for name, path in sources.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Predictions for {name} not found: {path}")
        print(f"Preparing {name} ...")
        models[name] = prepare_model(gt, path, check=True)
    print("Fast computation matches pycocotools on the full test set.")

    subsets = {"Full test": order}
    if args.pairs:
        for delta, clean in clean_stems_from_pairs(args.pairs, set(stems), args.deltas).items():
            subsets[f"Clean test (no near-duplicate in train, distance > {delta})"] = [i for i in order if stems[i] in clean]
    elif args.clean_csv:
        clean = clean_stems_from_csv(args.clean_csv)
        subsets["Clean test (no near-duplicate in train, distance > 4)"] = [i for i in order if stems[i] in clean]

    results = {}
    for label, positions in subsets.items():
        print(f"Bootstrapping: {label} ({len(positions)} images) ...")
        results[label] = analyse_subset(models, positions, len(image_ids), args.block_size, args.replicates, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "clean_test_bootstrap.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown = format_markdown(results, args.block_size, args.replicates)
    (args.output_dir / "clean_test_bootstrap.md").write_text(markdown + "\n", encoding="utf-8")
    print("\n" + markdown)
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
