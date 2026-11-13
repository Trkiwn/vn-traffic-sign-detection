#!/usr/bin/env python3
"""Plot test mAP@[.50:.95] against end-to-end inference latency for the three models.

Reads the output of benchmark_inference.py and the saved COCO test metrics.

Usage (repository root):
    python scripts/benchmark/plot_accuracy_latency.py
    python scripts/benchmark/plot_accuracy_latency.py --precision fp32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRIC_FILES = {
    "rtdetr": ROOT / "outputs" / "rtdetr_r50vd_comparison_50ep" / "evaluation" / "test_metrics.json",
    "faster_rcnn": ROOT / "outputs" / "faster_rcnn_comparison_50ep" / "test_metrics.json",
    "yolo11s": ROOT / "results" / "YOLO11s" / "min_val_loss" / "test_metrics.json",
}


def load_points(benchmark_path, precision, metric_files):
    benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    points = []
    for row in benchmark["rows"]:
        if row["precision"] != precision:
            continue
        metrics_path = Path(metric_files[row["key"]])
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Test metrics not found for {row['model']}: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        points.append({"label": row["model"], "map": float(metrics["map"]),
                       "p50": row["e2e_ms"]["p50"], "p95": row["e2e_ms"]["p95"]})
    if not points:
        raise ValueError(f"No benchmark rows with precision {precision!r} in {benchmark_path}")
    return points, benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "outputs" / "benchmark" / "benchmark_results.json")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--rtdetr-metrics", type=Path, default=METRIC_FILES["rtdetr"])
    parser.add_argument("--rcnn-metrics", type=Path, default=METRIC_FILES["faster_rcnn"])
    parser.add_argument("--yolo-metrics", type=Path, default=METRIC_FILES["yolo11s"])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "plots" / "accuracy_vs_latency.png")
    args = parser.parse_args()
    if not args.benchmark.is_file():
        parser.error(f"Benchmark results not found: {args.benchmark}")

    metric_files = {"rtdetr": args.rtdetr_metrics, "faster_rcnn": args.rcnn_metrics, "yolo11s": args.yolo_metrics}
    points, benchmark = load_points(args.benchmark, args.precision, metric_files)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for point in points:
        ax.errorbar(point["p50"], point["map"], xerr=[[0], [point["p95"] - point["p50"]]],
                    fmt="o", markersize=8, capsize=3)
        ax.annotate(point["label"], (point["p50"], point["map"]),
                    textcoords="offset points", xytext=(8, 6))
    gpu = benchmark.get("environment", {}).get("gpu_name", "GPU")
    ax.set(xlabel="End-to-end latency, median with p95 whisker (ms)",
           ylabel="Test mAP@[.50:.95]",
           title=f"Accuracy versus latency ({gpu}, batch size 1, {args.precision})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
