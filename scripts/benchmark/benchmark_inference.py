#!/usr/bin/env python3
"""Benchmark inference speed and model size for RT-DETR, Faster R-CNN and YOLO11s.

All three models run through the same harness in one session on one GPU:
batch size 1, the same 640 x 640 stretched input, real test images, a warm-up,
CUDA synchronization around every timed stage, and the evaluation score
threshold (0.001). Latency is reported end to end (decoded image in memory ->
detections on the host in original-image coordinates) and per stage.

This measures inference cost under PyTorch eager mode on the GPU used. It does
not cover export formats (ONNX, TensorRT), edge devices or power.

Usage (repository root):
    python scripts/benchmark/benchmark_inference.py \
        --rtdetr-checkpoint outputs/rtdetr_r50vd_comparison_50ep/checkpoint-4320 \
        --rcnn-checkpoint outputs/faster_rcnn_comparison_50ep/best.pt \
        --yolo-checkpoint /path/to/epoch36.pt \
        --device cuda:0 --precision both

Quick smoke test before the full run:
    ... --num-images 5 --warmup 2 --runs 10 --repeats 1
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import itertools
import json
import math
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_KEYS = ("rtdetr", "faster_rcnn", "yolo11s")
DISPLAY = {"rtdetr": "RT-DETR R50vd", "faster_rcnn": "Faster R-CNN", "yolo11s": "YOLO11s"}
IMAGE_SIZE = 640


# --------------------------------------------------------------------------
# Pure helpers (no torch): statistics, timing loop, formatting
# --------------------------------------------------------------------------
def percentile(values, q):
    xs = sorted(values)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarise(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": math.sqrt(var), "min": min(values),
            "p50": percentile(values, 50), "p95": percentile(values, 95),
            "p99": percentile(values, 99), "max": max(values)}


def run_timed_loop(adapter, images, warmup, runs, sync, clock=time.perf_counter):
    """Time `runs` iterations (after `warmup`), cycling through `images`.

    Each iteration is timed twice on the same image: end to end with a single
    synchronization at the end, and stage by stage with a synchronization after
    every stage. Returns lists of milliseconds.
    """
    out = {"e2e": [], "pre": [], "fwd": [], "post": []}
    for i in range(warmup + runs):
        img = images[i % len(images)]
        size = img.size  # (width, height)

        sync()
        t0 = clock()
        raw = adapter.forward(adapter.preprocess(img))
        adapter.postprocess(raw, size)
        sync()
        t1 = clock()

        sync()
        a = clock()
        inp = adapter.preprocess(img)
        sync()
        b = clock()
        raw = adapter.forward(inp)
        sync()
        c = clock()
        adapter.postprocess(raw, size)
        sync()
        d = clock()

        if i >= warmup:
            out["e2e"].append((t1 - t0) * 1e3)
            out["pre"].append((b - a) * 1e3)
            out["fwd"].append((c - b) * 1e3)
            out["post"].append((d - c) * 1e3)
    return out


def build_row(key, precision, acc):
    e2e = summarise(acc["e2e"])
    medians = acc["repeat_e2e_medians"]
    spread = (max(medians) - min(medians)) / e2e["p50"] * 100 if len(medians) > 1 else 0.0
    return {
        "model": DISPLAY[key], "key": key, "precision": precision,
        "params_m": acc["params"] / 1e6,
        "weights_mb": acc["weights_bytes"] / 1e6,
        "gflops": acc["gflops"],
        "peak_gpu_mem_mb": acc["peak_mem_mb"],
        "pre_ms": summarise(acc["pre"]), "fwd_ms": summarise(acc["fwd"]),
        "post_ms": summarise(acc["post"]), "e2e_ms": e2e,
        "fps": 1000.0 / e2e["p50"],
        "repeat_median_spread_pct": spread,
        "forward_includes_postprocess": acc["forward_includes_postprocess"],
    }


def _fmt(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_table(rows):
    header = ("| Model | Precision | Params (M) | Weights (MB) | GFLOPs | Peak GPU mem (MB) | "
              "Preprocess (ms) | Inference (ms) | Postprocess (ms) | End-to-end p50 / p95 (ms) | FPS (1/p50) |")
    lines = [header, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        dagger = "\u2020" if r["forward_includes_postprocess"] else ""
        post = "included" if r["forward_includes_postprocess"] else _fmt(r["post_ms"]["p50"], 2)
        lines.append(
            f"| {r['model']} | {r['precision']} | {r['params_m']:.1f} | {r['weights_mb']:.0f} | "
            f"{_fmt(r['gflops'])} | {_fmt(r['peak_gpu_mem_mb'], 0)} | {r['pre_ms']['p50']:.2f} | "
            f"{r['fwd_ms']['p50']:.2f}{dagger} | {post} | "
            f"{r['e2e_ms']['p50']:.2f} / {r['e2e_ms']['p95']:.2f} | {r['fps']:.1f} |")
    return "\n".join(lines)


TABLE_NOTES = (
    "Batch size 1, 640 x 640 stretched input, real test images; median (p50) and 95th percentile (p95) "
    "over all timed runs, pooled across repeats. Stage times are measured with a synchronization after "
    "every stage; end-to-end uses one synchronization at the end, so it is not the sum of the stages. "
    "End-to-end covers preprocessing, the forward pass, postprocessing (score threshold 0.001) and the "
    "copy of the detections to the host in original-image coordinates.\n"
    "\u2020 torchvision's Faster R-CNN performs NMS inside its forward pass, so its inference time includes "
    "postprocessing and its postprocessing column only rescales boxes and copies them to the host.\n"
    "GFLOPs are counted with torch.utils.flop_counter (convolution and matrix-multiply operations only; "
    "e.g. grid sampling in RT-DETR's deformable attention and RoIAlign are not counted), so they are "
    "approximate. Weights are the FP32 parameter and buffer sizes of the model as trained; checkpoint "
    "files are not comparable because some include optimizer state."
)


# --------------------------------------------------------------------------
# Runtime (the only place that touches torch for device handling)
# --------------------------------------------------------------------------
class TorchRuntime:
    def __init__(self, device_name):
        import torch
        self.torch = torch
        self.device = torch.device(device_name)
        self.is_cuda = self.device.type == "cuda"

    def sync(self):
        if self.is_cuda:
            self.torch.cuda.synchronize(self.device)

    def inference_context(self):
        return self.torch.inference_mode()

    def reset_peak(self):
        if self.is_cuda:
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def peak_mem_mb(self):
        if not self.is_cuda:
            return None
        return self.torch.cuda.max_memory_allocated(self.device) / 1e6

    def release(self):
        gc.collect()
        if self.is_cuda:
            self.torch.cuda.empty_cache()

    def device_index(self):
        return self.device.index if self.device.index is not None else 0


def make_runtime(device_name):
    return TorchRuntime(device_name)


def module_size(module):
    params = sum(p.numel() for p in module.parameters())
    weight_bytes = sum(t.numel() * t.element_size()
                       for t in itertools.chain(module.parameters(), module.buffers()))
    return params, weight_bytes


# --------------------------------------------------------------------------
# Model adapters. Each exposes preprocess / forward / postprocess.
# Detections leave postprocess as CPU tensors (boxes xyxy in original pixels).
# --------------------------------------------------------------------------
class TorchAdapter:
    forward_includes_postprocess = False

    def __init__(self, runtime, fp16):
        self.rt = runtime
        self.torch = runtime.torch
        self.device = runtime.device
        self.fp16 = fp16
        self.params = 0
        self.weights_bytes = 0

    def autocast(self):
        if self.fp16:
            return self.torch.autocast(device_type=self.device.type, dtype=self.torch.float16)
        return contextlib.nullcontext()

    def forward(self, inp):
        with self.autocast():
            return self._infer(inp)

    def example_input(self):
        raise NotImplementedError

    def gflops(self):
        """FLOPs of one forward pass at batch size 1 (FP32), or None if counting fails."""
        try:
            from torch.utils.flop_counter import FlopCounterMode
            example = self.example_input()
            with self.torch.inference_mode(), FlopCounterMode(display=False) as counter:
                self._infer(example)
            return counter.get_total_flops() / 1e9
        except Exception as error:  # counting is informative only
            print(f"  (FLOP counting failed: {error!r})")
            return None


class RTDetrAdapter(TorchAdapter):
    def __init__(self, runtime, fp16, checkpoint, score_threshold):
        super().__init__(runtime, fp16)
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        self.processor = RTDetrImageProcessor.from_pretrained(checkpoint)
        self.model = RTDetrForObjectDetection.from_pretrained(checkpoint).to(self.device).eval()
        self.threshold = score_threshold
        self.params, self.weights_bytes = module_size(self.model)

    def preprocess(self, img):
        return self.processor(images=img, return_tensors="pt")["pixel_values"].to(self.device)

    def _infer(self, pixel_values):
        return self.model(pixel_values=pixel_values)

    def postprocess(self, raw, size):
        width, height = size
        target = self.torch.tensor([[height, width]], device=self.device, dtype=self.torch.int64)
        det = self.processor.post_process_object_detection(
            raw, threshold=self.threshold, target_sizes=target)[0]
        return det["boxes"].cpu(), det["scores"].cpu(), det["labels"].cpu()

    def example_input(self):
        return self.torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=self.device)


class FasterRCNNAdapter(TorchAdapter):
    forward_includes_postprocess = True

    def __init__(self, runtime, fp16, checkpoint, score_threshold):
        super().__init__(runtime, fp16)
        sys.path.insert(0, str(ROOT / "scripts" / "Faster R-CNN" / "models"))
        from model_rcnn import build_model, load_checkpoint
        from torchvision.transforms import functional as TF
        self.TF = TF
        model = build_model(pretrained=False)
        load_checkpoint(Path(checkpoint), model)
        self.model = model.to(self.device).eval()
        # torchvision's default internal threshold is 0.05; match the evaluation setting.
        self.model.roi_heads.score_thresh = score_threshold
        self.params, self.weights_bytes = module_size(self.model)

    def preprocess(self, img):
        return self.TF.to_tensor(img.resize((IMAGE_SIZE, IMAGE_SIZE))).to(self.device)

    def _infer(self, tensor):
        return self.model([tensor])[0]

    def postprocess(self, det, size):
        width, height = size
        boxes = det["boxes"].clone()
        boxes[:, [0, 2]] *= width / IMAGE_SIZE
        boxes[:, [1, 3]] *= height / IMAGE_SIZE
        return boxes.cpu(), det["scores"].cpu(), det["labels"].cpu()

    def example_input(self):
        return self.torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE, device=self.device)


class Yolo11sAdapter(TorchAdapter):
    def __init__(self, runtime, fp16, checkpoint, score_threshold, iou, max_det, multi_label):
        super().__init__(runtime, fp16)
        from ultralytics import YOLO
        try:
            from ultralytics.utils.nms import non_max_suppression
        except ImportError:  # older Ultralytics releases
            from ultralytics.utils.ops import non_max_suppression
        self.nms = non_max_suppression
        yolo = YOLO(str(checkpoint), task="detect")
        if len(yolo.names) != 52:
            raise ValueError("Expected a fine-tuned 52-class YOLO11s checkpoint")
        net = yolo.model.float().to(self.device).eval()
        self.params, self.weights_bytes = module_size(net)  # as trained, before fusing
        net.fuse(verbose=False)  # Ultralytics fuses Conv+BatchNorm for inference
        self.net = net
        self.threshold, self.iou, self.max_det, self.multi_label = score_threshold, iou, max_det, multi_label

    def preprocess(self, img):
        import numpy as np
        from PIL import Image
        resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BICUBIC)
        arr = np.ascontiguousarray(np.asarray(resized))
        tensor = self.torch.from_numpy(arr).to(self.device)
        return tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)

    def _infer(self, tensor):
        return self.net(tensor)

    def postprocess(self, raw, size):
        width, height = size
        preds = raw[0] if isinstance(raw, (list, tuple)) else raw
        det = self.nms(preds.float(), conf_thres=self.threshold, iou_thres=self.iou,
                       max_det=self.max_det, multi_label=self.multi_label)[0]
        boxes = det[:, :4].clone()
        boxes[:, [0, 2]] *= width / IMAGE_SIZE
        boxes[:, [1, 3]] *= height / IMAGE_SIZE
        return boxes.cpu(), det[:, 4].cpu(), det[:, 5].cpu()

    def example_input(self):
        return self.torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=self.device)


def build_adapter(key, runtime, fp16, args):
    if key == "rtdetr":
        return RTDetrAdapter(runtime, fp16, args.rtdetr_checkpoint, args.score_threshold)
    if key == "faster_rcnn":
        return FasterRCNNAdapter(runtime, fp16, args.rcnn_checkpoint, args.score_threshold)
    return Yolo11sAdapter(runtime, fp16, args.yolo_checkpoint, args.score_threshold,
                          args.yolo_iou, args.yolo_max_det, args.yolo_multi_label)


ADAPTER_FACTORY = build_adapter  # replaced in unit tests


# --------------------------------------------------------------------------
# Environment, images, driver
# --------------------------------------------------------------------------
def gpu_snapshot(index):
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(index),
             "--query-gpu=name,driver_version,clocks.sm,clocks.max.sm,temperature.gpu,power.draw",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def collect_environment(runtime):
    from importlib import metadata
    versions = {}
    for package in ("torch", "torchvision", "transformers", "ultralytics", "numpy", "pillow"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "packages": versions, "device": str(runtime.device)}
    if runtime.is_cuda:
        env["gpu_name"] = runtime.torch.cuda.get_device_name(runtime.device)
        env["cuda"] = runtime.torch.version.cuda
        env["gpu_snapshot_start"] = gpu_snapshot(runtime.device_index())
    return env


def load_images(directory, count, seed):
    from PIL import Image
    files = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise FileNotFoundError(f"No images found in {directory}")
    chosen = random.Random(seed).sample(files, min(count, len(files)))
    images = []
    for path in chosen:
        with Image.open(path) as handle:
            images.append(handle.convert("RGB"))
    return images, [p.name for p in chosen]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    p.add_argument("--rtdetr-checkpoint", type=Path,
                   default=ROOT / "outputs" / "rtdetr_r50vd_comparison_50ep" / "checkpoint-4320")
    p.add_argument("--rcnn-checkpoint", type=Path,
                   default=ROOT / "outputs" / "faster_rcnn_comparison_50ep" / "best.pt")
    p.add_argument("--yolo-checkpoint", type=Path, help="Path to the selected YOLO11s weights (epoch36.pt).")
    p.add_argument("--images-dir", type=Path, default=ROOT / "dataset" / "model_ready" / "images" / "test")
    p.add_argument("--num-images", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--precision", choices=("fp32", "fp16", "both"), default="both",
                   help="fp16 uses torch.autocast, the mechanism used in training.")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--runs", type=int, default=300)
    p.add_argument("--repeats", type=int, default=3,
                   help="Full passes over all configurations, with the model order rotated each pass.")
    p.add_argument("--score-threshold", type=float, default=0.001)
    p.add_argument("--yolo-iou", type=float, default=0.7)
    p.add_argument("--yolo-max-det", type=int, default=100)
    p.add_argument("--yolo-multi-label", action="store_true",
                   help="Use multi-label NMS for YOLO11s (the reported evaluation did not).")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "benchmark")
    p.add_argument("--allow-cpu", action="store_true", help="Allow a CPU run (smoke tests only).")
    args = p.parse_args(argv)
    if args.runs < 1 or args.warmup < 0 or args.repeats < 1 or args.num_images < 1:
        p.error("--runs, --repeats and --num-images must be >= 1 and --warmup >= 0")
    if "yolo11s" in args.models and args.yolo_checkpoint is None:
        p.error("--yolo-checkpoint is required when benchmarking yolo11s (the weights are not in the repository)")
    return args


def main(argv=None):
    args = parse_args(argv)
    runtime = make_runtime(args.device)
    if not runtime.is_cuda and not args.allow_cpu:
        sys.exit("No CUDA device requested or available. Use --allow-cpu for a smoke test only.")
    precisions = ["fp32", "fp16"] if args.precision == "both" else [args.precision]
    if not runtime.is_cuda and "fp16" in precisions:
        sys.exit("FP16 autocast is only benchmarked on CUDA.")

    images, image_names = load_images(args.images_dir, args.num_images, args.seed)
    size_mix = Counter(f"{img.size[0]}x{img.size[1]}" for img in images)
    environment = collect_environment(runtime)
    print(f"Device: {environment.get('gpu_name', runtime.device)} | images: {len(images)} {dict(size_mix)}")

    configs = [(m, p) for p in precisions for m in args.models]
    acc = {}
    snapshots = []
    for repeat in range(args.repeats):
        shift = repeat % len(configs)
        for key, precision in configs[shift:] + configs[:shift]:
            print(f"[repeat {repeat + 1}/{args.repeats}] {DISPLAY[key]} {precision}")
            runtime.release()
            runtime.reset_peak()
            adapter = ADAPTER_FACTORY(key, runtime, precision == "fp16", args)
            entry = acc.setdefault((key, precision), {
                "e2e": [], "pre": [], "fwd": [], "post": [], "repeat_e2e_medians": [],
                "params": adapter.params, "weights_bytes": adapter.weights_bytes,
                "gflops": adapter.gflops() if repeat == 0 else None, "peak_mem_mb": None,
                "forward_includes_postprocess": adapter.forward_includes_postprocess})
            snapshots.append({"model": key, "precision": precision, "repeat": repeat + 1,
                              "before": gpu_snapshot(runtime.device_index()) if runtime.is_cuda else None})
            with runtime.inference_context():
                timings = run_timed_loop(adapter, images, args.warmup, args.runs, runtime.sync)
            snapshots[-1]["after"] = gpu_snapshot(runtime.device_index()) if runtime.is_cuda else None
            for stage in ("e2e", "pre", "fwd", "post"):
                entry[stage].extend(timings[stage])
            entry["repeat_e2e_medians"].append(percentile(timings["e2e"], 50))
            peak = runtime.peak_mem_mb()
            if peak is not None:
                entry["peak_mem_mb"] = max(peak, entry["peak_mem_mb"] or 0.0)
            del adapter
            runtime.release()

    rows = [build_row(key, precision, acc[(key, precision)])
            for precision in precisions for key in args.models]
    table = format_table(rows)
    print("\n" + table + "\n\n" + TABLE_NOTES)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    payload = {"created": datetime.now().isoformat(timespec="seconds"), "settings": settings,
               "environment": environment, "images": {"count": len(images), "sizes": dict(size_mix),
                                                       "names": image_names},
               "rows": rows, "gpu_snapshots": snapshots, "notes": TABLE_NOTES}
    (args.output_dir / "benchmark_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / "benchmark_table.md").write_text(table + "\n\n" + TABLE_NOTES + "\n", encoding="utf-8")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
