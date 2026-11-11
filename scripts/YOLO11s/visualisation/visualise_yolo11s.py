"""YOLO11s single-image demo using the historical 640x640 stretch geometry.

Not a COCO evaluator; threshold is for human-readable detections.
Usage (repo root):
    python scripts/YOLO11s/visualisation/visualise_yolo11s.py \
        --image path/to/image.jpg --checkpoint /path/to/epoch36.pt --threshold 0.5 \
        --output outputs/predictions/yolo11s/image_detected.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

IMAGE_SIZE = 640


def stretched_input(rgb_image: Image.Image) -> np.ndarray:
    """Match original YOLO11s 640x640 RGB bicubic stretch; Ultralytics expects BGR ndarray."""
    resized = rgb_image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BICUBIC)
    return np.ascontiguousarray(np.asarray(resized)[:, :, ::-1])


def original_box(box_xyxy: list[float], width: int, height: int) -> list[float]:
    """Map xyxy coordinates from 640x640 network input to original-image pixels."""
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return [
        max(0.0, min(float(width), x1 * width / IMAGE_SIZE)),
        max(0.0, min(float(height), y1 * height / IMAGE_SIZE)),
        max(0.0, min(float(width), x2 * width / IMAGE_SIZE)),
        max(0.0, min(float(height), y2 * height / IMAGE_SIZE)),
    ]


def inference(image: Path, checkpoint: Path, threshold: float, device: str) -> tuple[Image.Image, list[dict[str, Any]]]:
    from ultralytics import YOLO  # lazy import so geometry can be unit-tested without GPU

    original = Image.open(image).convert('RGB')
    width, height = original.size
    detector = YOLO(str(checkpoint), task='detect')
    if len(detector.names) != 52:
        raise ValueError('Expected a fine-tuned 52-class YOLO11s checkpoint, not the 80-class initializer')
    processed = stretched_input(original)
    result = detector.predict(
        source=processed,
        imgsz=IMAGE_SIZE,
        rect=False,
        conf=threshold,
        iou=0.7,
        max_det=100,
        agnostic_nms=False,
        device=device,
        verbose=False,
        save=False,
    )[0]
    detections = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.detach().cpu().tolist()
        confidences = result.boxes.conf.detach().cpu().tolist()
        classes = result.boxes.cls.detach().cpu().tolist()
        for bbox, confidence, class_index in zip(boxes, confidences, classes):
            x1, y1, x2, y2 = original_box(bbox, width, height)
            if x2 <= x1 or y2 <= y1:
                continue
            idx = int(class_index)
            detections.append({
                'class_id_0_based': idx,
                'class_name': str(detector.names[idx]),
                'confidence': float(confidence),
                'bbox_xyxy_original_pixels': [x1, y1, x2, y2],
            })
    return original, detections


def draw_predictions(original: Image.Image, detections: list[dict[str, Any]], destination: Path) -> None:
    image = original.copy()
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(image.size) / 300))
    for item in detections:
        box = item['bbox_xyxy_original_pixels']
        draw.rectangle(box, outline='red', width=line_width)
        text = f"{item['class_name']} {item['confidence']:.2f}"
        anchor_y = max(0, int(box[1]) - 16)
        text_width = min(len(text) * 7 + 6, image.width)
        draw.rectangle([int(box[0]), anchor_y, min(image.width, int(box[0]) + text_width), anchor_y + 16], fill='red')
        draw.text((int(box[0]) + 3, anchor_y + 1), text, fill='white')
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', default='cpu', help='CPU default; set 0 for first CUDA GPU')
    parser.add_argument('--output', type=Path, help='Optional annotated-image destination')
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f'Image does not exist: {args.image}')
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint does not exist: {args.checkpoint}. Download epoch36.pt separately.')
    if not 0.0 < args.threshold <= 1.0:
        parser.error('--threshold must be in (0,1]')

    original, detections = inference(args.image, args.checkpoint, args.threshold, args.device)
    print(json.dumps({'image': str(args.image), 'checkpoint': str(args.checkpoint),
                      'threshold': args.threshold, 'detections': detections}, indent=2, ensure_ascii=False))
    if args.output is not None:
        draw_predictions(original, detections, args.output)
        print(f'Annotated image saved to {args.output}')


if __name__ == '__main__':
    main()
