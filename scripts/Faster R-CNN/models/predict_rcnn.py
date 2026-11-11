import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model_rcnn import CocoDetectionDataset, build_model, collate_fn, load_checkpoint


def predict(checkpoint: Path, split: str, output: Path, batch_size: int, num_workers: int, score_threshold: float) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False).to(device)
    model.roi_heads.score_thresh = score_threshold  # torchvision default is 0.05
    load_checkpoint(checkpoint, model)
    model.eval()
    dataset = CocoDetectionDataset(split)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0, collate_fn=collate_fn,
    )
    results = []
    with torch.inference_mode():
        for images, targets in loader:
            detections = model([image.to(device, non_blocking=True) for image in images])
            for target, detection in zip(targets, detections):
                image_id = int(target["image_id"])
                original_width, original_height = dataset.image_sizes[image_id]
                x_scale = original_width / 640
                y_scale = original_height / 640
                for box, score, label in zip(detection["boxes"], detection["scores"], detection["labels"]):
                    if float(score) < score_threshold:
                        continue
                    x_min, y_min, x_max, y_max = box.tolist()
                    results.append({
                        "image_id": image_id, "category_id": int(label),
                        "bbox": [
                            x_min * x_scale,
                            y_min * y_scale,
                            max(0.0, x_max - x_min) * x_scale,
                            max(0.0, y_max - y_min) * y_scale,
                        ],
                        "score": float(score),
                    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(results)} predictions to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Faster R-CNN COCO predictions.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    args = parser.parse_args()
    output = args.output or args.checkpoint.parent / f"{args.split}_predictions.json"
    predict(args.checkpoint, args.split, output, args.batch_size, args.num_workers, args.score_threshold)
