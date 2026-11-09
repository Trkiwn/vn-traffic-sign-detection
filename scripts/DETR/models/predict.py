import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[3])))
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "rtdetr_r50vd_comparison_50ep" / "checkpoint-4320"


def load_detector(checkpoint: Path):
    if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint directory not found or invalid: {checkpoint}")
    processor = RTDetrImageProcessor.from_pretrained(checkpoint)
    model = RTDetrForObjectDetection.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, processor, device


def predict_image(
    image_path: Path,
    checkpoint: Path,
    score_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not 0 <= score_threshold <= 1:
        raise ValueError("Score threshold must be between 0 and 1.")

    model, processor, device = load_detector(checkpoint)
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
    encoded = processor(images=image, return_tensors="pt")
    target_sizes = torch.tensor(
        [[image.height, image.width]], device=device, dtype=torch.int64
    )
    with torch.inference_mode():
        outputs = model(pixel_values=encoded["pixel_values"].to(device))
        detection = processor.post_process_object_detection(
            outputs,
            threshold=score_threshold,
            target_sizes=target_sizes,
        )[0]

    id_to_label = model.config.id2label or {}
    predictions: list[dict[str, Any]] = []
    for box, score, label in zip(
        detection["boxes"], detection["scores"], detection["labels"]
    ):
        label_id = int(label)
        predictions.append(
            {
                "label_id": label_id,
                "label": id_to_label.get(str(label_id), id_to_label.get(label_id, str(label_id))),
                "score": float(score),
                "box": [float(value) for value in box.tolist()],
            }
        )
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RT-DETR inference on one image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output path; defaults to outputs/predictions/rtdetr/<image>.json.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictions = predict_image(args.image, args.checkpoint, args.threshold)
    if args.output is None:
        args.output = (
            PROJECT_ROOT
            / "outputs"
            / "predictions"
            / "rtdetr"
            / f"{args.image.stem}.json"
        )
    output = json.dumps(predictions, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(f"Predictions saved to: {args.output}")
    else:
        print(output, end="")
