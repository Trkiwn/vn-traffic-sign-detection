import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from model_rcnn import IMAGE_SIZE, build_model, load_checkpoint


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[3])))
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "faster_rcnn_comparison_50ep" / "best.pt"


def load_labels() -> dict[int, str]:
    annotation_path = (
        PROJECT_ROOT / "dataset" / "model_ready" / "annotations" / "instances_train.json"
    )
    with annotation_path.open(encoding="utf-8") as file:
        categories = json.load(file)["categories"]
    return {int(category["id"]): category["name"] for category in categories}


def predict_image(
    image_path: Path,
    checkpoint: Path,
    score_threshold: float,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not 0 <= score_threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    labels = load_labels()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False).to(device)
    load_checkpoint(checkpoint, model)
    model.eval()

    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
    original_width, original_height = image.size
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    with torch.inference_mode():
        detection = model([TF.to_tensor(resized).to(device)])[0]

    x_scale = original_width / IMAGE_SIZE
    y_scale = original_height / IMAGE_SIZE
    predictions: list[dict[str, Any]] = []
    for box, score, label in zip(
        detection["boxes"], detection["scores"], detection["labels"]
    ):
        confidence = float(score)
        if confidence < score_threshold:
            continue
        x_min, y_min, x_max, y_max = box.tolist()
        predictions.append(
            {
                "label_id": int(label),
                "label": labels.get(int(label), str(int(label))),
                "score": confidence,
                "box": [
                    x_min * x_scale,
                    y_min * y_scale,
                    x_max * x_scale,
                    y_max * y_scale,
                ],
            }
        )
    return image, predictions


def draw_predictions(
    image: Image.Image,
    predictions: list[dict[str, Any]],
    output_path: Path,
) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction in predictions:
        x_min, y_min, x_max, y_max = prediction["box"]
        label = f"{prediction['label']} {prediction['score']:.2f}"
        draw.rectangle((x_min, y_min, x_max, y_max), outline="red", width=3)
        text_box = draw.textbbox((x_min, y_min), label, font=font)
        draw.rectangle(text_box, fill="red")
        draw.text((x_min, y_min), label, fill="white", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Detected {len(predictions)} objects.")
    print(f"Visualization saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw Faster R-CNN predictions on one image."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = args.output or (
        PROJECT_ROOT
        / "outputs"
        / "predictions"
        / "faster_rcnn"
        / f"{args.image.stem}_detected.png"
    )
    image, predictions = predict_image(args.image, args.checkpoint, args.threshold)
    draw_predictions(image, predictions, output)
