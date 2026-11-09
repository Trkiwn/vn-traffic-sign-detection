import argparse
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from predict import DEFAULT_CHECKPOINT, predict_image


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[3])))


def draw_predictions(image_path: Path, predictions, output_path: Path) -> None:
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction in predictions:
        x_min, y_min, x_max, y_max = prediction["box"]
        label = f"{prediction['label']} {float(prediction['score']):.2f}"
        draw.rectangle((x_min, y_min, x_max, y_max), outline="red", width=3)
        text_box = draw.textbbox((x_min, y_min), label, font=font)
        draw.rectangle(text_box, fill="red")
        draw.text((x_min, y_min), label, fill="white", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Visualization saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw RT-DETR predictions on an image.")
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
        / "rtdetr"
        / f"{args.image.stem}_detected.png"
    )
    predictions = predict_image(args.image, args.checkpoint, args.threshold)
    draw_predictions(args.image, predictions, output)
