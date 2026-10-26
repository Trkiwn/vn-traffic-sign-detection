import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "model_ready"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "rtdetr_r50vd"
DEFAULT_BATCH_SIZE = 2
DEFAULT_NUM_WORKERS = 4
DEFAULT_SCORE_THRESHOLD = 0.001


class CocoImageDataset(Dataset):
    def __init__(
        self,
        annotation_file: Path,
        image_directory: Path,
        processor: RTDetrImageProcessor,
    ) -> None:
        with annotation_file.open(encoding="utf-8") as file:
            self.coco_data = json.load(file)

        self.images = self.coco_data["images"]
        self.image_directory = image_directory
        self.processor = processor
        missing_images = [
            image["file_name"]
            for image in self.images
            if not (self.image_directory / image["file_name"]).is_file()
        ]
        if missing_images:
            raise FileNotFoundError(
                f"{len(missing_images)} image(s) referenced by {annotation_file} "
                f"are missing from {self.image_directory}; example: {missing_images[0]}"
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_info = self.images[index]
        image_path = self.image_directory / image_info["file_name"]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            original_size = (image.height, image.width)
            encoded = self.processor(images=image, return_tensors="pt")

        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "image_id": image_info["id"],
            "original_size": original_size,
        }


def collate_images(examples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
        "image_ids": [example["image_id"] for example in examples],
        "original_sizes": [example["original_size"] for example in examples],
    }


def load_ground_truth(annotation_file: Path) -> COCO:
    coco = COCO(str(annotation_file))
    if not coco.dataset.get("images") or not coco.dataset.get("categories"):
        raise ValueError(f"COCO annotations contain no images or categories: {annotation_file}")

    category_ids = sorted(category["id"] for category in coco.dataset["categories"])
    if category_ids != list(range(1, len(category_ids) + 1)):
        raise ValueError(
            "RT-DETR evaluation expects COCO category IDs to be contiguous and "
            "start at 1 so they map to model labels 0..N-1."
        )
    return coco


def validate_model_labels(model: RTDetrForObjectDetection, coco: COCO) -> None:
    categories = sorted(coco.dataset["categories"], key=lambda category: category["id"])
    if model.config.num_labels != len(categories):
        raise ValueError(
            f"Checkpoint has {model.config.num_labels} labels, but the dataset has "
            f"{len(categories)} categories. Evaluate a checkpoint fine-tuned for "
            "this dataset, not the original COCO checkpoint."
        )

    model_label_ids = sorted(
        int(label_id) for label_id in (model.config.id2label or {})
    )
    if model_label_ids != list(range(len(categories))):
        raise ValueError(
            "Checkpoint label IDs must be contiguous from 0 to the number of "
            "COCO categories minus 1."
        )


def empty_coco_results(coco_ground_truth: COCO) -> COCO:
    coco_results = COCO()
    coco_results.dataset = {
        "images": coco_ground_truth.dataset["images"],
        "categories": coco_ground_truth.dataset["categories"],
        "annotations": [],
    }
    coco_results.createIndex()
    return coco_results


def evaluate(
    checkpoint: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    score_threshold: float,
    max_images: int | None = None,
) -> dict[str, float]:
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
    if num_workers < 0:
        raise ValueError("Number of workers cannot be negative.")
    if not 0 <= score_threshold <= 1:
        raise ValueError("Score threshold must be between 0 and 1.")
    if max_images is not None and max_images < 1:
        raise ValueError("Maximum image count must be at least 1.")

    annotation_file = DATASET_ROOT / "annotations" / f"instances_{split}.json"
    image_directory = DATASET_ROOT / "images" / split
    if not annotation_file.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_file}")
    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"Model checkpoint directory not found: {checkpoint}. "
            "Train the model first or pass a fine-tuned checkpoint with --checkpoint."
        )

    processor = RTDetrImageProcessor.from_pretrained(checkpoint)
    model = RTDetrForObjectDetection.from_pretrained(checkpoint)
    coco_ground_truth = load_ground_truth(annotation_file)
    validate_model_labels(model, coco_ground_truth)

    dataset = CocoImageDataset(annotation_file, image_directory, processor)
    image_ids = [image["id"] for image in dataset.images]
    if max_images is not None:
        dataset = Subset(dataset, range(min(max_images, len(dataset))))
        image_ids = image_ids[: len(dataset)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_images,
        pin_memory=device.type == "cuda",
    )

    predictions: list[dict[str, int | float | list[float]]] = []
    with torch.inference_mode():
        for batch in data_loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            outputs = model(pixel_values=pixel_values)
            target_sizes = torch.tensor(
                batch["original_sizes"], device=device, dtype=torch.int64
            )
            detections = processor.post_process_object_detection(
                outputs,
                threshold=score_threshold,
                target_sizes=target_sizes,
            )

            for image_id, image_size, detection in zip(
                batch["image_ids"], batch["original_sizes"], detections
            ):
                height, width = image_size
                for box, score, label in zip(
                    detection["boxes"], detection["scores"], detection["labels"]
                ):
                    x_min, y_min, x_max, y_max = box.tolist()
                    x_min = min(max(x_min, 0.0), float(width))
                    y_min = min(max(y_min, 0.0), float(height))
                    x_max = min(max(x_max, 0.0), float(width))
                    y_max = min(max(y_max, 0.0), float(height))
                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    if box_width <= 0 or box_height <= 0:
                        continue

                    predictions.append(
                        {
                            "image_id": int(image_id),
                            "category_id": int(label) + 1,
                            "bbox": [x_min, y_min, box_width, box_height],
                            "score": float(score),
                        }
                    )

    evaluated_ground_truth = coco_ground_truth
    if max_images is not None:
        evaluated_ground_truth = COCO()
        evaluated_ground_truth.dataset = {
            **coco_ground_truth.dataset,
            "images": [
                image
                for image in coco_ground_truth.dataset["images"]
                if image["id"] in set(image_ids)
            ],
            "annotations": [
                annotation
                for annotation in coco_ground_truth.dataset["annotations"]
                if annotation["image_id"] in set(image_ids)
            ],
        }
        evaluated_ground_truth.createIndex()

    coco_predictions = (
        evaluated_ground_truth.loadRes(predictions)
        if predictions
        else empty_coco_results(evaluated_ground_truth)
    )
    coco_evaluator = COCOeval(evaluated_ground_truth, coco_predictions, "bbox")
    coco_evaluator.params.imgIds = image_ids
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    metric_names = (
        "map",
        "map_50",
        "map_75",
        "map_small",
        "map_medium",
        "map_large",
        "mar_1",
        "mar_10",
        "mar_100",
        "mar_small",
        "mar_medium",
        "mar_large",
    )
    metrics = {
        name: float(value)
        for name, value in zip(metric_names, coco_evaluator.stats)
    }
    metrics["evaluated_images"] = len(image_ids)
    metrics["predictions"] = len(predictions)

    output_dir = checkpoint / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = output_dir / f"{split}_predictions.json"
    metrics_file = output_dir / f"{split}_metrics.json"
    prediction_file.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics_file.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Evaluated {len(image_ids)} {split} images on {device}.")
    print(f"Metrics saved to: {metrics_file}")
    print(f"Predictions saved to: {prediction_file}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Hugging Face RT-DETR checkpoint with COCO metrics."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Fine-tuned model directory (default: outputs/rtdetr_r50vd).",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help="Minimum confidence score; keep low for COCO mAP (default: 0.001).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Optional image limit for a quick smoke test; do not use for reported metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        checkpoint=args.checkpoint.resolve(),
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        score_threshold=args.score_threshold,
        max_images=args.max_images,
    )