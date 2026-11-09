import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[3])))
DATASET_ROOT = PROJECT_ROOT / "dataset" / "model_ready"
NUM_CLASSES = 53
IMAGE_SIZE = 640


class CocoDetectionDataset(Dataset):
    def __init__(self, split: str) -> None:
        annotation_path = DATASET_ROOT / "annotations" / f"instances_{split}.json"
        image_directory = DATASET_ROOT / "images" / split
        if not annotation_path.is_file():
            raise FileNotFoundError(f"COCO annotations not found: {annotation_path}")
        if not image_directory.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_directory}")
        with annotation_path.open(encoding="utf-8") as file:
            coco = json.load(file)
        self.images = coco["images"]
        self.image_directory = image_directory
        self.image_sizes = {
            int(image["id"]): (int(image["width"]), int(image["height"]))
            for image in self.images
        }
        self.annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for annotation in coco["annotations"]:
            self.annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image_info = self.images[index]
        with Image.open(self.image_directory / image_info["file_name"]) as image_file:
            image = image_file.convert("RGB")
            original_width, original_height = image.size
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

        x_scale = IMAGE_SIZE / original_width
        y_scale = IMAGE_SIZE / original_height
        boxes, labels, areas = [], [], []
        for annotation in self.annotations_by_image.get(image_info["id"], []):
            x, y, width, height = annotation["bbox"]
            x_min = max(0.0, x * x_scale)
            y_min = max(0.0, y * y_scale)
            x_max = min(float(IMAGE_SIZE), (x + width) * x_scale)
            y_max = min(float(IMAGE_SIZE), (y + height) * y_scale)
            if x_max <= x_min or y_max <= y_min:
                continue
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(int(annotation["category_id"]))
            areas.append((x_max - x_min) * (y_max - y_min))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(image_info["id"], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }
        return TF.to_tensor(image), target


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, Any]]]):
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_model(pretrained: bool = True) -> torch.nn.Module:
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        min_size=IMAGE_SIZE,
        max_size=IMAGE_SIZE,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def save_checkpoint(path, model, optimizer, scheduler, epoch, history) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({
        "model_state": state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "history": history,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return int(checkpoint.get("epoch", 0))
