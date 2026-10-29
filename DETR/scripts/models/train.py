import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import RTDetrImageProcessor, Trainer, TrainingArguments


torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("high")

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[3]))
)
MODEL_NAME = "PekingU/rtdetr_r50vd"
DATASET_ROOT = PROJECT_ROOT / "dataset" / "model_ready"
ANNOTATIONS_DIR = DATASET_ROOT / "annotations"
TRAIN_ANNOTATIONS = ANNOTATIONS_DIR / "instances_train.json"
VALIDATION_ANNOTATIONS = ANNOTATIONS_DIR / "instances_validation.json"
TEST_ANNOTATIONS = ANNOTATIONS_DIR / "instances_test.json"
NUM_LABELS = 52
IMAGE_SIZE = 640
OUTPUT_DIR = Path(
    os.environ.get(
        "OUTPUT_DIR",
        str(PROJECT_ROOT / "outputs" / "rtdetr_r50vd_comparison_50ep"),
    )
)
RESUME_CHECKPOINT = os.environ.get("RESUME_CHECKPOINT")
SEED = 42

TRAINING_ARGUMENTS = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=50,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    weight_decay=1e-4,
    warmup_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    fp16=True,
    bf16=False,
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    seed=SEED,
    data_seed=SEED,
    report_to="none",
)


class CocoDetectionDataset(Dataset):
    def __init__(
        self,
        annotation_file: Path,
        image_directory: Path,
        processor: RTDetrImageProcessor,
        expected_categories: dict[int, str],
    ) -> None:
        with annotation_file.open(encoding="utf-8") as file:
            coco = json.load(file)

        categories = sorted(coco["categories"], key=lambda category: category["id"])
        category_ids = [category["id"] for category in categories]
        if category_ids != list(range(1, len(categories) + 1)):
            raise ValueError(
                f"Category IDs in {annotation_file} must be contiguous and start at 1."
            )

        self.category_to_label = {
            category["id"]: category["id"] - 1 for category in categories
        }
        actual_categories = {
            category["id"] - 1: category["name"] for category in categories
        }
        if actual_categories != expected_categories:
            raise ValueError(
                f"Category names or order differ between {annotation_file} "
                "and the model label mapping."
            )

        self.image_directory = image_directory
        self.processor = processor
        self.images = coco["images"]
        self.annotations_by_image: dict[int, list[dict[str, object]]] = defaultdict(list)

        image_ids = {image["id"] for image in self.images}
        for annotation in coco["annotations"]:
            if annotation["image_id"] not in image_ids:
                raise ValueError(
                    f"Annotation references an unknown image in {annotation_file}."
                )
            if annotation["category_id"] not in self.category_to_label:
                raise ValueError(
                    f"Annotation uses an unknown category in {annotation_file}."
                )
            self.annotations_by_image[annotation["image_id"]].append(
                {
                    **annotation,
                    "category_id": self.category_to_label[annotation["category_id"]],
                }
            )

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

    def __getitem__(self, index: int) -> dict[str, object]:
        image_info = self.images[index]
        image_path = self.image_directory / image_info["file_name"]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        target = {
            "image_id": image_info["id"],
            "annotations": self.annotations_by_image[image_info["id"]],
        }
        encoded = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt",
        )
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"][0],
        }


def collate_fn(examples: list[dict[str, object]]) -> dict[str, object]:
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
        "labels": [example["labels"] for example in examples],
    }


def main() -> None:
    required_files = (
        TRAIN_ANNOTATIONS,
        VALIDATION_ANNOTATIONS,
        TEST_ANNOTATIONS,
    )
    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Prepared COCO annotations are missing; run the dataset import and "
            f"preprocessing scripts first. Missing: {missing_files[0]}"
        )

    processor = RTDetrImageProcessor.from_pretrained(
        MODEL_NAME,
        size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
    )

    from model import ID2LABEL, model

    if len(ID2LABEL) != NUM_LABELS:
        raise ValueError(
            f"Expected {NUM_LABELS} classes, found {len(ID2LABEL)} in the annotations."
        )

    train_dataset = CocoDetectionDataset(
        TRAIN_ANNOTATIONS,
        DATASET_ROOT / "images" / "train",
        processor,
        ID2LABEL,
    )
    validation_dataset = CocoDetectionDataset(
        VALIDATION_ANNOTATIONS,
        DATASET_ROOT / "images" / "validation",
        processor,
        ID2LABEL,
    )
    test_dataset = CocoDetectionDataset(
        TEST_ANNOTATIONS,
        DATASET_ROOT / "images" / "test",
        processor,
        ID2LABEL,
    )

    trainer = Trainer(
        model=model,
        args=TRAINING_ARGUMENTS,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collate_fn,
        processing_class=processor,
    )
    if RESUME_CHECKPOINT:
        resume_checkpoint = Path(RESUME_CHECKPOINT)
        if not resume_checkpoint.is_dir():
            raise FileNotFoundError(
                f"Requested resume checkpoint does not exist: {resume_checkpoint}"
            )
        trainer.train(resume_from_checkpoint=str(resume_checkpoint))
    else:
        trainer.train()

    final_model = trainer.model
    trainer.save_model(TRAINING_ARGUMENTS.output_dir)
    processor.save_pretrained(TRAINING_ARGUMENTS.output_dir)

    test_arguments = TrainingArguments(
        output_dir=TRAINING_ARGUMENTS.output_dir,
        per_device_eval_batch_size=TRAINING_ARGUMENTS.per_device_eval_batch_size,
        dataloader_num_workers=TRAINING_ARGUMENTS.dataloader_num_workers,
        dataloader_pin_memory=TRAINING_ARGUMENTS.dataloader_pin_memory,
        dataloader_persistent_workers=TRAINING_ARGUMENTS.dataloader_persistent_workers,
        bf16=TRAINING_ARGUMENTS.bf16,
        fp16=TRAINING_ARGUMENTS.fp16,
        remove_unused_columns=False,
        report_to="none",
    )
    test_trainer = Trainer(
        model=final_model,
        args=test_arguments,
        data_collator=collate_fn,
        processing_class=processor,
    )
    test_metrics = test_trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix="test",
    )
    test_trainer.log_metrics("test", test_metrics)
    test_trainer.save_metrics("test", test_metrics)


if __name__ == "__main__":
    main()
