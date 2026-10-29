import json
import os
from pathlib import Path

from transformers import RTDetrConfig, RTDetrForObjectDetection


PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[3]))
)
ANNOTATIONS_PATH = (
    PROJECT_ROOT / "dataset" / "model_ready" / "annotations" / "instances_train.json"
)
MODEL_NAME = "PekingU/rtdetr_r50vd"

if not ANNOTATIONS_PATH.is_file():
    raise FileNotFoundError(
        f"Prepared training annotations not found: {ANNOTATIONS_PATH}. "
        "Run the dataset import and preprocessing scripts first."
    )

with ANNOTATIONS_PATH.open(encoding="utf-8") as annotations_file:
    categories = json.load(annotations_file)["categories"]

categories = sorted(categories, key=lambda category: category["id"])
category_ids = [category["id"] for category in categories]
if category_ids != list(range(1, len(categories) + 1)):
    raise ValueError("Training category IDs must be contiguous and start at 1.")

ID2LABEL = {
    label_id: category["name"] for label_id, category in enumerate(categories)
}
LABEL2ID = {name: label_id for label_id, name in ID2LABEL.items()}

config = RTDetrConfig.from_pretrained(MODEL_NAME)
config.num_labels = len(ID2LABEL)
config.id2label = ID2LABEL
config.label2id = LABEL2ID
model = RTDetrForObjectDetection.from_pretrained(
    MODEL_NAME,
    config=config,
    ignore_mismatched_sizes=True,
)
