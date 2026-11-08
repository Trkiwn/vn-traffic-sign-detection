import argparse
import json
import os
import sys
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from predict_rcnn import predict


def evaluate(checkpoint: Path, split: str, batch_size: int, num_workers: int) -> None:
    predictions_path = checkpoint.parent / f"{split}_predictions.json"
    predict(checkpoint, split, predictions_path, batch_size, num_workers, 0.001)
    project_root = Path(
        os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[3]))
    )
    ground_truth = COCO(str(project_root / "dataset" / "model_ready" / "annotations" / f"instances_{split}.json"))
    predictions = ground_truth.loadRes(str(predictions_path))
    evaluator = COCOeval(ground_truth, predictions, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ("map", "map_50", "map_75", "map_small", "map_medium", "map_large", "mar_1", "mar_10", "mar_100", "mar_small", "mar_medium", "mar_large")
    metrics = {name: float(value) for name, value in zip(names, evaluator.stats)}
    metrics["evaluated_images"] = len(ground_truth.getImgIds())
    metrics["predictions"] = len(json.loads(predictions_path.read_text(encoding="utf-8")))
    metrics_path = checkpoint.parent / f"{split}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN with COCO metrics.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.split, args.batch_size, args.num_workers)
