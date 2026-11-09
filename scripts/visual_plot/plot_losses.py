import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
)


def load_rtdetr_history(output_dir: Path) -> tuple[list[float], list[float]]:
    state_paths = sorted(
        output_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda path: int(path.parent.name.removeprefix("checkpoint-")),
    )
    if not state_paths:
        raise FileNotFoundError(
            f"No RT-DETR trainer_state.json files found in {output_dir}."
        )
    state: dict[str, Any] = json.loads(
        state_paths[-1].read_text(encoding="utf-8")
    )
    train_points = [
        (float(entry["epoch"]), float(entry["loss"]))
        for entry in state.get("log_history", [])
        if "epoch" in entry and "loss" in entry
    ]
    validation_points = [
        (float(entry["epoch"]), float(entry["eval_loss"]))
        for entry in state.get("log_history", [])
        if "epoch" in entry and "eval_loss" in entry
    ]
    if not validation_points:
        raise ValueError(f"No validation-loss entries found in {state_paths[-1]}.")
    return train_points, validation_points


def load_faster_rcnn_history(output_dir: Path) -> tuple[list[float], list[float]]:
    history_path = output_dir / "training_history.json"
    if not history_path.is_file():
        raise FileNotFoundError(f"Faster R-CNN history not found: {history_path}")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    train_points = [
        (float(entry["epoch"]), float(entry["train_loss"]))
        for entry in history
        if "epoch" in entry and "train_loss" in entry
    ]
    validation_points = [
        (float(entry["epoch"]), float(entry["validation_loss"]))
        for entry in history
        if "epoch" in entry and "validation_loss" in entry
    ]
    if not validation_points:
        raise ValueError(f"No validation-loss entries found in {history_path}.")
    return train_points, validation_points


def plot_losses(
    output_dir: Path,
    model_name: str,
    output_path: Path,
) -> None:
    if (output_dir / "training_history.json").is_file():
        train_points, validation_points = load_faster_rcnn_history(output_dir)
    else:
        train_points, validation_points = load_rtdetr_history(output_dir)

    validation_epoch, validation_loss = zip(*validation_points)
    best_index = min(range(len(validation_loss)), key=validation_loss.__getitem__)
    best_epoch = validation_epoch[best_index]
    best_loss = validation_loss[best_index]

    plt.figure(figsize=(10, 6))
    if train_points:
        train_epoch, train_loss = zip(*train_points)
        plt.plot(train_epoch, train_loss, label="Training loss", linewidth=2)
    plt.plot(
        validation_epoch,
        validation_loss,
        label="Validation loss",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    plt.scatter(
        [best_epoch],
        [best_loss],
        color="red",
        zorder=3,
        label=f"Best checkpoint (epoch {best_epoch:g}, loss {best_loss:.4f})",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} training and validation loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Best validation loss: {best_loss:.6f} at epoch {best_epoch:g}")
    print(f"Loss plot saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training and validation loss for a completed run."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = args.output or (
        PROJECT_ROOT / "outputs" / "plots" / f"{args.model_name}_loss.png"
    )
    plot_losses(args.output_dir, args.model_name, output)
