import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model_rcnn import CocoDetectionDataset, build_model, collate_fn, load_checkpoint, save_checkpoint

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parents[3])))
OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    str(PROJECT_ROOT / "outputs" / "faster_rcnn_comparison_50ep"),
))
RESUME_CHECKPOINT = os.environ.get("RESUME_CHECKPOINT")
EPOCHS, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, NUM_WORKERS, SEED = 50, 2, 2, 2, 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validation_loss(model, loader, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {key: value.to(device, non_blocking=True) for key, value in target.items()}
                for target in targets
            ]
            losses = model(images, targets)
            total_loss += float(sum(losses.values()))
            batches += 1
    return total_loss / max(1, batches)


def main() -> None:
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
    set_seed(SEED + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=True).to(device)
    dataset = CocoDetectionDataset("train")
    validation_dataset = CocoDetectionDataset("validation")
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=sampler is None, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0, collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        collate_fn=collate_fn,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    start_epoch, history = 0, []
    if RESUME_CHECKPOINT:
        start_epoch = load_checkpoint(Path(RESUME_CHECKPOINT), model, optimizer, scheduler)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    for epoch in range(start_epoch, EPOCHS):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total, batches = 0.0, 0
        for batch_index, (images, targets) in enumerate(loader):
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                losses = model(images, targets)
                loss = sum(losses.values()) / GRADIENT_ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            if (batch_index + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_total += float(loss.detach()) * GRADIENT_ACCUMULATION_STEPS
            batches += 1
        if batches % GRADIENT_ACCUMULATION_STEPS:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        epoch_loss = loss_total / max(1, batches)
        if distributed:
            torch.distributed.barrier()
        validation_model = model.module if distributed else model
        validation_epoch_loss = validation_loss(
            validation_model,
            validation_loader,
            device,
        )
        if distributed:
            value = torch.tensor(epoch_loss, device=device)
            torch.distributed.all_reduce(value)
            epoch_loss = float(value.item() / torch.distributed.get_world_size())
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "validation_loss": validation_epoch_loss,
            }
        )
        if rank == 0:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            print(
                f"Epoch {epoch + 1}/{EPOCHS} - "
                f"train_loss: {epoch_loss:.4f} - "
                f"validation_loss: {validation_epoch_loss:.4f}"
            )
            if (epoch + 1) % 5 == 0 or epoch + 1 == EPOCHS:
                save_checkpoint(OUTPUT_DIR / f"checkpoint-{epoch + 1}.pt", model, optimizer, scheduler, epoch + 1, history)
            best_checkpoint = OUTPUT_DIR / "best.pt"
            previous_best = min(
                (entry["validation_loss"] for entry in history[:-1]),
                default=float("inf"),
            )
            if validation_epoch_loss < previous_best:
                save_checkpoint(
                    best_checkpoint,
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    history,
                )
        if distributed:
            torch.distributed.barrier()
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Train Faster R-CNN on the prepared COCO dataset."
    ).parse_args()
    main()
