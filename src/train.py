#!/usr/bin/env python3
"""PyTorch training and run artifact generation."""

# --- Import ---------------------------------------------------------------
# Imports used for training, dataset loading, and artifact writing.

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from backend import (
    BackendError,
    configure_torch_backend,
    resolve_amp_config,
    resolve_torch_device,
)
from results_writer import write_training_artifacts, write_training_plot

# --- Constants ------------------------------------------------------------
# Backend options and ImageNet normalization values.

BACKEND_CHOICES = ["auto", "cpu", "cuda", "rocm"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- CLI arguments --------------------------------------------------------
# Command line parser for training.

def build_parser():
    parser = argparse.ArgumentParser(prog="train", description="Train and create a run with artifacts.")
    parser.add_argument("--data_dir", required=True, help="Path with train/val/test.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default="auto",
        help="Backend for training (auto|cpu|cuda|rocm).",
    )
    return parser

# --- Support: filesystem --------------------------------------------------
# Validate expected dataset layout.

def _validate_dataset(data_dir):
    # Ensure train/val/test folders exist.
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists() or not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split: {split_dir}")

    # Require at least two classes.
    class_dirs = [p for p in (data_dir / "train").iterdir() if p.is_dir()]
    if len(class_dirs) < 2:
        raise ValueError("At least 2 classes are required in train/")

    # Ensure each split is not empty.
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not any(split_dir.rglob("*")):
            raise ValueError(f"Empty split: {split_dir}")

# --- Support: run ---------------------------------------------------------
# Create run directory and start timestamp.

def _setup_run():
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path("artifacts") / "runs" / run_id
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    return run_id, run_dir, start_time

# --- Support: data --------------------------------------------------------
# Build transforms, datasets, and dataloaders.

def _build_data(data_path, img_size, batch_size, num_workers, pin_memory):
    # Shared preprocessing for all splits.
    base_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_ds = datasets.ImageFolder(data_path / "train", transform=base_transform)
    val_ds = datasets.ImageFolder(data_path / "val", transform=base_transform)
    test_ds = datasets.ImageFolder(data_path / "test", transform=base_transform)
    # DataLoader tuning to reduce CPU-side input bottlenecks.
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs),
        train_ds.classes,
    )

# --- Support: model -------------------------------------------------------
# Build model and enable training on final layers only.

def _build_model(num_classes, device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Freeze backbone, train only layer4 and classifier head.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    return model.to(device)

# --- Support: metrics -----------------------------------------------------
# Evaluate loss and top-k accuracy.

def _eval(
    model,
    loader,
    criterion,
    device,
):
    """Evaluation: loss, top1/top3, and predictions for confusion matrix."""
    model.eval()
    total_loss, total, correct_top1, correct_top3 = 0.0, 0, 0, 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            _, top3 = logits.topk(k=min(3, logits.size(1)), dim=1)
            top1 = top3[:, 0]
            correct_top1 += (top1 == y).sum().item()
            correct_top3 += top3.eq(y.view(-1, 1)).any(dim=1).sum().item()

            y_true.extend(y.cpu().tolist())
            y_pred.extend(top1.cpu().tolist())

    denom = max(total, 1)
    return total_loss / denom, correct_top1 / denom, correct_top3 / denom, y_true, y_pred

# --- Training -------------------------------------------------------------
# Main flow: train, evaluate, export ONNX, and write artifacts.

def train(
    data_dir,
    epochs,
    batch_size,
    img_size,
    lr,
    seed,
    num_workers,
    backend,
):
    run_params = {
        "data_dir": data_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size,
        "lr": lr,
        "seed": seed,
        "num_workers": num_workers,
        "backend": backend,
    }

    data_path = Path(data_dir)
    _validate_dataset(data_path)

    run_id, run_dir, start_time = _setup_run()

    # Seed for reproducibility and select execution device.
    torch.manual_seed(seed)
    try:
        device = resolve_torch_device(backend)
    except BackendError as exc:
        raise SystemExit(f"Invalid backend for training: {exc}") from exc
    configure_torch_backend(backend, device, verbose=False)
    amp_config = resolve_amp_config(device)
    scaler = None
    if amp_config["use_grad_scaler"]:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Pin host memory only when using CUDA-like backends.
    use_pin_memory = device.type == "cuda"
    train_loader, val_loader, test_loader, labels = _build_data(
        data_path, img_size, batch_size, num_workers, use_pin_memory
    )

    model = _build_model(len(labels), device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)

    final_train_loss, final_val_loss, val_top1, val_top3 = 0.0, 0.0, 0.0, 0.0
    epoch_metrics = []

    # Training loop.
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        samples_seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(
                device_type=device.type,
                dtype=amp_config["dtype"],
                enabled=amp_config["enabled"],
            ):
                logits = model(x)
                loss = criterion(logits, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * x.size(0)
            samples_seen += x.size(0)

        # Validation after each epoch.
        final_train_loss = running_loss / max(samples_seen, 1)
        final_val_loss, val_top1, val_top3, _, _ = _eval(
            model, val_loader, criterion, device
        )
        # Epoch-level training log with losses only.
        print(
            f"[epoch {epoch + 1}/{epochs}] "
            f"train_loss={final_train_loss:.6f} "
            f"val_loss={final_val_loss:.6f}"
        )
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(final_train_loss, 6),
                "val_loss": round(final_val_loss, 6),
                "val_top1": round(val_top1, 6),
                "val_top3": round(val_top3, 6),
            }
        )

    # Final test and confusion matrix.
    test_loss, test_top1, test_top3, y_true, y_pred = _eval(
        model, test_loader, criterion, device
    )
    confusion_matrix = [[0] * len(labels) for _ in range(len(labels))]
    for target_idx, pred_idx in zip(y_true, y_pred):
        confusion_matrix[target_idx][pred_idx] += 1

    # Export ONNX for inference.
    model.eval()
    model_cpu = model.to("cpu")
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model_cpu,
        dummy,
        run_dir / "model.onnx",
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
    )

    # Save run artifacts and metrics.
    run_metrics = {
        "start_time": start_time,
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epoch_metrics": epoch_metrics,
        "final_train_loss": round(final_train_loss, 6),
        "final_val_loss": round(final_val_loss, 6),
        "val_top1": round(val_top1, 6),
        "val_top3": round(val_top3, 6),
        "test_loss": round(test_loss, 6),
        "test_top1": round(test_top1, 6),
        "test_top3": round(test_top3, 6),
    }
    write_training_artifacts(
        run_dir=run_dir,
        run_id=run_id,
        run_params=run_params,
        labels=labels,
        metrics=run_metrics,
        confusion_matrix=confusion_matrix,
    )

    # Save loss plot.
    write_training_plot(run_dir, epoch_metrics, run_params)
    print(f"Run created at: {run_dir}")

# --- Main -----------------------------------------------------------------
# Entry point: parse args and start training.

def main():
    args = build_parser().parse_args()
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
        backend=args.backend,
    )

# --- Script start ---------------------------------------------------------
# Script entrypoint executed from shell.

if __name__ == "__main__":
    main()
