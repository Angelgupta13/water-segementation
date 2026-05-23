import hashlib
import os
import random
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from segmentation_models_pytorch.losses import DiceLoss

from src.ingestion.dataset import get_loaders
from src.training.model import get_model
from src.training.utils import get_metrics

torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_DIR = "data/Images"
MASK_DIR = "data/Masks"
EPOCHS, LR, BATCH_SIZE = 50, 5e-5, 8

bce_fn = nn.BCEWithLogitsLoss()
dice_fn = DiceLoss(mode="binary")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def criterion(preds: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return 0.5 * bce_fn(preds, masks) + 0.5 * dice_fn(preds, masks)


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: Optional[torch.optim.Optimizer] = None):
    """One pass over the data. Trains if optimizer is provided, otherwise evaluates."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0
    total_metrics = {"iou": 0, "accuracy": 0, "precision": 0, "recall": 0}
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE).float()
            preds = model(imgs)
            loss = criterion(preds, masks)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            m = get_metrics(preds, masks)
            for k in total_metrics:
                total_metrics[k] += m[k]
    n = len(loader)
    return total_loss / n, {k: v / n for k, v in total_metrics.items()}


def _compute_dir_hash(directory: str) -> str:
    """SHA256 of sorted filenames + first 1KB of each file for data versioning."""
    hasher = hashlib.sha256()
    for fname in sorted(os.listdir(directory)):
        hasher.update(fname.encode())
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath):
            with open(fpath, "rb") as f:
                hasher.update(f.read(1024))
    return hasher.hexdigest()[:12]


def train(lr: float = LR, batch_size: int = BATCH_SIZE, epochs: int = EPOCHS,
          log_mlflow: bool = True, run_name: Optional[str] = None, seed: int = 42,
          img_dir: str = IMG_DIR, mask_dir: str = MASK_DIR) -> float:
    """Full training loop with MLflow tracking, early stopping, and best-model checkpointing."""
    set_seed(seed)
    train_loader, val_loader, test_loader = get_loaders(img_dir, mask_dir, batch_size=batch_size)
    model = get_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_iou, patience, no_improve = 0, 10, 0

    if log_mlflow:
        import mlflow
        import mlflow.pytorch
        mlflow.set_experiment("water-segmentation")
        ctx = mlflow.start_run(run_name=run_name)
    else:
        ctx = nullcontext()

    with ctx:
        if log_mlflow:
            dataset_hash = _compute_dir_hash(img_dir)
            mlflow.log_params({
                "model": "unet-resnet34", "loss": "bce+dice", "lr": lr,
                "batch_size": batch_size, "epochs": epochs, "optimizer": "AdamW",
                "dataset": "kaggle-satellite-water-bodies", "dataset_size": 2841,
                "dataset_hash": dataset_hash,
                "img_size": "256x256", "aug": "hflip,vflip,rotate90,colorjitter",
                "seed": seed,
            })

        for epoch in range(epochs):
            train_loss, train_m = run_epoch(model, train_loader, optimizer)
            val_loss, val_m = run_epoch(model, val_loader)
            scheduler.step()

            if log_mlflow:
                mlflow.log_metrics({
                    "train_loss": train_loss, "val_loss": val_loss,
                    "val_iou": val_m["iou"], "val_accuracy": val_m["accuracy"],
                    "val_precision": val_m["precision"], "val_recall": val_m["recall"],
                    "train_iou": train_m["iou"],
                }, step=epoch)

            print(f"Epoch {epoch+1:02d}/{epochs} | loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} | IOU={val_m['iou']:.4f} "
                  f"Acc={val_m['accuracy']:.4f} Prec={val_m['precision']:.4f} "
                  f"Rec={val_m['recall']:.4f}")

            if val_m["iou"] > best_iou:
                best_iou, no_improve = val_m["iou"], 0
                torch.save(model.state_dict(), "best_model.pth")
                print(f"saved (IOU={best_iou:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stop. Best IOU: {best_iou:.4f}")
                    break

        test_loss, test_m = run_epoch(model, test_loader)
        print(f"Test  | loss={test_loss:.4f} | IOU={test_m['iou']:.4f} "
              f"Acc={test_m['accuracy']:.4f} Prec={test_m['precision']:.4f} "
              f"Rec={test_m['recall']:.4f}")

        if log_mlflow:
            mlflow.pytorch.log_model(model, "model", registered_model_name="water-segmentation-unet", serialization_format="pt2")
            if os.path.exists("model.onnx"):
                mlflow.log_artifact("model.onnx", artifact_path="models")
            mlflow.log_metrics({
                "test_loss": test_loss, "test_iou": test_m["iou"],
                "test_accuracy": test_m["accuracy"], "test_precision": test_m["precision"],
                "test_recall": test_m["recall"],
            })
    return best_iou


if __name__ == "__main__":
    train(log_mlflow=True)
