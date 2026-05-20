import sys, os, torch, mlflow, mlflow.pytorch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch.nn as nn
from contextlib import nullcontext
from segmentation_models_pytorch.losses import DiceLoss
from src.ingestion.dataset import get_loaders
from src.training.model import get_model
from src.training.utils import get_metrics

torch.backends.cudnn.benchmark = True

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_DIR  = "data/Images"
MASK_DIR = "data/Masks"
EPOCHS, LR, BATCH_SIZE, VAL_SPLIT = 50, 5e-5, 8, 0.2

bce_fn  = nn.BCEWithLogitsLoss()
dice_fn = DiceLoss(mode="binary")

def criterion(preds, masks):
    return 0.5 * bce_fn(preds, masks) + 0.5 * dice_fn(preds, masks)


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0
    total_metrics = {"iou": 0, "accuracy": 0, "precision": 0, "recall": 0}
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE).float()
            preds = model(imgs)
            loss  = criterion(preds, masks)
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


def train(lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS, log_mlflow=True):
    train_loader, val_loader = get_loaders(IMG_DIR, MASK_DIR, val_split=VAL_SPLIT, batch_size=batch_size)
    model     = get_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_iou, patience, no_improve = 0, 10, 0

    if log_mlflow:
        mlflow.set_experiment("water-segmentation")
    ctx = mlflow.start_run() if log_mlflow else nullcontext()
    with ctx:
        mlflow.log_params({
            "model": "unet-resnet34", "loss": "bce+dice", "lr": lr,
            "batch_size": batch_size, "epochs": epochs, "optimizer": "AdamW",
            "dataset": "kaggle-satellite-water-bodies", "dataset_size": 2841,
            "img_size": "256x256", "aug": "hflip,vflip,rotate90,colorjitter",
        })

        for epoch in range(epochs):
            train_loss, train_m = run_epoch(model, train_loader, optimizer)
            val_loss,   val_m   = run_epoch(model, val_loader)
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss, "val_loss": val_loss,
                "val_iou": val_m["iou"], "val_accuracy": val_m["accuracy"],
                "val_precision": val_m["precision"], "val_recall": val_m["recall"],
                "train_iou": train_m["iou"],
            }, step=epoch)

            print(f"Epoch {epoch+1:02d}/{epochs} | loss={train_loss:.4f} val_loss={val_loss:.4f} | IOU={val_m['iou']:.4f} Acc={val_m['accuracy']:.4f} Prec={val_m['precision']:.4f} Rec={val_m['recall']:.4f}")

            if val_m["iou"] > best_iou:
                best_iou, no_improve = val_m["iou"], 0
                torch.save(model.state_dict(), "best_model.pth")
                mlflow.pytorch.log_model(model, "model")
                print(f"saved (IOU={best_iou:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stop. Best IOU: {best_iou:.4f}")
                    break
    return best_iou


if __name__ == "__main__":
    train(log_mlflow=True)
