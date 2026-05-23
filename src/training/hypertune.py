"""Hyperparameter search over learning rates and batch sizes."""
from typing import Dict, List
from src.training.train import train

SEARCH_SPACE: List[Dict] = [
    {"lr": 5e-4, "batch_size": 8, "epochs": 20, "run_name": "lr5e-4_bs8"},
    {"lr": 1e-4, "batch_size": 8, "epochs": 20, "run_name": "lr1e-4_bs8"},
    {"lr": 5e-5, "batch_size": 8, "epochs": 20, "run_name": "lr5e-5_bs8"},
    {"lr": 1e-5, "batch_size": 8, "epochs": 20, "run_name": "lr1e-5_bs8"},
    {"lr": 1e-4, "batch_size": 16, "epochs": 20, "run_name": "lr1e-4_bs16"},
    {"lr": 5e-5, "batch_size": 16, "epochs": 20, "run_name": "lr5e-5_bs16"},
]

if __name__ == "__main__":
    results = []
    for cfg in SEARCH_SPACE:
        run_name = cfg["run_name"]
        train_cfg = {k: v for k, v in cfg.items() if k != "run_name"}
        print(f"\n--- {run_name}: {train_cfg} ---")
        best_iou = train(**train_cfg, log_mlflow=True, run_name=run_name)
        results.append((best_iou, run_name))
    for iou, name in sorted(results, reverse=True):
        print(f"{name:<20} {iou:.4f}")
    print(f"\nBest: {results[0][1]} | IOU={results[0][0]:.4f}")
