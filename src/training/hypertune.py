"""
Hyperparameter tuning via MLflow.
Each config runs as a named MLflow run for easy comparison in the UI.
Run: python -m src.training.hypertune
Then view: mlflow ui
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlflow
from src.training.train import train

SEARCH_SPACE = [
    {"lr": 1e-4,  "batch_size": 8,  "epochs": 20, "run_name": "lr1e-4_bs8"},
    {"lr": 5e-5,  "batch_size": 8,  "epochs": 20, "run_name": "lr5e-5_bs8"},
    {"lr": 1e-5,  "batch_size": 16, "epochs": 20, "run_name": "lr1e-5_bs16"},
]

if __name__ == "__main__":
    mlflow.set_experiment("water-segmentation-hypertune")
    results = []

    for cfg in SEARCH_SPACE:
        run_name = cfg["run_name"]
        train_cfg = {k: v for k, v in cfg.items() if k != "run_name"}
        print(f"\n--- Run: {run_name} | Config: {train_cfg} ---")
        with mlflow.start_run(run_name=run_name):
            best_iou = train(**train_cfg, log_mlflow=False)
        results.append((best_iou, run_name, train_cfg))

    results.sort(reverse=True)
    print("\n=== Hypertuning Results ===")
    print(f"{'Run':<20} {'IOU':>8}  Config")
    for iou, name, cfg in results:
        print(f"{name:<20} {iou:>8.4f}  {cfg}")
    print(f"\nBest: {results[0][1]} — IOU={results[0][0]:.4f}")
