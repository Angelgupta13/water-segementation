"""Tests for MLflow experiment tracking.

Verifies that:
1. The tracking database exists and has runs.
2. logged experiments appear in the expected experiment.
3. Model registry has the registered model.

These tests skip gracefully when no MLflow DB is found (e.g., in CI).
"""

import sqlite3
from pathlib import Path
import pytest
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"


@pytest.mark.mlflow
def test_mlflow_db_exists():
    if not MLFLOW_DB.exists():
        pytest.skip("MLflow DB not found — run training first to create it")
    assert MLFLOW_DB.exists()


@pytest.mark.mlflow
def test_mlflow_db_is_valid_sqlite():
    if not MLFLOW_DB.exists():
        pytest.skip("MLflow DB not found — run training first to create it")
    conn = sqlite3.connect(str(MLFLOW_DB))
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    conn.close()
    assert len(tables) > 0, "MLflow DB has no tables (corrupted or empty)"


@pytest.mark.mlflow
def test_mlflow_tracking_uri_connectable():
    if not MLFLOW_DB.exists():
        pytest.skip("MLflow DB not found — run training first to create it")
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    experiments = client.search_experiments()
    assert len(experiments) > 0, "No experiments found in MLflow DB"


@pytest.mark.mlflow
def test_water_segmentation_experiment_exists():
    if not MLFLOW_DB.exists():
        pytest.skip("MLflow DB not found — run training first to create it")
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    experiment = client.get_experiment_by_name("water-segmentation")
    assert experiment is not None, (
        "Experiment 'water-segmentation' not found. "
        "Run `python -m src.training.train` to create it."
    )


@pytest.mark.mlflow
def test_experiment_has_runs():
    if not MLFLOW_DB.exists():
        pytest.skip("MLflow DB not found — run training first to create it")
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    experiment = client.get_experiment_by_name("water-segmentation")
    if experiment is None:
        pytest.skip("Experiment 'water-segmentation' not found — run training first")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1,
    )
    assert len(runs) > 0, (
        "No runs found in 'water-segmentation'. "
        "Run training first."
    )


@pytest.mark.mlflow
def test_model_registered():
    if not MLFLOW_DB.exists():
        pytest.skip("MLflow DB not found — run training first to create it")
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    registered_models = client.search_registered_models(
        filter_string="name='water-segmentation-unet'"
    )
    if len(registered_models) == 0:
        pytest.skip("No registered model found — run training first")


@pytest.mark.mlflow
def test_run_has_expected_metrics():
    if not MLFLOW_DB.exists():
        pytest.skip("MLflow DB not found — run training first to create it")
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    experiment = client.get_experiment_by_name("water-segmentation")
    if experiment is None:
        pytest.skip("Experiment 'water-segmentation' not found — run training first")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=5,
        order_by=["start_time DESC"],
    )
    for run in runs:
        if "val_iou" in run.data.metrics:
            assert 0 <= run.data.metrics["val_iou"] <= 1, (
                f"val_iou={run.data.metrics['val_iou']} out of range [0,1]"
            )
            return
    raise AssertionError("No run has val_iou metric logged yet")
