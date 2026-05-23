# Water Body Segmentation

Binary segmentation of water bodies from Sentinel-2 satellite imagery using U-Net + ResNet34.

## Quick Start

```bash
git clone https://github.com/Angelgupta13/water-segementation.git
cd water-segementation
git lfs pull
pip install -e .

docker build -t water-seg .
docker run -d -p 8000:8000 water-seg
curl -X POST http://localhost:8000/predict -F "file=@image.jpg" -o mask.png
curl http://localhost:8000/health
```

## Deliverables Mapping

| Deliverable | Location |
|---|---|
| Code Repository | GitHub with `src/ingestion/`, `src/training/`, `src/inference/`, `tests/` |
| Container Image | `Dockerfile` + `docker-compose.yml` + Docker Hub: `angelgupta/water-segmentation:latest` |
| Report & Presentation | `REPORT.md` + `presentation.pptx` + `notebooks/eda.ipynb` + `notebooks/demo.ipynb` |
| Working Demo | `POST /predict` endpoint + `notebooks/demo.ipynb` |

## Dataset

[Kaggle — Satellite Images of Water Bodies](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies)
- **2,841** RGB images + binary masks at ~2009x2007 resolution
- Sentinel-2 (Bands 8 & 3 for NDWI), masks thresholded at >200 to remove JPEG artifacts

### Preprocessing

Resize to 256x256 (bilinear images, nearest-neighbor masks) -> Normalize (ImageNet) -> Augment (flip 50%, rotate 50%, color jitter 30%).
Inference: sliding-window tiling (256x256 tiles, 32px overlap) for memory efficiency on large rasters.

## Project Structure

```
src/
 ingestion/    WaterDataset, transforms, tiling/merge
 training/     model.py, train.py, utils.py, hypertune.py, tests.py
 inference/    api.py (FastAPI), inference.py (ONNX + CLI)
tests/
 test_api.py         9 API endpoint tests
 test_mlflow.py      7 MLflow tracking tests (local DB)
training/tests.py    6 training unit tests
scripts/
 download_data.py    Kaggle downloader
 test_docker.ps1     Docker lifecycle test
 make_pptx.py        Presentation generator
```

## Setup

```bash
git lfs pull
pip install -e .
python scripts/download_data.py   # requires Kaggle API token
```

CLI entry point: `water-seg path/to/image.jpg --backend onnx`

## Training

```bash
python -m src.training.train              # 50 epochs, early stopping
python -m src.training.hypertune          # grid search: 6 configs
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Results

| Config | LR | Batch | IOU | Acc | Prec | Recall |
|---|---|---|---|---|---|---|
| lr1e-4_bs8 (best) | 1e-4 | 8 | **0.8018** | 0.9265 | 0.9093 | 0.8696 |
| lr5e-5_bs8 | 5e-5 | 8 | 0.7928 | 0.9231 | 0.9056 | 0.8621 |
| lr1e-4_bs16 | 1e-4 | 16 | 0.7542 | 0.9086 | 0.8781 | 0.8375 |
| lr5e-5_bs16 | 5e-5 | 16 | 0.7389 | 0.9012 | 0.8710 | 0.8189 |
| lr1e-5_bs16 | 1e-5 | 16 | 0.7142 | 0.8902 | 0.8629 | 0.8051 |
| lr5e-4_bs8 | 5e-4 | 8 | 0.6215 | 0.8523 | 0.7581 | 0.7342 |

**Final model (epoch 34):** Test IOU **0.8249** | Acc **93.8%** | Prec **91.9%** | Rec **88.7%**

Precision > Recall across all runs — model under-segments, which is the safer bias for flood mapping (fewer false positives).

### Failure Analysis

| Failure | Cause |
|---|---|
| Narrow waterways (<4px) | 256x256 resize destroys thin features |
| Shadow misclassified as water | RGB-only; NIR band needed |
| JPEG boundary noise | Compression artifacts in source masks |

Full analysis in `REPORT.md` and `notebooks/eda.ipynb`.

## Inference

```bash
# CLI
water-seg image.jpg --backend onnx --tta
python -m src.inference.inference image.jpg --backend pytorch

# API (Docker)
curl -X POST http://localhost:8000/predict -F "file=@image.jpg" -F "tta=true" -o mask.png
```

**Optimizations:** ONNX Runtime (~2.2x CPU speedup), sliding-window tiling, TTA (horizontal flip), multi-backend fallback.

## Docker

```bash
git lfs pull
docker build -t water-seg .
docker run -d -p 8000:8000 water-seg

# docker-compose (API + MLflow server)
docker-compose up -d
```

## CI/CD

On push to master: `flake8 src/ tests/` -> `pytest src/training/tests.py tests/test_api.py -v` -> Docker build & push to `angelgupta/water-segmentation:latest`.

## Testing

```bash
pytest src/training/tests.py tests/test_api.py -v    # 15 tests
pytest tests/test_mlflow.py -v -m mlflow              # 7 tests (needs local mlflow.db)
```

## MLOps

| Practice | Implementation |
|---|---|
| Version control | Git + Git LFS (model.onnx) |
| Experiment tracking | MLflow (params, metrics, artifacts per epoch) |
| Model registry | MLflow: `water-segmentation-unet` v1 |
| Data versioning | SHA256 dataset hash logged per run |
| CI/CD | GitHub Actions: lint -> test -> Docker push |
| Security | Path traversal protection, 100 MB cap, rate limiter, thread-safe cache |
