# Water Body Segmentation

Binary segmentation of water bodies from Sentinel-2 satellite imagery using U-Net + ResNet34.

## Quick Start

```bash
git clone https://github.com/Angelgupta13/water-segmentation.git
cd water-segmentation
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

## Pipeline

```
Data (Kaggle) → Preprocess (256x256, ImageNet norm, aug) → Train (U-Net + ResNet34)
→ MLflow tracking (metrics/epoch, model registry) → ONNX export
→ FastAPI inference server → Docker container → CI/CD (GitHub Actions → Docker Hub)
```

## Dataset

[Kaggle — Satellite Images of Water Bodies](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies)
- **2,841** RGB images + binary masks at ~2009x2007 resolution
- Sentinel-2 (Bands 8 & 3 for NDWI), masks thresholded at >200 to remove JPEG artifacts

### Preprocessing

Resize to 256x256 (bilinear images, nearest-neighbor masks) -> Normalize (ImageNet) -> Augment (flip 50%, rotate 50%, color jitter 30%).
Inference: sliding-window tiling (256x256 tiles, 32px overlap) for memory efficiency on large rasters.

## Project Structure

```
water-segmentation-project/
├── src/
│   ├── ingestion/
│   │   └── dataset.py           # WaterDataset, transforms, tiling/merge, dataloaders
│   ├── training/
│   │   ├── model.py             # U-Net + ResNet34 (21M params)
│   │   ├── train.py             # Training loop, MLflow tracking, early stopping
│   │   ├── tests.py             # 6 unit tests
│   │   ├── utils.py             # get_metrics, iou_score
│   │   └── hypertune.py         # Grid search (6 configs)
│   └── inference/
│       ├── api.py               # FastAPI server
│       └── inference.py         # ONNX/PyTorch inference, CLI entry point
├── tests/
│   ├── conftest.py              # Empty (cleaned)
│   ├── test_api.py              # 9 API tests
│   └── test_mlflow.py           # 7 MLflow tests (needs local DB)
├── scripts/
│   ├── download_data.py         # Kaggle dataset downloader
│   └── test_docker.ps1          # Docker lifecycle test
├── notebooks/
│   ├── eda.ipynb                # Dataset exploration (401 lines)
│   └── demo.ipynb               # End-to-end pipeline walkthrough (675 lines)
├── .github/workflows/lint.yml   # CI: lint -> test -> Docker push
├── Dockerfile                   # python:3.11-slim, ONNX baked in, HEALTHCHECK
├── docker-compose.yml           # API + MLflow server services
├── pyproject.toml               # Package config, [test] extras, CLI entry point
├── requirements.txt             # 13 deps, mlflow>=2.12.0
├── REPORT.md                    # 11 sections, 216 lines
├── presentation.pptx            # 8 slides, 5 embedded images, 5.3 MB
├── model.onnx                   # 97 MB, Git LFS tracked
├── .gitattributes               # LFS: *.onnx, *.pth
└── .gitignore                   # mlflow.db, best_model.pth, ~$*.pptx, data/
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

### Inference Optimizations

| Backend | Device | Latency/tile | Speedup |
|---|---|---|---|
| PyTorch | CPU | ~120 ms | 1.0x |
| ONNX Runtime | CPU | ~55 ms | **2.2x** |
| ONNX Runtime | GPU | ~15 ms | 8.0x |

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

## Docker

```bash
git lfs pull
docker build -t water-seg .
docker run -d -p 8000:8000 water-seg

# docker-compose (API + MLflow server)
docker-compose up -d
```

## Security

- **Path traversal**: `tempfile.NamedTemporaryFile` with safe suffix
- **Upload limit**: 100 MB cap returns HTTP 413
- **Rate limiter**: 60 req/min/IP sliding window, HTTP 429 with `Retry-After`, periodic IP eviction (300s)
- **Thread safety**: `threading.Lock` wraps all `MODEL_CACHE` reads/writes

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
| Version control | Git + Git LFS (model.onnx), conventional commits |
| Experiment tracking | MLflow (params, metrics, artifacts per epoch, 6 runs) |
| Model registry | MLflow: `water-segmentation-unet` v1, `serialization_format="pt2"` |
| Data versioning | SHA256 dataset hash logged per run |
| CI/CD | GitHub Actions: lint -> test -> Docker push |
| Security | Path traversal protection, 100 MB cap, rate limiter, thread-safe cache |

## CI Status

- Lint clean: `flake8 src/ tests/` → 0 errors
- 15/15 tests pass: 6 training + 9 API
- Docker image: `angelgupta/water-segmentation:latest` on Docker Hub
- All deliverables tracked: code, report, presentation, notebooks, container