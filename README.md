# Water Body Segmentation

Binary segmentation of water bodies from Sentinel-2 satellite imagery using U-Net + ResNet34.

## Dataset
- **Source:** [Kaggle — Satellite Images of Water Bodies](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies)
- **Size:** 2,841 RGB images + binary masks
- **Sensor:** Sentinel-2 (preprocessed with rasterio)
- **Masks:** Generated via NDWI (bands 8 & 3), thresholded at >200 to remove JPEG artifacts

## Project Structure
```
water-segmentation/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── dataset.py        # Data loading, augmentation, tiling utilities
│   ├── training/
│   │   ├── __init__.py
│   │   ├── model.py          # U-Net + ResNet34
│   │   ├── train.py          # Training loop + MLflow tracking
│   │   ├── hypertune.py      # Grid search over LR and batch size
│   │   ├── utils.py          # IOU, Accuracy, Precision, Recall
│   │   └── tests.py          # Unit tests
│   └── inference/
│       ├── __init__.py
│       └── inference.py      # ONNX + tiled inference service + CLI
├── notebooks/
│   ├── eda.ipynb             # Dataset exploration
│   └── demo.ipynb            # End-to-end pipeline demo
├── .github/
│   └── workflows/
│       └── lint.yml          # CI: lint + unit tests
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
# From project root
python -m src.training.train
```

## Hyperparameter Tuning
```bash
python -m src.training.hypertune
```
View all runs:
```bash
mlflow ui
# Open http://localhost:5000
```

## Inference

### PyTorch backend
```bash
python -m src.inference.inference path/to/image.jpg --backend pytorch --weights best_model.pth
```

### ONNX backend (~2x faster on CPU)
```bash
# Export once after training
python -c "from src.inference.inference import export_onnx; export_onnx()"

# Run
python -m src.inference.inference path/to/image.jpg --backend onnx
```



## Unit Tests
```bash
pytest src/training/tests.py -v
```

## Results

| Run | LR | Batch | IOU | Accuracy | Precision | Recall | Val Loss |
|-----|----|-------|-----|----------|-----------|--------|----------|
| lr1e-4_bs8 (best) | 1e-4 | 8  | **0.8018** | 0.9265 | 0.9093 | 0.8696 | 0.1710 |
| lr5e-5_bs8 | 5e-5 | 8  | 0.7928 | 0.9231 | 0.9056 | 0.8621 | 0.1812 |
| lr1e-5_bs16 | 1e-5 | 16 | 0.7142 | 0.8902 | 0.8629 | 0.8051 | 0.2633 |

## Model Weights

`best_model.pth` (~93 MB) and `model.onnx` (~93 MB) are not tracked in git (binary files exceed GitHub's recommended size). Retrain to obtain them:

```bash
python -m src.training.train
```

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Resize to 256×256 for training | Kaggle images are pre-cropped small JPEGs (~259×158px). Patching caused class collapse (IOU=0.40, Recall=1.0) — model predicted everything as water. Tiling preserved for inference on true large rasters. |
| Mask threshold > 200 | JPEG compression artifacts produce values 1–199. Threshold at 200 eliminates them without cutting real water boundaries. |
| BCE + Dice loss (50/50) | BCE optimizes pixel-wise accuracy; Dice optimizes overlap directly. Combined is standard for binary segmentation and sufficient for 32/68 class ratio. |
| ResNet34 encoder | 21M params, strong ImageNet pretraining, proven on dense prediction tasks. Fits RTX 4050 (6GB) at batch=8. |
| ONNX export | ~2x inference speedup on CPU. Exported at Docker build time for zero cold-start overhead. |
| AdamW + CosineAnnealingLR | AdamW weight decay prevents overfitting; cosine schedule avoids sharp LR drops that stall convergence. |

## Known Limitations
- RGB input only — true Sentinel-2 NIR band (Band 8) would improve NDWI-based boundary detection
- JPEG mask compression introduces boundary noise even after threshold fix
- Training on resized images loses fine spatial detail present in original 2009×2007 rasters

## Suggested Next Steps
1. **EfficientNet-B4 encoder** — larger capacity, potential +2–3% IOU
2. **Test-time augmentation (TTA)** — average predictions across horizontal/vertical flips
3. **Morphological post-processing** — closing fills small holes in output masks
4. **Multispectral input** — add NIR band if raw Sentinel-2 bands are available
5. **CRF post-processing** — sharpen water/non-water boundaries
