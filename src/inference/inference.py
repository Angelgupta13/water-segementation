"""
Optimized inference service for water body segmentation.

Optimizations:
1. ONNX export for runtime speedup (~2x vs PyTorch on CPU)
2. Tiled inference for large rasters (avoids memory overflow)
3. Batch processing support
4. Structured logging with timing per image
"""
import os
import sys
import time
import logging
import argparse

import cv2
import numpy as np
import torch
import onnxruntime as ort
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.training.model import get_model
from src.ingestion.dataset import tile_image, merge_tiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

TRANSFORM = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_onnx(weights_path="best_model.pth", onnx_path="model.onnx"):
    model = get_model()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["image"],
        output_names=["mask"],
        dynamic_axes={"image": {0: "batch"}, "mask": {0: "batch"}},
        opset_version=17,
    )
    logger.info(f"ONNX model exported to {onnx_path}")
    return onnx_path


# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------

def load_pytorch(weights_path="best_model.pth"):
    model = get_model()
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval().to(DEVICE)
    logger.info(f"PyTorch model loaded from {weights_path}")
    return model


def load_onnx(onnx_path="model.onnx"):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session   = ort.InferenceSession(onnx_path, providers=providers)
    logger.info(f"ONNX model loaded from {onnx_path}")
    return session


def preprocess_tile(tile):
    aug = TRANSFORM(image=tile)["image"]
    return aug.unsqueeze(0).numpy()  # (1, 3, 256, 256)


def predict_tile_pytorch(model, tile):
    tensor = torch.from_numpy(preprocess_tile(tile)).to(DEVICE)
    with torch.no_grad():
        logit = model(tensor)
    return torch.sigmoid(logit).squeeze().cpu().numpy()


def predict_tile_onnx(session, tile):
    inp    = preprocess_tile(tile).astype(np.float32)
    logit  = session.run(["mask"], {"image": inp})[0]
    return 1 / (1 + np.exp(-logit.squeeze()))  # sigmoid


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def predict(img_path, out_path=None, backend="onnx",
            weights_path="best_model.pth", onnx_path="model.onnx",
            tile_size=256, overlap=32):

    t0  = time.time()
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Load backend
    if backend == "onnx":
        if not os.path.exists(onnx_path):
            logger.info("ONNX model not found — exporting from PyTorch weights")
            export_onnx(weights_path, onnx_path)
        runner = load_onnx(onnx_path)
        pred_fn = lambda tile: predict_tile_onnx(runner, tile)
    else:
        runner  = load_pytorch(weights_path)
        pred_fn = lambda tile: predict_tile_pytorch(runner, tile)

    # Tiled inference
    tiles, coords = tile_image(img, tile_size=tile_size, overlap=overlap)
    logger.info(f"Image {os.path.basename(img_path)} ({h}x{w}) -> {len(tiles)} tiles")

    tile_preds = [pred_fn(tile) for tile in tiles]
    full_mask  = merge_tiles(tile_preds, coords, h, w)

    elapsed = time.time() - t0
    logger.info(f"Inference complete in {elapsed:.3f}s | backend={backend}")

    if out_path is None:
        base     = os.path.splitext(os.path.basename(img_path))[0]
        out_path = f"results/{base}_mask.png"
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    cv2.imwrite(out_path, full_mask * 255)
    logger.info(f"Mask saved to {out_path}")
    return full_mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Water body segmentation inference")
    parser.add_argument("img_path",       help="Path to input image")
    parser.add_argument("--out",          default=None,           help="Output mask path")
    parser.add_argument("--backend",      default="onnx",         choices=["pytorch", "onnx"])
    parser.add_argument("--weights",      default="best_model.pth")
    parser.add_argument("--onnx",         default="model.onnx")
    args = parser.parse_args()

    predict(
        img_path=args.img_path,
        out_path=args.out,
        backend=args.backend,
        weights_path=args.weights,
        onnx_path=args.onnx,
    )
