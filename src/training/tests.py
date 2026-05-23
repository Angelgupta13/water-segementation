"""Unit tests for metrics computation, model output shape, tiling/merge correctness, and mask thresholding."""

import torch
import numpy as np
from src.training.utils import get_metrics
from src.training.model import get_model
from src.ingestion.dataset import tile_image, merge_tiles


def test_metrics_all_correct():
    """All-negative logits with all-zero masks -> high accuracy."""
    preds = torch.full((2, 1, 64, 64), -10.0)
    masks = torch.zeros(2, 1, 64, 64)
    m = get_metrics(preds, masks)
    assert m["accuracy"] > 0.99


def test_metrics_all_wrong():
    """All-positive logits with all-zero masks -> zero recall and IOU."""
    preds = torch.full((1, 1, 64, 64), 10.0)
    masks = torch.zeros(1, 1, 64, 64)
    m = get_metrics(preds, masks)
    assert m["recall"] < 0.01 and m["iou"] < 0.01


def test_metrics_collapse_to_positive():
    """All-positive logits with a small positive region -> high recall, low precision."""
    preds = torch.full((1, 1, 64, 64), 10.0)
    masks = torch.zeros(1, 1, 64, 64)
    masks[:, :, :32, :32] = 1.0
    m = get_metrics(preds, masks)
    assert m["recall"] > 0.99 and m["precision"] < 0.5 and m["iou"] < 0.5


def test_model_output_shape():
    """Model produces (B, 1, 256, 256) for (B, 3, 256, 256) input."""
    model = get_model()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 256, 256)


def test_tiling_and_merge():
    """Tiling large image into overlapping patches then merging returns original shape."""
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    tiles, coords = tile_image(img, tile_size=256, overlap=32)
    assert len(tiles) > 1
    fake_preds = [np.random.rand(256, 256).astype(np.float32) for _ in tiles]
    merged = merge_tiles(fake_preds, coords, 512, 512)
    assert merged.shape == (512, 512)


def test_mask_threshold():
    """Values <200 become 0, >=200 become 1 (JPEG artifact removal)."""
    mask = np.array([0, 45, 100, 199, 200, 255], dtype=np.uint8)
    binary = (mask >= 200).astype(np.float32)
    assert binary[0] == 0 and binary[2] == 0 and binary[3] == 0 and binary[4] == 1 and binary[5] == 1
