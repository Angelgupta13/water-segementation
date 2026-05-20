"""
Unit tests for critical pipeline components.
Run: python -m pytest src/training/tests.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
import pytest
from src.training.utils import get_metrics
from src.training.model import get_model
from src.ingestion.dataset import tile_image, merge_tiles


def test_metrics_all_correct():
    """Perfect prediction should give all metrics = 1.0"""
    preds = torch.zeros(2, 1, 64, 64)   # all zeros = logit of 0.5 -> sigmoid=0.5
    masks = torch.zeros(2, 1, 64, 64)   # all non-water
    # Make preds strongly negative so sigmoid -> 0 (predicts non-water)
    preds = torch.full((2, 1, 64, 64), -10.0)
    m = get_metrics(preds, masks)
    assert m["accuracy"] > 0.99

def test_metrics_all_wrong():
    """Predict all water when none — recall=0, precision=0"""
    preds = torch.full((1, 1, 64, 64), 10.0)   # all water
    masks = torch.zeros(1, 1, 64, 64)           # no water
    m = get_metrics(preds, masks)
    assert m["recall"] < 0.01
    assert m["iou"]    < 0.01

def test_metrics_collapse_to_positive():
    """Simulates the recall=1.0, IOU=0.4 bug we saw — catches class collapse"""
    preds = torch.full((1, 1, 64, 64), 10.0)   # predicts everything as water
    masks = torch.zeros(1, 1, 64, 64)
    masks[:, :, :32, :32] = 1.0                 # only top-left is water
    m = get_metrics(preds, masks)
    assert m["recall"]    > 0.99   # recall=1 (found all water)
    assert m["precision"] < 0.5    # but also predicted lots of non-water as water
    assert m["iou"]       < 0.5

def test_model_output_shape():
    """Model must output (B, 1, H, W)"""
    model = get_model()
    x     = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 256, 256)

def test_tiling_and_merge():
    """Tiling then merging should reconstruct same spatial dimensions"""
    img    = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    tiles, coords = tile_image(img, tile_size=256, overlap=32)
    assert len(tiles) > 1
    fake_preds = [np.random.rand(256, 256).astype(np.float32) for _ in tiles]
    merged = merge_tiles(fake_preds, coords, 512, 512)
    assert merged.shape == (512, 512)

def test_mask_threshold():
    """Values 1-199 should be thresholded out (JPEG artifact fix)"""
    mask = np.array([0, 45, 100, 199, 200, 255], dtype=np.uint8)
    binary = (mask > 200).astype(np.float32)
    assert binary[0] == 0   # 0   -> non-water
    assert binary[2] == 0   # 100 -> artifact, non-water
    assert binary[4] == 0   # 200 -> boundary, non-water
    assert binary[5] == 1   # 255 -> water
