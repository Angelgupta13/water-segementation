"""Tests for the FastAPI inference server endpoints.

These tests verify that the API wrapper layer (file I/O, backend selection,
error handling, response formatting) works correctly. The actual model
inference is replaced with a mock at module level BEFORE the TestClient is
created, so both the test thread and the ASGI thread see the same mock.
"""

import numpy as np
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

# ── Module-level mocks (visible to ASGI thread) ─────────────────────────────
import src.inference.api as api
from src.inference.api import MODEL_CACHE, app

FAKE_MASK = np.zeros((32, 32), dtype=np.uint8)
FAKE_MASK[:, 16:] = 1

# Replace the real predict with a mock BEFORE TestClient is created
api.run_inference = MagicMock(return_value=FAKE_MASK)


@pytest.fixture(autouse=True)
def reset_cache():
    """Each test starts with a clean cache; test sets what it needs."""
    MODEL_CACHE.clear()
    yield


@pytest.fixture
def client():
    return TestClient(app)


def _jpg_bytes() -> bytes:
    """A tiny 32x32 RGB JPEG image (blue-ish) for test uploads."""
    import cv2
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 2] = 255
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ── /health ──────────────────────────────────────────────────────────────────


def test_health_returns_ok(client):
    MODEL_CACHE["pytorch"] = MagicMock()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_degraded_when_no_model(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "degraded"}


# ── /predict — success paths ────────────────────────────────────────────────


def test_predict_returns_png(client):
    MODEL_CACHE["onnx"] = MagicMock()
    resp = client.post("/predict", files={"file": ("test.jpg", _jpg_bytes(), "image/jpeg")})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
    assert resp.headers["content-type"] == "image/png"
    assert len(resp.content) > 100


def test_predict_tta_flag(client):
    MODEL_CACHE["pytorch"] = MagicMock()
    client.post("/predict?tta=true", files={"file": ("test.jpg", _jpg_bytes(), "image/jpeg")})
    assert api.run_inference.call_count >= 1
    _call_kwargs = api.run_inference.call_args[-1]
    assert _call_kwargs.get("tta") is True


def test_predict_default_backend_is_onnx(client):
    """When both backends available, ONNX is used by default."""
    MODEL_CACHE["onnx"] = MagicMock()
    MODEL_CACHE["pytorch"] = MagicMock()
    api.run_inference.reset_mock()
    client.post("/predict", files={"file": ("test.jpg", _jpg_bytes(), "image/jpeg")})
    _call_kwargs = api.run_inference.call_args[-1]
    assert _call_kwargs.get("backend") == "onnx"


def test_predict_fallback_onnx_to_pytorch(client):
    """When ONNX not cached but PyTorch is, fallback to pytorch."""
    MODEL_CACHE["pytorch"] = MagicMock()
    api.run_inference.reset_mock()
    resp = client.post("/predict?backend=onnx",
                       files={"file": ("t.jpg", _jpg_bytes(), "image/jpeg")})
    assert resp.status_code == 200
    _call_kwargs = api.run_inference.call_args[-1]
    assert _call_kwargs.get("backend") == "pytorch"


# ── /predict — error paths ──────────────────────────────────────────────────


def test_predict_without_file_returns_422(client):
    """FastAPI returns 422 for missing required file params (not 400)."""
    MODEL_CACHE["onnx"] = MagicMock()
    resp = client.post("/predict")
    assert resp.status_code == 422


def test_predict_invalid_backend_returns_400(client):
    MODEL_CACHE["onnx"] = MagicMock()
    resp = client.post("/predict?backend=tensorrt",
                       files={"file": ("t.jpg", b"fake", "image/jpeg")})
    assert resp.status_code == 400
    assert "Unsupported backend" in resp.json()["detail"]


def test_predict_503_when_neither_backend_loaded(client):
    resp = client.post("/predict", files={"file": ("t.jpg", _jpg_bytes(), "image/jpeg")})
    assert resp.status_code == 503, f"Expected 503, got {resp.status_code}: {resp.text[:200]}"
