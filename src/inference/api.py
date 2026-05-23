"""FastAPI inference server for water body segmentation."""

import os
import time
import math
import logging
import tempfile
import threading
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.inference.inference import predict as run_inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_CACHE: Dict[str, object] = {}
MODEL_LOCK = threading.Lock()
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
RATE_LIMIT = 60  # requests per minute per IP
RATE_WINDOW = 60.0  # seconds
RATE_CLEANUP_INTERVAL = 300  # evict stale IPs every 5 minutes
_rate_store: Dict[str, list] = defaultdict(list)
_rate_lock = threading.Lock()
_last_cleanup = [time.time()]


def _evict_stale_ips():
    """Remove IPs with no requests in the last RATE_WINDOW to prevent unbounded growth."""
    now = time.time()
    cutoff = now - RATE_WINDOW
    stale = [ip for ip, times in _rate_store.items() if not times or times[-1] < cutoff]
    for ip in stale:
        del _rate_store[ip]
    _last_cleanup[0] = now


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load model at startup. Tries: local ONNX -> local PyTorch -> MLflow registry."""
    from src.inference.inference import load_pytorch, load_onnx

    if os.path.exists("model.onnx"):
        try:
            with MODEL_LOCK:
                MODEL_CACHE["onnx"] = load_onnx()
            logger.info("ONNX model loaded from local file")
        except Exception as e:
            logger.warning("Failed to load local ONNX: %s", e)
    elif os.path.exists("best_model.pth"):
        try:
            with MODEL_LOCK:
                MODEL_CACHE["pytorch"] = load_pytorch()
            logger.info("PyTorch model loaded from local file")
        except Exception as e:
            logger.warning("Failed to load local PyTorch: %s", e)

    if not MODEL_CACHE:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        model_name = os.environ.get("MLFLOW_MODEL_NAME", "water-segmentation-unet")
        model_version = os.environ.get("MLFLOW_MODEL_VERSION", "latest")
        if tracking_uri:
            try:
                import mlflow.pytorch
                mlflow.set_tracking_uri(tracking_uri)
                model_uri = f"models:/{model_name}/{model_version}"
                logger.info("Downloading model from MLflow: %s", model_uri)
                model = mlflow.pytorch.load_model(model_uri)
                with MODEL_LOCK:
                    MODEL_CACHE["pytorch"] = model
                logger.info("Model loaded from MLflow Model Registry")
            except Exception as e:
                logger.error("Failed to load from MLflow registry: %s", e)

    if not MODEL_CACHE:
        logger.warning("No model loaded at startup (degraded mode)")
    yield


app = FastAPI(title="Water Segmentation API", version="0.1.0", lifespan=lifespan)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter per IP. Returns 429 when exceeded."""
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/predict":
            client_ip = request.client.host if request.client else "unknown"
            now = time.time()
            with _rate_lock:
                if now - _last_cleanup[0] > RATE_CLEANUP_INTERVAL:
                    _evict_stale_ips()
                window = _rate_store[client_ip]
                cutoff = now - RATE_WINDOW
                while window and window[0] < cutoff:
                    window.pop(0)
                if len(window) >= RATE_LIMIT:
                    retry_after = math.ceil(window[0] + RATE_WINDOW - now)
                    logger.warning("Rate limit exceeded for %s", client_ip)
                    headers = {"Retry-After": str(retry_after)}
                    return Response(status_code=429, headers=headers,
                                    content=f'{{"detail":"Rate limit exceeded. Retry after {retry_after}s"}}')
                window.append(now)
        return await call_next(request)


app.add_middleware(RateLimitMiddleware)


@app.get("/health")
def health() -> Dict[str, str]:
    with MODEL_LOCK:
        ok = bool(MODEL_CACHE)
    return {"status": "ok" if ok else "degraded"}


@app.post("/predict", response_class=Response)
def predict(file: UploadFile = File(...), backend: str = "onnx", tta: bool = False) -> Response:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    if backend not in ("onnx", "pytorch"):
        raise HTTPException(status_code=400, detail=f"Unsupported backend '{backend}'; use 'onnx' or 'pytorch'")

    with MODEL_LOCK:
        if backend not in MODEL_CACHE and backend == "onnx" and "pytorch" in MODEL_CACHE:
            logger.warning("ONNX backend not loaded, falling back to PyTorch")
            backend = "pytorch"
        elif backend not in MODEL_CACHE:
            raise HTTPException(status_code=503, detail=f"Backend '{backend}' not loaded; try again after model deployment")

    content_length = file.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large ({content_length} bytes); max {MAX_UPLOAD_SIZE} bytes")

    try:
        contents = file.file.read(MAX_UPLOAD_SIZE + 1)
        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_SIZE} byte limit")
        suffix = os.path.splitext(file.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(contents)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {e}")
    try:
        t0 = time.time()
        mask = run_inference(tmp_path, backend=backend, tta=tta)
        logger.info(f"Prediction done in {time.time()-t0:.3f}s (tta={tta})")
        _, buf = cv2.imencode(".png", mask * 255)
        return Response(content=buf.tobytes(), media_type="image/png")
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
