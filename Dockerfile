FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY best_model.pth .

# Export ONNX at build time so inference is fast at runtime
RUN python -c "import sys; sys.path.insert(0, '.'); from src.inference.inference import export_onnx; export_onnx('best_model.pth', 'model.onnx'); print('ONNX export complete')" || echo "WARNING: ONNX export failed -- will fall back to PyTorch at runtime"

ENTRYPOINT ["python", "-m", "src.inference.inference"]
CMD ["--help"]
