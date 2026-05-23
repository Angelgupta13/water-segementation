FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir "torch>=2.0.0" "torchvision>=0.15.0" --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY model.onnx ./model.onnx

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=3 --start-period=15s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
