# =============================================================================
# Multi-source YOLO detection + RTSP re-broadcast service
#
# Base image (auto-select by host architecture):
#   - Apple Silicon / arm64:  ultralytics/ultralytics:latest-arm64  (CPU)
#   - NVIDIA server / amd64:  build with --build-arg BASE_IMAGE=ultralytics/ultralytics:latest
# =============================================================================
ARG BASE_IMAGE=ultralytics/ultralytics:latest-arm64
FROM ${BASE_IMAGE}

# ffmpeg for RTSP/FLV/HLS/MSE decode + H264 encode, plus OpenCV system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY serve_vehicles.py .

# Pre-bundled custom models (non-ultralytics: helmet / fall detection)
# Official yolo11* models auto-download from ultralytics on first run.
COPY helmet.pt fall_detect.pt ./

# Directories for runtime state & uploaded models
RUN mkdir -p /app/data /app/models

ENV CONFIG_FILE=/app/data/sources_config.json \
    MODELS_DIR=/app/models \
    RTSP_HOST=mediamtx:8554 \
    HTTP_PORT=8000 \
    IMGSZ=480

EXPOSE 8000

CMD ["python", "serve_vehicles.py"]
