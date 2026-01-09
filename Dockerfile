# Multi-stage Dockerfile for Road Degradation Detection API

# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /tmp

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models/weights \
    /app/data/processed \
    /app/results/detections \
    /app/results/visualizations \
    /app/logs

# Download pretrained YOLOv8n model
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
