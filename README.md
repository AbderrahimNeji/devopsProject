# PROJECT 4: ROAD DEGRADATION DETECTION

## Overview

This project implements an automated road degradation detection system using YOLOv8 for object detection. The system was developed to address the critical need for efficient road infrastructure monitoring, detecting and classifying four types of road anomalies with associated GPS coordinates, and providing an interactive map dashboard for visualization and analysis.

### Technical Highlights

- **Deep Learning Framework**: YOLOv8 (Ultralytics) with PyTorch backend
- **Dataset**: RDD2022 Czech Republic (8,500+ professional annotations)
- **Model Architecture**: YOLOv8 Medium (25M parameters) optimized for road damage detection
- **Deployment**: Fully containerized with Docker, production-ready
- **Performance**: 0.434 mAP@0.5 on validation set, 154.9ms inference time (CPU)

## Features

✅ **4-Class Object Detection**: Longitudinal Cracks, Transverse Cracks, Alligator Cracks, Potholes  
✅ **GPS Integration**: Frame-level geolocation with timestamp synchronization  
✅ **Interactive Map Dashboard**: Leaflet.js-based visualization with OpenStreetMap tiles  
✅ **Comprehensive Metrics**: mAP@IoU thresholds, precision/recall, geolocation error, FPS throughput  
✅ **Docker Support**: One-command deployment with pre-configured environment  
✅ **Professional Dataset**: RDD2022 standard with consistent annotation quality

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Prepare Dataset (RDD2022 Czech)

The system uses the RDD2022 (Road Damage Dataset 2022) Czech Republic subset, which provides professional-grade annotations for road infrastructure monitoring.

```bash
cd road-degradation-detection
python convert_rdd2022_to_yolo.py
```

**Technical Details**:

- **Download Size**: 245 MB (compressed)
- **Processing Time**: 1-3 minutes depending on CPU
- **Output Format**: YOLO v8 format (normalized coordinates)
- **Dataset Split**: 80/20 train/val ratio (2829/214 images)
- **Total Annotations**: ~8,500+ bounding boxes
- **Classes**:
  docker-compose down

````
road-degradation-detection/
├── simple_detect_gps.py      # Video → GPS → GeoJSON pipeline
├── demo_project4.py          # Project verification
├── docker-compose.yml        # Docker Compose config
│   └── rdd2022_yolo/         # RDD2022 Czech dataset (4 classes)
│   └── detect/

  # Project: Road Degradation Detection

  ## Overview

  This project detects four kinds of road damage with YOLOv8 and shows results on an interactive map. It uses the RDD2022 Czech dataset and includes Docker for easy runs. The focus is fast detection, simple setup, and clear outputs.

  ## Quick Start

  Install dependencies:

  ```bash
  pip install -r requirements.txt
````

Download and prepare the dataset (Czech split of RDD2022). The script fetches, unzips, converts to YOLO labels, and creates train/val/test folders.

```bash
cd road-degradation-detection
python convert_rdd2022_to_yolo.py
```

Do a short CPU smoke test (slow, optional):

```bash
python simple_train.py
```

Use the provided production model `models/rdd2022_best.pt` for real work. It was trained on Colab GPU with yolov8m at 960px.

Run detections on images or folders:

```bash
python simple_detect.py data/rdd2022_yolo/val/images
python simple_detect.py data/potholes
```

Process a video with GPS and export GeoJSON:

```bash
python simple_detect_gps.py data/videos/road_video.mp4
```

Check metrics or FPS:

```bash
python evaluate_model.py
python evaluate_model.py data/videos/test.mp4
```

Open `map_dashboard.html` in your browser and load the GeoJSON file to view detections on the map.

## Docker

Build and start the service (runs `demo_project4.py` by default):

```bash

---
```

Run specific commands inside the container when needed:

```bash


## Dataset

Source: RDD2022 Czech. Classes are longitudinal crack (D00), transverse crack (D10), alligator crack (D20), and pothole (D40). After conversion the data lives in `data/rdd2022_yolo/` with `train/`, `val/`, `test/`, and `dataset.yaml`. There are about 8.5k boxes over 3752 images (train 2829, val 214, test 709).

## Model Notes

Production model: yolov8m trained for 80 epochs at 960px on a Colab T4. Validation mAP@0.5 was 0.434 and CPU inference is about 155 ms per image. A smaller yolov8n model is available for quick local checks.

## Project Layout

simple_train.py for training, simple_detect.py for image and folder inference, simple_detect_gps.py for video with GPS and GeoJSON export, evaluate_model.py for metrics and FPS, map_dashboard.html for the Leaflet map, Dockerfile and docker-compose.yml for containers, models/ for weights, data/ for the converted dataset, and runs/ plus resultats/ for outputs.

## Troubleshooting

If the model file is missing, confirm `models/rdd2022_best.pt` is present. Keep `numpy<2` as pinned to avoid import errors. Change the exposed port in `docker-compose.yml` if 3000 is busy. For real training speed, use GPU (Colab) instead of local CPU.

**Problem**: Docker Compose failed to start with multiple port binding errors:

```
Error response from daemon: ports are not available: exposing port TCP 0.0.0.0:9090
bind: Only one usage of each socket address is normally permitted
```

**Root Cause**: Port 9090 was already occupied by a background HTTP server from earlier testing:

```powershell
PS> netstat -ano | findstr :9090
TCP    0.0.0.0:9090    LISTENING    5234
```

**Attempted Solutions**:

1. **Port 8080**: Also occupied (PID 6012, likely IIS or another service)
2. **Kill process**: `taskkill /PID 6012 /F` → Access denied (system process)

**Final Solution**: Changed `docker-compose.yml` to use port 3000:

```yaml
# docker-compose.yml
ports:
  - "3000:9090" # External:Internal mapping
```

**Validation**: `docker-compose up -d` started successfully, dashboard accessible at `http://localhost:3000/map_dashboard.html`.

**Impact**: No functional impact, documentation updated to reflect port 3000.  
**Lesson learned**: Always check for port conflicts before deployment, especially on Windows where system services occupy common ports.

---

### 4. Dataset Migration: Small Custom Dataset → RDD2022

**Problem**: Initial dataset was insufficient for production-quality model:

- Only 493 images with 226 bounding boxes
- Mixed sources with inconsistent annotation quality
- Only 45 images per class average
- Poor generalization (71.8% mAP on validation but failed on real-world test images)

**Root Cause**: Insufficient training data leads to overfitting and poor generalization to unseen road conditions.

**Solution**: Migrated to RDD2022 Czech Republic dataset:

```python
# convert_rdd2022_to_yolo.py
# Downloads and converts 3752 images (2829 train, 214 val, 709 test)
# Total annotations: ~8500+ boxes across 4 classes
```

**Results After Migration**:

- Training images: 493 → 2829 (+476% increase)
- Annotations: 226 → ~8500+ (+3661% increase)
- Validation mAP@0.5: Baseline established at 0.434 (realistic, professional-grade metric)
- Real-world testing: Model generalizes to unseen Czech roads

**Impact**: Training time increased (38 minutes on GPU), but model quality improved significantly.  
**Lesson learned**: Dataset quality > model architecture. Always use professional-grade datasets for production systems.

---

### 5. CPU Training Performance: Impractical for Development

**Problem**: Initial attempt to train on local CPU:

```
Epoch 1/50: 2847s per epoch
Estimated total time: 39.5 hours
```

**Root Cause**: YOLOv8 Medium (25M parameters) with 2829 images requires GPU acceleration. CPU training is 15-20x slower than GPU.

**Investigation**:

- Monitored CPU usage: 100% on all cores (16 threads), still slow
- Memory: 7.2 GB RAM (acceptable)
- Calculation: (2847s × 80 epochs) / 3600 = ~63 hours total

**Solution**: Switched to Google Colab GPU (T4):

```bash
# Google Colab GPU training
# Completed in 38 minutes for 80 epochs
# GPU utilization: 85-95%
# VRAM usage: 12.4 / 15 GB
```

**Impact**: Development velocity increased dramatically (63 hours → 38 minutes = 99.5x speedup).  
**Lesson learned**: Always use GPU for deep learning training. CPU training is only viable for tiny datasets (<500 images) or small models.

---

## Verification

Run the demo script to verify all components:

```bash
python demo_project4.py
```

This checks:

- ✅ All required files present
- ✅ Dataset configuration
- ✅ Model availability
- ✅ Dependencies installed

---