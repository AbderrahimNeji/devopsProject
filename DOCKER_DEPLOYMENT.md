# Docker Deployment Guide - Road Degradation Detection

## Quick Start

### 1. Build the Docker Image

```bash
cd road-degradation-detection
docker-compose build
```

**Build time**: 5-10 minutes (downloads PyTorch, Ultralytics, models)  
**Image size**: ~2-3GB (Python 3.10 + PyTorch CPU + YOLOv8)

### 2. Verify Project Setup

```bash
docker-compose run degradation-detection python demo_project4.py
```

**Output**: Project compliance checklist (all objectives ✅)

### 3. Prepare Dataset (RDD2022 Czech)

```bash
docker-compose run degradation-detection python convert_rdd2022_to_yolo.py
```

**What it does**:

- Downloads RDD2022 Czech dataset (~245MB)
- Converts from PascalVOC to YOLO format
- Creates train/val/test splits
- Generates `data/rdd2022_yolo/dataset.yaml`

**Duration**: 5-10 minutes  
**Output**:

- Train: 2829 images (~8000 boxes)
- Val: 214 images (~500 boxes)
- Test: 709 images (no annotations)

---

## Usage Commands

### Detect Road Anomalies

**On validation images**:

```bash
docker-compose run degradation-detection python simple_detect.py data/rdd2022_yolo/val/images
```

**On custom images**:

```bash
# Mount your images directory
docker-compose run -v /path/to/images:/app/custom_images degradation-detection \
    python simple_detect.py custom_images
```

**Output**: `runs/detect/*/` (annotated images + labels with confidence)

### GPS Integration Pipeline

```bash
docker-compose run degradation-detection python simple_detect_gps.py data/videos/video.mp4
```

**Output**: `resultats/geojson/detections_*.geojson` (georeferenced detections)

### Evaluate Model

```bash
docker-compose run degradation-detection python evaluate_model.py
```

**Outputs**:

- `resultats/evaluation/metrics.json` (mAP, precision, recall)
- Console: FPS, geolocation error, throughput stats

### View Dashboard

**Start the web server**:

```bash
docker-compose up
```

**Access**: http://localhost:9090/map_dashboard.html

**Load GeoJSON**:

1. Click "Load GeoJSON"
2. Select `resultats/geojson/detections_*.geojson`
3. View anomalies on interactive map

---

## Advanced Usage

### Interactive Container

For debugging or running custom Python scripts:

```bash
docker-compose run -it degradation-detection bash
```

Then inside the container:

```bash
python simple_detect.py ...
python simple_train.py ...
python evaluate_model.py ...
```

### Train Custom Model

```bash
docker-compose run degradation-detection python simple_train.py
```

**Note**: CPU training is slow (12+ hours for 50 epochs). For GPU training, use Google Colab:

```bash
# In Colab notebook:
!git clone https://github.com/AbderrahimNeji/devopsProject.git
%cd devopsProject
!pip install -r requirements.txt
!python convert_rdd2022_to_yolo.py
!yolo train model=models/yolov8m.pt data=data/rdd2022_yolo/dataset.yaml epochs=80 imgsz=960 batch=16 cos_lr=True close_mosaic=10 device=0 name=rdd2022_yolov8m_80e
```

### Copy Results to Host

Results are automatically mounted to `./resultats/`. To access them:

```bash
ls -la resultats/detection/
ls -la resultats/geojson/
ls -la resultats/evaluation/
```

---

## Docker Compose Services

**Service**: `degradation-detection`

- **Image**: `degradation-detection:latest`
- **Container**: `road-degradation-detector`
- **Ports**:
  - `9090:9090` (dashboard/HTTP server)
- **Volumes**:
  - `./resultats:/app/resultats` (detection results, GeoJSON)
  - `./runs:/app/runs` (model training/runs)
  - `./data:/app/data` (dataset, videos)
- **Environment**: `PYTHONUNBUFFERED=1` (real-time logs)

---

## Troubleshooting

### Issue: "No space left on device"

**Solution**: Clean up Docker images and volumes

```bash
docker system prune -a
docker volume prune
```

### Issue: PyTorch download timeout

**Solution**: The Dockerfile pre-installs PyTorch CPU. If it fails, rebuild:

```bash
docker-compose build --no-cache
```

### Issue: Model file too large for container

**Solution**: The Dockerfile excludes large files (`.dockerignore`). Models are downloaded at runtime. Trained models are mounted via volumes.

### Issue: GPU support needed

**Solution**: Use Google Colab GPU instead (see "Train Custom Model" above).

---

## Project Structure Inside Container

```
/app/
├── simple_train.py              # Training script
├── simple_detect.py             # Detection on images/videos
├── simple_detect_gps.py         # Video → GPS → GeoJSON
├── evaluate_model.py            # Metrics evaluation
├── demo_project4.py             # Project verification
├── convert_rdd2022_to_yolo.py   # Dataset conversion
├── map_dashboard.html           # Web dashboard
│
├── data/
│   └── rdd2022_yolo/           # RDD2022 Czech (downloaded on demand)
│       ├── dataset.yaml
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
│   ├── yolov8n.pt              # Base model (3M params)
│   └── yolov8m.pt              # Medium model (25M params)
│
├── resultats/                   # Output directory (mounted)
│   ├── detection/               # Detection results
│   ├── geojson/                 # GeoJSON files (GPS)
│   └── evaluation/              # Metrics JSON
│
└── runs/                        # Training runs (mounted)
    └── detect/
        └── rdd2022_best_pred_*/  # Detection outputs
```

---

## Performance Benchmarks

| Metric           | CPU (Container) | GPU (Colab) | Notes                 |
| ---------------- | --------------- | ----------- | --------------------- |
| Detection FPS    | 8-10 FPS        | 50-65 FPS   | 960x960 input         |
| Dataset prep     | 1-3 min         | 1-3 min     | Download + convert    |
| Model eval       | 5-10 min        | <1 min      | Full validation split |
| Training (50 ep) | 12+ hours       | 30-45 min   | YOLOv8m on RDD2022    |

---

## Environment Variables

Inside the container, the following are automatically set:

```bash
PYTHONUNBUFFERED=1      # Real-time output
PIP_DISABLE_PIP_VERSION_CHECK=1
PIP_DEFAULT_TIMEOUT=120 # For slow networks
PIP_PROGRESS_BAR=off
```

---

## Support & Documentation

- **README.md**: Project overview, features, quick start
- **REPORT.md**: Technical report, metrics, analysis
- **docker-compose.yml**: Container orchestration
- **Dockerfile**: Image definition
- **.dockerignore**: Excluded files (keeps image lean)
- **requirements.txt**: Python dependencies
