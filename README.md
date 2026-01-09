# Road Degradation Detection System

**A Complete Computer Vision Solution for Autonomous Road Monitoring**

Computer Vision project for detecting and classifying road infrastructure anomalies using YOLOv8, GPS geolocation, and real-time web dashboard.

**Status:** ✅ Production Ready | **License:** MIT | **Python:** 3.8-3.11

---

## 🎯 Objectives

- **Detect anomalies** in road infrastructure automatically
- **Classify degradations** into 4 distinct categories (potholes, cracks, crazing, faded markings)
- **Geolocation** of each detection for mapping and targeted maintenance
- **Real-time visualization** via interactive dashboard
- **Scalable deployment** for production environments

## 🏗️ Architecture

```
road-degradation-detection/
├── data/                      # Données d'entraînement et test
│   ├── raw/                   # Vidéos/images brutes
│   ├── processed/             # Frames extraites
│   └── annotations/           # Labels YOLO format
├── models/                    # Modèles entraînés
├── src/
│   ├── data_preparation/      # Scripts extraction & annotation
│   ├── training/              # Scripts d'entraînement
│   ├── inference/             # Détection et géolocalisation
│   └── api/                   # FastAPI backend
├── dashboard/                 # Frontend web
├── configs/                   # Configurations YOLO
├── notebooks/                 # Jupyter notebooks pour expérimentation
└── docker/                    # Dockerfiles
```

## 🚀 Classes détectées

1. **Potholes** (nids-de-poule)
2. **Longitudinal cracks** (fissures longitudinales)
3. **Crazing** (faïençage)
4. **Faded markings** (marquages effacés)

## 📅 Planning (6 semaines)

- **Semaine 1**: Collection & annotation des données
- **Semaine 2**: Entraînement modèle baseline
- **Semaine 3**: Augmentation de données et équilibrage
- **Semaine 4**: Implémentation géolocalisation GPS
- **Semaine 5**: Pipeline de traitement batch + visualisation
- **Semaine 6**: Métriques, optimisation, démo finale

## 🛠️ Stack Technique

- **ML/CV**: Python, PyTorch, YOLOv8, OpenCV
- **Backend**: FastAPI, Uvicorn
- **Frontend**: HTML/JS, Leaflet/Mapbox
- **Data**: GeoJSON, pandas, numpy
- **Deployment**: Docker, Docker Compose

## 📦 Installation

```bash
# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer dépendances
pip install -r requirements.txt

# Installer Ultralytics YOLOv8
pip install ultralytics
```

## 🎓 Entraînement

```bash
# Extraire frames d'une vidéo
python src/data_preparation/extract_frames.py --video data/raw/road_video.mp4

# Entraîner le modèle
python src/training/train_yolo.py --config configs/yolov8_config.yaml

# Évaluer le modèle
python src/training/evaluate.py --model models/best.pt --data data/test
```

## 🔍 Inférence

```bash
# Traitement vidéo avec géolocalisation
python src/inference/detect_video.py \
  --video data/raw/test_video.mp4 \
  --gps data/raw/gps_log.csv \
  --output results/detections.geojson

# Traitement batch
python src/inference/batch_process.py --input data/raw/ --output results/
```

## 🌐 API & Dashboard

```bash
# Lancer l'API FastAPI
cd src/api
uvicorn main:app --reload --port 8000

# Lancer le dashboard
cd dashboard
python -m http.server 8080
```

Accéder à:

- API: http://localhost:8000
- Dashboard: http://localhost:8080
- API Docs: http://localhost:8000/docs

## 🐳 Docker & Deployment

### Quick Start with Docker Compose

The easiest way to run the entire system (API + Dashboard):

```bash
# Build and start all services
docker-compose up --build

# Services will be available at:
# - API: http://localhost:8000
# - Dashboard: http://localhost:80
# - API Docs: http://localhost:8000/docs
```

### Build Docker Image

Build a production-ready image with optimizations:

```bash
# Build the image
docker build -t road-degradation-api:latest .

# View image info
docker images | grep road-degradation-api
```

**Image Details:**

- **Base:** Python 3.10-slim
- **Size:** ~1.2 GB (multi-stage optimized)
- **Includes:** Pre-downloaded YOLOv8 model
- **Health checks:** Automated monitoring enabled

### Run Docker Container

#### Option 1: Basic Run

```bash
docker run -d \
  -p 8000:8000 \
  --name road-detection \
  road-degradation-api:latest
```

#### Option 2: With GPU Support

```bash
docker run -d \
  -p 8000:8000 \
  --gpus all \
  --memory 4g \
  --cpus 2 \
  --name road-detection \
  road-degradation-api:latest
```

#### Option 3: With Volume Mounts

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  --name road-detection \
  road-degradation-api:latest
```

### Test Docker Container

```bash
# Check if container is running
docker ps | grep road-detection

# View logs
docker logs -f road-detection

# Test health endpoint
curl http://localhost:8000/health

# Test API
curl -X GET http://localhost:8000/classes

# Stop container
docker stop road-detection

# Remove container
docker rm road-detection
```

### Docker Compose - Advanced Configuration

#### Scale Multiple API Instances

```bash
# Start with 3 API instances
docker-compose up --build -d
docker-compose up -d --scale api=3

# View all containers
docker-compose ps
```

#### Environment Variables

Create `.env` file:

```bash
# API Configuration
API_WORKERS=4
API_PORT=8000
CONFIDENCE_THRESHOLD=0.25

# Model Configuration
MODEL_PATH=/app/models/best.pt
IOU_THRESHOLD=0.45

# Logging
LOG_LEVEL=INFO
```

#### Custom Configuration

Edit `docker-compose.yml`:

```yaml
services:
  api:
    environment:
      - API_WORKERS=4
      - CONFIDENCE_THRESHOLD=0.3
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Production Deployment

#### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f kubernetes/

# View deployment status
kubectl get pods -n road-detection

# Scale replicas
kubectl scale deployment/api --replicas=5 -n road-detection

# Access via port-forward
kubectl port-forward -n road-detection svc/api-service 8000:80
```

#### AWS ECS Deployment

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push image
docker tag road-degradation-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/road-detection-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/road-detection-api:latest

# Deploy to ECS
aws ecs create-service \
  --cluster road-detection \
  --service-name api \
  --task-definition api:1 \
  --desired-count 2 \
  --launch-type FARGATE
```

### Troubleshooting Docker

**Container won't start:**

```bash
# Check logs
docker logs road-detection

# Check docker daemon
docker info
```

**GPU not detected:**

```bash
# Install NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

**Port already in use:**

```bash
# Find process using port 8000
lsof -i :8000

# Use different port
docker run -p 9000:8000 road-degradation-api:latest
```

### Monitoring Docker Deployment

```bash
# View resource usage
docker stats road-detection

# Monitor logs in real-time
docker logs -f --tail 100 road-detection

# Check health
docker inspect road-detection | grep -A 5 "Health"
```

For detailed deployment documentation, see [DEPLOYMENT.md](docs/DEPLOYMENT.md)

## 📊 Project Reports & Results

### Technical Report

See [REPORT.md](REPORT.md) for comprehensive documentation including:

- **Experimental Results:** Detailed performance metrics

  - Model accuracy: mAP@0.5 = 66.8%
  - Inference speed: 62.5 FPS (GPU)
  - Per-class metrics and analysis
  - GPS geolocation accuracy: 3.2m mean error

- **Analysis & Interpretation:**

  - Strengths and limitations
  - Performance analysis per class
  - API & dashboard evaluation
  - Cost-benefit analysis

- **Deployment Performance:**

  - Docker optimization details
  - Kubernetes specifications
  - Resource consumption estimates
  - AWS cost calculations

- **Recommendations:**
  - Short-term enhancements (1-2 months)
  - Medium-term improvements (3-6 months)
  - Long-term vision (6-12 months)

### Key Metrics

| Metric             | Value | Notes                       |
| ------------------ | ----- | --------------------------- |
| **mAP@0.5**        | 66.8% | Test set average precision  |
| **Inference FPS**  | 62.5  | GPU (RTX 3080)              |
| **Inference FPS**  | 125   | With TensorRT               |
| **GPS Accuracy**   | 3.2m  | Mean position error         |
| **API Latency**    | 45ms  | Image detection (p50)       |
| **Container Size** | 1.2GB | Optimized multi-stage build |

## 📁 Format des données

### Annotations YOLO

```
<class_id> <x_center> <y_center> <width> <height>
```

### GeoJSON Output

```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "geometry": {"type": "Point", "coordinates": [lon, lat]},
    "properties": {
      "class": "pothole",
      "confidence": 0.95,
      "timestamp": "2026-01-09T10:30:00",
      "image_path": "frame_0001.jpg"
    }
  }]
}
```

## 🤝 Contribution & Development

### Getting Started with Development

```bash
# Setup development environment
make dev-setup

# Run tests
make test

# Run linting and formatting
make lint
make format

# Full CI check
make ci
```

### Project Structure

```
road-degradation-detection/
├── src/                          # Main source code
│   ├── data_preparation/         # Data extraction & preparation
│   ├── training/                 # Model training pipeline
│   ├── inference/                # Detection & inference
│   └── api/                      # FastAPI backend
├── dashboard/                    # Web UI (HTML/CSS/JS)
├── tests/                        # Unit tests (pytest)
├── scripts/                      # Utility scripts
│   ├── benchmark_models.py       # Performance benchmarking
│   ├── export_model.py           # ONNX/TensorRT export
│   ├── generate_visualizations.py # Result visualization
│   ├── monitor.py                # System monitoring
│   ├── setup.py                  # Automated setup
│   └── download_examples.py      # Sample data download
├── docs/                         # Comprehensive documentation
│   ├── week1_guide.md-week6_guide.md  # 6-week implementation guides
│   ├── API.md                    # REST API documentation
│   ├── DEPLOYMENT.md             # Deployment guide
│   └── QUICKSTART.md             # Quick start tutorial
├── configs/                      # YOLOv8 configuration
├── examples/                     # Example data & notebooks
├── Dockerfile                    # Production Docker image
├── docker-compose.yml            # Multi-service composition
├── Makefile                      # 40+ automation commands
├── REPORT.md                     # Technical report with results
├── CONTRIBUTING.md               # Contributing guidelines
├── LICENSE                       # MIT License
└── README.md                     # This file
```

### Development Workflow

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

- Code of conduct
- Development setup
- Coding standards
- Testing requirements
- Pull request process
- Issue guidelines

### Documentation

**Available Resources:**

- **Quick Start:** [QUICKSTART.md](docs/QUICKSTART.md)
- **API Reference:** [API.md](docs/API.md)
- **Deployment Guide:** [DEPLOYMENT.md](docs/DEPLOYMENT.md)
- **Week-by-Week Guides:** `docs/week{1-6}_guide.md`
- **Technical Report:** [REPORT.md](REPORT.md)
- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🔗 Repository Information

- **GitHub:** [Your GitHub URL]
- **Issues & Feedback:** [GitHub Issues]
- **Documentation:** https://your-docs-url.com
- **Demo:** https://your-demo-url.com

## 📧 Support

For questions, issues, or contributions:

1. **Check Documentation:** Start with [QUICKSTART.md](docs/QUICKSTART.md)
2. **Review Issues:** Check [GitHub Issues](../../issues)
3. **Read Guides:** See `/docs/` for detailed guides
4. **Technical Report:** See [REPORT.md](REPORT.md) for results & analysis

## 🎓 Educational Value

This project demonstrates:

- **Machine Learning:** YOLOv8, transfer learning, model evaluation
- **Computer Vision:** Object detection, image processing, visualization
- **Backend Development:** FastAPI, async processing, REST APIs
- **Frontend Development:** HTML/CSS/JS, geospatial visualization
- **DevOps:** Docker, Kubernetes, CI/CD pipelines
- **Software Engineering:** Testing, documentation, best practices

Suitable for:

- University capstone projects
- Portfolio demonstrations
- Production deployments
- Research & development

---

**Last Updated:** January 9, 2026 | **Version:** 1.0.0 | **Status:** ✅ Production Ready
