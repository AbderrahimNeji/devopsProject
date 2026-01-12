# PROJECT 4: DEGRADATION DETECTION - Dockerfile
FROM python:3.10-slim

# Installer les dépendances système pour OpenCV (Debian trixie slim)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les dépendances et installer
COPY requirements.txt .

# Réduire les erreurs de timeout réseau et forcer les roues CPU de PyTorch (plus légères)
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_PROGRESS_BAR=off

# Installer PyTorch CPU avant ultralytics pour éviter le téléchargement des roues CUDA (~900MB)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

# Installer le reste des dépendances (ultralytics, opencv, etc.)
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copier le projet
COPY . .

# Télécharger les modèles YOLOv8 de base si absents
RUN mkdir -p models && \
    if [ ! -f yolov8n.pt ]; then \
    wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt -O yolov8n.pt; \
    fi && \
    if [ ! -f models/yolov8m.pt ]; then \
    wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt -O models/yolov8m.pt; \
    fi

# Créer les dossiers de résultats et data
RUN mkdir -p resultats/detection resultats/geojson resultats/evaluation \
    data/rdd2022_yolo temp_download

# Exposer le port pour le dashboard (serveur HTTP)
EXPOSE 9090

# Commande par défaut : afficher les instructions
CMD ["python", "demo_project4.py"]
