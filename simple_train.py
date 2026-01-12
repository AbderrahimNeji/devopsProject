"""
SCRIPT SIMPLE - Entraînement YOLOv8 pour détection d'anomalies routières
Détecte: Potholes, Fissures longitudinales, Faïençage, Marquages effacés
Usage: python simple_train.py
"""

from ultralytics import YOLO

print("=" * 60)
print("  ENTRAÎNEMENT - Détection d'Anomalies Routières")
print("=" * 60)

# Charger le modèle YOLOv8 Nano (rapide et léger)
print("\n📦 Chargement du modèle YOLOv8n...")
model = YOLO('yolov8n.pt')

# Entraîner sur le dataset
print("\n🚀 Démarrage de l'entraînement...")
print("   Classes: Potholes, Fissures, Faïençage, Marquages")
print("   (Durée: 10-30 minutes)\n")

results = model.train(
    data='data/rdd2022_yolo/dataset.yaml',
    epochs=20,
    imgsz=640,
    batch=16,
    name='rdd2022_model',
    device='cpu',
    project='runs/detect',
    patience=5
)

print("\n" + "=" * 60)
print("  ✅ ENTRAÎNEMENT TERMINÉ !")
print("=" * 60)
print(f"\n📁 Modèle: runs/detect/simple_model/weights/best.pt")
print("\n🎯 Prochaine étape:")
print("   python simple_detect.py image.jpg")
