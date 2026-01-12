"""
SCRIPT SIMPLE - Détection d'anomalies sur images/vidéos
Usage: python simple_detect.py image.jpg
       python simple_detect.py video.mp4
"""

from ultralytics import YOLO
import sys
import os

print("=" * 60)
print("  DÉTECTION - Anomalies Routières")
print("=" * 60)

# Paramètres
source = sys.argv[1] if len(sys.argv) > 1 else 'data/potholes'
model_path = 'models/rdd2022_best.pt'

# Vérifier le modèle
if not os.path.exists(model_path):
    print(f"\n❌ Modèle non trouvé: {model_path}")
    print("   Entraînez d'abord: python simple_train.py")
    sys.exit(1)

# Charger et détecter
print(f"\n📦 Modèle: {model_path}")
print(f"🔍 Source: {source}\n")

model = YOLO(model_path)
results = model.predict(
    source=source,
    save=True,
    conf=0.25,
    project='resultats',
    name='detection'
)

print("\n" + "=" * 60)
print("  ✅ DÉTECTION TERMINÉE !")
print("=" * 60)
print(f"\n📁 Résultats: resultats/detection/")
print(f"\n💡 Classes détectées: Potholes, Fissures, Faïençage, Marquages")
