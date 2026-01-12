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
source = sys.argv[1] if len(sys.argv) > 1 else 'data/rdd2022_yolo/val/images'
model_path = 'models/rdd2022_best.pt'

# Vérifier le modèle
if not os.path.exists(model_path):
    print(f"\n❌ Modèle non trouvé: {model_path}")
    print("   Entraînez d'abord: python simple_train.py")
    sys.exit(1)

# Vérifier la source
if not os.path.exists(source):
    print(f"\n❌ Source introuvable: {source}")
    print("   Utilise un chemin valide (image, vidéo ou dossier)")
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
    name='detection',
    verbose=False
)

# Afficher les détections avec confidences
print("\n" + "=" * 60)
print("  RÉSULTATS DÉTECTION")
print("=" * 60)

total_detections = 0
for i, r in enumerate(results):
    boxes = r.boxes
    n_det = len(boxes)
    total_detections += n_det
    
    if n_det > 0:
        img_name = r.path.split('\\')[-1] if '\\' in r.path else r.path.split('/')[-1]
        print(f"\n📷 {img_name}: {n_det} détection(s)")
        
        for j, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            print(f"   [{j+1}] {cls_name}: {conf*100:.1f}%")

print("\n" + "=" * 60)
print(f"  ✅ TOTAL: {total_detections} détection(s)")
print("=" * 60)
print(f"\n📁 Résultats sauvegardés: resultats/detection/")
print(f"\n💡 Classes: {', '.join(model.names.values())}")
