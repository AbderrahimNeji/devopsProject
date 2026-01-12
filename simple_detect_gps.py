"""
SCRIPT SIMPLE - Détection avec GPS et génération GeoJSON
Traite une vidéo, détecte les anomalies, et génère un fichier GeoJSON géolocalisé
Usage: python simple_detect_gps.py video.mp4
"""

from ultralytics import YOLO
import cv2
import json
import sys
import os
from datetime import datetime
from pathlib import Path

print("=" * 70)
print("  PIPELINE VIDÉO → GPS → GeoJSON")
print("=" * 70)

def generate_sample_gps(num_points=100):
    """Générer des coordonnées GPS simulées pour démo"""
    base_lat, base_lon = 48.8566, 2.3522  # Paris
    gps_data = []
    
    for i in range(num_points):
        lat = base_lat + (i * 0.0001)
        lon = base_lon + (i * 0.00015)
        gps_data.append({
            'frame': i,
            'lat': lat,
            'lon': lon,
            'timestamp': datetime.now().isoformat()
        })
    
    return gps_data

def process_video(video_path, model_path):
    """Pipeline: Frames → Détection → GPS → GeoJSON"""
    
    # Charger le modèle
    if not os.path.exists(model_path):
        print(f"\n❌ Modèle non trouvé: {model_path}")
        print("   Entraînez d'abord: python simple_train.py")
        sys.exit(1)
    
    print(f"\n📦 Modèle: {model_path}")
    model = YOLO(model_path)
    
    # Ouvrir la vidéo
    if not os.path.exists(video_path):
        print(f"\n❌ Vidéo non trouvée: {video_path}")
        sys.exit(1)
    
    print(f"🎥 Vidéo: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📊 Frames: {total_frames}, FPS: {fps}")
    print("📍 Génération GPS simulée...")
    
    gps_data = generate_sample_gps(total_frames)
    
    # GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    frame_count = 0
    detections_count = 0
    
    print("\n🔍 Détection en cours...\n")
    
    # Traiter toutes les 30 frames pour accélérer
    stride = 30
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % stride == 0:
            results = model.predict(frame, conf=0.25, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    gps_point = next((g for g in gps_data if g['frame'] == frame_count), None)
                    
                    if gps_point:
                        feature = {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [gps_point['lon'], gps_point['lat']]
                            },
                            "properties": {
                                "frame": frame_count,
                                "timestamp": gps_point.get('timestamp', ''),
                                "class": model.names[int(box.cls[0])],
                                "confidence": float(box.conf[0]),
                                "bbox": box.xyxy[0].tolist()
                            }
                        }
                        geojson["features"].append(feature)
                        detections_count += 1
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"   {frame_count}/{total_frames} frames - {detections_count} détections")
    
    cap.release()
    
    print(f"\n✅ Traitement terminé!")
    print(f"   Total détections: {detections_count}")
    
    return geojson

# Programme principal
video_path = sys.argv[1] if len(sys.argv) > 1 else 'data/videos/test.mp4'
model_path = 'models/rdd2022_best.pt'

geojson = process_video(video_path, model_path)

if geojson:
    # Sauvegarder GeoJSON
    output_dir = Path("resultats/geojson")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"detections_{Path(video_path).stem}.geojson"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"\n📁 GeoJSON: {output_file}")
    
    # Sauvegarder résumé
    summary = {
        "video": video_path,
        "total_detections": len(geojson["features"]),
        "model": model_path,
        "timestamp": datetime.now().isoformat()
    }
    
    summary_file = output_dir / f"summary_{Path(video_path).stem}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Résumé: {summary_file}")
    print(f"\n🗺️ Visualisez: map_dashboard.html")
