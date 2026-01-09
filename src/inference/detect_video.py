"""
Detect road degradations in video with GPS geolocation.
"""

import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd
from .gps_utils import GPSProcessor


class VideoDetector:
    """Detect road degradations in video with geolocation."""
    
    def __init__(self, model_path, gps_file=None, gps_format='csv', conf_threshold=0.25):
        """
        Initialize video detector.
        
        Args:
            model_path: Path to trained YOLO model
            gps_file: Path to GPS data file
            gps_format: GPS file format ('csv', 'gpx', 'json')
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        
        # Load model
        print(f"📥 Loading model: {model_path}")
        self.model = YOLO(str(model_path))
        
        # Initialize GPS processor
        self.gps_processor = None
        if gps_file:
            self.gps_processor = GPSProcessor(gps_file, gps_format)
        
        self.class_names = ['pothole', 'longitudinal_crack', 'crazing', 'faded_marking']
    
    def process_video(self, video_path, output_path=None, save_video=False, 
                     video_start_time=None, skip_frames=1):
        """
        Process video and detect degradations.
        
        Args:
            video_path: Path to input video
            output_path: Path for output GeoJSON
            save_video: Whether to save annotated video
            video_start_time: Video start datetime
            skip_frames: Process every Nth frame
        
        Returns:
            List of detections with geolocation
        """
        video_path = Path(video_path)
        print(f"🎬 Processing video: {video_path.name}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video info: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
        
        # Setup video writer if needed
        video_writer = None
        if save_video:
            output_video_path = video_path.parent / f"{video_path.stem}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        # Prepare GPS synchronization
        gps_data = None
        if self.gps_processor:
            gps_data = self.gps_processor.gps_data
            if video_start_time is None:
                video_start_time = gps_data['timestamp'].iloc[0]
        
        # Process frames
        detections = []
        frame_count = 0
        processed_count = 0
        
        pbar = tqdm(total=total_frames)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_count % skip_frames == 0:
                # Get timestamp
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_sec = timestamp_ms / 1000.0
                
                # Calculate absolute timestamp
                if video_start_time:
                    frame_timestamp = video_start_time + timedelta(seconds=timestamp_sec)
                else:
                    frame_timestamp = datetime.now()
                
                # Get GPS coordinates
                gps_coords = None
                if self.gps_processor:
                    gps_coords = self.gps_processor.interpolate_gps(frame_timestamp, gps_data)
                
                # Run detection
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    verbose=False
                )
                
                # Process detections
                result = results[0]
                boxes = result.boxes
                
                for box in boxes:
                    # Get box data
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    detection = {
                        'frame_number': frame_count,
                        'timestamp': frame_timestamp.isoformat(),
                        'timestamp_sec': timestamp_sec,
                        'class_id': cls,
                        'class_name': self.class_names[cls],
                        'confidence': conf,
                        'bbox': {
                            'xmin': float(xyxy[0]),
                            'ymin': float(xyxy[1]),
                            'xmax': float(xyxy[2]),
                            'ymax': float(xyxy[3])
                        }
                    }
                    
                    # Add GPS coordinates
                    if gps_coords:
                        detection['latitude'] = gps_coords['latitude']
                        detection['longitude'] = gps_coords['longitude']
                        detection['altitude'] = gps_coords.get('altitude', 0.0)
                    
                    detections.append(detection)
                
                # Draw boxes on frame for video
                if save_video:
                    annotated_frame = result.plot()
                    video_writer.write(annotated_frame)
                
                processed_count += 1
            elif save_video:
                # Write original frame if not processed
                video_writer.write(frame)
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        if video_writer:
            video_writer.release()
            print(f"✅ Annotated video saved to {output_video_path}")
        
        print(f"✅ Processed {processed_count} frames, found {len(detections)} detections")
        
        # Save detections
        if output_path:
            self.save_detections(detections, output_path)
        
        return detections
    
    def save_detections(self, detections, output_path, format='geojson'):
        """
        Save detections to file.
        
        Args:
            detections: List of detection dicts
            output_path: Output file path
            format: Output format ('geojson' or 'json')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'geojson':
            self._save_geojson(detections, output_path)
        else:
            self._save_json(detections, output_path)
        
        print(f"✅ Detections saved to {output_path}")
    
    def _save_geojson(self, detections, output_path):
        """Save detections as GeoJSON."""
        features = []
        
        for det in detections:
            # Only include detections with GPS coordinates
            if 'latitude' not in det or 'longitude' not in det:
                continue
            
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [det['longitude'], det['latitude']]
                },
                'properties': {
                    'class': det['class_name'],
                    'class_id': det['class_id'],
                    'confidence': det['confidence'],
                    'timestamp': det['timestamp'],
                    'frame_number': det['frame_number'],
                    'altitude': det.get('altitude', 0.0)
                }
            }
            
            features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features,
            'metadata': {
                'total_detections': len(features),
                'generated_at': datetime.now().isoformat(),
                'classes': self.class_names
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
    
    def _save_json(self, detections, output_path):
        """Save detections as JSON."""
        output = {
            'detections': detections,
            'metadata': {
                'total_detections': len(detections),
                'generated_at': datetime.now().isoformat(),
                'classes': self.class_names
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Detect road degradations in video')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--gps', type=str, default=None, help='Path to GPS data file')
    parser.add_argument('--gps-format', type=str, choices=['csv', 'gpx', 'json'], 
                        default='csv', help='GPS file format')
    parser.add_argument('--output', type=str, default='results/detections.geojson',
                        help='Output file path')
    parser.add_argument('--save-video', action='store_true', help='Save annotated video')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--skip-frames', type=int, default=1, help='Process every Nth frame')
    parser.add_argument('--start-time', type=str, default=None,
                        help='Video start time (ISO format: 2026-01-09T10:30:00)')
    
    args = parser.parse_args()
    
    # Parse start time
    start_time = None
    if args.start_time:
        start_time = datetime.fromisoformat(args.start_time)
    
    detector = VideoDetector(
        model_path=args.model,
        gps_file=args.gps,
        gps_format=args.gps_format,
        conf_threshold=args.conf
    )
    
    detector.process_video(
        video_path=args.video,
        output_path=args.output,
        save_video=args.save_video,
        video_start_time=start_time,
        skip_frames=args.skip_frames
    )


if __name__ == "__main__":
    main()
