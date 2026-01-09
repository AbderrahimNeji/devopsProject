"""
Extract frames from video files for annotation and training.
Supports various video formats and provides options for frame sampling.
"""

import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import json


class FrameExtractor:
    """Extract frames from video files."""
    
    def __init__(self, video_path, output_dir, fps=None, max_frames=None, skip_frames=1):
        """
        Initialize the frame extractor.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted frames
            fps: Target FPS for extraction (None = original FPS)
            max_frames: Maximum number of frames to extract
            skip_frames: Extract every Nth frame
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.max_frames = max_frames
        self.skip_frames = skip_frames
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract(self):
        """Extract frames from video."""
        print(f"📹 Processing video: {self.video_path.name}")
        
        # Open video
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video info:")
        print(f"  - Resolution: {width}x{height}")
        print(f"  - FPS: {video_fps:.2f}")
        print(f"  - Total frames: {total_frames}")
        
        # Calculate frame interval
        if self.fps:
            frame_interval = int(video_fps / self.fps)
        else:
            frame_interval = self.skip_frames
            
        # Extract frames
        frame_count = 0
        extracted_count = 0
        metadata = []
        
        pbar = tqdm(total=min(total_frames, self.max_frames or total_frames))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Check if we should extract this frame
            if frame_count % frame_interval == 0:
                # Generate filename
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                filename = f"frame_{extracted_count:06d}.jpg"
                filepath = self.output_dir / filename
                
                # Save frame
                cv2.imwrite(str(filepath), frame)
                
                # Store metadata
                metadata.append({
                    "filename": filename,
                    "frame_number": frame_count,
                    "timestamp_ms": timestamp_ms,
                    "timestamp_sec": timestamp_ms / 1000.0,
                    "width": width,
                    "height": height
                })
                
                extracted_count += 1
                
                # Check max frames limit
                if self.max_frames and extracted_count >= self.max_frames:
                    break
                    
            frame_count += 1
            pbar.update(1)
            
        pbar.close()
        cap.release()
        
        # Save metadata
        metadata_path = self.output_dir / "frames_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                "video_name": self.video_path.name,
                "video_fps": video_fps,
                "extraction_interval": frame_interval,
                "total_frames_extracted": extracted_count,
                "frames": metadata
            }, f, indent=2)
        
        print(f"✅ Extracted {extracted_count} frames to {self.output_dir}")
        print(f"📝 Metadata saved to {metadata_path}")
        
        return extracted_count


def main():
    parser = argparse.ArgumentParser(description='Extract frames from video for annotation')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, required=True, help='Output directory for frames')
    parser.add_argument('--fps', type=float, default=None, help='Target FPS for extraction')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum frames to extract')
    parser.add_argument('--skip', type=int, default=1, help='Extract every Nth frame')
    
    args = parser.parse_args()
    
    extractor = FrameExtractor(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps,
        max_frames=args.max_frames,
        skip_frames=args.skip
    )
    
    extractor.extract()


if __name__ == "__main__":
    main()
