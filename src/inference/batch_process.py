"""
Batch processing script for multiple videos.
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import json
from src.inference.detect_video import VideoDetector
import concurrent.futures


class BatchProcessor:
    """Process multiple videos in batch."""
    
    def __init__(self, model_path, conf_threshold=0.25, max_workers=2):
        """
        Initialize batch processor.
        
        Args:
            model_path: Path to trained model
            conf_threshold: Confidence threshold
            max_workers: Number of parallel workers
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.max_workers = max_workers
    
    def process_directory(self, input_dir, output_dir, gps_dir=None, save_videos=False):
        """
        Process all videos in a directory.
        
        Args:
            input_dir: Directory with input videos
            output_dir: Directory for outputs
            gps_dir: Directory with GPS files (optional)
            save_videos: Whether to save annotated videos
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_dir.glob(f'*{ext}'))
        
        print(f"Found {len(video_files)} videos to process")
        
        # Process videos
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for video_path in video_files:
                # Find corresponding GPS file if available
                gps_file = None
                if gps_dir:
                    gps_path = Path(gps_dir) / f"{video_path.stem}.csv"
                    if gps_path.exists():
                        gps_file = str(gps_path)
                
                output_path = output_dir / f"{video_path.stem}_detections.geojson"
                
                # Submit task
                future = executor.submit(
                    self._process_single_video,
                    video_path,
                    output_path,
                    gps_file,
                    save_videos
                )
                futures[future] = video_path.name
            
            # Wait for completion
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                video_name = futures[future]
                try:
                    result = future.result()
                    results.append({
                        'video': video_name,
                        'status': 'success',
                        'detections': result
                    })
                except Exception as e:
                    results.append({
                        'video': video_name,
                        'status': 'failed',
                        'error': str(e)
                    })
                    print(f"Error processing {video_name}: {e}")
        
        # Save summary
        summary_path = output_dir / 'batch_summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'total_videos': len(video_files),
                'successful': sum(1 for r in results if r['status'] == 'success'),
                'failed': sum(1 for r in results if r['status'] == 'failed'),
                'results': results
            }, f, indent=2)
        
        print(f"\n✅ Batch processing complete!")
        print(f"Summary saved to {summary_path}")
        
        return results
    
    def _process_single_video(self, video_path, output_path, gps_file, save_video):
        """Process a single video."""
        detector = VideoDetector(
            model_path=str(self.model_path),
            gps_file=gps_file,
            conf_threshold=self.conf_threshold
        )
        
        detections = detector.process_video(
            video_path=str(video_path),
            output_path=str(output_path),
            save_video=save_video
        )
        
        return len(detections)


def main():
    parser = argparse.ArgumentParser(description='Batch process videos')
    parser.add_argument('--input', type=str, required=True, help='Input directory with videos')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--gps-dir', type=str, default=None, help='Directory with GPS files')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save-videos', action='store_true', help='Save annotated videos')
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    processor = BatchProcessor(
        model_path=args.model,
        conf_threshold=args.conf,
        max_workers=args.workers
    )
    
    processor.process_directory(
        input_dir=args.input,
        output_dir=args.output,
        gps_dir=args.gps_dir,
        save_videos=args.save_videos
    )


if __name__ == "__main__":
    main()
