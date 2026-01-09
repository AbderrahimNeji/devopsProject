"""
Train YOLOv8 model for road degradation detection.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml
import torch
import os


class YOLOTrainer:
    """YOLOv8 training wrapper."""
    
    def __init__(self, config_path, model_name='yolov8n.pt', resume=False):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to training configuration YAML
            model_name: Pretrained model name or path
            resume: Resume training from checkpoint
        """
        self.config_path = Path(config_path)
        self.model_name = model_name
        self.resume = resume
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model
        if resume and (Path('runs/detect') / self.config.get('name', 'road_degradation') / 'weights/last.pt').exists():
            checkpoint_path = Path('runs/detect') / self.config.get('name', 'road_degradation') / 'weights/last.pt'
            print(f"📥 Resuming from checkpoint: {checkpoint_path}")
            self.model = YOLO(str(checkpoint_path))
        else:
            print(f"🏗️ Loading pretrained model: {model_name}")
            self.model = YOLO(model_name)
        
        # Print device info
        self._print_device_info()
    
    def _print_device_info(self):
        """Print device information."""
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("💻 Using CPU")
    
    def train(self):
        """Start training."""
        print("🚀 Starting training...")
        print(f"📝 Config: {self.config_path}")
        
        # Training parameters
        results = self.model.train(
            data=self.config['data'],
            epochs=self.config.get('epochs', 100),
            batch=self.config.get('batch', 16),
            imgsz=self.config.get('imgsz', 640),
            
            # Optimizer
            optimizer=self.config.get('optimizer', 'SGD'),
            lr0=self.config.get('lr0', 0.01),
            lrf=self.config.get('lrf', 0.01),
            momentum=self.config.get('momentum', 0.937),
            weight_decay=self.config.get('weight_decay', 0.0005),
            
            # Augmentation
            hsv_h=self.config.get('hsv_h', 0.015),
            hsv_s=self.config.get('hsv_s', 0.7),
            hsv_v=self.config.get('hsv_v', 0.4),
            degrees=self.config.get('degrees', 0.0),
            translate=self.config.get('translate', 0.1),
            scale=self.config.get('scale', 0.5),
            shear=self.config.get('shear', 0.0),
            perspective=self.config.get('perspective', 0.0),
            flipud=self.config.get('flipud', 0.0),
            fliplr=self.config.get('fliplr', 0.5),
            mosaic=self.config.get('mosaic', 1.0),
            mixup=self.config.get('mixup', 0.0),
            copy_paste=self.config.get('copy_paste', 0.0),
            
            # Training settings
            device=self.config.get('device', 0),
            workers=self.config.get('workers', 8),
            project=self.config.get('project', 'runs/detect'),
            name=self.config.get('name', 'road_degradation'),
            exist_ok=self.config.get('exist_ok', False),
            pretrained=self.config.get('pretrained', True),
            verbose=self.config.get('verbose', True),
            seed=self.config.get('seed', 0),
            deterministic=self.config.get('deterministic', True),
            single_cls=self.config.get('single_cls', False),
            rect=self.config.get('rect', False),
            cos_lr=self.config.get('cos_lr', False),
            close_mosaic=self.config.get('close_mosaic', 10),
            resume=self.resume,
            amp=self.config.get('amp', True),
            fraction=self.config.get('fraction', 1.0),
            profile=self.config.get('profile', False),
            
            # Validation
            val=self.config.get('val', True),
            save=self.config.get('save', True),
            save_period=self.config.get('save_period', -1),
            cache=self.config.get('cache', False),
            plots=self.config.get('plots', True),
        )
        
        print("✅ Training completed!")
        print(f"📊 Results saved to: {results.save_dir}")
        
        return results
    
    def validate(self, weights_path=None):
        """
        Validate model on test set.
        
        Args:
            weights_path: Path to model weights (default: best.pt from training)
        """
        if weights_path is None:
            weights_path = Path(self.config.get('project', 'runs/detect')) / \
                          self.config.get('name', 'road_degradation') / 'weights/best.pt'
        
        print(f"🔍 Validating model: {weights_path}")
        
        # Load model
        model = YOLO(str(weights_path))
        
        # Validate
        metrics = model.val(
            data=self.config['data'],
            batch=self.config.get('batch', 16),
            imgsz=self.config.get('imgsz', 640),
            device=self.config.get('device', 0),
            workers=self.config.get('workers', 8),
            plots=True
        )
        
        # Print metrics
        print("\n📈 Validation Metrics:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
        
        # Per-class metrics
        print("\n📊 Per-Class Metrics:")
        class_names = ['pothole', 'longitudinal_crack', 'crazing', 'faded_marking']
        for i, class_name in enumerate(class_names):
            if i < len(metrics.box.maps):
                print(f"  {class_name}:")
                print(f"    AP50: {metrics.box.maps[i]:.4f}")
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for road degradation detection')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                        help='Pretrained model (yolov8n/s/m/l/x.pt)')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--validate-only', action='store_true', help='Only run validation')
    parser.add_argument('--weights', type=str, default=None, help='Path to weights for validation')
    
    args = parser.parse_args()
    
    trainer = YOLOTrainer(
        config_path=args.config,
        model_name=args.model,
        resume=args.resume
    )
    
    if args.validate_only:
        trainer.validate(args.weights)
    else:
        trainer.train()
        # Auto-validate after training
        trainer.validate()


if __name__ == "__main__":
    main()
