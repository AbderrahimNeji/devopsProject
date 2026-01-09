"""
Prepare dataset for YOLO training.
Splits data into train/val/test sets and creates dataset YAML configuration.
"""

import os
import shutil
import argparse
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random


class DatasetPreparator:
    """Prepare dataset for YOLO training."""
    
    def __init__(self, images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """
        Initialize dataset preparator.
        
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing YOLO format labels
            output_dir: Output directory for organized dataset
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"
        
    def prepare(self):
        """Prepare and split dataset."""
        print("📦 Preparing dataset...")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(self.images_dir.glob(f'*{ext}')))
            image_files.extend(list(self.images_dir.glob(f'*{ext.upper()}')))
        
        print(f"Found {len(image_files)} images")
        
        # Filter images that have corresponding labels
        valid_pairs = []
        for img_path in image_files:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                valid_pairs.append((img_path, label_path))
        
        print(f"Found {len(valid_pairs)} image-label pairs")
        
        if len(valid_pairs) == 0:
            raise ValueError("No valid image-label pairs found!")
        
        # Shuffle data
        random.shuffle(valid_pairs)
        
        # Split dataset
        train_val, test = train_test_split(
            valid_pairs, 
            test_size=self.test_ratio,
            random_state=42
        )
        
        val_size = self.val_ratio / (self.train_ratio + self.val_ratio)
        train, val = train_test_split(
            train_val,
            test_size=val_size,
            random_state=42
        )
        
        print(f"📊 Dataset split:")
        print(f"  - Train: {len(train)} samples")
        print(f"  - Val: {len(val)} samples")
        print(f"  - Test: {len(test)} samples")
        
        # Create directory structure
        splits = {
            'train': train,
            'val': val,
            'test': test
        }
        
        for split_name, pairs in splits.items():
            # Create directories
            images_split_dir = self.output_dir / split_name / 'images'
            labels_split_dir = self.output_dir / split_name / 'labels'
            images_split_dir.mkdir(parents=True, exist_ok=True)
            labels_split_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            print(f"Copying {split_name} files...")
            for img_path, label_path in tqdm(pairs):
                # Copy image
                shutil.copy2(img_path, images_split_dir / img_path.name)
                # Copy label
                shutil.copy2(label_path, labels_split_dir / label_path.name)
        
        # Create dataset YAML
        self._create_yaml()
        
        print(f"✅ Dataset prepared in {self.output_dir}")
    
    def _create_yaml(self):
        """Create YOLO dataset configuration YAML."""
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 4,  # number of classes
            'names': [
                'pothole',
                'longitudinal_crack',
                'crazing',
                'faded_marking'
            ]
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✅ Created dataset config: {yaml_path}")
        
        return yaml_path


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for YOLO training')
    parser.add_argument('--images', type=str, required=True, help='Directory with images')
    parser.add_argument('--labels', type=str, required=True, help='Directory with YOLO labels')
    parser.add_argument('--output', type=str, required=True, help='Output directory for organized dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test set ratio')
    
    args = parser.parse_args()
    
    preparator = DatasetPreparator(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    preparator.prepare()


if __name__ == "__main__":
    main()
