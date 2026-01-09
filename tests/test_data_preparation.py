"""
Unit tests for data preparation modules.
"""

import pytest
import tempfile
from pathlib import Path
import cv2
import numpy as np
import json
from src.data_preparation.extract_frames import FrameExtractor
from src.data_preparation.convert_annotations import AnnotationConverter
from src.data_preparation.prepare_dataset import DatasetPreparator


class TestFrameExtractor:
    """Test frame extraction."""
    
    @pytest.fixture
    def sample_video(self):
        """Create a sample video for testing."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
        
        # Create simple video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))
        
        for i in range(30):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        yield video_path
        
        # Cleanup
        Path(video_path).unlink(missing_ok=True)
    
    def test_extract_frames(self, sample_video):
        """Test frame extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = FrameExtractor(
                video_path=sample_video,
                output_dir=tmpdir,
                skip_frames=3,
                max_frames=5
            )
            
            count = extractor.extract()
            
            assert count == 5
            assert len(list(Path(tmpdir).glob('*.jpg'))) == 5
            assert (Path(tmpdir) / 'frames_metadata.json').exists()
    
    def test_metadata_creation(self, sample_video):
        """Test metadata file creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = FrameExtractor(
                video_path=sample_video,
                output_dir=tmpdir,
                max_frames=3
            )
            
            extractor.extract()
            
            metadata_path = Path(tmpdir) / 'frames_metadata.json'
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            assert 'frames' in metadata
            assert len(metadata['frames']) == 3
            assert 'filename' in metadata['frames'][0]
            assert 'timestamp_ms' in metadata['frames'][0]


class TestAnnotationConverter:
    """Test annotation conversion."""
    
    @pytest.fixture
    def sample_coco_annotation(self):
        """Create sample COCO annotation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            coco_data = {
                'images': [
                    {'id': 1, 'file_name': 'image1.jpg', 'width': 640, 'height': 480}
                ],
                'annotations': [
                    {
                        'id': 1,
                        'image_id': 1,
                        'category_id': 0,
                        'bbox': [100, 100, 200, 150]  # x, y, w, h
                    }
                ]
            }
            
            ann_file = tmpdir / 'annotations.json'
            with open(ann_file, 'w') as f:
                json.dump(coco_data, f)
            
            yield tmpdir, ann_file
    
    def test_coco_conversion(self, sample_coco_annotation):
        """Test COCO to YOLO conversion."""
        input_dir, ann_file = sample_coco_annotation
        
        with tempfile.TemporaryDirectory() as output_dir:
            converter = AnnotationConverter(
                input_dir=input_dir,
                output_dir=output_dir,
                format_type='coco'
            )
            
            converter.convert(ann_file)
            
            yolo_file = Path(output_dir) / 'image1.txt'
            assert yolo_file.exists()
            
            with open(yolo_file, 'r') as f:
                line = f.readline().strip()
                parts = line.split()
                
                assert len(parts) == 5
                assert int(parts[0]) == 0  # class id
                assert all(0 <= float(p) <= 1 for p in parts[1:])  # normalized


class TestDatasetPreparator:
    """Test dataset preparation."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample images and labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            images_dir = tmpdir / 'images'
            labels_dir = tmpdir / 'labels'
            images_dir.mkdir()
            labels_dir.mkdir()
            
            # Create 10 sample images and labels
            for i in range(10):
                # Image
                img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                cv2.imwrite(str(images_dir / f'img_{i:03d}.jpg'), img)
                
                # Label
                label = f"0 0.5 0.5 0.2 0.3\n"
                with open(labels_dir / f'img_{i:03d}.txt', 'w') as f:
                    f.write(label)
            
            yield tmpdir, images_dir, labels_dir
    
    def test_dataset_split(self, sample_dataset):
        """Test dataset splitting."""
        tmpdir, images_dir, labels_dir = sample_dataset
        
        with tempfile.TemporaryDirectory() as output_dir:
            preparator = DatasetPreparator(
                images_dir=images_dir,
                labels_dir=labels_dir,
                output_dir=output_dir,
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2
            )
            
            preparator.prepare()
            
            output_path = Path(output_dir)
            
            # Check structure
            assert (output_path / 'train' / 'images').exists()
            assert (output_path / 'train' / 'labels').exists()
            assert (output_path / 'val' / 'images').exists()
            assert (output_path / 'val' / 'labels').exists()
            assert (output_path / 'test' / 'images').exists()
            assert (output_path / 'test' / 'labels').exists()
            
            # Check split ratios
            train_count = len(list((output_path / 'train' / 'images').glob('*.jpg')))
            val_count = len(list((output_path / 'val' / 'images').glob('*.jpg')))
            test_count = len(list((output_path / 'test' / 'images').glob('*.jpg')))
            
            assert train_count == 6
            assert val_count == 2
            assert test_count == 2
    
    def test_yaml_creation(self, sample_dataset):
        """Test YAML config creation."""
        tmpdir, images_dir, labels_dir = sample_dataset
        
        with tempfile.TemporaryDirectory() as output_dir:
            preparator = DatasetPreparator(
                images_dir=images_dir,
                labels_dir=labels_dir,
                output_dir=output_dir
            )
            
            preparator.prepare()
            
            yaml_path = Path(output_dir) / 'dataset.yaml'
            assert yaml_path.exists()
            
            import yaml
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert 'train' in config
            assert 'val' in config
            assert 'nc' in config
            assert config['nc'] == 4
            assert 'names' in config


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
