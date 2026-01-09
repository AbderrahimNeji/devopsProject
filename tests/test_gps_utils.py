"""
Unit tests for GPS utilities.
"""

import pytest
import tempfile
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from src.inference.gps_utils import GPSProcessor


class TestGPSProcessor:
    """Test GPS processing."""
    
    @pytest.fixture
    def sample_gps_csv(self):
        """Create sample GPS CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('timestamp,latitude,longitude,altitude,speed\n')
            
            base_time = datetime.now()
            for i in range(10):
                timestamp = base_time + timedelta(seconds=i)
                lat = 48.8566 + i * 0.0001
                lon = 2.3522 + i * 0.0001
                f.write(f'{timestamp.isoformat()},{lat},{lon},100.0,30.0\n')
            
            yield f.name
        
        Path(f.name).unlink(missing_ok=True)
    
    def test_load_csv(self, sample_gps_csv):
        """Test loading GPS from CSV."""
        processor = GPSProcessor(sample_gps_csv, 'csv')
        
        assert processor.gps_data is not None
        assert len(processor.gps_data) == 10
        assert 'latitude' in processor.gps_data.columns
        assert 'longitude' in processor.gps_data.columns
    
    def test_interpolation(self, sample_gps_csv):
        """Test GPS interpolation."""
        processor = GPSProcessor(sample_gps_csv, 'csv')
        
        # Get time between two points
        t1 = processor.gps_data['timestamp'].iloc[0]
        t2 = processor.gps_data['timestamp'].iloc[1]
        t_mid = t1 + (t2 - t1) / 2
        
        # Interpolate
        result = processor.interpolate_gps(t_mid)
        
        assert result is not None
        assert 'latitude' in result
        assert 'longitude' in result
        
        # Check interpolated value is between the two points
        lat1 = processor.gps_data['latitude'].iloc[0]
        lat2 = processor.gps_data['latitude'].iloc[1]
        assert lat1 <= result['latitude'] <= lat2 or lat2 <= result['latitude'] <= lat1
    
    def test_distance_calculation(self, sample_gps_csv):
        """Test distance calculation."""
        processor = GPSProcessor(sample_gps_csv, 'csv')
        
        coord1 = (48.8566, 2.3522)
        coord2 = (48.8576, 2.3532)
        
        distance = processor.calculate_distance(coord1, coord2)
        
        assert distance > 0
        assert distance < 200  # Should be less than 200m
    
    def test_synchronization(self, sample_gps_csv):
        """Test frame synchronization."""
        processor = GPSProcessor(sample_gps_csv, 'csv')
        
        # Create fake frame metadata
        base_time = datetime.now()
        frames_metadata = [
            {
                'frame_number': i,
                'filename': f'frame_{i:06d}.jpg',
                'timestamp_sec': i * 0.5
            }
            for i in range(5)
        ]
        
        synced = processor.synchronize_with_frames(
            frames_metadata,
            video_start_time=base_time
        )
        
        assert len(synced) == 5
        assert 'latitude' in synced.columns
        assert 'longitude' in synced.columns
    
    def test_create_sample_gps(self):
        """Test sample GPS creation."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            output_path = f.name
        
        GPSProcessor.create_sample_gps_csv(output_path, num_points=20)
        
        df = pd.read_csv(output_path)
        
        assert len(df) == 20
        assert 'latitude' in df.columns
        assert 'longitude' in df.columns
        assert 'timestamp' in df.columns
        
        Path(output_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
