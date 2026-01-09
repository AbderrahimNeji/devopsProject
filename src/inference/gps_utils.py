"""
GPS utilities for geolocation of detections.
Supports multiple GPS data formats and synchronization methods.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import gpxpy
import gpxpy.gpx
from geopy.distance import geodesic


class GPSProcessor:
    """Process and synchronize GPS data with video frames."""
    
    def __init__(self, gps_file=None, gps_format='csv'):
        """
        Initialize GPS processor.
        
        Args:
            gps_file: Path to GPS data file (CSV, GPX, or JSON)
            gps_format: Format of GPS file ('csv', 'gpx', 'json')
        """
        self.gps_file = Path(gps_file) if gps_file else None
        self.gps_format = gps_format
        self.gps_data = None
        
        if gps_file:
            self.load_gps_data()
    
    def load_gps_data(self):
        """Load GPS data from file."""
        print(f"📍 Loading GPS data from {self.gps_file}")
        
        if self.gps_format == 'csv':
            self.gps_data = self._load_csv()
        elif self.gps_format == 'gpx':
            self.gps_data = self._load_gpx()
        elif self.gps_format == 'json':
            self.gps_data = self._load_json()
        else:
            raise ValueError(f"Unsupported GPS format: {self.gps_format}")
        
        print(f"✅ Loaded {len(self.gps_data)} GPS points")
        
        return self.gps_data
    
    def _load_csv(self):
        """
        Load GPS data from CSV.
        Expected columns: timestamp, latitude, longitude, altitude (optional), speed (optional)
        """
        df = pd.read_csv(self.gps_file)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def _load_gpx(self):
        """Load GPS data from GPX file."""
        with open(self.gps_file, 'r') as f:
            gpx = gpxpy.parse(f)
        
        gps_points = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    gps_points.append({
                        'timestamp': point.time,
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'altitude': point.elevation if point.elevation else 0.0,
                        'speed': point.speed if hasattr(point, 'speed') else 0.0
                    })
        
        return pd.DataFrame(gps_points)
    
    def _load_json(self):
        """Load GPS data from JSON file."""
        with open(self.gps_file, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def synchronize_with_frames(self, frames_metadata, video_start_time=None):
        """
        Synchronize GPS data with video frames.
        
        Args:
            frames_metadata: List of frame metadata with timestamps
            video_start_time: Video start datetime (if known)
        
        Returns:
            DataFrame with frame_number, timestamp, latitude, longitude
        """
        print("🔄 Synchronizing GPS data with frames...")
        
        if self.gps_data is None:
            raise ValueError("GPS data not loaded")
        
        # Sort GPS data by timestamp
        gps_sorted = self.gps_data.sort_values('timestamp').reset_index(drop=True)
        
        synchronized_data = []
        
        for frame_info in frames_metadata:
            frame_num = frame_info['frame_number']
            frame_time_sec = frame_info['timestamp_sec']
            
            # Calculate absolute timestamp
            if video_start_time:
                frame_timestamp = video_start_time + timedelta(seconds=frame_time_sec)
            else:
                # Use relative timestamps
                frame_timestamp = gps_sorted['timestamp'].iloc[0] + timedelta(seconds=frame_time_sec)
            
            # Find closest GPS point
            gps_point = self._find_closest_gps_point(frame_timestamp, gps_sorted)
            
            if gps_point is not None:
                synchronized_data.append({
                    'frame_number': frame_num,
                    'filename': frame_info['filename'],
                    'timestamp': frame_timestamp,
                    'latitude': gps_point['latitude'],
                    'longitude': gps_point['longitude'],
                    'altitude': gps_point.get('altitude', 0.0),
                    'speed': gps_point.get('speed', 0.0)
                })
        
        df = pd.DataFrame(synchronized_data)
        print(f"✅ Synchronized {len(df)} frames with GPS")
        
        return df
    
    def _find_closest_gps_point(self, target_time, gps_data, max_time_diff_sec=5):
        """
        Find closest GPS point to target timestamp.
        
        Args:
            target_time: Target timestamp
            gps_data: Sorted GPS DataFrame
            max_time_diff_sec: Maximum allowed time difference in seconds
        
        Returns:
            GPS point dict or None
        """
        # Calculate time differences
        time_diffs = (gps_data['timestamp'] - target_time).abs()
        
        # Find minimum
        min_idx = time_diffs.idxmin()
        min_diff = time_diffs[min_idx]
        
        # Check if within threshold
        if min_diff.total_seconds() <= max_time_diff_sec:
            return gps_data.iloc[min_idx].to_dict()
        else:
            return None
    
    def interpolate_gps(self, timestamp, gps_data=None):
        """
        Interpolate GPS coordinates for a specific timestamp.
        
        Args:
            timestamp: Target timestamp
            gps_data: GPS DataFrame (uses self.gps_data if None)
        
        Returns:
            Interpolated GPS coordinates dict
        """
        if gps_data is None:
            gps_data = self.gps_data
        
        if gps_data is None:
            raise ValueError("GPS data not loaded")
        
        # Sort by timestamp
        gps_sorted = gps_data.sort_values('timestamp').reset_index(drop=True)
        
        # Find bracketing points
        before = gps_sorted[gps_sorted['timestamp'] <= timestamp]
        after = gps_sorted[gps_sorted['timestamp'] > timestamp]
        
        if len(before) == 0:
            # Before start - use first point
            return gps_sorted.iloc[0].to_dict()
        elif len(after) == 0:
            # After end - use last point
            return gps_sorted.iloc[-1].to_dict()
        else:
            # Interpolate
            point1 = before.iloc[-1]
            point2 = after.iloc[0]
            
            # Time interpolation factor
            t1 = point1['timestamp']
            t2 = point2['timestamp']
            t = timestamp
            
            factor = (t - t1).total_seconds() / (t2 - t1).total_seconds()
            
            # Linear interpolation
            lat = point1['latitude'] + factor * (point2['latitude'] - point1['latitude'])
            lon = point1['longitude'] + factor * (point2['longitude'] - point1['longitude'])
            alt = point1.get('altitude', 0) + factor * (point2.get('altitude', 0) - point1.get('altitude', 0))
            
            return {
                'timestamp': t,
                'latitude': lat,
                'longitude': lon,
                'altitude': alt,
                'interpolated': True
            }
    
    def calculate_distance(self, coord1, coord2):
        """
        Calculate distance between two GPS coordinates.
        
        Args:
            coord1: (latitude, longitude) tuple
            coord2: (latitude, longitude) tuple
        
        Returns:
            Distance in meters
        """
        return geodesic(coord1, coord2).meters
    
    @staticmethod
    def create_sample_gps_csv(output_path, num_points=100, start_lat=48.8566, start_lon=2.3522):
        """
        Create a sample GPS CSV file for testing.
        
        Args:
            output_path: Output file path
            num_points: Number of GPS points
            start_lat: Starting latitude
            start_lon: Starting longitude
        """
        print(f"📝 Creating sample GPS file: {output_path}")
        
        timestamps = [datetime.now() + timedelta(seconds=i) for i in range(num_points)]
        
        # Simulate movement (random walk)
        latitudes = [start_lat]
        longitudes = [start_lon]
        
        for i in range(1, num_points):
            # Add small random variations
            lat_change = np.random.normal(0, 0.0001)
            lon_change = np.random.normal(0, 0.0001)
            latitudes.append(latitudes[-1] + lat_change)
            longitudes.append(longitudes[-1] + lon_change)
        
        altitudes = np.random.uniform(50, 100, num_points)
        speeds = np.random.uniform(0, 50, num_points)  # km/h
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'latitude': latitudes,
            'longitude': longitudes,
            'altitude': altitudes,
            'speed': speeds
        })
        
        df.to_csv(output_path, index=False)
        print(f"✅ Sample GPS file created with {num_points} points")


if __name__ == "__main__":
    # Create sample GPS data for testing
    GPSProcessor.create_sample_gps_csv(
        'data/sample_gps.csv',
        num_points=100,
        start_lat=48.8566,
        start_lon=2.3522
    )
