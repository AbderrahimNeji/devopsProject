"""
Pytest configuration and fixtures.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture(scope='session')
def test_data_dir():
    """Create temporary test data directory."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image():
    """Create sample image."""
    import numpy as np
    import cv2
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(f.name, img)
        yield f.name
    
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def sample_video():
    """Create sample video."""
    import numpy as np
    import cv2
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = f.name
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))
    
    for _ in range(30):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    yield video_path
    
    Path(video_path).unlink(missing_ok=True)
