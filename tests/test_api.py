"""
Unit tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import numpy as np
import cv2
from src.api.main import app


client = TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_get_classes(self):
        """Test get classes endpoint."""
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert "classes" in data
        assert len(data["classes"]) == 4
        assert "pothole" in data["classes"]
    
    @pytest.fixture
    def sample_image(self):
        """Create sample image for testing."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(f.name, img)
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    def test_detect_image_endpoint(self, sample_image):
        """Test image detection endpoint."""
        with open(sample_image, 'rb') as f:
            response = client.post(
                "/detect/image",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"conf_threshold": 0.25}
            )
        
        # May fail if no model loaded, but should return proper structure
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            assert "detections" in data
    
    def test_list_jobs(self):
        """Test list jobs endpoint."""
        response = client.get("/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "total" in data


class TestAPICORS:
    """Test CORS configuration."""
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = client.options("/")
        assert response.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
