"""
Test configuration and fixtures
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pytest
    from fastapi.testclient import TestClient
    from backend.api import app
    
    @pytest.fixture
    def client():
        """Test client for FastAPI"""
        return TestClient(app)
    
    @pytest.fixture
    def api_url():
        """API URL for testing"""
        return "http://localhost:8000"
    
    @pytest.fixture
    def sample_image_path():
        """Path to sample test image"""
        return project_root / "dataset" / "สีวลี" / "พระสิวลี-จัมโบ้-ด้านหน้าหลัง-1คู่.jpg"
    
    @pytest.fixture
    def test_config():
        """Test configuration"""
        return {
            "test_mode": True,
            "simulation_mode": True,
            "confidence_threshold": 0.5
        }

except ImportError as e:
    print(f"Warning: Some test dependencies not available: {e}")
    print("Please install test dependencies: pip install pytest fastapi[all]")
    
    # Minimal fixtures for when pytest is not available
    class MockFixture:
        def __init__(self, value):
            self.value = value
        
        def __call__(self, func):
            return lambda: self.value
    
    def pytest_fixture_fallback(value):
        return MockFixture(value)
    
    api_url = pytest_fixture_fallback("http://localhost:8000")
    sample_image_path = pytest_fixture_fallback(project_root / "dataset" / "สีวลี" / "พระสิวลี-จัมโบ้-ด้านหน้าหลัง-1คู่.jpg")
