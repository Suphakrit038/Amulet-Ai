"""
Test configuration for Amulet-AI
การตั้งค่าสำหรับการทดสอบระบบ Amulet-AI
"""
import pytest
import sys
import os
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def test_config():
    """Fixture for test configuration"""
    return {
        "test_mode": True,
        "use_mock_data": True
    }

@pytest.fixture
def sample_image_path():
    """Fixture that returns a path to a sample test image"""
    images_dir = Path(__file__).parent / "test_images"
    images_dir.mkdir(exist_ok=True)
    
    # Return a path even if the file doesn't exist yet
    return images_dir / "sample_amulet.jpg"
