"""
API Tests for Amulet-AI
การทดสอบ API ของระบบ Amulet-AI
"""
import pytest
import sys
import os
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_api_imports():
    """Test that API modules can be imported"""
    # This will raise an exception if imports fail
    from backend import api
    assert api is not None

def test_config_imports():
    """Test that configuration modules can be imported"""
    from utils.config_manager import Config, get_config
    assert Config is not None
    assert get_config is not None

def test_config_functionality(test_config):
    """Test configuration functionality"""
    from utils.config_manager import Config
    
    # Create a test config file
    config = Config("test_config.json")
    config.set("test_key", "test_value")
    
    # Verify the value was set
    assert config.get("test_key") == "test_value"
