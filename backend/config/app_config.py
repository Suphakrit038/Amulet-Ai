"""
Application Configuration for Amulet-AI System
กำหนดค่าการทำงานของแอปพลิเคชันโดยรวม
"""
import os
from typing import Dict, List, Optional
from pathlib import Path

# Import the main configuration
from .config import SystemConfig, get_config, config

# Create app specific configurations from the main config
api_config = config.api
model_config = config.model
data_config = config.data

# Function to get configuration - reexport from main config
def get_config() -> SystemConfig:
    """Get the current system configuration"""
    from .config import get_config as main_get_config
    return main_get_config()

# Additional application-specific settings
APP_DEBUG = config.debug
APP_ENVIRONMENT = config.environment
APP_VERSION = "1.0.0"
APP_NAME = "Amulet AI"

# Application paths
APP_ROOT = Path(__file__).parent.parent.parent
STATIC_PATH = APP_ROOT / "frontend" / "static"
TEMPLATES_PATH = APP_ROOT / "frontend" / "templates"
UPLOADS_PATH = APP_ROOT / "uploads"

# Ensure directories exist
UPLOADS_PATH.mkdir(parents=True, exist_ok=True)

# Performance settings
MAX_CONCURRENT_REQUESTS = config.max_concurrent_requests
REQUEST_TIMEOUT = config.request_timeout
CACHE_TTL = config.cache_ttl

# Export commonly used configs and values
def get_api_config():
    """Get API configuration"""
    return api_config

def get_model_config():
    """Get AI model configuration"""
    return model_config

def get_data_config():
    """Get data processing configuration"""
    return data_config

def is_production():
    """Check if running in production mode"""
    return config.is_production()

def is_development():
    """Check if running in development mode"""
    return config.is_development()

def get_log_level():
    """Get current log level"""
    return config.log_level

def get_log_format():
    """Get log format string"""
    return config.log_format
