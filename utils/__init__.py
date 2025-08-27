"""
Utils Package
ฟังก์ชันช่วยเหลือสำหรับระบบ Amulet AI
"""

# Import main utilities
from .config_manager import config, get_config, set_config
from .logger import default_logger, performance_tracker, log_info, log_error, log_warning
from .error_handler import (
    AmuletError, ValidationError, ModelError, ImageProcessingError,
    handle_error, create_error_response, validate_required_fields
)
from .image_utils import validate_image, preprocess_image, image_to_bytes

# Version
__version__ = "1.0.0"

# Export main utilities
__all__ = [
    # Config
    'config', 'get_config', 'set_config',
    
    # Logging
    'default_logger', 'performance_tracker', 
    'log_info', 'log_error', 'log_warning',
    
    # Error handling
    'AmuletError', 'ValidationError', 'ModelError', 'ImageProcessingError',
    'handle_error', 'create_error_response', 'validate_required_fields',
    
    # Image processing
    'validate_image', 'preprocess_image', 'image_to_bytes'
]
