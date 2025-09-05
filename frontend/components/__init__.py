"""
Frontend Components Package
รวมคอมโพเนนต์ต่างๆ สำหรับ UI
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from image_validator import ImageValidator, image_validator, validate_and_convert_image
    from result_display import ResultDisplayer, result_displayer, display_results
except ImportError as e:
    print(f"Warning: Could not import some components: {e}")
    # Create dummy classes to prevent import errors
    class ImageValidator:
        pass
    class ResultDisplayer:
        pass
    def image_validator(*args, **kwargs):
        pass
    def validate_and_convert_image(*args, **kwargs):
        return None
    def result_displayer(*args, **kwargs):
        pass
    def display_results(*args, **kwargs):
        pass

__all__ = [
    'ImageValidator',
    'image_validator', 
    'validate_and_convert_image',
    'ResultDisplayer',
    'result_displayer',
    'display_results'
]
