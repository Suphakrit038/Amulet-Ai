"""
Utils module (Legacy compatibility)
This file re-exports functionality from the utils directory for backwards compatibility
"""

from .utils import (
    validate_and_convert_image,
    send_predict_request,
    SUPPORTED_FORMATS,
    FORMAT_DISPLAY
)
