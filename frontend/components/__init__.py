# ===== MAIN COMPONENT MODULE =====

"""
Frontend Components Module
===========================

This module contains all frontend components for the Amulet AI application,
organized into logical categories for easy maintenance and usage.

Structure:
- ui/: Core UI components with mystical theming
- layouts/: Layout management components
- forms/: Form components and validation
- data/: Data display and visualization components
"""

# Import UI components
from .ui import *

# Version information
__version__ = "1.0.0"
__author__ = "Amulet AI Development Team"

# Module exports
__all__ = [
    # Re-export all UI components
    "mystical_header",
    "mystical_card", 
    "mystical_button",
    "mystical_progress",
    "mystical_alert",
    "mystical_upload_zone",
    "mystical_metrics",
    "mystical_tabs",
    "confidence_indicator",
    "result_display_card",
    "create_sidebar_navigation",
    "create_main_layout",
    "create_hero_section",
    "create_upload_section",
    "create_results_section",
    "create_comparison_section",
    "create_footer"
]
