# ===== COMPONENT EXPORTS =====

"""
Mystical UI Components for Amulet AI
====================================

This module provides a complete set of UI components with mystical theming
for the Buddhist amulet analysis application.

Components included:
- Mystical headers and cards
- Progress indicators with glow effects
- Upload zones with drag-and-drop
- Alert and notification systems
- Layout management components
- Navigation and sidebar elements

Usage:
    from frontend.components.ui import (
        mystical_header,
        mystical_card,
        mystical_progress,
        create_sidebar_navigation,
        create_main_layout
    )
"""

# Import all components for easy access
from .mystical_components import (
    mystical_header,
    mystical_card,
    mystical_button,
    mystical_progress,
    mystical_alert,
    mystical_upload_zone,
    mystical_metrics,
    mystical_tabs,
    confidence_indicator,
    result_display_card
)

from .layout_components import (
    create_sidebar_navigation,
    create_main_layout,
    create_hero_section,
    create_upload_section,
    create_results_section,
    create_comparison_section,
    create_footer
)

# Component categories for organized imports
MYSTICAL_COMPONENTS = [
    'mystical_header',
    'mystical_card', 
    'mystical_button',
    'mystical_progress',
    'mystical_alert',
    'mystical_upload_zone',
    'mystical_metrics',
    'mystical_tabs',
    'confidence_indicator',
    'result_display_card'
]

LAYOUT_COMPONENTS = [
    'create_sidebar_navigation',
    'create_main_layout',
    'create_hero_section',
    'create_upload_section',
    'create_results_section',
    'create_comparison_section',
    'create_footer'
]

# All available components
__all__ = MYSTICAL_COMPONENTS + LAYOUT_COMPONENTS

# Version information
__version__ = "1.0.0"
__author__ = "Amulet AI Team"
__description__ = "Mystical UI components with glassmorphism and golden theme"
