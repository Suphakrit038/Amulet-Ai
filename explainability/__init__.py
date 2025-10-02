"""
Explainability Module for Amulet-AI

Provides visualization tools for understanding model decisions:
- Grad-CAM: Gradient-weighted Class Activation Mapping
- Grad-CAM++: Improved version with better localization
- Saliency maps: Simple gradient-based visualization
- Integration utilities for UI

Author: Amulet-AI Team
Date: October 2, 2025
"""

from .gradcam import (
    GradCAM,
    GradCAMPlusPlus,
    visualize_gradcam,
    overlay_heatmap,
    generate_explanation
)

from .saliency import (
    generate_saliency_map,
    generate_smoothgrad,
    visualize_saliency
)

__all__ = [
    'GradCAM',
    'GradCAMPlusPlus',
    'visualize_gradcam',
    'overlay_heatmap',
    'generate_explanation',
    'generate_saliency_map',
    'generate_smoothgrad',
    'visualize_saliency'
]
