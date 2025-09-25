"""
🧠 AI Models Module - Python 3.13 Compatible Version

This module contains AI components that work with Python 3.13:
- Lightweight ML System (scikit-learn + OpenCV)
- Compatible Data Pipeline 
- Compatible Visualizer
- AI Project Diagnostics
"""

import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import compatible modules only
try:
    from .lightweight_ml_system import LightweightMLSystem, LightweightMLConfig
except ImportError as e:
    print(f"⚠️ Could not import lightweight_ml_system: {e}")
    LightweightMLSystem = None
    LightweightMLConfig = None

try:
    from .compatible_data_pipeline import CompatibleDataPipeline, CompatibleDataPipelineConfig
except ImportError as e:
    print(f"⚠️ Could not import compatible_data_pipeline: {e}")
    CompatibleDataPipeline = None
    CompatibleDataPipelineConfig = None

try:
    from .compatible_visualizer import CompatibleVisualizer
except ImportError as e:
    print(f"⚠️ Could not import compatible_visualizer: {e}")
    CompatibleVisualizer = None

try:
    from .ai_project_diagnostics import SystemDiagnostics
except ImportError as e:
    print(f"⚠️ Could not import ai_project_diagnostics: {e}")
    SystemDiagnostics = None

__version__ = "3.0.0-compatible"
__author__ = "Amulet AI Team - Python 3.13 Compatible"

__all__ = [
    # Compatible ML System
    'LightweightMLSystem',
    'LightweightMLConfig',
    
    # Compatible Data Pipeline
    'CompatibleDataPipeline',
    'CompatibleDataPipelineConfig',
    
    # Compatible Visualization
    'CompatibleVisualizer',
    
    # Diagnostics
    'SystemDiagnostics'
]
