"""
🖼️ Image Preprocessing Module
============================

Advanced image preprocessing for Amulet-AI:
- Basic preprocessing (resize, normalize)
- Advanced preprocessing (CLAHE, denoising)
- Quality checking and validation
- Batch processing utilities

Author: Amulet-AI Team
Date: October 2025
"""

# Basic preprocessing
from .image_processor import (
    ImageProcessor,
    ColorSpaceConverter,
    BasicPreprocessor,
    create_standard_processor,
    create_basic_preprocessor
)

# Advanced preprocessing
from .advanced_processor import (
    CLAHEProcessor,
    DenoisingProcessor,
    EdgeEnhancer,
    AdvancedPreprocessor,
    create_medical_preprocessor,
    create_artifact_preprocessor
)

# Quality checking
from .quality_checker import (
    BlurDetector,
    BrightnessContrastChecker,
    ResolutionChecker,
    ImageQualityChecker,
    QualityMetrics,
    create_strict_checker,
    create_lenient_checker
)

__all__ = [
    # Basic processing
    'ImageProcessor',
    'ColorSpaceConverter',
    'BasicPreprocessor',
    'create_standard_processor',
    'create_basic_preprocessor',
    
    # Advanced processing
    'CLAHEProcessor',
    'DenoisingProcessor',
    'EdgeEnhancer',
    'AdvancedPreprocessor',
    'create_medical_preprocessor',
    'create_artifact_preprocessor',
    
    # Quality checking
    'BlurDetector',
    'BrightnessContrastChecker',
    'ResolutionChecker',
    'ImageQualityChecker',
    'QualityMetrics',
    'create_strict_checker',
    'create_lenient_checker',
]
