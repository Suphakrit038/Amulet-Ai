"""
🎨 Data Augmentation Module
==========================

Advanced data augmentation techniques for Amulet-AI:
- MixUp: Linear interpolation between images
- CutMix: Cut and paste between images
- RandAugment: Random augmentation policies
- RandomErasing: Random occlusion
- Custom augmentations for amulet images

Author: Amulet-AI Team
Date: October 2025
"""

from .advanced_augmentation import (
    MixUpAugmentation,
    CutMixAugmentation,
    RandAugmentPipeline,
    RandomErasingTransform,
    MixUpCutMixCollator,
    create_training_augmentation,
    create_validation_transform
)
from .augmentation_pipeline import (
    AugmentationPipeline,
    create_pipeline_from_preset
)

__all__ = [
    'MixUpAugmentation',
    'CutMixAugmentation',
    'RandAugmentPipeline',
    'RandomErasingTransform',
    'MixUpCutMixCollator',
    'create_training_augmentation',
    'create_validation_transform',
    'AugmentationPipeline',
    'create_pipeline_from_preset',
]
