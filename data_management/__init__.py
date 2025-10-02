"""
🗂️ Data Management Module
=====================

โมดูลจัดการข้อมูลแบบครอบคลุม สำหรับ Amulet-AI

Components:
-----------
- augmentation: Advanced data augmentation (MixUp, CutMix, RandAugment)
- preprocessing: Image preprocessing and quality control
- dataset: PyTorch Dataset loaders and samplers
- validation: Dataset validation (coming in Phase 1.5)
- utils: Utilities (coming in Phase 1.5)

Author: Amulet-AI Team
Date: October 2025
"""

__version__ = "1.0.0"

# Import main classes for easy access
from .augmentation.augmentation_pipeline import AugmentationPipeline, create_pipeline_from_preset
from .preprocessing.image_processor import ImageProcessor
from .preprocessing.quality_checker import ImageQualityChecker
from .dataset.dataset_loader import AmuletDataset

__all__ = [
    'AugmentationPipeline',
    'create_pipeline_from_preset',
    'ImageProcessor',
    'ImageQualityChecker',
    'AmuletDataset',
]
