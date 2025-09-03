"""
🧠 AI Models Module for Advanced Amulet Recognition System

This module contains all AI-related components for maximum quality image recognition:
- Advanced Image Processing System
- Self-Supervised Learning with Contrastive Learning 
- Advanced Data Pipeline
- Dataset Organization & Embedding Database
- Master Training System
- Training Scripts
"""

# ป้องกัน TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import core modules
from .advanced_image_processor import AdvancedImageProcessor, advanced_processor, process_image_max_quality
from .self_supervised_learning import (
    ContrastiveLearningModel,
    SelfSupervisedTrainer,
    AdvancedEmbeddingSystem,
    EmbeddingConfig,
    create_self_supervised_system
)
from .advanced_data_pipeline import AdvancedDataPipeline, DataPipelineConfig, HighQualityAmuletDataset
from .dataset_organizer import DatasetOrganizer, EmbeddingDatabase, create_embedding_record
from .master_training_system import (
    MasterTrainingSystem,
    MasterTrainingConfig,
    create_master_training_system
)

__version__ = "2.0.0"
__author__ = "Advanced Amulet AI Team"

__all__ = [
    # Image Processing
    'AdvancedImageProcessor',
    'advanced_processor',
    'process_image_max_quality',
    
    # Self-Supervised Learning
    'ContrastiveLearningModel',
    'SelfSupervisedTrainer', 
    'AdvancedEmbeddingSystem',
    'EmbeddingConfig',
    'create_self_supervised_system',
    
    # Data Pipeline
    'AdvancedDataPipeline',
    'DataPipelineConfig',
    'HighQualityAmuletDataset',
    
    # Dataset Organization
    'DatasetOrganizer',
    'EmbeddingDatabase',
    'create_embedding_record',
    
    # Training System
    'MasterTrainingSystem',
    'MasterTrainingConfig',
    'create_master_training_system'
]
