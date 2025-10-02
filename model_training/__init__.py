"""
🏗️ Model Training Module
========================

Transfer learning and model training for Amulet-AI.

Author: Amulet-AI Team
Date: October 2025
"""

from .transfer_learning import (
    AmuletTransferModel,
    create_transfer_model,
    freeze_backbone,
    unfreeze_layers,
    TwoStageTrainer
)

from .trainer import (
    AmuletTrainer,
    TrainingConfig,
    create_trainer
)

from .callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    MetricsLogger
)

__all__ = [
    # Transfer learning
    'AmuletTransferModel',
    'create_transfer_model',
    'freeze_backbone',
    'unfreeze_layers',
    'TwoStageTrainer',
    
    # Trainer
    'AmuletTrainer',
    'TrainingConfig',
    'create_trainer',
    
    # Callbacks
    'EarlyStopping',
    'ModelCheckpoint',
    'LearningRateScheduler',
    'MetricsLogger',
]
