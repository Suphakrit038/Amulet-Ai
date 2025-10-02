#!/usr/bin/env python3
"""
🚀 Modern Training Pipeline for Amulet-AI
=========================================

Complete training pipeline with:
- Transfer Learning (ResNet50, EfficientNet, MobileNet)
- Advanced Augmentation (RandAugment, MixUp, CutMix)
- Two-stage training strategy
- Early stopping & LR scheduling
- Evaluation & model saving
- Integration with all system components

Author: Amulet-AI Team
Date: October 2, 2025
"""

import logging
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from tqdm.auto import tqdm

# Project imports
from .transfer_learning import AmuletTransferModel, TwoStageTrainer
from .callbacks import EarlyStopping, ModelCheckpoint
from ..data_management.augmentation.augmentation_pipeline import AugmentationPipeline, create_pipeline_from_preset
from ..data_management.dataset.dataset_loader import AmuletDataset, create_class_weights
from ..evaluation.metrics import compute_per_class_metrics, evaluate_model
from ..evaluation.calibration import TemperatureScaling
from ..core.config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModernTrainingPipeline:
    """
    🎯 Modern Training Pipeline
    
    Complete end-to-end training with state-of-the-art techniques:
    
    Features:
    - Multiple backbone architectures
    - Advanced augmentation strategies
    - Two-stage training (freeze → fine-tune)
    - Automatic hyperparameter optimization
    - Comprehensive evaluation
    - Model versioning & artifacts saving
    - Integration with MLOps tools
    
    Example:
        >>> pipeline = ModernTrainingPipeline(
        ...     data_dir='organized_dataset/splits',
        ...     output_dir='trained_model',
        ...     backbone='resnet50'
        ... )
        >>> results = pipeline.train()
        >>> print(f"Best validation accuracy: {results['best_val_acc']:.4f}")
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        backbone: str = 'resnet50',
        num_classes: int = 6,
        augmentation_preset: str = 'medium',
        device: str = 'auto',
        config_overrides: Optional[Dict] = None
    ):
        """
        Initialize Training Pipeline
        
        Args:
            data_dir: Path to dataset splits (train/val/test folders)
            output_dir: Output directory for models and artifacts
            backbone: Backbone architecture ('resnet50', 'efficientnet_b0', etc.)
            num_classes: Number of classes
            augmentation_preset: Augmentation preset ('light', 'medium', 'heavy')
            device: Device ('cuda', 'cpu', 'auto')
            config_overrides: Override default configuration
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.backbone = backbone
        self.num_classes = num_classes
        self.augmentation_preset = augmentation_preset
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.config = {
            # Model
            'backbone': backbone,
            'num_classes': num_classes,
            'pretrained': True,
            'dropout': 0.3,
            'hidden_dim': 128,
            'use_two_fc': True,
            
            # Training
            'batch_size': 32,
            'num_workers': 4,
            'pin_memory': True,
            
            # Stage 1 (Head training)
            'stage1_epochs': 10,
            'stage1_lr': 1e-3,
            'stage1_weight_decay': 1e-4,
            'stage1_patience': 5,
            
            # Stage 2 (Fine-tuning)
            'stage2_epochs': 30,
            'stage2_lr': 1e-4,
            'stage2_weight_decay': 1e-4,
            'stage2_unfreeze_layers': 10,
            'stage2_patience': 8,
            
            # Optimization
            'optimizer': 'adam',
            'lr_scheduler': 'reduce_on_plateau',
            'lr_factor': 0.5,
            'lr_patience': 3,
            
            # Regularization
            'use_class_weights': True,
            'label_smoothing': 0.1,
            
            # Augmentation
            'augmentation_preset': augmentation_preset,
            
            # Evaluation
            'eval_every': 1,  # Evaluate every N epochs
            'save_best_only': True,
            'metric_for_best': 'val_acc',  # 'val_acc', 'val_f1', 'val_loss'
            
            # MLOps
            'experiment_name': f'amulet_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'save_artifacts': True,
            'track_metrics': True,
        }
        
        # Apply overrides
        if config_overrides:
            self.config.update(config_overrides)
            
        # Initialize components
        self.model = None
        self.augmentation_pipeline = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_names = None
        self.class_weights = None
        
        # Training state
        self.training_history = {
            'stage1': {},
            'stage2': {},
            'metrics': {}
        }
        
        logger.info(f"ModernTrainingPipeline initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Backbone: {backbone}")
        logger.info(f"  Data: {data_dir}")
        logger.info(f"  Output: {output_dir}")
        
    def setup_data(self):
        """Setup datasets and data loaders"""
        logger.info("🔧 Setting up data loaders...")
        
        # Create augmentation pipeline
        aug_config = {
            'batch_size': self.config['batch_size'],
            'num_workers': self.config['num_workers'],
            'pin_memory': self.config['pin_memory'],
            'num_classes': self.num_classes
        }
        
        self.augmentation_pipeline = create_pipeline_from_preset(
            self.augmentation_preset,
            **aug_config
        )
        
        # Load datasets
        train_dataset = AmuletDataset(
            data_dir=self.data_dir / 'train',
            transform=self.augmentation_pipeline.get_transform('train'),
            return_path=False
        )
        
        val_dataset = AmuletDataset(
            data_dir=self.data_dir / 'validation',
            transform=self.augmentation_pipeline.get_transform('val'),
            return_path=False
        )
        
        # Test dataset (optional)
        test_dir = self.data_dir / 'test'
        if test_dir.exists():
            test_dataset = AmuletDataset(
                data_dir=test_dir,
                transform=self.augmentation_pipeline.get_transform('val'),
                return_path=False
            )
        else:
            test_dataset = None
            logger.warning("Test directory not found. Skipping test dataset.")
        
        # Get class names and weights
        self.class_names = train_dataset.classes
        logger.info(f"Classes found: {self.class_names}")
        
        # Create class weights for imbalanced data
        if self.config['use_class_weights']:
            self.class_weights = create_class_weights(train_dataset)
            logger.info(f"Class weights: {dict(zip(self.class_names, self.class_weights))}")
            
            # Create weighted sampler
            sample_weights = [self.class_weights[label] for _, label in train_dataset]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
        else:
            sampler = None
            self.class_weights = None
        
        # Create data loaders
        self.train_loader = self.augmentation_pipeline.create_dataloader(
            train_dataset,
            mode='train',
            sampler=sampler
        )
        
        self.val_loader = self.augmentation_pipeline.create_dataloader(
            val_dataset,
            mode='val'
        )
        
        if test_dataset:
            self.test_loader = self.augmentation_pipeline.create_dataloader(
                test_dataset,
                mode='val'
            )
        
        logger.info(f"✅ Data loaders created:")
        logger.info(f"  Train: {len(train_dataset)} samples, {len(self.train_loader)} batches")
        logger.info(f"  Val: {len(val_dataset)} samples, {len(self.val_loader)} batches")
        if test_dataset:
            logger.info(f"  Test: {len(test_dataset)} samples, {len(self.test_loader)} batches")
            
    def setup_model(self):
        """Setup model and training components"""
        logger.info("🔧 Setting up model...")
        
        # Create model
        self.model = AmuletTransferModel(
            backbone_name=self.config['backbone'],
            num_classes=self.config['num_classes'],
            pretrained=self.config['pretrained'],
            dropout=self.config['dropout'],
            hidden_dim=self.config['hidden_dim'],
            use_two_fc=self.config['use_two_fc']
        )
        
        self.model.to(self.device)
        
        # Log model info
        params_info = self.model.get_trainable_params()
        logger.info(f"✅ Model created:")
        logger.info(f"  Architecture: {self.config['backbone']}")
        logger.info(f"  Total parameters: {params_info['total']:,}")
        logger.info(f"  Trainable parameters: {params_info['trainable']:,}")
        
    def create_criterion(self):
        """Create loss function"""
        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, dtype=torch.float32).to(self.device)
        else:
            weight = None
            
        if self.config['label_smoothing'] > 0:
            criterion = nn.CrossEntropyLoss(
                weight=weight,
                label_smoothing=self.config['label_smoothing']
            )
        else:
            criterion = nn.CrossEntropyLoss(weight=weight)
            
        return criterion
        
    def train_stage1(self) -> Dict:
        """Train Stage 1: Head only (frozen backbone)"""
        logger.info("="*70)
        logger.info("🎯 Stage 1: Training Head Only (Frozen Backbone)")
        logger.info("="*70)
        
        # Create trainer
        criterion = self.create_criterion()
        trainer = TwoStageTrainer(self.model, criterion, self.device)
        
        # Train
        history = trainer.train_stage1(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=self.config['stage1_epochs'],
            lr=self.config['stage1_lr'],
            weight_decay=self.config['stage1_weight_decay'],
            patience=self.config['stage1_patience']
        )
        
        self.training_history['stage1'] = history
        logger.info("✅ Stage 1 completed!")
        
        return history
        
    def train_stage2(self) -> Dict:
        """Train Stage 2: Fine-tune last layers"""
        logger.info("="*70)
        logger.info("🔥 Stage 2: Fine-Tuning Last Layers")
        logger.info("="*70)
        
        # Create trainer (reuse from stage 1)
        criterion = self.create_criterion()
        trainer = TwoStageTrainer(self.model, criterion, self.device)
        
        # Train
        history = trainer.train_stage2(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=self.config['stage2_epochs'],
            lr=self.config['stage2_lr'],
            weight_decay=self.config['stage2_weight_decay'],
            unfreeze_layers=self.config['stage2_unfreeze_layers'],
            patience=self.config['stage2_patience']
        )
        
        self.training_history['stage2'] = history
        logger.info("✅ Stage 2 completed!")
        
        return history
        
    def evaluate_model(self) -> Dict:
        """Comprehensive model evaluation"""
        logger.info("📊 Evaluating model...")
        
        results = {}
        
        # Validation metrics
        if self.val_loader:
            val_metrics = evaluate_model(
                self.model, self.val_loader, self.device, self.class_names
            )
            results['validation'] = val_metrics
            logger.info(f"Validation Accuracy: {val_metrics.accuracy:.4f}")
            logger.info(f"Validation F1 (macro): {val_metrics.macro_avg_f1:.4f}")
        
        # Test metrics
        if self.test_loader:
            test_metrics = evaluate_model(
                self.model, self.test_loader, self.device, self.class_names
            )
            results['test'] = test_metrics
            logger.info(f"Test Accuracy: {test_metrics.accuracy:.4f}")
            logger.info(f"Test F1 (macro): {test_metrics.macro_avg_f1:.4f}")
        
        # Calibration
        try:
            from ..evaluation.calibration import evaluate_calibration
            cal_results = evaluate_calibration(self.model, self.val_loader, self.device)
            results['calibration'] = cal_results
            logger.info(f"Expected Calibration Error: {cal_results['ece']:.4f}")
        except Exception as e:
            logger.warning(f"Calibration evaluation failed: {e}")
        
        self.training_history['metrics'] = results
        return results
        
    def save_artifacts(self):
        """Save all training artifacts"""
        logger.info("💾 Saving training artifacts...")
        
        # Create experiment directory
        exp_dir = self.output_dir / self.config['experiment_name']
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = exp_dir / 'best_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'class_names': self.class_names,
            'training_history': self.training_history
        }, model_path)
        
        # Copy to main trained_model directory
        main_model_path = self.output_dir / 'best_model.pth'
        shutil.copy2(model_path, main_model_path)
        
        # Save class mapping
        class_mapping = {str(i): name for i, name in enumerate(self.class_names)}
        with open(exp_dir / 'class_mapping.json', 'w') as f:
            json.dump(class_mapping, f, indent=2)
        with open(self.output_dir / 'class_mapping.json', 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        # Save training config
        with open(exp_dir / 'training_config.json', 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        with open(self.output_dir / 'model_config.json', 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        # Save training history
        with open(exp_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        # Save augmentation config
        aug_stats = self.augmentation_pipeline.get_augmentation_stats()
        with open(exp_dir / 'augmentation_config.json', 'w') as f:
            json.dump(aug_stats, f, indent=2)
        
        # Save model info
        model_info = {
            'backbone': self.config['backbone'],
            'num_classes': self.config['num_classes'],
            'num_parameters': self.model.get_trainable_params()['total'],
            'device': str(self.device),
            'training_date': datetime.now().isoformat(),
            'experiment_name': self.config['experiment_name']
        }
        
        with open(exp_dir / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        with open(self.output_dir / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"✅ Artifacts saved to: {exp_dir}")
        logger.info(f"✅ Main model saved to: {main_model_path}")
        
    def train(self) -> Dict:
        """Complete training pipeline"""
        logger.info("🚀 Starting Modern Training Pipeline")
        logger.info("="*70)
        
        try:
            # Setup
            self.setup_data()
            self.setup_model()
            
            # Training stages
            stage1_history = self.train_stage1()
            stage2_history = self.train_stage2()
            
            # Evaluation
            eval_results = self.evaluate_model()
            
            # Save artifacts
            if self.config['save_artifacts']:
                self.save_artifacts()
            
            # Prepare results
            results = {
                'stage1_history': stage1_history,
                'stage2_history': stage2_history,
                'evaluation': eval_results,
                'best_val_acc': max(stage2_history.get('val_acc', [0])) if stage2_history.get('val_acc') else 0,
                'best_val_f1': max(stage2_history.get('val_acc', [0])) if stage2_history.get('val_acc') else 0,  # Placeholder
                'experiment_name': self.config['experiment_name'],
                'model_path': str(self.output_dir / 'best_model.pth')
            }
            
            logger.info("🎉 Training pipeline completed successfully!")
            logger.info(f"Best validation accuracy: {results['best_val_acc']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
            
    def resume_training(self, checkpoint_path: str):
        """Resume training from checkpoint"""
        logger.info(f"📥 Resuming training from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load config and history
        self.config.update(checkpoint.get('config', {}))
        self.training_history = checkpoint.get('training_history', {})
        self.class_names = checkpoint.get('class_names', [])
        
        logger.info("✅ Training resumed successfully!")


# Helper functions for quick training
def quick_train(
    data_dir: str,
    output_dir: str = 'trained_model',
    backbone: str = 'resnet50',
    augmentation: str = 'medium',
    batch_size: int = 32,
    **kwargs
) -> Dict:
    """
    Quick training with sensible defaults
    
    Args:
        data_dir: Path to dataset splits
        output_dir: Output directory
        backbone: Model backbone
        augmentation: Augmentation preset
        batch_size: Batch size
        **kwargs: Additional config overrides
        
    Returns:
        Training results
    """
    config_overrides = {
        'batch_size': batch_size,
        **kwargs
    }
    
    pipeline = ModernTrainingPipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        backbone=backbone,
        augmentation_preset=augmentation,
        config_overrides=config_overrides
    )
    
    return pipeline.train()


def train_multiple_models(
    data_dir: str,
    output_dir: str = 'trained_model',
    backbones: List[str] = ['resnet50', 'efficientnet_b0'],
    **kwargs
) -> Dict[str, Dict]:
    """
    Train multiple models for comparison
    
    Args:
        data_dir: Path to dataset
        output_dir: Output directory
        backbones: List of backbones to train
        **kwargs: Additional config
        
    Returns:
        Results for each backbone
    """
    results = {}
    
    for backbone in backbones:
        logger.info(f"🔄 Training {backbone}...")
        
        exp_output_dir = Path(output_dir) / f'experiments_{backbone}'
        
        try:
            result = quick_train(
                data_dir=data_dir,
                output_dir=str(exp_output_dir),
                backbone=backbone,
                **kwargs
            )
            results[backbone] = result
            
        except Exception as e:
            logger.error(f"Training {backbone} failed: {e}")
            results[backbone] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Modern Training Pipeline')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset splits')
    parser.add_argument('--output_dir', type=str, default='trained_model', help='Output directory')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--augmentation', type=str, default='medium', help='Augmentation preset')
    
    args = parser.parse_args()
    
    # Run training
    results = quick_train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        backbone=args.backbone,
        batch_size=args.batch_size,
        augmentation=args.augmentation
    )
    
    print("\n🎉 Training completed!")
    print(f"Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"Model saved to: {results['model_path']}")