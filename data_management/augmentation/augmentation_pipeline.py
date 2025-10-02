"""
🔄 Complete Augmentation Pipeline
================================

High-level augmentation pipeline that combines all techniques:
- On-the-fly augmentation during training
- MixUp/CutMix collation
- Quality checks
- Configurable pipelines

Usage:
    pipeline = AugmentationPipeline(config)
    dataloader = pipeline.create_dataloader(dataset)
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
import numpy as np
from pathlib import Path
import json

from .advanced_augmentation import (
    MixUpAugmentation,
    CutMixAugmentation,
    RandAugmentPipeline,
    RandomErasingTransform,
    MixUpCutMixCollator,
    create_training_augmentation,
    create_validation_transform
)


class AugmentationPipeline:
    """
    Complete augmentation pipeline manager
    
    Features:
    - Automatic pipeline creation
    - MixUp/CutMix support
    - Configurable parameters
    - Training vs validation modes
    
    Args:
        config: Configuration dictionary or path to config file
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Default configuration
        self.config = {
            # Image parameters
            'image_size': 224,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            
            # RandAugment parameters
            'rand_augment_n': 2,
            'rand_augment_m': 9,
            'rand_augment_enabled': True,
            
            # RandomErasing parameters
            'random_erasing_p': 0.5,
            'random_erasing_scale': (0.02, 0.33),
            'random_erasing_ratio': (0.3, 3.3),
            'random_erasing_enabled': True,
            
            # MixUp parameters
            'mixup_alpha': 0.2,
            'mixup_enabled': True,
            
            # CutMix parameters
            'cutmix_alpha': 1.0,
            'cutmix_enabled': True,
            
            # MixUp/CutMix switching
            'mixup_cutmix_prob': 0.5,  # 50% MixUp, 50% CutMix
            'mixup_cutmix_switch_prob': 1.0,  # Always apply one of them
            
            # DataLoader parameters
            'batch_size': 32,
            'num_workers': 4,
            'pin_memory': True,
            
            # Number of classes
            'num_classes': 6
        }
        
        # Update with provided config
        if config:
            if isinstance(config, (str, Path)):
                config = self._load_config(config)
            self.config.update(config)
            
        # Create transforms
        self.train_transform = None
        self.val_transform = None
        self._build_transforms()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
            
    def _build_transforms(self):
        """Build training and validation transforms"""
        # Training transform
        if self.config['rand_augment_enabled']:
            self.train_transform = create_training_augmentation(
                image_size=self.config['image_size'],
                mean=self.config['mean'],
                std=self.config['std'],
                rand_augment_n=self.config['rand_augment_n'],
                rand_augment_m=self.config['rand_augment_m'],
                random_erasing_p=self.config['random_erasing_p'] if self.config['random_erasing_enabled'] else 0.0
            )
        else:
            # Basic transform without RandAugment
            import torchvision.transforms as transforms
            transform_list = [
                transforms.Resize((self.config['image_size'], self.config['image_size'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config['mean'], std=self.config['std'])
            ]
            if self.config['random_erasing_enabled']:
                transform_list.append(
                    RandomErasingTransform(p=self.config['random_erasing_p'])
                )
            self.train_transform = transforms.Compose(transform_list)
            
        # Validation transform (no augmentation)
        self.val_transform = create_validation_transform(
            image_size=self.config['image_size'],
            mean=self.config['mean'],
            std=self.config['std']
        )
        
    def create_collate_fn(self, mode: str = 'train'):
        """
        Create collate function for DataLoader
        
        Args:
            mode: 'train' or 'val'
            
        Returns:
            Collate function
        """
        if mode == 'train' and (self.config['mixup_enabled'] or self.config['cutmix_enabled']):
            return MixUpCutMixCollator(
                mixup_alpha=self.config['mixup_alpha'] if self.config['mixup_enabled'] else 0,
                cutmix_alpha=self.config['cutmix_alpha'] if self.config['cutmix_enabled'] else 0,
                prob=self.config['mixup_cutmix_prob'],
                num_classes=self.config['num_classes']
            )
        else:
            # Default collate
            return None
            
    def create_dataloader(
        self,
        dataset,
        mode: str = 'train',
        shuffle: Optional[bool] = None,
        sampler: Optional[Any] = None,
        **kwargs
    ):
        """
        Create DataLoader with appropriate augmentation
        
        Args:
            dataset: PyTorch Dataset
            mode: 'train' or 'val'
            shuffle: Shuffle data (default: True for train, False for val)
            sampler: Optional sampler (e.g., WeightedRandomSampler)
            **kwargs: Additional DataLoader arguments
            
        Returns:
            torch.utils.data.DataLoader
        """
        # Set default shuffle
        if shuffle is None:
            shuffle = (mode == 'train')
            
        # Get collate function
        collate_fn = self.create_collate_fn(mode)
        
        # Create DataLoader
        loader_kwargs = {
            'batch_size': self.config['batch_size'],
            'shuffle': shuffle if sampler is None else False,
            'num_workers': self.config['num_workers'],
            'pin_memory': self.config['pin_memory'],
            'sampler': sampler,
            'collate_fn': collate_fn
        }
        loader_kwargs.update(kwargs)
        
        return DataLoader(dataset, **loader_kwargs)
        
    def get_transform(self, mode: str = 'train'):
        """
        Get transform for dataset
        
        Args:
            mode: 'train' or 'val'
            
        Returns:
            Transform function
        """
        if mode == 'train':
            return self.train_transform
        else:
            return self.val_transform
            
    def update_config(self, **kwargs):
        """Update configuration"""
        self.config.update(kwargs)
        self._build_transforms()
        
    def save_config(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4)
            
    def get_augmentation_stats(self) -> Dict[str, Any]:
        """Get augmentation statistics"""
        return {
            'image_size': self.config['image_size'],
            'rand_augment_enabled': self.config['rand_augment_enabled'],
            'rand_augment_ops': self.config['rand_augment_n'],
            'rand_augment_magnitude': self.config['rand_augment_m'],
            'random_erasing_enabled': self.config['random_erasing_enabled'],
            'random_erasing_prob': self.config['random_erasing_p'],
            'mixup_enabled': self.config['mixup_enabled'],
            'mixup_alpha': self.config['mixup_alpha'],
            'cutmix_enabled': self.config['cutmix_enabled'],
            'cutmix_alpha': self.config['cutmix_alpha'],
        }
        
    def __repr__(self):
        stats = self.get_augmentation_stats()
        return f"AugmentationPipeline({stats})"


# Preset configurations
PRESET_CONFIGS = {
    'light': {
        'rand_augment_n': 2,
        'rand_augment_m': 5,
        'random_erasing_p': 0.25,
        'mixup_alpha': 0.1,
        'cutmix_alpha': 0.5,
    },
    'medium': {
        'rand_augment_n': 2,
        'rand_augment_m': 9,
        'random_erasing_p': 0.5,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 1.0,
    },
    'heavy': {
        'rand_augment_n': 3,
        'rand_augment_m': 12,
        'random_erasing_p': 0.7,
        'mixup_alpha': 0.4,
        'cutmix_alpha': 1.0,
    },
    'minimal': {
        'rand_augment_enabled': False,
        'random_erasing_enabled': False,
        'mixup_enabled': False,
        'cutmix_enabled': False,
    }
}


def create_pipeline_from_preset(preset: str = 'medium', **kwargs):
    """
    Create pipeline from preset configuration
    
    Args:
        preset: 'light', 'medium', 'heavy', or 'minimal'
        **kwargs: Additional config overrides
        
    Returns:
        AugmentationPipeline
    """
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(PRESET_CONFIGS.keys())}")
        
    config = PRESET_CONFIGS[preset].copy()
    config.update(kwargs)
    
    return AugmentationPipeline(config)


# Example usage
if __name__ == "__main__":
    # Create pipeline with medium preset
    pipeline = create_pipeline_from_preset('medium', batch_size=64)
    
    print("Pipeline configuration:")
    print(pipeline.get_augmentation_stats())
    
    # Get transforms
    train_transform = pipeline.get_transform('train')
    val_transform = pipeline.get_transform('val')
    
    print("\nTrain transform:", train_transform)
    print("Val transform:", val_transform)
