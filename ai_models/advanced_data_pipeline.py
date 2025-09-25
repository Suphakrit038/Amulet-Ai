"""
Advanced Data Pipeline for Amulet-AI
ระบบการประมวลผลข้อมูลขั้นสูงสำหรับการจดจำพระเครื่อง
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataPipelineConfig:
    """Configuration for data pipeline"""
    data_path: str = "ai_models/dataset_split"
    batch_size: int = 16
    num_workers: int = 4
    image_size: int = 224
    normalize_mean: List[float] = None
    normalize_std: List[float] = None
    
    def __post_init__(self):
        if self.normalize_mean is None:
            self.normalize_mean = [0.485, 0.456, 0.406]
        if self.normalize_std is None:
            self.normalize_std = [0.229, 0.224, 0.225]

class AmuletDataset(Dataset):
    """Dataset class for amulet images"""
    
    def __init__(self, data_path: str, transform=None, split: str = "train"):
        self.data_path = Path(data_path)
        self.transform = transform
        self.split = split
        
        # Load data
        self.samples = []
        self.class_to_idx = {}
        self._load_data()
    
    def _load_data(self):
        """Load data from directory structure"""
        split_path = self.data_path / self.split
        
        if not split_path.exists():
            raise FileNotFoundError(f"Split path not found: {split_path}")
        
        # Get class directories
        class_dirs = [d for d in split_path.iterdir() if d.is_dir()]
        class_dirs.sort()
        
        # Create class mapping
        self.class_to_idx = {cls_dir.name: idx for idx, cls_dir in enumerate(class_dirs)}
        
        # Load samples
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            
            # Get all image files
            for img_path in class_dir.glob("*.jpg"):
                if img_path.is_file():
                    self.samples.append((str(img_path), class_idx))
            
            for img_path in class_dir.glob("*.png"):
                if img_path.is_file():
                    self.samples.append((str(img_path), class_idx))
        
        logger.info(f"Loaded {len(self.samples)} samples for {self.split} split")
        logger.info(f"Found {len(self.class_to_idx)} classes: {list(self.class_to_idx.keys())}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

class AdvancedDataPipeline:
    """Advanced data pipeline for training"""
    
    def __init__(self, config: DataPipelineConfig):
        self.config = config
        self.datasets = {}
        self.dataloaders = {}
        
        # Setup transforms
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup data transforms"""
        # Training transforms with augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])
        
        # Validation/test transforms without augmentation
        self.val_transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])
    
    def setup_datasets(self):
        """Setup train/val/test datasets"""
        # Training dataset
        self.datasets['train'] = AmuletDataset(
            self.config.data_path,
            transform=self.train_transform,
            split='train'
        )
        
        # Validation dataset
        val_path = Path(self.config.data_path) / 'validation'
        if val_path.exists():
            self.datasets['val'] = AmuletDataset(
                self.config.data_path,
                transform=self.val_transform,
                split='validation'
            )
        
        # Test dataset
        test_path = Path(self.config.data_path) / 'test'
        if test_path.exists():
            self.datasets['test'] = AmuletDataset(
                self.config.data_path,
                transform=self.val_transform,
                split='test'
            )
        
        return self.datasets
    
    def setup_dataloaders(self):
        """Setup data loaders"""
        if not self.datasets:
            self.setup_datasets()
        
        for split, dataset in self.datasets.items():
            shuffle = (split == 'train')
            self.dataloaders[split] = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
        
        return self.dataloaders
    
    def get_class_mapping(self):
        """Get class index mapping"""
        if 'train' in self.datasets:
            return self.datasets['train'].class_to_idx
        return {}
    
    def get_dataset_info(self):
        """Get dataset information"""
        info = {
            'num_classes': len(self.get_class_mapping()),
            'class_mapping': self.get_class_mapping(),
            'splits': {}
        }
        
        for split, dataset in self.datasets.items():
            info['splits'][split] = len(dataset)
        
        return info

def create_data_pipeline(config: Optional[DataPipelineConfig] = None) -> AdvancedDataPipeline:
    """Create data pipeline with default configuration"""
    if config is None:
        config = DataPipelineConfig()
    
    return AdvancedDataPipeline(config)