"""
🎯 Advanced Transfer Learning for Amulet Classification
ระบบฝึกฝนโมเดลด้วย Transfer Learning สำหรับการจำแนกพระเครื่อง
"""
import os
import sys
import numpy as np
import json
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights, EfficientNet_B3_Weights

from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.manifold import TSNE

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger("transfer_learning")

# Configuration
@dataclass
class TransferLearningConfig:
    """Configuration for transfer learning model training"""
    # Data configuration
    data_path: str = "ai_models/dataset_split"  # Uses the dataset split structure
    img_size: int = 224  # Input image size
    
    # Model configuration
    model_type: str = "efficientnet_b3"  # Options: efficientnet_b0, efficientnet_b3, resnet50
    num_classes: int = 10  # Will be updated based on actual dataset
    freeze_backbone: bool = True  # Initially freeze backbone for head-only training
    use_pretrained: bool = True  # Use pretrained weights
    
    # Training configuration
    batch_size: int = 16
    num_epochs: int = 50
    head_only_epochs: int = 10  # Train only the head for these epochs
    learning_rate: float = 1e-3
    head_learning_rate: float = 1e-2  # Higher LR for head-only training
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    
    # Early stopping
    patience: int = 7
    min_delta: float = 0.001
    
    # Class weights for handling class imbalance
    use_class_weights: bool = True
    
    # Data augmentation
    use_augmentation: bool = True
    # No rotate or flip as specified
    use_rotate: bool = False
    use_flip: bool = False
    
    # Hardware configuration
    use_cuda: bool = True
    mixed_precision: bool = True
    num_workers: int = 4
    
    # Output configuration
    output_dir: str = "training_output"
    save_best_only: bool = True
    log_interval: int = 5
    
    # Evaluation configuration
    eval_train: bool = True  # Whether to evaluate on training set too
    save_confusion_matrix: bool = True
    
    # Custom handling for small classes
    small_class_threshold: int = 20  # Classes with samples less than this are considered small
    small_class_strategy: str = "weighted_loss"  # Options: weighted_loss, oversample, all_train

class AmuletDataset(Dataset):
    """Custom dataset for loading amulet images with advanced preprocessing"""
    
    def __init__(self, 
                 image_paths: List[str], 
                 labels: List[int], 
                 transform=None, 
                 class_names: List[str] = None,
                 metadata: List[Dict] = None):
        """
        Initialize the dataset
        
        Args:
            image_paths: List of paths to images
            labels: List of class indices
            transform: Image transformations
            class_names: List of class names
            metadata: List of metadata dictionaries for each sample
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names
        self.metadata = metadata or [{}] * len(image_paths)
        
        # Validate
        assert len(self.image_paths) == len(self.labels), "Paths and labels must have same length"
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load image - handle errors gracefully
            from PIL import Image
            
            # Handle potential image loading issues
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                logger.warning(f"Error loading {img_path}: {e}, using black image instead")
                img = Image.new('RGB', (224, 224), color=0)
            
            # Apply transformations
            if self.transform:
                img = self.transform(img)
            
            # Create sample with metadata
            sample = {
                'image': img,
                'label': label,
                'path': img_path,
                'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
            }
            
            return sample
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            # Return a placeholder to keep the dataloader running
            import torch
            return {
                'image': torch.zeros((3, 224, 224)),
                'label': label,
                'path': img_path,
                'metadata': {'error': str(e)}
            }

class TransferLearningModel(nn.Module):
    """Transfer learning model for amulet classification"""
    
    def __init__(self, config: TransferLearningConfig):
        """Initialize the model"""
        super(TransferLearningModel, self).__init__()
        self.config = config
        
        # Initialize backbone based on config
        if config.model_type.startswith('efficientnet'):
            self._init_efficientnet_backbone()
        elif config.model_type == 'resnet50':
            self._init_resnet_backbone()
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        # Freeze backbone if required
        self._set_backbone_freeze_state(config.freeze_backbone)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model initialized: {config.model_type}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.1%})")
    
    def _init_efficientnet_backbone(self):
        """Initialize EfficientNet backbone"""
        if self.config.model_type == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.DEFAULT if self.config.use_pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif self.config.model_type == 'efficientnet_b3':
            weights = EfficientNet_B3_Weights.DEFAULT if self.config.use_pretrained else None
            self.backbone = models.efficientnet_b3(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported EfficientNet type: {self.config.model_type}")
        
        # Create classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, self.config.num_classes)
        )
    
    def _init_resnet_backbone(self):
        """Initialize ResNet backbone"""
        weights = ResNet50_Weights.DEFAULT if self.config.use_pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Create classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, self.config.num_classes)
        )
    
    def _set_backbone_freeze_state(self, freeze: bool):
        """Set the freeze state of the backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
    
    def unfreeze_last_layers(self):
        """Unfreeze the last few layers of the backbone for fine-tuning"""
        # For EfficientNet, unfreeze the last 2 stages
        if self.config.model_type.startswith('efficientnet'):
            for name, param in self.backbone.named_parameters():
                if 'features.6' in name or 'features.7' in name or 'features.8' in name:
                    param.requires_grad = True
        
        # For ResNet, unfreeze the last layer (layer4)
        elif self.config.model_type == 'resnet50':
            for name, param in self.backbone.named_parameters():
                if 'layer4' in name:
                    param.requires_grad = True
        
        # Log the number of trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"After unfreezing: Trainable parameters: {trainable_params:,}")
    
    def unfreeze_all(self):
        """Unfreeze all backbone layers for full fine-tuning"""
        self._set_backbone_freeze_state(False)
        logger.info("Unfrozen all backbone layers for fine-tuning")
    
    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)
        return self.classifier(features)

class TransferLearningTrainer:
    """Trainer for transfer learning model"""
    
    def __init__(self, config: TransferLearningConfig):
        """Initialize the trainer"""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')
        
        # Setup output directories
        self.setup_directories()
        
        # Training state
        self.best_val_metric = 0.0
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.current_epoch = 0
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rates': [],
            'confusion_matrices': []
        }
        
        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard'))
        
        logger.info(f"Trainer initialized on {self.device}")
    
    def setup_directories(self):
        """Setup output directories"""
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
        
        logger.info(f"Output directory setup: {self.output_dir}")
    
    def prepare_data(self):
        """Prepare datasets and dataloaders"""
        logger.info("Preparing datasets...")
        
        # Get data path
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        # Load labels.json to get class mapping
        labels_path = data_path / "labels.json"
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.class_mapping = json.load(f)
        
        # Update num_classes in config
        self.config.num_classes = len(self.class_mapping)
        logger.info(f"Found {self.config.num_classes} classes")
        
        # Check for class_info.json for small class handling
        class_info_path = data_path / "class_info.json"
        self.small_classes = []
        if class_info_path.exists():
            with open(class_info_path, 'r', encoding='utf-8') as f:
                class_info = json.load(f)
                
                # Identify small classes
                for class_name, info in class_info.items():
                    if info.get('total_files', 0) < self.config.small_class_threshold:
                        self.small_classes.append(class_name)
                        logger.info(f"Small class detected: {class_name} with {info.get('total_files')} samples")
        
        # Create transformations
        self.train_transform = self._create_train_transform()
        self.val_transform = self._create_val_transform()
        
        # Load datasets
        self.datasets = {}
        self.dataloaders = {}
        
        # Load training data
        self.datasets['train'] = self._load_split_data('train', self.train_transform)
        self.dataloaders['train'] = DataLoader(
            self.datasets['train'],
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        # Load validation data
        self.datasets['val'] = self._load_split_data('validation', self.val_transform)
        self.dataloaders['val'] = DataLoader(
            self.datasets['val'],
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        # Load test data
        self.datasets['test'] = self._load_split_data('test', self.val_transform)
        self.dataloaders['test'] = DataLoader(
            self.datasets['test'],
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        # Calculate class weights for handling imbalance
        if self.config.use_class_weights:
            self.class_weights = self._calculate_class_weights()
        
        logger.info("Data preparation completed")
        return self.dataloaders
    
    def _create_train_transform(self):
        """Create training data transformations"""
        transform_list = [
            transforms.Resize((self.config.img_size, self.config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        # Add augmentations if enabled
        if self.config.use_augmentation:
            aug_list = [
                transforms.RandomResizedCrop(self.config.img_size, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
            ]
            
            # Only add rotation if explicitly enabled
            if self.config.use_rotate:
                aug_list.append(transforms.RandomRotation(15))
            
            # Only add horizontal flip if explicitly enabled
            if self.config.use_flip:
                aug_list.append(transforms.RandomHorizontalFlip())
            
            # Insert augmentations at the beginning of the list
            transform_list = aug_list + transform_list
        
        return transforms.Compose(transform_list)
    
    def _create_val_transform(self):
        """Create validation data transformations"""
        return transforms.Compose([
            transforms.Resize((self.config.img_size, self.config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_split_data(self, split_name: str, transform):
        """Load data for a specific split"""
        split_dir = Path(self.config.data_path) / split_name
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            return None
        
        image_paths = []
        labels = []
        metadata = []
        
        # Iterate through class directories
        for class_idx, class_name in self.class_mapping.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Handle small classes based on strategy
            is_small_class = class_name in self.small_classes
            if is_small_class and self.config.small_class_strategy == 'all_train' and split_name != 'train':
                logger.info(f"Skipping small class {class_name} in {split_name} split (all samples in train)")
                continue
            
            # Get all image files
            image_files = [f for f in class_dir.glob('**/*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
            # Log the number of images found
            logger.info(f"Found {len(image_files)} images for class {class_name} in {split_name}")
            
            for img_path in image_files:
                image_paths.append(str(img_path))
                labels.append(int(class_idx))
                
                # Add metadata
                meta = {
                    'class_name': class_name,
                    'split': split_name,
                    'view_type': self._determine_view_type(img_path),
                    'is_small_class': is_small_class
                }
                metadata.append(meta)
        
        logger.info(f"Loaded {len(image_paths)} images for {split_name} split")
        
        # Create dataset
        class_names = [self.class_mapping.get(str(i), f"Class_{i}") for i in range(self.config.num_classes)]
        return AmuletDataset(image_paths, labels, transform, class_names, metadata)
    
    def _determine_view_type(self, img_path: Path) -> str:
        """Determine view type from filename"""
        filename = img_path.name.lower()
        
        if 'front' in filename or '-f' in filename:
            return 'front'
        elif 'back' in filename or '-b' in filename:
            return 'back'
        else:
            return 'unknown'
    
    def _calculate_class_weights(self):
        """Calculate class weights for handling imbalance"""
        if 'train' not in self.datasets:
            return None
        
        # Count samples per class
        class_counts = torch.zeros(self.config.num_classes)
        for label in [item['label'] for item in self.datasets['train']]:
            class_counts[label] += 1
        
        # Add small offset to avoid division by zero
        class_counts = torch.max(class_counts, torch.ones_like(class_counts))
        
        # Inverse frequency weighting
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * self.config.num_classes
        
        # Extra weight for small classes
        if self.small_classes:
            for class_name in self.small_classes:
                class_idx = [idx for idx, name in self.class_mapping.items() if name == class_name]
                if class_idx:
                    idx = int(class_idx[0])
                    weights[idx] *= 1.5  # Extra weight
        
        logger.info(f"Class weights: {weights}")
        return weights.to(self.device)
    
    def initialize_model(self):
        """Initialize the model"""
        logger.info("Initializing model...")
        self.model = TransferLearningModel(self.config)
        self.model = self.model.to(self.device)
        
        return self.model
    
    def setup_training(self):
        """Setup optimizers and schedulers for training"""
        logger.info("Setting up training components...")
        
        # Use different learning rates for head and backbone
        if self.config.freeze_backbone:
            # Only train the classifier head initially
            self.optimizer = optim.AdamW(
                self.model.classifier.parameters(),
                lr=self.config.head_learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            # Train all parameters with separate learning rates
            self.optimizer = optim.AdamW([
                {'params': self.model.backbone.parameters(), 'lr': self.config.learning_rate},
                {'params': self.model.classifier.parameters(), 'lr': self.config.head_learning_rate}
            ], weight_decay=self.config.weight_decay)
        
        # Create loss function
        if self.config.use_class_weights and hasattr(self, 'class_weights'):
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            logger.info("Using weighted CrossEntropyLoss")
        else:
            self.criterion = nn.CrossEntropyLoss()
            logger.info("Using standard CrossEntropyLoss")
        
        # Initialize mixed precision scaler if needed
        self.scaler = GradScaler() if self.config.mixed_precision else None
        
        logger.info("Training setup completed")
    
    def train(self, resume_path=None):
        """Execute the full training loop"""
        logger.info("Starting training...")
        
        if resume_path:
            self._load_checkpoint(resume_path)
        
        # Phase 1: Train only the head
        if self.config.freeze_backbone:
            logger.info("Phase 1: Training only the classifier head...")
            
            # Configure optimizer for head-only training
            self.optimizer = optim.AdamW(
                self.model.classifier.parameters(),
                lr=self.config.head_learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            # Train for specified number of head-only epochs
            for epoch in range(self.config.head_only_epochs):
                logger.info(f"Head-only epoch {epoch+1}/{self.config.head_only_epochs}")
                
                # Training
                train_metrics = self._train_epoch(epoch)
                
                # Validation
                val_metrics = self._validate_epoch(epoch)
                
                # Update learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Log metrics
                self._log_metrics(epoch, train_metrics, val_metrics, current_lr, phase="head")
                
                # Save checkpoint
                is_best = self._save_checkpoint(epoch, train_metrics, val_metrics, phase="head")
                
                # Update epoch counter
                self.current_epoch += 1
            
            # Prepare for fine-tuning
            logger.info("Head-only training completed, preparing for fine-tuning...")
            
            # Reset early stopping counter
            self.early_stop_counter = 0
        
        # Phase 2: Fine-tune the model gradually
        logger.info("Phase 2: Fine-tuning the model...")
        
        # First unfreeze last layers
        self.model.unfreeze_last_layers()
        
        # Configure optimizer for fine-tuning with lower learning rate
        self.optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Create scheduler with warmup and cosine annealing
        total_steps = (self.config.num_epochs - self.current_epoch) * len(self.dataloaders['train'])
        warmup_steps = self.config.warmup_epochs * len(self.dataloaders['train'])
        
        # Linear warmup followed by cosine annealing
        warmup_scheduler = LinearLR(
            self.optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.learning_rate * 0.01
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        # Continue training for remaining epochs
        for epoch in range(self.current_epoch, self.config.num_epochs):
            logger.info(f"Fine-tuning epoch {epoch+1}/{self.config.num_epochs}")
            
            # Unfreeze all layers after additional epochs
            if epoch == self.current_epoch + 5:  # After 5 more epochs of fine-tuning
                logger.info("Unfreezing all layers for final fine-tuning...")
                self.model.unfreeze_all()
            
            # Training
            train_metrics = self._train_epoch(epoch)
            
            # Validation
            val_metrics = self._validate_epoch(epoch)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics, current_lr, phase="finetune")
            
            # Save checkpoint
            is_best = self._save_checkpoint(epoch, train_metrics, val_metrics, phase="finetune")
            
            # Check for early stopping
            if self._check_early_stopping(val_metrics['loss']):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # Update epoch counter
            self.current_epoch += 1
        
        # Final evaluation
        self._final_evaluation()
        
        # Close tensorboard writer
        self.writer.close()
        
        logger.info("Training completed!")
        return self.history
    
    def _train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch+1} (Train)")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward and backward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Update learning rate if scheduler exists
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            # Store predictions for metrics calculation
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds, total_loss, len(self.dataloaders['train']))
        
        return metrics
    
    def _validate_epoch(self, epoch: int) -> Dict:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.dataloaders['val'], desc=f"Epoch {epoch+1} (Val)")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Get data
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                
                # Store predictions for metrics calculation
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({'loss': f"{avg_loss:.4f}"})
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds, total_loss, len(self.dataloaders['val']))
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=range(self.config.num_classes))
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def _calculate_metrics(self, labels, predictions, total_loss, num_batches) -> Dict:
        """Calculate metrics from predictions and labels"""
        # Calculate accuracy
        accuracy = np.mean(np.array(predictions) == np.array(labels))
        
        # Calculate F1 score (weighted due to potential class imbalance)
        f1 = f1_score(labels, predictions, average='weighted')
        
        # Calculate precision and recall
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict, 
                    current_lr: float, phase: str):
        """Log metrics to console and tensorboard"""
        # Update history
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_acc'].append(train_metrics['accuracy'])
        self.history['train_f1'].append(train_metrics['f1'])
        self.history['learning_rates'].append(current_lr)
        
        if val_metrics:
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            
            if 'confusion_matrix' in val_metrics:
                self.history['confusion_matrices'].append(val_metrics['confusion_matrix'])
        
        # Log to tensorboard
        step = len(self.history['train_loss'])
        self.writer.add_scalar(f'{phase}/Train/Loss', train_metrics['loss'], step)
        self.writer.add_scalar(f'{phase}/Train/Accuracy', train_metrics['accuracy'], step)
        self.writer.add_scalar(f'{phase}/Train/F1', train_metrics['f1'], step)
        self.writer.add_scalar(f'{phase}/LR', current_lr, step)
        
        if val_metrics:
            self.writer.add_scalar(f'{phase}/Val/Loss', val_metrics['loss'], step)
            self.writer.add_scalar(f'{phase}/Val/Accuracy', val_metrics['accuracy'], step)
            self.writer.add_scalar(f'{phase}/Val/F1', val_metrics['f1'], step)
            
            if 'confusion_matrix' in val_metrics and epoch % 5 == 0:
                # Plot and log confusion matrix periodically
                cm = val_metrics['confusion_matrix']
                fig = plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - Epoch {epoch+1}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                
                self.writer.add_figure(f'{phase}/Val/ConfusionMatrix', fig, step)
                
                # Save confusion matrix visualization
                if self.config.save_confusion_matrix:
                    cm_path = self.output_dir / 'visualizations' / f'cm_epoch_{epoch+1}.png'
                    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                plt.close()
        
        # Log to console
        logger.info(
            f"Epoch {epoch+1} ({phase}): "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Train F1: {train_metrics['f1']:.4f}, "
            f"LR: {current_lr:.2e}"
        )
        
        if val_metrics:
            logger.info(
                f"Epoch {epoch+1} ({phase}): "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}"
            )
    
    def _save_checkpoint(self, epoch: int, train_metrics: Dict, val_metrics: Dict, phase: str) -> bool:
        """Save model checkpoint"""
        # Determine if this is the best model based on validation metrics
        current_metric = val_metrics['f1']  # Use F1 score as the primary metric
        current_loss = val_metrics['loss']
        
        is_best_metric = current_metric > self.best_val_metric
        is_best_loss = current_loss < self.best_val_loss
        
        if is_best_metric:
            self.best_val_metric = current_metric
        
        if is_best_loss:
            self.best_val_loss = current_loss
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_metric': self.best_val_metric,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': asdict(self.config)
        }
        
        # Add scheduler state if it exists
        if hasattr(self, 'scheduler'):
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / 'models' / f'{phase}_checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model (by metric)
        if is_best_metric:
            best_model_path = self.output_dir / 'models' / f'{phase}_best_model_metric.pth'
            torch.save(checkpoint, best_model_path)
            logger.info(f"Saved new best model (by F1 score): {current_metric:.4f}")
        
        # Save best model (by loss)
        if is_best_loss:
            best_model_path = self.output_dir / 'models' / f'{phase}_best_model_loss.pth'
            torch.save(checkpoint, best_model_path)
            logger.info(f"Saved new best model (by loss): {current_loss:.4f}")
        
        return is_best_metric or is_best_loss
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping should be triggered"""
        if val_loss > self.best_val_loss - self.config.min_delta:
            self.early_stop_counter += 1
        else:
            self.early_stop_counter = 0
        
        return self.early_stop_counter >= self.config.patience
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load a model checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if it exists
        if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_metric = checkpoint.get('best_val_metric', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Load history if it exists
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        logger.info(f"Checkpoint loaded. Resuming from epoch {self.current_epoch}")
    
    def _final_evaluation(self):
        """Perform final evaluation on test set"""
        logger.info("Performing final evaluation on test set...")
        
        # Load best model
        best_model_path = self.output_dir / 'models' / 'finetune_best_model_metric.pth'
        if best_model_path.exists():
            self._load_checkpoint(str(best_model_path))
            logger.info("Loaded best model for evaluation")
        
        # Evaluate on test set
        self.model.eval()
        all_preds = []
        all_labels = []
        all_paths = []
        
        with torch.no_grad():
            for batch in tqdm(self.dataloaders['test'], desc="Evaluating on test set"):
                images = batch['image'].to(self.device)
                labels = batch['label']
                paths = batch['path']
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_paths.extend(paths)
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Calculate per-class metrics
        class_names = [self.class_mapping.get(str(i), f"Class_{i}") for i in range(self.config.num_classes)]
        class_report = classification_report(
            all_labels, all_preds, 
            labels=range(self.config.num_classes),
            target_names=class_names,
            output_dict=True
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=range(self.config.num_classes))
        
        # Create detailed results for each sample
        sample_results = []
        for i, (path, label, pred) in enumerate(zip(all_paths, all_labels, all_preds)):
            sample_results.append({
                'path': path,
                'true_label': int(label),
                'true_class': self.class_mapping.get(str(label), f"Class_{label}"),
                'predicted_label': int(pred),
                'predicted_class': self.class_mapping.get(str(pred), f"Class_{pred}"),
                'correct': label == pred
            })
        
        # Save results
        results = {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'class_report': class_report,
            'confusion_matrix': cm.tolist(),
            'test_samples': len(all_labels),
            'num_classes': self.config.num_classes,
            'model_type': self.config.model_type,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to JSON
        results_path = self.output_dir / 'reports' / 'test_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save detailed sample results to CSV
        sample_results_df = pd.DataFrame(sample_results)
        csv_path = self.output_dir / 'reports' / 'sample_predictions.csv'
        sample_results_df.to_csv(csv_path, index=False)
        
        # Create confusion matrix visualization
        self._create_confusion_matrix_plot(cm, class_names)
        
        # Create per-class metrics visualization
        self._create_class_metrics_plot(class_report)
        
        # Log summary
        logger.info(f"Final Test Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        
        # Create a comprehensive report
        self._create_final_report(results)
    
    def _create_confusion_matrix_plot(self, cm, class_names):
        """Create and save confusion matrix visualization"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save the figure
        cm_path = self.output_dir / 'visualizations' / 'final_confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_class_metrics_plot(self, class_report):
        """Create and save per-class metrics visualization"""
        # Extract class metrics
        classes = []
        precisions = []
        recalls = []
        f1_scores = []
        supports = []
        
        for class_name, metrics in class_report.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            
            classes.append(class_name)
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1-score'])
            supports.append(metrics['support'])
        
        # Create figure
        plt.figure(figsize=(14, 7))
        
        # Create bar chart
        x = np.arange(len(classes))
        width = 0.25
        
        plt.bar(x - width, precisions, width, label='Precision')
        plt.bar(x, recalls, width, label='Recall')
        plt.bar(x + width, f1_scores, width, label='F1 Score')
        
        # Add support information as text
        for i, support in enumerate(supports):
            plt.text(i, max(precisions[i], recalls[i], f1_scores[i]) + 0.05, 
                    f"n={support}", ha='center')
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Per-Class Metrics')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the figure
        metrics_path = self.output_dir / 'visualizations' / 'per_class_metrics.png'
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_final_report(self, results):
        """Create a comprehensive final report"""
        # Create training curves
        plt.figure(figsize=(15, 10))
        
        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot F1 score
        plt.subplot(2, 2, 3)
        plt.plot(self.history['train_f1'], label='Train F1')
        plt.plot(self.history['val_f1'], label='Validation F1')
        plt.title('F1 Score Curves')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(2, 2, 4)
        plt.plot(self.history['learning_rates'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        plt.tight_layout()
        curves_path = self.output_dir / 'visualizations' / 'training_curves.png'
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save final comprehensive report
        final_report = {
            'model_info': {
                'model_type': self.config.model_type,
                'num_classes': self.config.num_classes,
                'img_size': self.config.img_size,
                'parameters': sum(p.numel() for p in self.model.parameters())
            },
            'training_info': {
                'total_epochs': self.current_epoch,
                'best_val_metric': float(self.best_val_metric),
                'best_val_loss': float(self.best_val_loss),
                'early_stopping_triggered': self.early_stop_counter >= self.config.patience
            },
            'dataset_info': {
                'train_samples': len(self.datasets['train']) if 'train' in self.datasets else 0,
                'val_samples': len(self.datasets['val']) if 'val' in self.datasets else 0,
                'test_samples': len(self.datasets['test']) if 'test' in self.datasets else 0,
                'small_classes': self.small_classes
            },
            'test_results': results,
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save comprehensive report
        report_path = self.output_dir / 'reports' / 'FINAL_COMPREHENSIVE_REPORT.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False, default=str)
        
        # Create a summary text file
        summary_path = self.output_dir / 'EXECUTIVE_SUMMARY.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("AMULET-AI MODEL TRAINING EXECUTIVE SUMMARY\n")
            f.write("==========================================\n\n")
            f.write(f"Model: {self.config.model_type}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("MODEL PERFORMANCE\n")
            f.write("-----------------\n")
            f.write(f"Test Accuracy: {results['accuracy']:.2%}\n")
            f.write(f"Test F1 Score: {results['f1_score']:.4f}\n")
            f.write(f"Test Precision: {results['precision']:.4f}\n")
            f.write(f"Test Recall: {results['recall']:.4f}\n\n")
            
            f.write("TRAINING DETAILS\n")
            f.write("----------------\n")
            f.write(f"Total Epochs: {self.current_epoch}\n")
            f.write(f"Best Validation F1: {self.best_val_metric:.4f}\n")
            f.write(f"Best Validation Loss: {self.best_val_loss:.4f}\n\n")
            
            f.write("CLASS PERFORMANCE\n")
            f.write("-----------------\n")
            for class_name, metrics in results['class_report'].items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    f.write(f"{class_name}: F1={metrics['f1-score']:.4f}, "
                           f"Precision={metrics['precision']:.4f}, "
                           f"Recall={metrics['recall']:.4f}, "
                           f"Samples={metrics['support']}\n")
            
            f.write("\nRECOMMENDATIONS\n")
            f.write("--------------\n")
            
            # Add recommendations based on results
            low_performing_classes = [
                class_name for class_name, metrics in results['class_report'].items()
                if class_name not in ['accuracy', 'macro avg', 'weighted avg'] 
                and metrics['f1-score'] < 0.7
            ]
            
            if low_performing_classes:
                f.write(f"Focus on improving the following classes: {', '.join(low_performing_classes)}\n")
            
            if results['accuracy'] < 0.85:
                f.write("Consider further training with more data augmentation\n")
            
            if self.small_classes:
                f.write(f"Collect more samples for small classes: {', '.join(self.small_classes)}\n")
        
        logger.info(f"Final report and executive summary created")
        logger.info(f"Final test accuracy: {results['accuracy']:.2%}")

# Export model in production format
def export_for_production(model, output_path, class_mapping, img_size=224):
    """Export model for production use"""
    logger.info(f"Exporting model for production...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 3, img_size, img_size)
    
    # Export to TorchScript
    try:
        script_model = torch.jit.trace(model, example_input)
        script_path = Path(output_path) / "model_scripted.pt"
        script_model.save(str(script_path))
        logger.info(f"TorchScript model saved to {script_path}")
    except Exception as e:
        logger.error(f"Failed to export TorchScript model: {e}")
    
    # Export class mapping
    class_mapping_path = Path(output_path) / "production_class_mapping.json"
    with open(class_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)
    
    # Create a production info file
    info = {
        'model_type': model.config.model_type if hasattr(model, 'config') else "unknown",
        'input_size': img_size,
        'num_classes': len(class_mapping),
        'classes': class_mapping,
        'date_exported': datetime.now().isoformat(),
        'framework': f"PyTorch {torch.__version__}"
    }
    
    info_path = Path(output_path) / "PRODUCTION_MODEL_INFO.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Model exported for production use")
    return {
        'script_path': str(script_path),
        'class_mapping_path': str(class_mapping_path),
        'info_path': str(info_path)
    }

# Main function to run the complete process
def run_transfer_learning(config_override=None):
    """Run the complete transfer learning process"""
    # Create configuration
    config = TransferLearningConfig()
    
    # Apply overrides if provided
    if config_override:
        for key, value in config_override.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    logger.info(f"Starting transfer learning with {config.model_type}...")
    
    # Create trainer
    trainer = TransferLearningTrainer(config)
    
    # Prepare data
    trainer.prepare_data()
    
    # Initialize model
    model = trainer.initialize_model()
    
    # Setup training components
    trainer.setup_training()
    
    # Train model
    history = trainer.train()
    
    # Export model for production
    export_path = trainer.output_dir
    export_info = export_for_production(
        model, 
        export_path, 
        trainer.class_mapping, 
        config.img_size
    )
    
    logger.info("Transfer learning completed successfully!")
    return trainer, model, history

if __name__ == "__main__":
    # Example configuration overrides
    config_overrides = {
        "model_type": "efficientnet_b3",
        "batch_size": 16,
        "num_epochs": 50,
        "head_only_epochs": 10,
        "use_rotate": False,  # No rotation as specified
        "use_flip": False,    # No flipping as specified
        "data_path": "unified_dataset"  # Use unified dataset
    }
    
    # Run transfer learning
    trainer, model, history = run_transfer_learning(config_overrides)
