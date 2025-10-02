"""
Complete Training System for Amulet-AI

Implements production-ready trainer with:
- Mixed precision training (AMP)
- Gradient clipping
- Progress tracking
- Callback support
- Comprehensive logging

Author: Amulet-AI Team
Date: October 2, 2025
"""

import logging
from typing import Optional, Dict, Tuple, List, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from .callbacks import CallbackList, create_default_callbacks

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Configuration for training process.
    
    Parameters:
    -----------
    epochs : int
        Number of training epochs
    learning_rate : float
        Initial learning rate
    weight_decay : float
        L2 regularization weight
    gradient_clip_norm : float
        Maximum norm for gradient clipping (0 = disabled)
    mixed_precision : bool
        Enable automatic mixed precision (AMP)
    accumulation_steps : int
        Gradient accumulation steps (for large batch simulation)
    warmup_epochs : int
        Number of warmup epochs for learning rate
    save_dir : str
        Directory to save models and logs
    experiment_name : str
        Name of experiment
    device : str
        Device to use ('cuda' or 'cpu')
    """
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    accumulation_steps: int = 1
    warmup_epochs: int = 0
    save_dir: str = 'trained_model'
    experiment_name: Optional[str] = None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class AmuletTrainer:
    """
    Complete training system with callbacks, mixed precision, and monitoring.
    
    Features:
    ---------
    - Automatic mixed precision (AMP) for faster training
    - Gradient accumulation for effective large batch sizes
    - Gradient clipping for stability
    - Learning rate warmup
    - Progress bars with tqdm
    - Comprehensive logging
    - Callback support (early stopping, checkpointing, etc.)
    
    Example:
    --------
    >>> config = TrainingConfig(epochs=50, learning_rate=1e-3)
    >>> trainer = AmuletTrainer(model, criterion, optimizer, config)
    >>> history = trainer.fit(train_loader, val_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        callbacks: Optional[CallbackList] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Initialize trainer.
        
        Parameters:
        -----------
        model : nn.Module
            PyTorch model to train
        criterion : nn.Module
            Loss function
        optimizer : torch.optim.Optimizer
            Optimizer
        config : TrainingConfig
            Training configuration
        callbacks : CallbackList, optional
            List of callbacks
        scheduler : torch.optim.lr_scheduler, optional
            Learning rate scheduler (managed separately from callbacks)
        """
        self.model = model.to(config.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.scheduler = scheduler
        
        # Setup callbacks
        if callbacks is None:
            self.callbacks = create_default_callbacks(
                save_dir=config.save_dir,
                monitor='val_loss',
                patience=5,
                experiment_name=config.experiment_name
            )
        else:
            self.callbacks = callbacks
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        logger.info(f"Trainer initialized on device: {config.device}")
        logger.info(f"Mixed precision: {config.mixed_precision}")
        logger.info(f"Gradient accumulation steps: {config.accumulation_steps}")
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        initial_epoch: int = 0
    ) -> Dict:
        """
        Train the model.
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader
        initial_epoch : int
            Starting epoch (for resuming training)
            
        Returns:
        --------
        dict : Training history
        
        Example:
        --------
        >>> history = trainer.fit(train_loader, val_loader)
        >>> print(f"Best val loss: {min(history['val_loss']):.4f}")
        """
        self.current_epoch = initial_epoch
        
        # Training start callback
        self.callbacks.on_train_start(logs={'model': self.model})
        
        try:
            for epoch in range(initial_epoch, self.config.epochs):
                self.current_epoch = epoch
                
                # Epoch start callback
                self.callbacks.on_epoch_start(epoch)
                
                # Learning rate warmup
                if epoch < self.config.warmup_epochs:
                    lr_scale = (epoch + 1) / self.config.warmup_epochs
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.config.learning_rate * lr_scale
                
                # Training phase
                train_metrics = self._train_epoch(train_loader, epoch)
                
                # Validation phase
                val_metrics = {}
                if val_loader is not None:
                    val_metrics = self._validate_epoch(val_loader, epoch)
                
                # Update history
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['train_acc'].append(train_metrics['accuracy'])
                self.history['learning_rates'].append(current_lr)
                
                if val_metrics:
                    self.history['val_loss'].append(val_metrics['loss'])
                    self.history['val_acc'].append(val_metrics['accuracy'])
                
                # Prepare logs for callbacks
                logs = {
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_acc': train_metrics['accuracy'],
                    'learning_rate': current_lr,
                    'model': self.model,
                    'optimizer': self.optimizer
                }
                
                if val_metrics:
                    logs.update({
                        'val_loss': val_metrics['loss'],
                        'val_acc': val_metrics['accuracy']
                    })
                
                # Epoch end callback
                self.callbacks.on_epoch_end(epoch, logs)
                
                # Print epoch summary
                self._print_epoch_summary(epoch, train_metrics, val_metrics, current_lr)
                
                # Check if should stop
                if self.callbacks.should_stop_training():
                    logger.info(f"Training stopped early at epoch {epoch}")
                    break
                
                # Step scheduler (if not ReduceLROnPlateau)
                if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                elif self.scheduler is not None and val_metrics:
                    self.scheduler.step(val_metrics['loss'])
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            # Training end callback
            self.callbacks.on_train_end(logs={'model': self.model})
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch (handle MixUp/CutMix if present)
            if len(batch) == 2:
                inputs, targets = batch
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                is_mixed = False
            else:
                # MixUp/CutMix format
                inputs, (targets_a, targets_b, lam) = batch
                inputs = inputs.to(self.config.device)
                targets_a = targets_a.to(self.config.device)
                targets_b = targets_b.to(self.config.device)
                is_mixed = True
            
            # Forward pass with mixed precision
            if self.config.mixed_precision:
                with autocast():
                    outputs = self.model(inputs)
                    
                    if is_mixed:
                        loss = lam * self.criterion(outputs, targets_a) + \
                               (1 - lam) * self.criterion(outputs, targets_b)
                    else:
                        loss = self.criterion(outputs, targets)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.accumulation_steps
            else:
                outputs = self.model(inputs)
                
                if is_mixed:
                    loss = lam * self.criterion(outputs, targets_a) + \
                           (1 - lam) * self.criterion(outputs, targets_b)
                else:
                    loss = self.criterion(outputs, targets)
                
                loss = loss / self.config.accumulation_steps
            
            # Backward pass
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights (with accumulation)
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    if self.config.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                # Optimizer step
                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Statistics
            running_loss += loss.item() * self.config.accumulation_steps * inputs.size(0)
            
            # Accuracy (use targets_a for mixed samples as approximation)
            if is_mixed:
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets_a).sum().item()
            else:
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            
            total += inputs.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / total,
                'acc': 100. * correct / total
            })
        
        metrics = {
            'loss': running_loss / total,
            'accuracy': 100. * correct / total
        }
        
        return metrics
    
    @torch.no_grad()
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict:
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.config.epochs} [Val]")
        
        for batch in pbar:
            inputs, targets = batch
            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)
            
            # Forward pass
            if self.config.mixed_precision:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / total,
                'acc': 100. * correct / total
            })
        
        metrics = {
            'loss': running_loss / total,
            'accuracy': 100. * correct / total
        }
        
        return metrics
    
    def _print_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        learning_rate: float
    ):
        """Print epoch summary"""
        summary = f"\nEpoch {epoch+1}/{self.config.epochs} Summary:"
        summary += f"\n  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%"
        
        if val_metrics:
            summary += f"\n  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%"
        
        summary += f"\n  Learning Rate: {learning_rate:.2e}"
        
        logger.info(summary)
    
    def save_checkpoint(self, filepath: str, include_optimizer: bool = True):
        """
        Save training checkpoint.
        
        Parameters:
        -----------
        filepath : str
            Path to save checkpoint
        include_optimizer : bool
            Whether to include optimizer state
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'history': self.history,
            'config': self.config
        }
        
        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            
            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """
        Load training checkpoint.
        
        Parameters:
        -----------
        filepath : str
            Path to checkpoint
        load_optimizer : bool
            Whether to load optimizer state
        """
        checkpoint = torch.load(filepath, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.history = checkpoint['history']
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if 'scaler_state_dict' in checkpoint and self.scaler is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint loaded: {filepath}")
        logger.info(f"Resuming from epoch {self.current_epoch}")


def create_trainer(
    model: nn.Module,
    num_classes: int,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    device: str = 'cuda',
    experiment_name: Optional[str] = None
) -> Tuple[AmuletTrainer, TrainingConfig]:
    """
    Quick setup function for creating a trainer with defaults.
    
    Parameters:
    -----------
    model : nn.Module
        Model to train
    num_classes : int
        Number of classes
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader, optional
        Validation data loader
    learning_rate : float
        Initial learning rate
    epochs : int
        Number of epochs
    device : str
        Device to use
    experiment_name : str, optional
        Name of experiment
        
    Returns:
    --------
    trainer : AmuletTrainer
        Configured trainer
    config : TrainingConfig
        Training configuration
        
    Example:
    --------
    >>> trainer, config = create_trainer(
    ...     model=model,
    ...     num_classes=6,
    ...     train_loader=train_loader,
    ...     val_loader=val_loader,
    ...     learning_rate=1e-3,
    ...     epochs=50
    ... )
    >>> history = trainer.fit(train_loader, val_loader)
    """
    # Create config
    config = TrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        mixed_precision=torch.cuda.is_available(),
        device=device,
        experiment_name=experiment_name
    )
    
    # Create criterion with class weights if dataset is imbalanced
    try:
        labels = [label for _, label in train_loader.dataset]
        class_counts = np.bincount(labels, minlength=num_classes)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * num_classes
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Using weighted loss with class weights: {class_weights}")
    except:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using standard CrossEntropyLoss")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Create trainer
    trainer = AmuletTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        scheduler=scheduler
    )
    
    return trainer, config
