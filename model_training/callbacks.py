"""
Training Callbacks for Amulet-AI Model Training

Implements essential callbacks for production training:
- EarlyStopping: Stop training when validation metric stops improving
- ModelCheckpoint: Save best/last models automatically
- LearningRateScheduler: Adjust learning rate based on metrics
- MetricsLogger: Log metrics to TensorBoard/WandB/CSV

Author: Amulet-AI Team
Date: October 2, 2025
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union, Literal, Callable
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Callback:
    """Base callback class"""
    
    def on_epoch_start(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the start of an epoch"""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the end of an epoch"""
        pass
    
    def on_train_start(self, logs: Optional[Dict] = None):
        """Called at the start of training"""
        pass
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Called at the end of training"""
        pass
    
    def on_batch_start(self, batch: int, logs: Optional[Dict] = None):
        """Called at the start of a batch"""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Called at the end of a batch"""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback to stop training when monitored metric stops improving.
    
    Parameters:
    -----------
    monitor : str
        Metric to monitor (e.g., 'val_loss', 'val_f1')
    patience : int
        Number of epochs with no improvement after which training will be stopped
    mode : str
        'min' for metrics where lower is better, 'max' for higher is better
    min_delta : float
        Minimum change in monitored value to qualify as an improvement
    restore_best_weights : bool
        Whether to restore model weights from best epoch
    verbose : bool
        Whether to print messages
        
    Example:
    --------
    >>> early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    >>> trainer.add_callback(early_stop)
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 5,
        mode: Literal['min', 'max'] = 'min',
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.should_stop = False
        
        # Comparison function
        if mode == 'min':
            self.monitor_op = lambda current, best: current < (best - min_delta)
        else:
            self.monitor_op = lambda current, best: current > (best + min_delta)
    
    def on_train_start(self, logs: Optional[Dict] = None):
        """Reset counters at training start"""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = np.inf if self.mode == 'min' else -np.inf
        self.should_stop = False
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Check if should stop at epoch end"""
        if logs is None:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            logger.warning(f"Early stopping monitor '{self.monitor}' not found in logs")
            return
        
        # Check if improved
        if self.monitor_op(current, self.best_value):
            self.best_value = current
            self.wait = 0
            
            # Save best weights
            if self.restore_best_weights and 'model' in logs:
                self.best_weights = {k: v.cpu().clone() for k, v in logs['model'].state_dict().items()}
            
            if self.verbose:
                logger.info(f"Epoch {epoch}: {self.monitor} improved to {current:.6f}")
        else:
            self.wait += 1
            if self.verbose:
                logger.info(
                    f"Epoch {epoch}: {self.monitor} did not improve from {self.best_value:.6f} "
                    f"(current: {current:.6f}, patience: {self.wait}/{self.patience})"
                )
            
            # Check if should stop
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True
                
                # Restore best weights
                if self.restore_best_weights and self.best_weights is not None and 'model' in logs:
                    logger.info(f"Restoring model weights from epoch {epoch - self.patience}")
                    logs['model'].load_state_dict(self.best_weights)
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Print summary at training end"""
        if self.stopped_epoch > 0:
            logger.info(f"Early stopping triggered at epoch {self.stopped_epoch}")
            logger.info(f"Best {self.monitor}: {self.best_value:.6f}")


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.
    
    Parameters:
    -----------
    filepath : str
        Path to save model (can include {epoch} and {metric} placeholders)
    monitor : str
        Metric to monitor for saving best model
    mode : str
        'min' for metrics where lower is better, 'max' for higher is better
    save_best_only : bool
        If True, only save when monitor improves
    save_last : bool
        If True, always save last checkpoint
    verbose : bool
        Whether to print messages
        
    Example:
    --------
    >>> checkpoint = ModelCheckpoint(
    ...     filepath='models/model_epoch{epoch}_val{val_loss:.4f}.pth',
    ...     monitor='val_loss',
    ...     save_best_only=True
    ... )
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: Literal['min', 'max'] = 'min',
        save_best_only: bool = True,
        save_last: bool = True,
        save_weights_only: bool = False,
        verbose: bool = True
    ):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.monitor_op = lambda current, best: current < best if mode == 'min' else current > best
        
        # Create directory
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Save checkpoint at epoch end"""
        if logs is None or 'model' not in logs:
            return
        
        model = logs['model']
        current = logs.get(self.monitor)
        
        # Format filepath
        filepath = self.filepath.format(epoch=epoch, **{k: v for k, v in logs.items() if isinstance(v, (int, float))})
        
        # Check if should save
        should_save = False
        is_best = False
        
        if current is not None:
            if self.monitor_op(current, self.best_value):
                self.best_value = current
                is_best = True
                should_save = True
        
        if not self.save_best_only:
            should_save = True
        
        # Save checkpoint
        if should_save:
            self._save_checkpoint(model, filepath, logs, is_best)
        
        # Save last checkpoint
        if self.save_last:
            last_path = str(Path(self.filepath).parent / 'last_checkpoint.pth')
            self._save_checkpoint(model, last_path, logs, is_best=False)
    
    def _save_checkpoint(self, model: nn.Module, filepath: str, logs: Dict, is_best: bool):
        """Save checkpoint to disk"""
        try:
            if self.save_weights_only:
                torch.save(model.state_dict(), filepath)
            else:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'epoch': logs.get('epoch', 0),
                    'monitor_value': logs.get(self.monitor),
                    'logs': {k: v for k, v in logs.items() if isinstance(v, (int, float, str))},
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add optimizer state if available
                if 'optimizer' in logs:
                    checkpoint['optimizer_state_dict'] = logs['optimizer'].state_dict()
                
                torch.save(checkpoint, filepath)
            
            if self.verbose:
                status = " (best)" if is_best else ""
                logger.info(f"Saved checkpoint: {filepath}{status}")
                
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")


class LearningRateScheduler(Callback):
    """
    Adjust learning rate during training.
    
    Supports multiple scheduler types:
    - 'reduce_on_plateau': Reduce LR when metric plateaus
    - 'cosine': Cosine annealing
    - 'step': Step decay
    
    Parameters:
    -----------
    scheduler_type : str
        Type of scheduler
    optimizer : torch.optim.Optimizer
        Optimizer to adjust
    monitor : str
        Metric to monitor (for ReduceLROnPlateau)
    **kwargs : dict
        Scheduler-specific arguments
        
    Example:
    --------
    >>> scheduler = LearningRateScheduler(
    ...     scheduler_type='reduce_on_plateau',
    ...     optimizer=optimizer,
    ...     monitor='val_loss',
    ...     factor=0.5,
    ...     patience=3
    ... )
    """
    
    def __init__(
        self,
        scheduler_type: Literal['reduce_on_plateau', 'cosine', 'step'],
        optimizer: torch.optim.Optimizer,
        monitor: str = 'val_loss',
        verbose: bool = True,
        **kwargs
    ):
        super().__init__()
        self.scheduler_type = scheduler_type
        self.optimizer = optimizer
        self.monitor = monitor
        self.verbose = verbose
        
        # Create scheduler
        if scheduler_type == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 3),
                verbose=verbose,
                min_lr=kwargs.get('min_lr', 1e-7)
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 50),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 10),
                gamma=kwargs.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Update learning rate at epoch end"""
        if logs is None:
            return
        
        # Get current LR
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Update scheduler
        if self.scheduler_type == 'reduce_on_plateau':
            metric = logs.get(self.monitor)
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()
        
        # Get new LR
        new_lr = self.optimizer.param_groups[0]['lr']
        
        # Log if changed
        if new_lr != current_lr and self.verbose:
            logger.info(f"Learning rate adjusted: {current_lr:.2e} -> {new_lr:.2e}")


class MetricsLogger(Callback):
    """
    Log training metrics to various backends.
    
    Supports:
    - CSV file
    - JSON file
    - TensorBoard (if installed)
    - Weights & Biases (if installed)
    
    Parameters:
    -----------
    log_dir : str
        Directory to save logs
    backends : list
        List of backends to use ['csv', 'json', 'tensorboard', 'wandb']
    experiment_name : str
        Name of experiment
        
    Example:
    --------
    >>> logger = MetricsLogger(
    ...     log_dir='logs',
    ...     backends=['csv', 'tensorboard'],
    ...     experiment_name='resnet50_medium_aug'
    ... )
    """
    
    def __init__(
        self,
        log_dir: str = 'logs',
        backends: list = ['csv', 'json'],
        experiment_name: Optional[str] = None,
        verbose: bool = True
    ):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.backends = backends
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.verbose = verbose
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize backends
        self.csv_file = None
        self.json_logs = []
        self.tb_writer = None
        self.wandb_run = None
        
        if 'csv' in backends:
            self.csv_file = open(self.log_dir / f"{self.experiment_name}.csv", 'w')
            self.csv_header_written = False
        
        if 'tensorboard' in backends:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / self.experiment_name))
                if self.verbose:
                    logger.info(f"TensorBoard logging enabled: {self.log_dir / self.experiment_name}")
            except ImportError:
                logger.warning("TensorBoard not available. Install with: pip install tensorboard")
        
        if 'wandb' in backends:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project="amulet-ai",
                    name=self.experiment_name,
                    dir=str(self.log_dir)
                )
                if self.verbose:
                    logger.info(f"W&B logging enabled: {self.experiment_name}")
            except ImportError:
                logger.warning("Weights & Biases not available. Install with: pip install wandb")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Log metrics at epoch end"""
        if logs is None:
            return
        
        # Filter logs (only numeric values)
        filtered_logs = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        filtered_logs['epoch'] = epoch
        
        # CSV logging
        if 'csv' in self.backends and self.csv_file is not None:
            if not self.csv_header_written:
                self.csv_file.write(','.join(filtered_logs.keys()) + '\n')
                self.csv_header_written = True
            self.csv_file.write(','.join(map(str, filtered_logs.values())) + '\n')
            self.csv_file.flush()
        
        # JSON logging
        if 'json' in self.backends:
            self.json_logs.append(filtered_logs)
        
        # TensorBoard logging
        if 'tensorboard' in self.backends and self.tb_writer is not None:
            for key, value in filtered_logs.items():
                if key != 'epoch':
                    self.tb_writer.add_scalar(key, value, epoch)
        
        # W&B logging
        if 'wandb' in self.backends and self.wandb_run is not None:
            self.wandb_run.log(filtered_logs)
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Clean up at training end"""
        # Save JSON
        if 'json' in self.backends and self.json_logs:
            json_path = self.log_dir / f"{self.experiment_name}.json"
            with open(json_path, 'w') as f:
                json.dump(self.json_logs, f, indent=2)
            if self.verbose:
                logger.info(f"Logs saved to: {json_path}")
        
        # Close files
        if self.csv_file is not None:
            self.csv_file.close()
        
        if self.tb_writer is not None:
            self.tb_writer.close()
        
        if self.wandb_run is not None:
            self.wandb_run.finish()


class CallbackList:
    """
    Container for managing multiple callbacks.
    
    Example:
    --------
    >>> callbacks = CallbackList([early_stop, checkpoint, logger])
    >>> callbacks.on_epoch_end(epoch=5, logs={'val_loss': 0.25})
    """
    
    def __init__(self, callbacks: list = None):
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback):
        """Add a callback"""
        self.callbacks.append(callback)
    
    def __iter__(self):
        return iter(self.callbacks)
    
    def on_epoch_start(self, epoch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_epoch_start(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_train_start(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_train_start(logs)
    
    def on_train_end(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_batch_start(self, batch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_batch_start(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)
    
    def should_stop_training(self) -> bool:
        """Check if any callback requests stopping"""
        for callback in self.callbacks:
            if isinstance(callback, EarlyStopping) and callback.should_stop:
                return True
        return False


# Helper function to create common callback configurations
def create_default_callbacks(
    save_dir: str = 'trained_model',
    monitor: str = 'val_loss',
    patience: int = 5,
    experiment_name: Optional[str] = None
) -> CallbackList:
    """
    Create a default set of callbacks for training.
    
    Parameters:
    -----------
    save_dir : str
        Directory to save models and logs
    monitor : str
        Metric to monitor
    patience : int
        Early stopping patience
    experiment_name : str
        Name of experiment
        
    Returns:
    --------
    CallbackList with EarlyStopping, ModelCheckpoint, and MetricsLogger
    
    Example:
    --------
    >>> callbacks = create_default_callbacks(
    ...     save_dir='trained_model',
    ...     monitor='val_f1',
    ...     patience=10,
    ...     experiment_name='resnet50_heavy_aug'
    ... )
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    mode = 'min' if 'loss' in monitor else 'max'
    
    callbacks = CallbackList([
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            restore_best_weights=True,
            verbose=True
        ),
        ModelCheckpoint(
            filepath=str(save_path / 'best_model.pth'),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_last=True,
            verbose=True
        ),
        MetricsLogger(
            log_dir=str(save_path / 'logs'),
            backends=['csv', 'json'],
            experiment_name=experiment_name,
            verbose=True
        )
    ])
    
    return callbacks
