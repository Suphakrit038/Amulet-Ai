"""
🚀 Improved Step 5: Memory-Efficient Self-Supervised Training
Step 5 ปรับปรุงใหม่: ระบบฝึกสอน Self-Supervised ที่ประหยัดหน่วยความจำ
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
import json
import logging
import time
import gc
import psutil
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MemoryEfficientConfig:
    """Memory-efficient configuration"""
    # Severely reduced settings to prevent memory issues
    image_size: Tuple[int, int] = (224, 224)  # Reduced from 512x512
    batch_size: int = 2                       # Much smaller batch
    embedding_dim: int = 256                  # Reduced embedding
    projection_dim: int = 64                  # Smaller projection
    temperature: float = 0.1
    learning_rate: float = 1e-4
    max_epochs: int = 3                       # Very limited epochs
    save_interval: int = 1
    memory_threshold: float = 0.80            # Stop if memory > 80%

class MemoryMonitor:
    """Monitor and manage memory usage"""
    
    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold
        self.cleanup_count = 0
    
    def check_memory(self) -> Tuple[float, bool]:
        """Check current memory usage"""
        memory = psutil.virtual_memory()
        usage = memory.percent / 100.0
        is_critical = usage > self.threshold
        return usage, is_critical
    
    def emergency_cleanup(self):
        """Emergency memory cleanup"""
        self.cleanup_count += 1
        logger.warning(f"🧹 Emergency memory cleanup #{self.cleanup_count}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def safe_operation(self, operation_name: str = "operation"):
        """Check memory before critical operations"""
        usage, is_critical = self.check_memory()
        if is_critical:
            logger.warning(f"⚠️ Memory critical before {operation_name}: {usage:.1%}")
            self.emergency_cleanup()
            return False
        return True

class LightweightContrastiveModel(nn.Module):
    """Lightweight contrastive learning model"""
    
    def __init__(self, config: MemoryEfficientConfig):
        super().__init__()
        self.config = config
        
        # Very lightweight backbone
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),  # 56x56
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 28x28
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 14x14
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # 4x4
        )
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(256 * 4 * 4, config.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.embedding_dim, config.projection_dim),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Project to embedding space
        projections = self.projection_head(features)
        
        # L2 normalize
        projections = F.normalize(projections, p=2, dim=1)
        
        return projections

class MemoryEfficientDataset:
    """Memory-efficient dataset with on-demand loading"""
    
    def __init__(self, data_dir: Path, config: MemoryEfficientConfig, memory_monitor: MemoryMonitor):
        self.data_dir = data_dir
        self.config = config
        self.memory_monitor = memory_monitor
        self.image_paths = []
        self.labels = []
        
        self._load_paths()
    
    def _load_paths(self):
        """Load image paths without loading actual images"""
        train_dir = self.data_dir / "dataset_split" / "train"
        
        if not train_dir.exists():
            logger.error(f"❌ Training directory not found: {train_dir}")
            return
        
        label_to_idx = {}
        current_idx = 0
        
        for category_dir in train_dir.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                if category_name not in label_to_idx:
                    label_to_idx[category_name] = current_idx
                    current_idx += 1
                
                # Collect image paths
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    for img_path in category_dir.glob(ext):
                        self.image_paths.append(img_path)
                        self.labels.append(label_to_idx[category_name])
        
        logger.info(f"📂 Found {len(self.image_paths)} images in {len(label_to_idx)} categories")
        self.label_to_idx = label_to_idx
        self.idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    def __len__(self):
        return len(self.image_paths)
    
    def load_image(self, idx: int) -> Optional[torch.Tensor]:
        """Load and preprocess single image"""
        try:
            img_path = self.image_paths[idx]
            
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to target size
                img = img.resize(self.config.image_size, Image.Resampling.LANCZOS)
                
                # Convert to tensor
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                
                return img_tensor
                
        except Exception as e:
            logger.error(f"❌ Failed to load {img_path}: {e}")
            return None
    
    def get_batch(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a batch of images"""
        images = []
        labels = []
        
        for idx in indices:
            # Check memory before each image
            if not self.memory_monitor.safe_operation(f"loading image {idx}"):
                continue
            
            img_tensor = self.load_image(idx)
            if img_tensor is not None:
                images.append(img_tensor)
                labels.append(self.labels[idx])
        
        if not images:
            # Return dummy batch if no images loaded
            dummy_img = torch.zeros(1, 3, *self.config.image_size)
            dummy_label = torch.zeros(1, dtype=torch.long)
            return dummy_img, dummy_label
        
        # Stack into batches
        images_batch = torch.stack(images)
        labels_batch = torch.tensor(labels, dtype=torch.long)
        
        return images_batch, labels_batch

def contrastive_loss(projections: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Compute contrastive loss"""
    batch_size = projections.size(0)
    
    if batch_size < 2:
        # Return zero loss for single sample
        return torch.tensor(0.0, device=projections.device, requires_grad=True)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(projections, projections.T) / temperature
    
    # Create positive mask (same labels)
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    
    # Remove diagonal (self-similarity)
    mask = mask - torch.eye(batch_size, device=mask.device)
    
    # Compute loss
    exp_sim = torch.exp(sim_matrix)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    
    # Apply mask and compute mean
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    
    return loss

class MemoryEfficientTrainer:
    """Memory-efficient self-supervised trainer"""
    
    def __init__(self, config: MemoryEfficientConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.memory_monitor = MemoryMonitor(config.memory_threshold)
        
        # Create model
        self.model = LightweightContrastiveModel(config)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"🧠 Model created with {total_params:,} parameters")
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Training stats
        self.training_stats = {
            'epoch': 0,
            'batch_count': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'total_loss': 0.0,
            'memory_cleanups': 0
        }
    
    def train_step_5(self, data_dir: Path) -> Dict:
        """Execute Step 5: Self-Supervised Training"""
        logger.info("🚀 Starting Step 5: Memory-Efficient Self-Supervised Training")
        
        # Create dataset
        dataset = MemoryEfficientDataset(data_dir, self.config, self.memory_monitor)
        
        if len(dataset) == 0:
            logger.error("❌ No training data found")
            return self.training_stats
        
        # Training loop
        self.model.train()
        
        for epoch in range(self.config.max_epochs):
            logger.info(f"🏋️ Epoch {epoch + 1}/{self.config.max_epochs}")
            epoch_loss = 0.0
            epoch_batches = 0
            
            # Create random batches
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            
            # Process in small batches
            for i in range(0, len(indices), self.config.batch_size):
                # Check memory before each batch
                usage, is_critical = self.memory_monitor.check_memory()
                if is_critical:
                    logger.warning(f"🛑 Stopping due to critical memory: {usage:.1%}")
                    self.training_stats['memory_cleanups'] = self.memory_monitor.cleanup_count
                    break
                
                batch_indices = indices[i:i + self.config.batch_size]
                
                try:
                    # Load batch
                    images, labels = dataset.get_batch(batch_indices)
                    
                    if images.size(0) < 2:
                        # Skip batches with less than 2 samples
                        continue
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    projections = self.model(images)
                    loss = contrastive_loss(projections, labels, self.config.temperature)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    # Update stats
                    epoch_loss += loss.item()
                    epoch_batches += 1
                    self.training_stats['successful_batches'] += 1
                    
                    if epoch_batches % 5 == 0:
                        logger.info(f"📊 Batch {epoch_batches} - Loss: {loss.item():.4f} - Memory: {usage:.1%}")
                    
                    # Cleanup
                    del images, labels, projections, loss
                    self.memory_monitor.emergency_cleanup()
                    
                except Exception as e:
                    logger.error(f"❌ Batch failed: {e}")
                    self.training_stats['failed_batches'] += 1
                    self.memory_monitor.emergency_cleanup()
                    continue
            
            # Epoch summary
            if epoch_batches > 0:
                avg_loss = epoch_loss / epoch_batches
                logger.info(f"✅ Epoch {epoch + 1} completed - Avg Loss: {avg_loss:.4f}")
                self.training_stats['total_loss'] += epoch_loss
            
            self.training_stats['epoch'] = epoch + 1
            
            # Save model checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch + 1)
        
        # Final stats
        logger.info("🎯 Step 5 Training Completed!")
        logger.info(f"✅ Successful batches: {self.training_stats['successful_batches']}")
        logger.info(f"❌ Failed batches: {self.training_stats['failed_batches']}")
        
        # Save final model
        self.save_final_model()
        
        return self.training_stats
    
    def save_checkpoint(self, epoch: int):
        """Save training checkpoint"""
        checkpoint_path = self.output_dir / f"step5_checkpoint_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'config': self.config
        }, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def save_final_model(self):
        """Save final trained model"""
        model_path = self.output_dir / "step5_final_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_stats': self.training_stats,
            'config': self.config,
            'model_architecture': 'LightweightContrastiveModel'
        }, model_path)
        
        # Save training report
        report_path = self.output_dir / "step5_training_report.json"
        report = {
            'step': 5,
            'training_type': 'memory_efficient_self_supervised',
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'training_stats': self.training_stats,
            'config': {
                'image_size': self.config.image_size,
                'batch_size': self.config.batch_size,
                'embedding_dim': self.config.embedding_dim,
                'max_epochs': self.config.max_epochs,
                'learning_rate': self.config.learning_rate
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Final model saved: {model_path}")
        logger.info(f"📝 Training report saved: {report_path}")

def run_step_5_improved():
    """Run improved Step 5 with memory optimization"""
    logger.info("🚀 STARTING IMPROVED STEP 5: MEMORY-EFFICIENT SELF-SUPERVISED TRAINING")
    
    # Setup
    config = MemoryEfficientConfig()
    output_dir = Path("training_output")
    output_dir.mkdir(exist_ok=True)
    
    data_dir = Path(".")
    
    # Check initial memory
    memory_monitor = MemoryMonitor()
    usage, _ = memory_monitor.check_memory()
    logger.info(f"💾 Initial memory usage: {usage:.1%}")
    
    # Create trainer and run
    trainer = MemoryEfficientTrainer(config, output_dir)
    
    try:
        stats = trainer.train_step_5(data_dir)
        logger.info("🎯 Step 5 completed successfully!")
        return stats
        
    except Exception as e:
        logger.error(f"💥 Step 5 failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    stats = run_step_5_improved()
    if stats:
        logger.info("✅ Improved Step 5 completed successfully!")
    else:
        logger.error("❌ Improved Step 5 failed!")
