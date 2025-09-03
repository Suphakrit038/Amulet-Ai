"""
🚀 Emergency Low-Memory Training System
ระบบฝึกสอนฉุกเฉินสำหรับระบบที่มีหน่วยความจำจำกัด
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import logging
import time
import gc
from memory_optimized_training import (
    MemoryOptimizedConfig, 
    EmergencyMemoryManager, 
    MemoryEfficientDataset
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplifiedModel(nn.Module):
    """Simplified model for low-memory training"""
    
    def __init__(self, num_classes=10, embedding_dim=128):
        super().__init__()
        # Much smaller model to save memory
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128x128
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x64
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # 4x4
        )
        
        # Feature dimension: 256 * 4 * 4 = 4096
        self.embedding = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embedding_dim, num_classes)
        )
    
    def forward(self, x, return_embedding=False):
        features = self.features(x)
        features_flat = features.view(features.size(0), -1)
        embedding = self.embedding(features_flat)
        
        if return_embedding:
            return embedding
        
        output = self.classifier(embedding)
        return output

class EmergencyTrainer:
    """Emergency trainer for low-memory systems"""
    
    def __init__(self, model, config, memory_manager):
        self.model = model
        self.config = config
        self.memory_manager = memory_manager
        
        # Use very small learning rate for stability
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training stats
        self.training_stats = {
            'epoch': 0,
            'batch_count': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'memory_cleanups': 0
        }
    
    def train_batch(self, images, labels):
        """Train on a single batch with memory monitoring"""
        try:
            # Check memory before training
            usage, is_critical = self.memory_manager.check_memory()
            if is_critical:
                self.memory_manager.emergency_cleanup()
                self.training_stats['memory_cleanups'] += 1
                return False
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Clear gradients to save memory
            self.optimizer.zero_grad()
            
            self.training_stats['successful_batches'] += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch training failed: {e}")
            self.memory_manager.emergency_cleanup()
            self.training_stats['failed_batches'] += 1
            return False
    
    def emergency_training_loop(self, dataset, max_batches=50):
        """Emergency training loop with severe memory constraints"""
        logger.info("🚨 Starting emergency training loop")
        
        # Create minimal dataloader
        try:
            # Use very small batch size
            dataloader = DataLoader(
                dataset, 
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,  # No multiprocessing
                pin_memory=False  # Don't use pinned memory
            )
        except Exception as e:
            logger.error(f"❌ Failed to create dataloader: {e}")
            return
        
        self.model.train()
        batch_count = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            if batch_count >= max_batches:
                logger.info(f"🛑 Reached maximum batch limit: {max_batches}")
                break
            
            try:
                # Handle different batch formats
                if isinstance(batch_data, tuple):
                    images, labels = batch_data
                else:
                    images = batch_data
                    # Create dummy labels for unsupervised learning
                    labels = torch.zeros(images.size(0), dtype=torch.long)
                
                batch_count += 1
                self.training_stats['batch_count'] = batch_count
                
                logger.info(f"🏋️ Training batch {batch_count}/{max_batches} - Images: {images.shape}")
                
                # Train the batch
                success = self.train_batch(images, labels)
                
                if success:
                    logger.info(f"✅ Batch {batch_count} completed successfully")
                else:
                    logger.warning(f"⚠️ Batch {batch_count} failed or skipped")
                
                # Memory cleanup after every batch
                del images, labels
                gc.collect()
                
                # Check memory status
                usage, is_critical = self.memory_manager.check_memory()
                logger.info(f"💾 Memory usage: {usage:.1%}")
                
                if is_critical:
                    logger.warning("🚨 Critical memory detected, performing cleanup")
                    self.memory_manager.emergency_cleanup()
                
                # Short pause to let system recover
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"💥 Batch {batch_count} processing failed: {e}")
                self.memory_manager.emergency_cleanup()
                continue
        
        # Final stats
        logger.info("📊 Emergency training completed!")
        logger.info(f"✅ Successful batches: {self.training_stats['successful_batches']}")
        logger.info(f"❌ Failed batches: {self.training_stats['failed_batches']}")
        logger.info(f"🧹 Memory cleanups: {self.training_stats['memory_cleanups']}")
        
        return self.training_stats

def run_emergency_training():
    """Run emergency training with memory optimization"""
    logger.info("🚨 EMERGENCY TRAINING MODE ACTIVATED 🚨")
    
    # Setup memory-optimized components
    config = MemoryOptimizedConfig()
    memory_manager = EmergencyMemoryManager()
    
    # Check initial memory
    usage, is_critical = memory_manager.check_memory()
    logger.info(f"💾 Starting memory usage: {usage:.1%}")
    
    if is_critical:
        logger.warning("⚠️ Starting with critical memory!")
        memory_manager.emergency_cleanup()
    
    # Find training data
    dataset_path = Path("dataset_split/train")
    if not dataset_path.exists():
        logger.error(f"❌ Training dataset not found at {dataset_path}")
        return
    
    # Collect image paths
    image_paths = []
    labels = []
    label_to_idx = {}
    idx = 0
    
    for category_dir in dataset_path.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            if category_name not in label_to_idx:
                label_to_idx[category_name] = idx
                idx += 1
            
            for img_path in category_dir.glob("*.jpg"):
                image_paths.append(img_path)
                labels.append(label_to_idx[category_name])
            for img_path in category_dir.glob("*.png"):
                image_paths.append(img_path)
                labels.append(label_to_idx[category_name])
    
    logger.info(f"📂 Found {len(image_paths)} images in {len(label_to_idx)} categories")
    
    # Create simplified dataset
    class SimplifiedDataset:
        def __init__(self, paths, labels, config, memory_manager):
            self.paths = paths
            self.labels = labels
            self.config = config
            self.memory_manager = memory_manager
        
        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, idx):
            image_path = self.paths[idx]
            label = self.labels[idx]
            
            # Load with memory management
            image = self.memory_manager.safe_image_load(
                image_path, 
                self.config.image_size
            )
            
            if image is None:
                image = np.zeros((*self.config.image_size, 3), dtype=np.float32)
            
            return torch.from_numpy(image).permute(2, 0, 1), torch.tensor(label, dtype=torch.long)
    
    # Create dataset and model
    dataset = SimplifiedDataset(image_paths, labels, config, memory_manager)
    
    # Create simplified model
    num_classes = len(label_to_idx)
    model = SimplifiedModel(num_classes=num_classes, embedding_dim=64)  # Very small embedding
    
    logger.info(f"🧠 Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = EmergencyTrainer(model, config, memory_manager)
    
    # Run emergency training
    stats = trainer.emergency_training_loop(dataset, max_batches=20)  # Very limited training
    
    # Save results
    results = {
        'emergency_training_stats': stats,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'categories': label_to_idx,
        'total_images': len(image_paths)
    }
    
    output_path = Path("training_output/emergency_training_results.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Results saved to {output_path}")
    
    # Save model
    model_path = Path("training_output/emergency_model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"🧠 Model saved to {model_path}")
    
    logger.info("🎯 Emergency training completed successfully!")
    
    return stats

if __name__ == "__main__":
    try:
        stats = run_emergency_training()
        logger.info("✅ Emergency training system completed successfully!")
    except Exception as e:
        logger.error(f"💥 Emergency training failed: {e}")
        import traceback
        traceback.print_exc()
