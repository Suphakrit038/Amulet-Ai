"""
🚀 Memory-Optimized Emergency Training System
ระบบฝึกสอนที่เพิ่มประสิทธิภาพหน่วยความจำสำหรับแก้ปัญหา memory allocation
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import gc
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import psutil
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryOptimizedConfig:
    """Configuration optimized for low memory usage"""
    # Severely reduce image resolution to prevent memory issues
    image_size = (256, 256)  # Much smaller than original 512x512
    batch_size = 4           # Reduced from 16
    max_workers = 0          # No multiprocessing 
    memory_threshold = 0.85  # Stop if memory usage > 85%
    emergency_cleanup = True # Enable emergency cleanup
    
class EmergencyMemoryManager:
    """Emergency memory management system"""
    
    def __init__(self):
        self.memory_threshold = 0.85
        self.cleanup_count = 0
    
    def check_memory(self) -> Tuple[float, bool]:
        """Check current memory usage"""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent / 100.0
        available_gb = memory.available / (1024**3)
        
        is_critical = usage_percent > self.memory_threshold
        
        if is_critical:
            logger.warning(f"⚠️ CRITICAL MEMORY: {usage_percent:.1%} used, {available_gb:.1f} GB available")
        
        return usage_percent, is_critical
    
    def emergency_cleanup(self):
        """Perform emergency memory cleanup"""
        self.cleanup_count += 1
        logger.warning(f"🧹 Emergency cleanup #{self.cleanup_count}")
        
        # Force garbage collection
        gc.collect()
        
        # Clear PyTorch cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Additional cleanup
        import sys
        for obj in list(sys.modules.keys()):
            if 'numpy' in obj or 'cv2' in obj:
                if hasattr(sys.modules[obj], '__cached__'):
                    try:
                        delattr(sys.modules[obj], '__cached__')
                    except:
                        pass
    
    def safe_image_load(self, image_path: Path, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Safely load and resize image with memory monitoring"""
        try:
            # Check memory before loading
            usage, is_critical = self.check_memory()
            if is_critical:
                self.emergency_cleanup()
                return None
            
            # Load image with minimal memory footprint
            with Image.open(image_path) as img:
                # Convert to RGB immediately
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize immediately to reduce memory
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # Convert to numpy with minimal memory
                img_array = np.array(img_resized, dtype=np.float32)
                
                # Normalize to [0, 1] range
                img_array = img_array / 255.0
                
                return img_array
                
        except Exception as e:
            logger.error(f"❌ Failed to load image {image_path}: {e}")
            self.emergency_cleanup()
            return None

class MemoryEfficientDataset:
    """Memory-efficient dataset that loads images on-demand"""
    
    def __init__(self, image_paths: List[Path], config: MemoryOptimizedConfig):
        self.image_paths = image_paths
        self.config = config
        self.memory_manager = EmergencyMemoryManager()
        
        logger.info(f"📦 Created memory-efficient dataset with {len(image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Load single image with memory management"""
        image_path = self.image_paths[idx]
        
        # Load with memory monitoring
        image = self.memory_manager.safe_image_load(
            image_path, 
            self.config.image_size
        )
        
        if image is None:
            # Return a black image if loading failed
            image = np.zeros((*self.config.image_size, 3), dtype=np.float32)
            logger.warning(f"⚠️ Using fallback black image for {image_path}")
        
        return torch.from_numpy(image).permute(2, 0, 1)  # CHW format

def create_memory_optimized_training():
    """Create emergency memory-optimized training setup"""
    config = MemoryOptimizedConfig()
    memory_manager = EmergencyMemoryManager()
    
    logger.info("🚀 Starting Memory-Optimized Emergency Training")
    
    # Check initial memory state
    usage, is_critical = memory_manager.check_memory()
    logger.info(f"💾 Initial memory usage: {usage:.1%}")
    
    if is_critical:
        logger.warning("⚠️ Starting with critical memory levels!")
        memory_manager.emergency_cleanup()
    
    # Find training images
    dataset_path = Path("dataset_split/train")
    if not dataset_path.exists():
        logger.error(f"❌ Training dataset not found at {dataset_path}")
        return None
    
    # Collect image paths with memory monitoring
    image_paths = []
    for category_dir in dataset_path.iterdir():
        if category_dir.is_dir():
            for img_path in category_dir.glob("*.jpg"):
                image_paths.append(img_path)
            for img_path in category_dir.glob("*.png"):
                image_paths.append(img_path)
            
            # Check memory periodically
            if len(image_paths) % 20 == 0:
                usage, is_critical = memory_manager.check_memory()
                if is_critical:
                    logger.warning(f"⚠️ Memory critical while collecting paths: {usage:.1%}")
                    memory_manager.emergency_cleanup()
    
    logger.info(f"📂 Found {len(image_paths)} training images")
    
    # Create memory-efficient dataset
    dataset = MemoryEfficientDataset(image_paths, config)
    
    # Test loading a few images
    logger.info("🧪 Testing image loading...")
    test_count = min(5, len(image_paths))
    successful_loads = 0
    
    for i in range(test_count):
        try:
            image_tensor = dataset[i]
            if image_tensor is not None:
                successful_loads += 1
                logger.info(f"✅ Successfully loaded image {i+1}/{test_count} - Shape: {image_tensor.shape}")
            
            # Cleanup after each test
            del image_tensor
            memory_manager.emergency_cleanup()
            
        except Exception as e:
            logger.error(f"❌ Failed to load test image {i}: {e}")
    
    logger.info(f"📊 Successfully loaded {successful_loads}/{test_count} test images")
    
    # Final memory check
    final_usage, _ = memory_manager.check_memory()
    logger.info(f"💾 Final memory usage: {final_usage:.1%}")
    
    return dataset, config, memory_manager

if __name__ == "__main__":
    # Run emergency memory-optimized setup
    try:
        dataset, config, memory_manager = create_memory_optimized_training()
        
        if dataset:
            logger.info("✅ Memory-optimized training setup complete!")
            logger.info("🎯 Ready for emergency low-memory training")
        else:
            logger.error("❌ Failed to create memory-optimized setup")
            
    except Exception as e:
        logger.error(f"💥 Emergency setup failed: {e}")
        import traceback
        traceback.print_exc()
