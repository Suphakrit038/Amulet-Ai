"""
🚀 Ultra-Simple Emergency Training
ระบบฝึกสอนที่เรียบง่ายที่สุดสำหรับแก้ปัญหาหน่วยความจำ
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from pathlib import Path
import json
import logging
import time
import gc
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_memory():
    """Check current memory usage"""
    memory = psutil.virtual_memory()
    usage_percent = memory.percent
    available_gb = memory.available / (1024**3)
    return usage_percent, available_gb

def emergency_cleanup():
    """Force memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_single_image(image_path, target_size=(128, 128)):
    """Load and process a single image with minimal memory"""
    try:
        with Image.open(image_path) as img:
            # Convert and resize immediately
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            
            # Convert to torch tensor
            tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # CHW format
            return tensor
            
    except Exception as e:
        logger.error(f"❌ Failed to load {image_path}: {e}")
        # Return black image
        return torch.zeros(3, target_size[0], target_size[1])

class UltraSimpleModel(nn.Module):
    """Ultra-simple CNN for emergency training"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Very simple architecture
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),  # 32x32 -> 8x8
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
            
            nn.AdaptiveAvgPool2d((2, 2))  # Force to 2x2
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(32 * 2 * 2, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def ultra_simple_training():
    """Ultra-simple training without DataLoader"""
    logger.info("🚨 ULTRA-SIMPLE EMERGENCY TRAINING 🚨")
    
    # Check initial memory
    usage, available = check_memory()
    logger.info(f"💾 Initial memory: {usage:.1f}% used, {available:.1f} GB available")
    
    # Find training images
    dataset_path = Path("dataset_split/train")
    if not dataset_path.exists():
        logger.error(f"❌ Dataset not found: {dataset_path}")
        return
    
    # Collect image paths and labels
    image_paths = []
    labels = []
    label_to_idx = {}
    current_idx = 0
    
    for category_dir in dataset_path.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            if category_name not in label_to_idx:
                label_to_idx[category_name] = current_idx
                current_idx += 1
            
            # Find image files
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in category_dir.glob(ext):
                    image_paths.append(img_path)
                    labels.append(label_to_idx[category_name])
    
    logger.info(f"📂 Found {len(image_paths)} images in {len(label_to_idx)} categories")
    logger.info(f"📝 Categories: {list(label_to_idx.keys())}")
    
    # Create model
    num_classes = len(label_to_idx)
    model = UltraSimpleModel(num_classes=num_classes)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"🧠 Model created with {total_params:,} parameters")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    # Training statistics
    successful_samples = 0
    failed_samples = 0
    total_loss = 0.0
    
    # Train on individual samples (no batching)
    max_samples = min(50, len(image_paths))  # Very limited training
    
    logger.info(f"🏋️ Starting training on {max_samples} samples...")
    
    for sample_idx in range(max_samples):
        try:
            # Check memory before each sample
            usage, available = check_memory()
            if usage > 80.0:  # If memory usage > 80%
                logger.warning(f"⚠️ High memory usage: {usage:.1f}%")
                emergency_cleanup()
            
            # Load single image
            img_path = image_paths[sample_idx]
            label = labels[sample_idx]
            
            logger.info(f"📸 Processing sample {sample_idx+1}/{max_samples}: {img_path.name}")
            
            # Load image
            image_tensor = load_single_image(img_path, target_size=(128, 128))
            
            if image_tensor is None:
                failed_samples += 1
                continue
            
            # Add batch dimension
            image_batch = image_tensor.unsqueeze(0)  # 1 x 3 x 128 x 128
            label_tensor = torch.tensor([label], dtype=torch.long)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(image_batch)
            loss = criterion(outputs, label_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track statistics
            total_loss += loss.item()
            successful_samples += 1
            
            logger.info(f"✅ Sample {sample_idx+1} - Loss: {loss.item():.4f}")
            
            # Cleanup
            del image_tensor, image_batch, label_tensor, outputs, loss
            emergency_cleanup()
            
            # Small pause to let system recover
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"❌ Failed to process sample {sample_idx}: {e}")
            failed_samples += 1
            emergency_cleanup()
            continue
    
    # Final statistics
    avg_loss = total_loss / max(successful_samples, 1)
    
    logger.info("📊 TRAINING COMPLETED!")
    logger.info(f"✅ Successful samples: {successful_samples}")
    logger.info(f"❌ Failed samples: {failed_samples}")
    logger.info(f"📉 Average loss: {avg_loss:.4f}")
    
    # Check final memory
    final_usage, final_available = check_memory()
    logger.info(f"💾 Final memory: {final_usage:.1f}% used, {final_available:.1f} GB available")
    
    # Save model
    output_dir = Path("training_output")
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / "ultra_simple_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'categories': label_to_idx,
        'model_params': total_params,
        'training_stats': {
            'successful_samples': successful_samples,
            'failed_samples': failed_samples,
            'avg_loss': avg_loss,
            'total_samples_attempted': max_samples
        }
    }, model_path)
    
    logger.info(f"💾 Model saved to: {model_path}")
    
    # Save training report
    report = {
        'training_type': 'ultra_simple_emergency',
        'model_parameters': total_params,
        'categories': label_to_idx,
        'dataset_size': len(image_paths),
        'samples_trained': max_samples,
        'successful_samples': successful_samples,
        'failed_samples': failed_samples,
        'average_loss': avg_loss,
        'final_memory_usage': final_usage
    }
    
    report_path = output_dir / "ultra_simple_training_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📝 Training report saved to: {report_path}")
    logger.info("🎯 ULTRA-SIMPLE TRAINING COMPLETED SUCCESSFULLY!")
    
    return {
        'successful_samples': successful_samples,
        'failed_samples': failed_samples,
        'avg_loss': avg_loss,
        'model_path': str(model_path),
        'report_path': str(report_path)
    }

if __name__ == "__main__":
    try:
        results = ultra_simple_training()
        logger.info("✅ Emergency training system completed!")
        
    except Exception as e:
        logger.error(f"💥 Emergency training failed: {e}")
        import traceback
        traceback.print_exc()
