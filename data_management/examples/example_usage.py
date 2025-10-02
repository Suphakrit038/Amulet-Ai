#!/usr/bin/env python3
"""
🧪 Data Augmentation Examples & Testing
======================================

ตัวอย่างการใช้งาน Data Augmentation System

Usage:
    python example_usage.py
    
Author: Amulet-AI Team
Date: October 2025
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Import our augmentation modules
from data_management.augmentation import (
    create_pipeline_from_preset,
    MixUpAugmentation,
    CutMixAugmentation,
    RandAugmentPipeline
)


# ============================================================================
# Example 1: Basic Pipeline Usage
# ============================================================================

def example_1_basic_pipeline():
    """ตัวอย่างการใช้งาน Pipeline พื้นฐาน"""
    print("\n" + "="*60)
    print("📋 Example 1: Basic Pipeline Usage")
    print("="*60)
    
    # สร้าง pipeline แบบ medium preset
    pipeline = create_pipeline_from_preset(
        'medium',
        batch_size=32,
        num_classes=6,
        image_size=224
    )
    
    # ดู configuration
    print("\n📊 Pipeline Configuration:")
    stats = pipeline.get_augmentation_stats()
    for key, value in stats.items():
        print(f"  • {key}: {value}")
    
    # ดู transforms
    print("\n🔄 Training Transform:")
    print(f"  {pipeline.get_transform('train')}")
    
    print("\n✅ Validation Transform:")
    print(f"  {pipeline.get_transform('val')}")
    
    return pipeline


# ============================================================================
# Example 2: Custom Dataset with Augmentation
# ============================================================================

class SimpleAmuletDataset(Dataset):
    """ตัวอย่าง Dataset สำหรับทดสอบ"""
    
    def __init__(self, num_samples: int = 100, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        
        # สร้างข้อมูลปลอมสำหรับทดสอบ
        self.images = []
        self.labels = []
        
        for i in range(num_samples):
            # สร้างภาพสุ่ม (3 channels, 224x224)
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img)
            self.images.append(img)
            
            # สร้าง label สุ่ม (6 classes)
            label = np.random.randint(0, 6)
            self.labels.append(label)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label


def example_2_dataset_with_augmentation():
    """ตัวอย่างการใช้ Dataset กับ Augmentation"""
    print("\n" + "="*60)
    print("📋 Example 2: Dataset with Augmentation")
    print("="*60)
    
    # สร้าง pipeline
    pipeline = create_pipeline_from_preset('medium')
    
    # สร้าง dataset
    train_dataset = SimpleAmuletDataset(
        num_samples=100,
        transform=pipeline.get_transform('train')
    )
    
    val_dataset = SimpleAmuletDataset(
        num_samples=20,
        transform=pipeline.get_transform('val')
    )
    
    print(f"\n📊 Dataset Info:")
    print(f"  • Training samples: {len(train_dataset)}")
    print(f"  • Validation samples: {len(val_dataset)}")
    
    # สร้าง DataLoader
    train_loader = pipeline.create_dataloader(
        train_dataset,
        mode='train',
        batch_size=16
    )
    
    val_loader = pipeline.create_dataloader(
        val_dataset,
        mode='val',
        batch_size=16
    )
    
    # ดู batch แรก
    for batch_data in train_loader:
        if len(batch_data) == 4:  # MixUp/CutMix
            images, labels_a, labels_b, lam = batch_data
            print(f"\n🔄 MixUp/CutMix Batch:")
            print(f"  • Images shape: {images.shape}")
            print(f"  • Labels A: {labels_a[:5].tolist()}")
            print(f"  • Labels B: {labels_b[:5].tolist()}")
            print(f"  • Lambda: {lam:.3f}")
        else:
            images, labels = batch_data
            print(f"\n📦 Normal Batch:")
            print(f"  • Images shape: {images.shape}")
            print(f"  • Labels: {labels[:5].tolist()}")
        break
    
    return train_loader, val_loader


# ============================================================================
# Example 3: Training Loop with MixUp/CutMix
# ============================================================================

def example_3_training_loop():
    """ตัวอย่าง Training Loop พร้อม MixUp/CutMix"""
    print("\n" + "="*60)
    print("📋 Example 3: Training Loop with Augmentation")
    print("="*60)
    
    # สร้าง pipeline และ dataset
    pipeline = create_pipeline_from_preset('medium', batch_size=8)
    dataset = SimpleAmuletDataset(num_samples=50, transform=pipeline.get_transform('train'))
    loader = pipeline.create_dataloader(dataset, mode='train')
    
    # สร้างโมเดลง่ายๆ สำหรับทดสอบ
    class SimpleModel(nn.Module):
        def __init__(self, num_classes=6):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, num_classes)
        
        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n🏋️ Starting training simulation...")
    
    # Training loop (1 epoch สำหรับทดสอบ)
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch_data in enumerate(loader):
        if len(batch_data) == 4:  # MixUp/CutMix
            images, labels_a, labels_b, lam = batch_data
            
            # Forward
            outputs = model(images)
            
            # MixUp/CutMix loss
            loss = lam * criterion(outputs, labels_a) + \
                   (1 - lam) * criterion(outputs, labels_b)
            
            print(f"  Batch {batch_idx+1}: MixUp/CutMix loss = {loss.item():.4f}, λ = {lam:.3f}")
        else:
            images, labels = batch_data
            outputs = model(images)
            loss = criterion(outputs, labels)
            print(f"  Batch {batch_idx+1}: Normal loss = {loss.item():.4f}")
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx >= 3:  # แสดงแค่ 4 batches
            break
    
    avg_loss = total_loss / num_batches
    print(f"\n✅ Average loss: {avg_loss:.4f}")


# ============================================================================
# Example 4: Comparing Augmentation Presets
# ============================================================================

def example_4_compare_presets():
    """เปรียบเทียบ presets ต่างๆ"""
    print("\n" + "="*60)
    print("📋 Example 4: Comparing Augmentation Presets")
    print("="*60)
    
    presets = ['minimal', 'light', 'medium', 'heavy']
    
    for preset in presets:
        pipeline = create_pipeline_from_preset(preset)
        stats = pipeline.get_augmentation_stats()
        
        print(f"\n🎯 Preset: {preset.upper()}")
        print(f"  RandAugment: {'✅' if stats['rand_augment_enabled'] else '❌'} "
              f"(n={stats['rand_augment_ops']}, m={stats['rand_augment_magnitude']})")
        print(f"  RandomErasing: {'✅' if stats['random_erasing_enabled'] else '❌'} "
              f"(p={stats['random_erasing_prob']:.2f})")
        print(f"  MixUp: {'✅' if stats['mixup_enabled'] else '❌'} "
              f"(α={stats['mixup_alpha']:.2f})")
        print(f"  CutMix: {'✅' if stats['cutmix_enabled'] else '❌'} "
              f"(α={stats['cutmix_alpha']:.2f})")


# ============================================================================
# Example 5: Visualize Augmentations
# ============================================================================

def example_5_visualize_augmentations():
    """แสดงผลของ augmentations (requires matplotlib)"""
    print("\n" + "="*60)
    print("📋 Example 5: Visualize Augmentations")
    print("="*60)
    
    try:
        # สร้างภาพตัวอย่าง
        img = Image.new('RGB', (224, 224), color=(100, 150, 200))
        
        # สร้าง RandAugment
        rand_aug = RandAugmentPipeline(n=2, m=9)
        
        print("\n🎨 Applying RandAugment transformations...")
        
        # Apply หลายรอบ
        augmented_images = [img]
        for i in range(4):
            aug_img = rand_aug(img.copy())
            augmented_images.append(aug_img)
        
        print(f"  Generated {len(augmented_images)-1} augmented versions")
        print("  Note: Use matplotlib to visualize if needed")
        
    except Exception as e:
        print(f"  ⚠️  Visualization skipped: {e}")


# ============================================================================
# Example 6: Custom Configuration
# ============================================================================

def example_6_custom_configuration():
    """สร้าง custom configuration"""
    print("\n" + "="*60)
    print("📋 Example 6: Custom Configuration")
    print("="*60)
    
    # สร้าง custom config
    custom_config = {
        'image_size': 256,  # Larger images
        'batch_size': 64,   # Larger batch
        'rand_augment_n': 3,
        'rand_augment_m': 12,
        'random_erasing_p': 0.6,
        'mixup_alpha': 0.3,
        'cutmix_alpha': 1.2,
        'num_classes': 6,
        'num_workers': 8
    }
    
    # สร้าง pipeline
    pipeline = create_pipeline_from_preset('medium', **custom_config)
    
    print("\n⚙️ Custom Configuration:")
    stats = pipeline.get_augmentation_stats()
    for key, value in stats.items():
        print(f"  • {key}: {value}")
    
    print(f"\n📦 DataLoader Settings:")
    print(f"  • Batch size: {pipeline.config['batch_size']}")
    print(f"  • Num workers: {pipeline.config['num_workers']}")
    print(f"  • Image size: {pipeline.config['image_size']}")


# ============================================================================
# Main: Run All Examples
# ============================================================================

def main():
    """รัน examples ทั้งหมด"""
    print("\n" + "="*70)
    print("🚀 Amulet-AI Data Augmentation Examples")
    print("="*70)
    
    try:
        # Example 1: Basic pipeline
        pipeline = example_1_basic_pipeline()
        
        # Example 2: Dataset with augmentation
        train_loader, val_loader = example_2_dataset_with_augmentation()
        
        # Example 3: Training loop
        example_3_training_loop()
        
        # Example 4: Compare presets
        example_4_compare_presets()
        
        # Example 5: Visualize (optional)
        example_5_visualize_augmentations()
        
        # Example 6: Custom config
        example_6_custom_configuration()
        
        print("\n" + "="*70)
        print("✅ All examples completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
