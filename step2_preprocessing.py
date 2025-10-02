#!/usr/bin/env python3
"""
🔄 Step 2: Data Preprocessing & Augmentation
============================================

Prepare data for training with preprocessing and augmentation.
"""

import os
import shutil
from pathlib import Path
import json
import random
from PIL import Image
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

def create_data_splits(data_path, output_path, splits={'train': 0.8, 'val': 0.15, 'test': 0.05}):
    """Create train/val/test splits"""
    
    print("📂 Creating Data Splits")
    print("=" * 60)
    
    data_path = Path(data_path)
    output_path = Path(output_path)
    
    # Create output directories
    for split in splits.keys():
        split_dir = output_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all image files with their classes
    all_files = []
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    
    # Get files from root directory (with class extracted from filename)
    for item in data_path.iterdir():
        if item.is_file() and item.suffix.lower() in image_extensions:
            # Extract class from filename
            parts = item.stem.split('_')
            if len(parts) >= 2:
                class_name = '_'.join(parts[:-2])
                all_files.append((str(item), class_name))
        
        elif item.is_dir():
            # Files in subdirectories
            class_name = item.name
            for file in item.rglob('*'):
                if file.is_file() and file.suffix.lower() in image_extensions:
                    all_files.append((str(file), class_name))
    
    print(f"📊 Total files found: {len(all_files)}")
    
    # Group by class
    class_files = {}
    for file_path, class_name in all_files:
        if class_name not in class_files:
            class_files[class_name] = []
        class_files[class_name].append(file_path)
    
    print(f"🏷️  Classes found: {list(class_files.keys())}")
    
    # Split each class proportionally
    split_stats = {split: {} for split in splits.keys()}
    
    for class_name, files in class_files.items():
        print(f"\n📋 Processing class: {class_name} ({len(files)} files)")
        
        # Random shuffle
        random.shuffle(files)
        
        # Calculate split sizes
        n_files = len(files)
        n_train = int(n_files * splits['train'])
        n_val = int(n_files * splits['val'])
        n_test = n_files - n_train - n_val
        
        # Split files
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        # Create class directories in each split
        for split in splits.keys():
            class_dir = output_path / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files to respective splits
        for split, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            split_stats[split][class_name] = len(split_files)
            
            for i, file_path in enumerate(split_files):
                src_path = Path(file_path)
                dst_path = output_path / split / class_name / f"{class_name}_{split}_{i:03d}{src_path.suffix}"
                shutil.copy2(src_path, dst_path)
        
        print(f"   ✅ Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Print split statistics
    print(f"\n📊 Split Statistics:")
    for split in splits.keys():
        total = sum(split_stats[split].values())
        print(f"   {split.upper():5}: {total:3d} files")
        for class_name, count in split_stats[split].items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"      {class_name:30} | {count:3d} ({percentage:5.1f}%)")
    
    # Save split info
    split_info = {
        'splits': splits,
        'statistics': split_stats,
        'total_files': len(all_files),
        'classes': list(class_files.keys())
    }
    
    with open(output_path / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    
    return split_info

def setup_augmentations():
    """Setup augmentation pipelines"""
    
    print("\n🔄 Setting up Augmentation Pipelines")
    print("=" * 60)
    
    # Training augmentations (stronger)
    train_augment = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.Rotate(limit=15, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Validation/Test augmentations (minimal)
    val_augment = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    print("✅ Training augmentations: Resize, Crop, Flip, Rotate, Color, Noise, Blur")
    print("✅ Validation augmentations: Resize only")
    
    return train_augment, val_augment

def test_augmentations(data_path, output_path="augmentation_test"):
    """Test augmentation pipeline with sample images"""
    
    print(f"\n🧪 Testing Augmentation Pipeline")
    print("=" * 60)
    
    # Create test output directory
    test_dir = Path(output_path)
    test_dir.mkdir(exist_ok=True)
    
    # Get sample images
    data_path = Path(data_path)
    sample_files = []
    
    for item in data_path.rglob("*.png"):
        sample_files.append(item)
        if len(sample_files) >= 3:  # Test with 3 samples
            break
    
    if not sample_files:
        print("❌ No sample images found")
        return
    
    # Setup augmentations
    train_augment, val_augment = setup_augmentations()
    
    # Test each sample
    for i, sample_file in enumerate(sample_files):
        print(f"📸 Testing with: {sample_file.name}")
        
        # Load image
        try:
            image = Image.open(sample_file).convert('RGB')
            original_size = image.size
            
            # Convert to numpy for albumentations
            import numpy as np
            image_np = np.array(image)
            
            # Apply training augmentations (5 variations)
            for j in range(5):
                augmented = train_augment(image=image_np)
                tensor_image = augmented['image']
                
                # Convert back to PIL for saving
                # Denormalize
                tensor_denorm = tensor_image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                tensor_denorm = torch.clamp(tensor_denorm, 0, 1)
                
                # Convert to PIL
                pil_image = transforms.ToPILImage()(tensor_denorm)
                
                # Save
                output_file = test_dir / f"sample_{i}_aug_{j}.png"
                pil_image.save(output_file)
            
            # Apply validation augmentation
            val_augmented = val_augment(image=image_np)
            val_tensor = val_augmented['image']
            
            # Denormalize and save
            val_denorm = val_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            val_denorm = torch.clamp(val_denorm, 0, 1)
            val_pil = transforms.ToPILImage()(val_denorm)
            
            val_output = test_dir / f"sample_{i}_val.png"
            val_pil.save(val_output)
            
            print(f"   ✅ Original: {original_size} -> Processed: {val_pil.size}")
            
        except Exception as e:
            print(f"   ❌ Error processing {sample_file.name}: {e}")
    
    print(f"\n✅ Augmentation test completed. Check {output_path}/ for results")

if __name__ == "__main__":
    print("🔄 Data Preprocessing & Augmentation")
    print("=" * 80)
    
    # Paths
    data_path = "organized_dataset/DATA SET"
    splits_path = "organized_dataset/splits"
    
    # Step 1: Create data splits
    if os.path.exists(data_path):
        print("📂 Step 1: Creating data splits...")
        split_info = create_data_splits(data_path, splits_path)
        print("✅ Data splits created")
        
        # Step 2: Setup and test augmentations
        print("\n🔄 Step 2: Setting up augmentations...")
        train_aug, val_aug = setup_augmentations()
        
        # Step 3: Test augmentations
        print("\n🧪 Step 3: Testing augmentations...")
        test_augmentations(data_path)
        
        print(f"\n🎉 Preprocessing completed successfully!")
        print(f"📁 Data splits saved to: {splits_path}")
        print(f"🔄 Augmentation test saved to: augmentation_test/")
        
    else:
        print(f"❌ Dataset path not found: {data_path}")