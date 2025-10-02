#!/usr/bin/env python3
"""
📊 Step 1: Dataset Analysis
==========================

Analyze the current dataset structure and prepare for training.
"""

import os
from pathlib import Path
from collections import defaultdict
import json

def analyze_dataset(data_path):
    """Analyze dataset structure and statistics"""
    
    print("🔍 Dataset Analysis")
    print("=" * 60)
    
    data_path = Path(data_path)
    
    # Count files by class
    class_counts = defaultdict(int)
    total_files = 0
    file_extensions = defaultdict(int)
    
    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    
    for item in data_path.iterdir():
        if item.is_file():
            # Files in root directory
            if item.suffix.lower() in image_extensions:
                # Extract class from filename
                parts = item.stem.split('_')
                if len(parts) >= 2:
                    class_name = '_'.join(parts[:-2])  # Remove last 2 parts (back/front and number)
                    class_counts[class_name] += 1
                    total_files += 1
                    file_extensions[item.suffix.lower()] += 1
        
        elif item.is_dir():
            # Folders
            class_name = item.name
            folder_count = 0
            
            for file in item.rglob('*'):
                if file.is_file() and file.suffix.lower() in image_extensions:
                    folder_count += 1
                    file_extensions[file.suffix.lower()] += 1
            
            if folder_count > 0:
                class_counts[class_name] += folder_count
                total_files += folder_count
    
    # Display results
    print(f"📁 Dataset Path: {data_path}")
    print(f"📊 Total Images: {total_files}")
    print(f"🏷️  Total Classes: {len(class_counts)}")
    
    print(f"\n📋 Class Distribution:")
    for class_name, count in sorted(class_counts.items()):
        percentage = (count / total_files) * 100
        print(f"   {class_name:30} | {count:3d} images ({percentage:5.1f}%)")
    
    print(f"\n📝 File Extensions:")
    for ext, count in sorted(file_extensions.items()):
        percentage = (count / total_files) * 100
        print(f"   {ext:10} | {count:3d} files ({percentage:5.1f}%)")
    
    # Calculate balance
    counts = list(class_counts.values())
    min_count = min(counts)
    max_count = max(counts)
    balance_ratio = min_count / max_count if max_count > 0 else 0
    
    print(f"\n⚖️  Dataset Balance:")
    print(f"   Min class size: {min_count}")
    print(f"   Max class size: {max_count}")
    print(f"   Balance ratio: {balance_ratio:.2f}")
    
    if balance_ratio < 0.5:
        print(f"   ⚠️  Dataset is imbalanced - consider data augmentation")
    else:
        print(f"   ✅ Dataset is reasonably balanced")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    # Training split recommendations
    if total_files < 100:
        print(f"   📊 Small dataset - use 70/15/15 split with strong augmentation")
        train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    elif total_files < 500:
        print(f"   📊 Medium dataset - use 70/20/10 split with moderate augmentation")
        train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
    else:
        print(f"   📊 Large dataset - use 80/15/5 split")
        train_ratio, val_ratio, test_ratio = 0.8, 0.15, 0.05
    
    # Augmentation recommendations
    if min_count < 20:
        print(f"   🔄 Strong augmentation needed (rotation, flip, color, crop)")
    elif min_count < 50:
        print(f"   🔄 Moderate augmentation recommended (rotation, flip)")
    else:
        print(f"   🔄 Light augmentation sufficient")
    
    # Model recommendations
    if total_files < 200:
        print(f"   🎯 Use transfer learning with frozen features")
        print(f"   🎯 Start with MobileNet or EfficientNet-B0")
    else:
        print(f"   🎯 Use transfer learning with fine-tuning")
        print(f"   🎯 Consider ResNet50 or EfficientNet-B3")
    
    return {
        'total_files': total_files,
        'classes': dict(class_counts),
        'balance_ratio': balance_ratio,
        'recommended_splits': {
            'train': train_ratio,
            'val': val_ratio, 
            'test': test_ratio
        }
    }

if __name__ == "__main__":
    # Analyze the dataset
    data_path = "organized_dataset/DATA SET"
    
    if os.path.exists(data_path):
        stats = analyze_dataset(data_path)
        
        # Save analysis results
        with open("dataset_analysis.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✅ Analysis saved to dataset_analysis.json")
    else:
        print(f"❌ Dataset path not found: {data_path}")