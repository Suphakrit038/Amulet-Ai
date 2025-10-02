"""
🧪 Complete Data Management Examples
====================================

ตัวอย่างการใช้งาน Data Management System ทั้งหมด

Includes:
- Augmentation pipeline
- Preprocessing
- Quality checking
- Dataset loading
- Sampling & splitting

Author: Amulet-AI Team
Date: October 2025
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data_management.augmentation import create_pipeline_from_preset
from data_management.preprocessing import (
    create_basic_preprocessor,
    create_artifact_preprocessor,
    create_strict_checker
)
from data_management.dataset import (
    AmuletDataset,
    create_balanced_sampler,
    split_dataset_stratified,
    analyze_distribution,
    print_distribution
)


# ============================================================================
# Example 1: Complete Preprocessing Pipeline
# ============================================================================

def example_1_complete_preprocessing():
    """ตัวอย่าง preprocessing pipeline แบบสมบูรณ์"""
    print("\n" + "="*70)
    print("📋 Example 1: Complete Preprocessing Pipeline")
    print("="*70)
    
    # Create test image
    test_img = Image.new('RGB', (400, 400), color=(100, 120, 140))
    
    # Step 1: Quality check
    print("\n🔍 Step 1: Quality Check")
    quality_checker = create_strict_checker()
    metrics = quality_checker.check_quality(test_img)
    
    print(f"  Blur score: {metrics.blur_score:.2f}")
    print(f"  Brightness: {metrics.brightness:.1f}")
    print(f"  Contrast: {metrics.contrast:.1f}")
    print(f"  Overall score: {metrics.overall_score:.1f}/100")
    print(f"  Status: {'✅ PASS' if metrics.passed else '❌ FAIL'}")
    
    # Step 2: Advanced preprocessing
    print("\n🔬 Step 2: Advanced Preprocessing")
    preprocessor = create_artifact_preprocessor()
    enhanced_img = preprocessor(test_img)
    print(f"  Applied: {preprocessor.get_config()}")
    
    # Step 3: Basic preprocessing
    print("\n⚙️ Step 3: Basic Preprocessing & Normalization")
    basic_prep = create_basic_preprocessor(image_size=224)
    tensor = basic_prep(enhanced_img)
    print(f"  Output shape: {tensor.shape}")
    print(f"  Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    
    print("\n✅ Preprocessing pipeline complete!")


# ============================================================================
# Example 2: Dataset Creation & Splitting
# ============================================================================

def example_2_dataset_splitting():
    """ตัวอย่างการสร้าง dataset และแบ่ง train/val/test"""
    print("\n" + "="*70)
    print("📋 Example 2: Dataset Creation & Splitting")
    print("="*70)
    
    # Create dummy labels
    print("\n📦 Creating dummy dataset...")
    labels = [0]*100 + [1]*80 + [2]*120 + [3]*60 + [4]*40 + [5]*90
    print(f"  Total samples: {len(labels)}")
    print(f"  Number of classes: {len(set(labels))}")
    
    # Analyze original distribution
    print("\n📊 Original Distribution:")
    original_stats = analyze_distribution(labels)
    print_distribution(original_stats)
    
    # Split dataset
    print("\n✂️ Splitting dataset (70/15/15)...")
    train_idx, val_idx, test_idx = split_dataset_stratified(
        labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )
    
    # Analyze train split
    print("\n📊 Training Split Distribution:")
    train_stats = analyze_distribution(labels, train_idx)
    print_distribution(train_stats)
    
    # Verify stratification
    print("\n✅ Stratification Verification:")
    print(f"  Original imbalance ratio: {original_stats['imbalance_ratio']:.2f}:1")
    print(f"  Train imbalance ratio: {train_stats['imbalance_ratio']:.2f}:1")
    print(f"  Difference: {abs(original_stats['imbalance_ratio'] - train_stats['imbalance_ratio']):.3f}")


# ============================================================================
# Example 3: Balanced Sampling
# ============================================================================

def example_3_balanced_sampling():
    """ตัวอย่างการสร้าง balanced sampler"""
    print("\n" + "="*70)
    print("📋 Example 3: Balanced Sampling")
    print("="*70)
    
    # Create imbalanced dataset
    labels = [0]*10 + [1]*100 + [2]*200 + [3]*30 + [4]*5 + [5]*50
    
    print("\n📊 Original Distribution (Highly Imbalanced):")
    from collections import Counter
    counts = Counter(labels)
    for cls, count in sorted(counts.items()):
        print(f"  Class {cls}: {count:3d} samples ({count/len(labels)*100:5.1f}%)")
    
    # Create weighted sampler
    print("\n⚖️ Creating weighted sampler...")
    sampler = create_balanced_sampler(labels, strategy='weighted')
    
    # Simulate sampling
    print("\n🎲 Simulating 1000 samples...")
    sampled_labels = []
    for i, idx in enumerate(sampler):
        sampled_labels.append(labels[idx])
        if i >= 999:
            break
    
    # Analyze sampled distribution
    print("\n📊 Sampled Distribution (After Balancing):")
    sampled_counts = Counter(sampled_labels)
    for cls, count in sorted(sampled_counts.items()):
        print(f"  Class {cls}: {count:3d} samples ({count/len(sampled_labels)*100:5.1f}%)")
    
    print("\n✅ Classes are now more balanced!")


# ============================================================================
# Example 4: Full Training Pipeline
# ============================================================================

def example_4_full_pipeline():
    """ตัวอย่าง pipeline สมบูรณ์สำหรับ training"""
    print("\n" + "="*70)
    print("📋 Example 4: Full Training Pipeline")
    print("="*70)
    
    print("\n🔧 Pipeline Components:")
    print("  1️⃣  Augmentation: medium preset")
    print("  2️⃣  Preprocessing: artifact preprocessor")
    print("  3️⃣  Quality check: strict checker")
    print("  4️⃣  Sampling: weighted balanced sampler")
    
    # Create augmentation pipeline
    print("\n🎨 Creating augmentation pipeline...")
    aug_pipeline = create_pipeline_from_preset(
        'medium',
        batch_size=32,
        num_classes=6,
        image_size=224
    )
    
    stats = aug_pipeline.get_augmentation_stats()
    print(f"  ✅ RandAugment: n={stats['rand_augment_ops']}, m={stats['rand_augment_magnitude']}")
    print(f"  ✅ RandomErasing: p={stats['random_erasing_prob']:.2f}")
    print(f"  ✅ MixUp: α={stats['mixup_alpha']:.2f}")
    print(f"  ✅ CutMix: α={stats['cutmix_alpha']:.2f}")
    
    # Create preprocessing
    print("\n🔬 Creating preprocessing pipeline...")
    preprocessor = create_artifact_preprocessor()
    config = preprocessor.get_config()
    print(f"  ✅ Denoising: {config['enable_denoise']}")
    print(f"  ✅ CLAHE: {config['enable_clahe']}")
    print(f"  ✅ Edge enhance: {config['enable_edge_enhance']}")
    
    # Create quality checker
    print("\n🔍 Creating quality checker...")
    quality_checker = create_strict_checker()
    print(f"  ✅ Strict quality thresholds enabled")
    
    # Create dummy labels
    labels = [0]*100 + [1]*80 + [2]*120 + [3]*60 + [4]*40 + [5]*90
    
    # Create sampler
    print("\n⚖️ Creating balanced sampler...")
    sampler = create_balanced_sampler(labels, strategy='weighted')
    print(f"  ✅ Weighted sampling for {len(set(labels))} classes")
    
    print("\n🚀 Pipeline ready for training!")
    print("\n📝 Usage:")
    print("  1. Load images")
    print("  2. Run quality check (filter bad images)")
    print("  3. Apply preprocessing (CLAHE, denoising, edge enhance)")
    print("  4. Apply augmentation (RandAugment, MixUp/CutMix)")
    print("  5. Use balanced sampler in DataLoader")
    print("  6. Train model!")


# ============================================================================
# Example 5: Quality Check Batch
# ============================================================================

def example_5_quality_check_batch():
    """ตัวอย่างการตรวจสอบ quality แบบ batch"""
    print("\n" + "="*70)
    print("📋 Example 5: Batch Quality Checking")
    print("="*70)
    
    # Create checker
    quality_checker = create_strict_checker()
    
    # Create test images with different qualities
    print("\n🖼️ Creating test images...")
    test_images = []
    
    # Good image
    good_img = Image.new('RGB', (400, 400), color=(120, 140, 160))
    test_images.append(("Good", good_img))
    
    # Low resolution
    low_res = Image.new('RGB', (100, 100), color=(120, 140, 160))
    test_images.append(("Low Res", low_res))
    
    # Low contrast
    low_contrast = Image.new('RGB', (400, 400), color=(128, 128, 128))
    test_images.append(("Low Contrast", low_contrast))
    
    # Check each image
    print("\n🔍 Quality Check Results:")
    print("-" * 70)
    
    for name, img in test_images:
        metrics = quality_checker.check_quality(img)
        status = "✅ PASS" if metrics.passed else "❌ FAIL"
        
        print(f"\n{name}:")
        print(f"  Status: {status}")
        print(f"  Overall score: {metrics.overall_score:.1f}/100")
        print(f"  Resolution: {metrics.resolution[0]}x{metrics.resolution[1]}")
        print(f"  Blur score: {metrics.blur_score:.2f}")
        print(f"  Brightness: {metrics.brightness:.1f}")
        print(f"  Contrast: {metrics.contrast:.1f}")
        
        if metrics.issues:
            print(f"  Issues:")
            for issue in metrics.issues:
                print(f"    • {issue}")
    
    print("\n" + "-" * 70)
    print("✅ Batch quality check complete!")


# ============================================================================
# Example 6: Performance Comparison
# ============================================================================

def example_6_performance_comparison():
    """เปรียบเทียบ performance ของ presets ต่างๆ"""
    print("\n" + "="*70)
    print("📋 Example 6: Augmentation Preset Comparison")
    print("="*70)
    
    presets = ['minimal', 'light', 'medium', 'heavy']
    
    print("\n📊 Preset Comparison:")
    print("-" * 70)
    
    for preset in presets:
        pipeline = create_pipeline_from_preset(preset, batch_size=32)
        stats = pipeline.get_augmentation_stats()
        
        print(f"\n🎯 {preset.upper()} Preset:")
        print(f"  RandAugment: {'✅' if stats['rand_augment_enabled'] else '❌'} "
              f"(n={stats['rand_augment_ops']}, m={stats['rand_augment_magnitude']})")
        print(f"  RandomErasing: {'✅' if stats['random_erasing_enabled'] else '❌'} "
              f"(p={stats['random_erasing_prob']:.2f})")
        print(f"  MixUp: {'✅' if stats['mixup_enabled'] else '❌'} "
              f"(α={stats['mixup_alpha']:.2f})")
        print(f"  CutMix: {'✅' if stats['cutmix_enabled'] else '❌'} "
              f"(α={stats['cutmix_alpha']:.2f})")
    
    print("\n" + "-" * 70)
    print("\n💡 Recommendations:")
    print("  • minimal: Fast inference, no augmentation")
    print("  • light: Small datasets (100-500 images)")
    print("  • medium: Medium datasets (500-2000 images) ⭐ RECOMMENDED")
    print("  • heavy: Large datasets (>2000 images) or high augmentation needed")


# ============================================================================
# Main: Run All Examples
# ============================================================================

def main():
    """รัน examples ทั้งหมด"""
    print("\n" + "="*70)
    print("🚀 Amulet-AI Complete Data Management Examples")
    print("="*70)
    
    try:
        # Run all examples
        example_1_complete_preprocessing()
        example_2_dataset_splitting()
        example_3_balanced_sampling()
        example_4_full_pipeline()
        example_5_quality_check_batch()
        example_6_performance_comparison()
        
        print("\n" + "="*70)
        print("✅ All examples completed successfully!")
        print("="*70)
        
        print("\n📚 Next Steps:")
        print("  1. Try with real Amulet images from organized_dataset/")
        print("  2. Tune augmentation presets for your dataset size")
        print("  3. Implement Phase 2: Transfer Learning")
        print("  4. Start training your model!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
