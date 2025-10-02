"""
Complete Training Example: End-to-End Amulet Classification

This example demonstrates the complete pipeline:
1. Data loading with augmentation
2. Balanced sampling for imbalanced classes
3. Transfer learning model creation
4. Two-stage training (freeze → fine-tune)
5. Comprehensive evaluation
6. Model calibration
7. OOD detection setup
8. Model saving

Run with:
    python -m examples.complete_training_example

Author: Amulet-AI Team
Date: October 2, 2025
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Amulet-AI modules
from data_management.augmentation import create_pipeline_from_preset
from data_management.dataset import (
    create_amulet_dataset,
    create_balanced_sampler,
    split_dataset_stratified,
    analyze_distribution,
    print_distribution
)
from model_training.transfer_learning import create_transfer_model, TwoStageTrainer
from model_training.trainer import AmuletTrainer, TrainingConfig
from model_training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    MetricsLogger,
    CallbackList
)
from evaluation.metrics import evaluate_model, compute_per_class_metrics
from evaluation.calibration import calibrate_model, evaluate_calibration
from evaluation.ood_detection import IsolationForestDetector, extract_features


def main():
    """
    Complete training pipeline with best practices.
    """
    print("=" * 80)
    print("AMULET-AI COMPLETE TRAINING EXAMPLE")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------------------
    
    # Paths
    DATASET_ROOT = 'organized_dataset'
    SAVE_DIR = 'trained_model'
    
    # Hyperparameters
    NUM_CLASSES = 6  # Adjust to your dataset
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    
    # Model config
    BACKBONE = 'resnet50'  # 'resnet50', 'efficientnet_b0', 'mobilenet_v2'
    
    # Augmentation preset
    AUG_PRESET = 'medium'  # 'minimal', 'light', 'medium', 'heavy'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Using device: {device}")
    
    # -------------------------------------------------------------------------
    # STEP 1: DATA LOADING & AUGMENTATION
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("STEP 1: DATA LOADING & AUGMENTATION")
    print("=" * 80)
    
    # Create augmentation pipeline
    print(f"\n[INFO] Creating augmentation pipeline (preset: {AUG_PRESET})...")
    aug_pipeline = create_pipeline_from_preset(
        preset=AUG_PRESET,
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        image_size=224
    )
    
    # View augmentation stats
    print("\n[INFO] Augmentation configuration:")
    stats = aug_pipeline.get_augmentation_stats()
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    # Load datasets
    print("\n[INFO] Loading datasets...")
    
    try:
        train_dataset = create_amulet_dataset(
            root_dir=DATASET_ROOT,
            split='train',
            transform=aug_pipeline.get_transform('train')
        )
        
        val_dataset = create_amulet_dataset(
            root_dir=DATASET_ROOT,
            split='val',
            transform=aug_pipeline.get_transform('val')
        )
        
        test_dataset = create_amulet_dataset(
            root_dir=DATASET_ROOT,
            split='test',
            transform=aug_pipeline.get_transform('val')
        )
        
        print(f"  ✓ Train: {len(train_dataset)} images")
        print(f"  ✓ Val: {len(val_dataset)} images")
        print(f"  ✓ Test: {len(test_dataset)} images")
        
    except Exception as e:
        print(f"\n[ERROR] Could not load dataset: {e}")
        print("\n[INFO] Dataset should have structure:")
        print("  organized_dataset/")
        print("    train/")
        print("      class_0/")
        print("      class_1/")
        print("      ...")
        print("    val/")
        print("    test/")
        return
    
    # -------------------------------------------------------------------------
    # STEP 2: ANALYZE DISTRIBUTION & CREATE BALANCED SAMPLER
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("STEP 2: CLASS BALANCE ANALYSIS")
    print("=" * 80)
    
    # Get labels
    train_labels = [label for _, label in train_dataset]
    
    # Analyze distribution
    print("\n[INFO] Training set class distribution:")
    train_stats = analyze_distribution(train_labels, list(range(len(train_labels))))
    print_distribution(train_stats)
    
    # Create balanced sampler
    print("\n[INFO] Creating weighted sampler for balanced training...")
    sampler = create_balanced_sampler(
        train_labels,
        strategy='weighted',  # Options: 'weighted', 'stratified', 'balanced_batch'
        num_samples=len(train_labels)
    )
    
    # -------------------------------------------------------------------------
    # STEP 3: CREATE DATA LOADERS
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("STEP 3: DATA LOADERS")
    print("=" * 80)
    
    print("\n[INFO] Creating data loaders...")
    
    # Train loader with MixUp/CutMix and balanced sampler
    train_loader = aug_pipeline.create_dataloader(
        train_dataset,
        mode='train',
        sampler=sampler  # Use balanced sampler
    )
    
    # Val/Test loaders (no augmentation, no sampling)
    val_loader = aug_pipeline.create_dataloader(
        val_dataset,
        mode='val'
    )
    
    test_loader = aug_pipeline.create_dataloader(
        test_dataset,
        mode='val'
    )
    
    print(f"  ✓ Train batches: {len(train_loader)} (with MixUp/CutMix)")
    print(f"  ✓ Val batches: {len(val_loader)}")
    print(f"  ✓ Test batches: {len(test_loader)}")
    
    # -------------------------------------------------------------------------
    # STEP 4: CREATE MODEL
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("STEP 4: MODEL CREATION")
    print("=" * 80)
    
    print(f"\n[INFO] Creating transfer learning model ({BACKBONE})...")
    model = create_transfer_model(
        backbone=BACKBONE,
        num_classes=NUM_CLASSES,
        pretrained=True,
        device=device
    )
    
    # Print model info
    params = model.get_trainable_params()
    print(f"\n[INFO] Model parameters:")
    print(f"  - Total: {params['total']:,}")
    print(f"  - Trainable: {params['trainable']:,}")
    print(f"  - Frozen: {params['frozen']:,}")
    
    # -------------------------------------------------------------------------
    # STEP 5: TWO-STAGE TRAINING
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("STEP 5: TWO-STAGE TRAINING")
    print("=" * 80)
    
    # Create criterion with class weights
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print("\n[INFO] Class weights for loss:")
    for i, weight in enumerate(class_weights):
        print(f"  - Class {i}: {weight:.4f}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Create trainer
    two_stage_trainer = TwoStageTrainer(
        model=model,
        criterion=criterion,
        device=device
    )
    
    # Stage 1: Train head only
    print("\n" + "-" * 80)
    print("STAGE 1: TRAIN HEAD ONLY (FROZEN BACKBONE)")
    print("-" * 80)
    
    history1 = two_stage_trainer.train_stage1(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        lr=1e-3,
        patience=3
    )
    
    print("\n[INFO] Stage 1 complete!")
    print(f"  - Best val loss: {min(history1['val_loss']):.4f}")
    print(f"  - Best val acc: {max(history1['val_acc']):.2f}%")
    
    # Stage 2: Fine-tune last N layers
    print("\n" + "-" * 80)
    print("STAGE 2: FINE-TUNE LAST 10 LAYERS")
    print("-" * 80)
    
    history2 = two_stage_trainer.train_stage2(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        lr=1e-4,
        unfreeze_layers=10,
        patience=5
    )
    
    print("\n[INFO] Stage 2 complete!")
    print(f"  - Best val loss: {min(history2['val_loss']):.4f}")
    print(f"  - Best val acc: {max(history2['val_acc']):.2f}%")
    
    # -------------------------------------------------------------------------
    # STEP 6: COMPREHENSIVE EVALUATION
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("STEP 6: COMPREHENSIVE EVALUATION")
    print("=" * 80)
    
    print("\n[INFO] Evaluating on test set...")
    
    # Get class names (or use indices)
    class_names = [f'Class {i}' for i in range(NUM_CLASSES)]
    
    # Evaluate
    test_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        class_names=class_names
    )
    
    # Print detailed report
    print("\n" + "-" * 80)
    test_metrics.print_report()
    print("-" * 80)
    
    # Save metrics
    metrics_dict = test_metrics.to_dict()
    import json
    metrics_path = Path(SAVE_DIR) / 'test_metrics.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\n[INFO] Metrics saved to: {metrics_path}")
    
    # -------------------------------------------------------------------------
    # STEP 7: MODEL CALIBRATION
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("STEP 7: MODEL CALIBRATION")
    print("=" * 80)
    
    print("\n[INFO] Calibrating model with temperature scaling...")
    
    # Use validation set for calibration
    temp_scaler = calibrate_model(
        model=model,
        data_loader=val_loader,
        device=device
    )
    
    print(f"  ✓ Optimal temperature: {temp_scaler.temperature.item():.4f}")
    
    # Evaluate calibration
    print("\n[INFO] Evaluating calibration...")
    
    before_calib = evaluate_calibration(
        model=model,
        data_loader=test_loader,
        device=device,
        temp_scaler=None
    )
    
    after_calib = evaluate_calibration(
        model=model,
        data_loader=test_loader,
        device=device,
        temp_scaler=temp_scaler
    )
    
    print("\n[INFO] Calibration results:")
    print(f"  Before calibration:")
    print(f"    - ECE: {before_calib['ece']:.4f}")
    print(f"    - Brier Score: {before_calib['brier_score']:.4f}")
    print(f"    - Accuracy: {before_calib['accuracy']:.2%}")
    print(f"  After calibration:")
    print(f"    - ECE: {after_calib['ece']:.4f} (↓ {before_calib['ece'] - after_calib['ece']:.4f})")
    print(f"    - Brier Score: {after_calib['brier_score']:.4f}")
    print(f"    - Accuracy: {after_calib['accuracy']:.2%}")
    
    # -------------------------------------------------------------------------
    # STEP 8: OOD DETECTION SETUP
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("STEP 8: OOD DETECTION SETUP")
    print("=" * 80)
    
    print("\n[INFO] Training OOD detector on training set embeddings...")
    
    # Extract features
    train_features, train_targets = extract_features(
        model=model,
        data_loader=train_loader,
        device=device
    )
    
    print(f"  ✓ Extracted features: {train_features.shape}")
    
    # Train OOD detector
    ood_detector = IsolationForestDetector(contamination=0.01)
    ood_detector.fit(train_features)
    
    print(f"  ✓ OOD detector trained (contamination=0.01)")
    
    # Test on in-distribution data
    test_features, _ = extract_features(
        model=model,
        data_loader=test_loader,
        device=device
    )
    
    ood_scores = ood_detector.score(test_features)
    ood_predictions = ood_detector.predict(test_features)
    
    ood_rate = (ood_predictions == -1).mean()
    print(f"\n[INFO] OOD detection on test set (should be low):")
    print(f"  - OOD rate: {ood_rate:.2%}")
    print(f"  - Mean OOD score: {ood_scores.mean():.4f}")
    
    # -------------------------------------------------------------------------
    # STEP 9: SAVE MODEL & COMPONENTS
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("STEP 9: SAVE MODEL & COMPONENTS")
    print("=" * 80)
    
    save_path = Path(SAVE_DIR)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = save_path / 'best_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\n[INFO] Model saved: {model_path}")
    
    # Save temperature scaler
    temp_path = save_path / 'temperature_scaler.pth'
    torch.save(temp_scaler.state_dict(), temp_path)
    print(f"[INFO] Temperature scaler saved: {temp_path}")
    
    # Save OOD detector
    import joblib
    ood_path = save_path / 'ood_detector.joblib'
    joblib.dump(ood_detector, ood_path)
    print(f"[INFO] OOD detector saved: {ood_path}")
    
    # Save configuration
    config = {
        'backbone': BACKBONE,
        'num_classes': NUM_CLASSES,
        'image_size': 224,
        'class_names': class_names,
        'augmentation_preset': AUG_PRESET,
        'temperature': temp_scaler.temperature.item(),
        'test_metrics': metrics_dict,
        'calibration': {
            'before': before_calib,
            'after': after_calib
        }
    }
    
    config_path = save_path / 'model_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Configuration saved: {config_path}")
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    
    print(f"\n✓ Model: {BACKBONE}")
    print(f"✓ Test Accuracy: {test_metrics.accuracy:.2f}%")
    print(f"✓ Macro F1: {test_metrics.macro_avg_f1:.4f}")
    print(f"✓ Balanced Accuracy: {test_metrics.balanced_accuracy:.4f}")
    print(f"✓ ECE (calibrated): {after_calib['ece']:.4f}")
    print(f"✓ All artifacts saved to: {save_path.absolute()}")
    
    print("\n[INFO] To use the model:")
    print(f"""
    # Load model
    model = create_transfer_model('{BACKBONE}', {NUM_CLASSES})
    model.load_state_dict(torch.load('{model_path}'))
    
    # Load temperature scaler
    temp_scaler = TemperatureScaling()
    temp_scaler.load_state_dict(torch.load('{temp_path}'))
    
    # Load OOD detector
    ood_detector = joblib.load('{ood_path}')
    
    # Inference with calibration & OOD check
    with torch.no_grad():
        logits = model(image)
        probs = temp_scaler(logits)
        
        features = model.get_features(image)
        is_ood = ood_detector.predict(features.cpu().numpy())[0] == -1
        
        if is_ood:
            print("Warning: Out-of-distribution input detected!")
    """)
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
