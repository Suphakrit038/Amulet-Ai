"""
Integration Tests for Complete ML System

Tests the integration of all Phase 1 & 2 components:
- Data loading → Augmentation → Training → Evaluation → Calibration → OOD → Explainability

Author: Amulet-AI Team
Date: October 2, 2025
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_dummy_dataset(root_dir: str, num_classes: int = 3, images_per_class: int = 20):
    """Create dummy dataset for testing"""
    print(f"[TEST] Creating dummy dataset: {num_classes} classes, {images_per_class} images each")
    
    root = Path(root_dir)
    
    for split in ['train', 'val', 'test']:
        for class_idx in range(num_classes):
            class_dir = root / split / f'class_{class_idx}'
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Create dummy images
            n_images = images_per_class if split == 'train' else images_per_class // 2
            
            for img_idx in range(n_images):
                # Create random image
                img = Image.fromarray(
                    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                )
                img.save(class_dir / f'img_{img_idx:03d}.jpg')
    
    print(f"  ✓ Dataset created at: {root_dir}")


def test_phase1_data_management():
    """Test Phase 1: Data Management"""
    print("\n" + "=" * 80)
    print("TEST 1: PHASE 1 - DATA MANAGEMENT")
    print("=" * 80)
    
    from data_management.augmentation import create_pipeline_from_preset
    from data_management.dataset import (
        create_amulet_dataset,
        create_balanced_sampler,
        split_dataset_stratified
    )
    
    # Create temp dataset
    temp_dir = tempfile.mkdtemp()
    
    try:
        create_dummy_dataset(temp_dir, num_classes=3, images_per_class=20)
        
        # Test 1.1: Augmentation Pipeline
        print("\n[1.1] Testing Augmentation Pipeline...")
        pipeline = create_pipeline_from_preset('medium', batch_size=4, num_classes=3)
        print("  ✓ Pipeline created")
        
        # Test 1.2: Dataset Loading
        print("\n[1.2] Testing Dataset Loading...")
        train_dataset = create_amulet_dataset(
            root_dir=temp_dir,
            split='train',
            transform=pipeline.get_transform('train')
        )
        print(f"  ✓ Loaded {len(train_dataset)} training images")
        
        # Test 1.3: Balanced Sampler
        print("\n[1.3] Testing Balanced Sampler...")
        labels = [label for _, label in train_dataset]
        sampler = create_balanced_sampler(labels, strategy='weighted')
        print("  ✓ Balanced sampler created")
        
        # Test 1.4: Data Loader with MixUp/CutMix
        print("\n[1.4] Testing DataLoader with MixUp/CutMix...")
        train_loader = pipeline.create_dataloader(
            train_dataset,
            mode='train',
            sampler=sampler
        )
        
        # Get one batch
        batch_data = next(iter(train_loader))
        
        # Check batch structure
        if len(batch_data) == 4:
            # MixUp/CutMix applied: (images, labels_a, labels_b, lambda)
            images, labels_a, labels_b, lam = batch_data
            print(f"  ✓ MixUp/CutMix batch: images {images.shape}, λ={lam:.3f}")
        elif len(batch_data) == 2:
            # Standard batch: (images, labels)
            images, labels = batch_data
            print(f"  ✓ Standard batch: images {images.shape}, labels {labels.shape}")
        else:
            raise ValueError(f"Unexpected batch format: {len(batch_data)} items")
        
        print("\n✅ Phase 1 Tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 1 Tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_phase2_transfer_learning():
    """Test Phase 2: Transfer Learning"""
    print("\n" + "=" * 80)
    print("TEST 2: PHASE 2 - TRANSFER LEARNING")
    print("=" * 80)
    
    from model_training.transfer_learning import (
        create_transfer_model,
        freeze_backbone,
        unfreeze_layers
    )
    
    try:
        # Test 2.1: Model Creation
        print("\n[2.1] Testing Model Creation...")
        device = 'cpu'  # Use CPU for testing
        
        for backbone in ['resnet50', 'efficientnet_b0', 'mobilenet_v2']:
            print(f"  Testing {backbone}...")
            model = create_transfer_model(
                backbone=backbone,
                num_classes=3,
                pretrained=True,
                device=device
            )
            
            # Test forward pass
            dummy_input = torch.randn(2, 3, 224, 224)
            output = model(dummy_input)
            
            assert output.shape == (2, 3), f"Wrong output shape: {output.shape}"
            print(f"    ✓ {backbone} works (output: {output.shape})")
        
        # Test 2.2: Freeze/Unfreeze
        print("\n[2.2] Testing Freeze/Unfreeze...")
        model = create_transfer_model('resnet50', num_classes=3, device=device)
        
        # Check trainable params
        params_before = model.get_trainable_params()
        print(f"  Before freeze: {params_before['trainable']:,} trainable")
        
        # Freeze
        freeze_backbone(model)
        params_after_freeze = model.get_trainable_params()
        print(f"  After freeze: {params_after_freeze['trainable']:,} trainable")
        
        assert params_after_freeze['trainable'] < params_before['trainable']
        
        # Unfreeze last 10 layers
        unfreeze_layers(model, n=10)
        params_after_unfreeze = model.get_trainable_params()
        print(f"  After unfreeze 10: {params_after_unfreeze['trainable']:,} trainable")
        
        assert params_after_unfreeze['trainable'] > params_after_freeze['trainable']
        
        print("\n✅ Phase 2 Transfer Learning Tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 2 Transfer Learning Tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase2_evaluation():
    """Test Phase 2: Evaluation"""
    print("\n" + "=" * 80)
    print("TEST 3: PHASE 2 - EVALUATION")
    print("=" * 80)
    
    from evaluation.metrics import compute_per_class_metrics
    from evaluation.calibration import TemperatureScaling, compute_ece
    from evaluation.ood_detection import IsolationForestDetector
    from evaluation.fid_kid import compute_fid
    
    try:
        # Test 3.1: Metrics
        print("\n[3.1] Testing Metrics...")
        y_true = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0]
        y_pred = [0, 0, 1, 2, 2, 2, 0, 1, 1, 0]
        
        metrics = compute_per_class_metrics(
            y_true, y_pred,
            class_names=['Class 0', 'Class 1', 'Class 2']
        )
        
        print(f"  Macro F1: {metrics.macro_avg_f1:.4f}")
        print(f"  Balanced Accuracy: {metrics.balanced_accuracy:.4f}")
        print("  ✓ Metrics computed")
        
        # Test 3.2: Temperature Scaling
        print("\n[3.2] Testing Temperature Scaling...")
        temp_scaler = TemperatureScaling()
        
        # Dummy logits
        logits = torch.randn(10, 3)
        scaled_logits = temp_scaler(logits)
        
        assert scaled_logits.shape == logits.shape
        print("  ✓ Temperature scaling works")
        
        # Test ECE
        probs = torch.softmax(logits, dim=1)
        labels = torch.randint(0, 3, (10,))
        ece = compute_ece(probs.numpy(), labels.numpy(), n_bins=5)
        print(f"  ECE: {ece:.4f}")
        
        # Test 3.3: OOD Detection
        print("\n[3.3] Testing OOD Detection...")
        detector = IsolationForestDetector(contamination=0.1)
        
        # Dummy features
        train_features = np.random.randn(50, 128)
        detector.fit(train_features)
        
        test_features = np.random.randn(10, 128)
        scores = detector.score(test_features)
        predictions = detector.predict(test_features)
        
        print(f"  OOD scores: {scores[:3]}")
        print(f"  Predictions: {predictions[:3]}")
        print("  ✓ OOD detector works")
        
        # Test 3.4: FID
        print("\n[3.4] Testing FID...")
        real_images = torch.randn(20, 3, 224, 224)
        fake_images = torch.randn(20, 3, 224, 224)
        
        fid = compute_fid(real_images, fake_images, device='cpu')
        print(f"  FID: {fid:.2f}")
        print("  ✓ FID computation works")
        
        print("\n✅ Phase 2 Evaluation Tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 2 Evaluation Tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase2_explainability():
    """Test Phase 2: Explainability"""
    print("\n" + "=" * 80)
    print("TEST 4: PHASE 2 - EXPLAINABILITY")
    print("=" * 80)
    
    from explainability.gradcam import GradCAM, get_target_layer
    from explainability.saliency import generate_saliency_map
    from model_training.transfer_learning import create_transfer_model
    
    try:
        # Create model
        device = 'cpu'
        model = create_transfer_model('resnet50', num_classes=3, device=device)
        model.eval()
        
        # Test 4.1: Grad-CAM
        print("\n[4.1] Testing Grad-CAM...")
        
        # Get target layer
        target_layer = get_target_layer(model, architecture='resnet')
        print(f"  Auto-detected target layer: {target_layer.__class__.__name__}")
        
        # Create Grad-CAM
        gradcam = GradCAM(model, target_layer, device=device)
        
        # Generate heatmap
        dummy_input = torch.randn(1, 3, 224, 224)
        heatmap = gradcam.generate(dummy_input, target_class=0)
        
        assert heatmap.shape == (224, 224)
        assert heatmap.min() >= 0 and heatmap.max() <= 1
        print(f"  ✓ Grad-CAM heatmap: {heatmap.shape}, range [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        
        # Test 4.2: Saliency Map
        print("\n[4.2] Testing Saliency Map...")
        
        saliency = generate_saliency_map(model, dummy_input, target_class=0)
        
        assert saliency.shape == (224, 224)
        print(f"  ✓ Saliency map: {saliency.shape}")
        
        print("\n✅ Phase 2 Explainability Tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 2 Explainability Tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_integration():
    """Test complete end-to-end integration"""
    print("\n" + "=" * 80)
    print("TEST 5: END-TO-END INTEGRATION")
    print("=" * 80)
    
    from data_management.augmentation import create_pipeline_from_preset
    from data_management.dataset import create_amulet_dataset, create_balanced_sampler
    from model_training.transfer_learning import create_transfer_model, TwoStageTrainer
    from evaluation.metrics import evaluate_model
    from evaluation.calibration import calibrate_model
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create dataset
        print("\n[5.1] Creating test dataset...")
        create_dummy_dataset(temp_dir, num_classes=3, images_per_class=10)
        
        # Setup pipeline
        print("\n[5.2] Setting up data pipeline...")
        pipeline = create_pipeline_from_preset('minimal', batch_size=4, num_classes=3)
        
        train_dataset = create_amulet_dataset(
            temp_dir, 'train',
            transform=pipeline.get_transform('train')
        )
        
        val_dataset = create_amulet_dataset(
            temp_dir, 'val',
            transform=pipeline.get_transform('val')
        )
        
        labels = [label for _, label in train_dataset]
        sampler = create_balanced_sampler(labels, strategy='weighted')
        
        train_loader = pipeline.create_dataloader(train_dataset, mode='train', sampler=sampler)
        val_loader = pipeline.create_dataloader(val_dataset, mode='val')
        
        print(f"  Train: {len(train_loader)} batches")
        print(f"  Val: {len(val_loader)} batches")
        
        # Create model
        print("\n[5.3] Creating model...")
        device = 'cpu'
        model = create_transfer_model('mobilenet_v2', num_classes=3, device=device)
        
        # Mini training
        print("\n[5.4] Mini training (2 epochs)...")
        criterion = nn.CrossEntropyLoss()
        trainer = TwoStageTrainer(model, criterion, device)
        
        # Just stage 1 with 2 epochs for testing
        history = trainer.train_stage1(
            train_loader, val_loader,
            epochs=2, lr=1e-3, patience=5
        )
        
        print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
        
        # Evaluation
        print("\n[5.5] Evaluation...")
        metrics = evaluate_model(
            model, val_loader, device,
            class_names=['Class 0', 'Class 1', 'Class 2']
        )
        
        print(f"  Accuracy: {metrics.accuracy:.2f}%")
        print(f"  Macro F1: {metrics.macro_avg_f1:.4f}")
        
        # Calibration
        print("\n[5.6] Calibration...")
        temp_scaler = calibrate_model(model, val_loader, device)
        print(f"  Optimal temperature: {temp_scaler.temperature.item():.4f}")
        
        print("\n✅ End-to-End Integration Test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ End-to-End Integration Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    """Run all integration tests"""
    print("=" * 80)
    print("AMULET-AI INTEGRATION TEST SUITE")
    print("Testing Phase 1 & 2 Components")
    print("=" * 80)
    
    results = {}
    
    # Run tests
    results['Phase 1: Data Management'] = test_phase1_data_management()
    results['Phase 2: Transfer Learning'] = test_phase2_transfer_learning()
    results['Phase 2: Evaluation'] = test_phase2_evaluation()
    results['Phase 2: Explainability'] = test_phase2_explainability()
    results['End-to-End Integration'] = test_end_to_end_integration()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print("\n" + "=" * 80)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 80)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is production-ready! 🚀")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == '__main__':
    exit(main())
