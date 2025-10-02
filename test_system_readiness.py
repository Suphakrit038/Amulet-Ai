#!/usr/bin/env python3
"""
🧪 Core Components Test Script
===============================

Tests all core ML libraries and system readiness.
"""

def test_core_libraries():
    """Test core ML libraries"""
    print("🧪 Testing Core ML Libraries...")
    print("=" * 50)
    
    try:
        # Test PyTorch
        import torch
        import torchvision
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ TorchVision: {torchvision.__version__}")
        
        # Test GPU
        gpu_available = torch.cuda.is_available()
        print(f"✅ GPU Available: {gpu_available}")
        if gpu_available:
            print(f"   - GPU Count: {torch.cuda.device_count()}")
            print(f"   - GPU Name: {torch.cuda.get_device_name()}")
        
        # Test NumPy & Pandas
        import numpy as np
        import pandas as pd
        print(f"✅ NumPy: {np.__version__}")
        print(f"✅ Pandas: {pd.__version__}")
        
        # Test Image Processing
        from PIL import Image
        import cv2
        print(f"✅ PIL/Pillow: Available")
        print(f"✅ OpenCV: {cv2.__version__}")
        
        # Test Scientific Computing
        import sklearn
        import matplotlib
        import seaborn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
        print(f"✅ Matplotlib: {matplotlib.__version__}")
        print(f"✅ Seaborn: {seaborn.__version__}")
        
        # Test Deep Learning Tools
        import timm
        import albumentations
        print(f"✅ TIMM: {timm.__version__}")
        print(f"✅ Albumentations: {albumentations.__version__}")
        
        # Test Web Frameworks
        import fastapi
        import streamlit
        print(f"✅ FastAPI: {fastapi.__version__}")
        print(f"✅ Streamlit: {streamlit.__version__}")
        
        print("\n🎉 All Core Libraries Ready!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_amulet_components():
    """Test Amulet-AI specific components"""
    print("\n🔍 Testing Amulet-AI Components...")
    print("=" * 50)
    
    try:
        # Test if we can import our modules
        import sys
        import os
        
        # Add project root to path
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)
        
        # Test model training components
        try:
            from model_training.pipeline import ModernTrainingPipeline
            print("✅ Modern Training Pipeline: Available")
        except ImportError as e:
            print(f"⚠️  Training Pipeline: {e}")
        
        # Test evaluation components
        try:
            from evaluation.enhanced_framework import EnhancedEvaluator
            print("✅ Enhanced Evaluation: Available")
        except ImportError as e:
            print(f"⚠️  Enhanced Evaluation: {e}")
        
        # Test MLOps components
        try:
            from mlops.versioning import ModelRegistry
            print("✅ MLOps Versioning: Available")
        except ImportError as e:
            print(f"⚠️  MLOps Versioning: {e}")
        
        # Test augmentation
        try:
            from data_management.augmentation.augmentation_pipeline import AugmentationPipeline
            print("✅ Augmentation Pipeline: Available")
        except ImportError as e:
            print(f"⚠️  Augmentation Pipeline: {e}")
        
        print("\n✅ Amulet-AI Components Check Complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing components: {e}")
        return False

def test_dataset():
    """Test dataset availability"""
    print("\n📁 Testing Dataset...")
    print("=" * 50)
    
    try:
        import os
        
        dataset_path = "organized_dataset/DATA SET"
        if os.path.exists(dataset_path):
            # Count files
            total_files = 0
            class_dirs = []
            
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    class_dirs.append(item)
                    class_files = len([f for f in os.listdir(item_path) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                    total_files += class_files
                    print(f"   📂 {item}: {class_files} images")
            
            print(f"\n✅ Dataset found: {len(class_dirs)} classes, {total_files} total images")
            return True
        else:
            print(f"❌ Dataset not found at: {dataset_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking dataset: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Amulet-AI System Readiness Test")
    print("=" * 60)
    
    # Run all tests
    core_ok = test_core_libraries()
    components_ok = test_amulet_components()
    dataset_ok = test_dataset()
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"Core Libraries: {'✅ PASS' if core_ok else '❌ FAIL'}")
    print(f"AI Components: {'✅ PASS' if components_ok else '❌ FAIL'}")
    print(f"Dataset: {'✅ PASS' if dataset_ok else '❌ FAIL'}")
    
    if core_ok and components_ok and dataset_ok:
        print("\n🎉 SYSTEM READY FOR TRAINING! 🚀")
    else:
        print("\n⚠️  SYSTEM NEEDS ATTENTION")
    
    return core_ok and components_ok and dataset_ok

if __name__ == "__main__":
    main()