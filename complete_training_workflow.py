#!/usr/bin/env python3
"""
🚀 Complete Data Processing & Training Pipeline
==============================================

รัน data augmentation, image preprocessing และ training ด้วยระบบใหม่

Usage:
    python complete_training_workflow.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main workflow execution"""
    print("🚀 Amulet-AI Complete Training Workflow")
    print("=" * 60)
    
    # Check dataset
    data_path = project_root / "organized_dataset" / "DATA SET"
    if not data_path.exists():
        print(f"❌ Dataset not found: {data_path}")
        return
    
    # Count files
    image_files = list(data_path.glob("*.png")) + list(data_path.glob("*.jpg"))
    print(f"📊 Found {len(image_files)} images in dataset")
    
    try:
        # Import modern training pipeline
        from model_training.pipeline import ModernTrainingPipeline
        
        print("\n🎯 Step 1: Initialize Modern Training Pipeline")
        pipeline = ModernTrainingPipeline()
        
        print("\n🔄 Step 2: Starting Complete Training Process")
        print("   - Data loading and preprocessing")
        print("   - Advanced augmentation")
        print("   - Transfer learning training")
        print("   - Model evaluation")
        print("   - Model saving")
        
        # Start training
        results = pipeline.quick_train(str(data_path))
        
        print("\n✅ Training Completed Successfully!")
        print("=" * 60)
        print(f"📈 Final Results:")
        print(f"   - Accuracy: {results.get('accuracy', 'N/A'):.4f}")
        print(f"   - F1 Score: {results.get('f1_score', 'N/A'):.4f}")
        print(f"   - Training Time: {results.get('training_time', 'N/A'):.2f}s")
        print(f"   - Model Path: {results.get('model_path', 'N/A')}")
        
        # Save results
        results_path = project_root / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_path}")
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required modules are available")
        return None
    except Exception as e:
        print(f"❌ Training error: {e}")
        logger.exception("Training failed")
        return None

if __name__ == "__main__":
    results = main()
    
    if results:
        print("\n🎉 Workflow completed successfully!")
    else:
        print("\n💔 Workflow failed!")
        sys.exit(1)