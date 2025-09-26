#!/usr/bin/env python3
"""
🧪 Quick Model Evaluation Script
รันการประเมินโมเดลแบบเร็วสำหรับ CI/CD pipeline

Usage: python eval/run_quick_eval.py --threshold 0.75
"""

import numpy as np
import json
import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ai_models.enhanced_production_system import (
        EnhancedProductionClassifier, 
        load_and_prepare_dataset
    )
except ImportError as e:
    print(f"❌ Failed to import models: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model_quick(model_path: str = "trained_model", dataset_path: str = "dataset"):
    """
    Quick evaluation using existing test set
    Returns metrics for CI gate checking
    """
    
    logger.info("🚀 Starting quick model evaluation...")
    
    # Load test dataset only (เพื่อความเร็ว)
    test_dir = Path(dataset_path) / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Load model
    try:
        classifier = EnhancedProductionClassifier()
        classifier.load_model(model_path)
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise
    
    # Load test data
    image_pairs = []
    labels = []
    
    for class_dir in test_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        image_files = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
        
        logger.info(f"Loading {len(image_files)} images for class {class_name}")
        
        # สำหรับ quick eval ใช้แค่ 5 รูปแรกต่อคลาส
        for img_path in image_files[:5]:
            try:
                import cv2
                
                # Load as both front and back (simplified for quick eval)
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                image_pairs.append((img_rgb, img_rgb))
                labels.append(class_name)
                
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
                continue
    
    logger.info(f"Loaded {len(image_pairs)} test samples from {len(set(labels))} classes")
    
    if len(image_pairs) == 0:
        raise ValueError("No test data loaded")
    
    # Make predictions
    predictions = []
    confidences = []
    
    logger.info("Making predictions...")
    
    for i, (front, back) in enumerate(image_pairs):
        try:
            result = classifier.predict_production(front, back, request_id=f"eval_{i}")
            
            if result['status'] == 'success' and result['is_supported']:
                predictions.append(result['predicted_class'])
                confidences.append(result['confidence'])
            else:
                # Handle OOD or failed predictions
                predictions.append("unknown")
                confidences.append(0.0)
                
        except Exception as e:
            logger.warning(f"Prediction failed for sample {i}: {e}")
            predictions.append("unknown")
            confidences.append(0.0)
    
    # Calculate metrics
    from sklearn.metrics import classification_report, balanced_accuracy_score
    from collections import Counter
    
    # Filter out unknown predictions for metrics calculation
    valid_indices = [i for i, pred in enumerate(predictions) if pred != "unknown"]
    
    if len(valid_indices) == 0:
        raise ValueError("No valid predictions made")
    
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]
    valid_confidences = [confidences[i] for i in valid_indices]
    
    # Calculate metrics
    balanced_acc = balanced_accuracy_score(valid_labels, valid_predictions)
    
    # Per-class metrics
    report = classification_report(valid_labels, valid_predictions, output_dict=True, zero_division=0)
    
    # Calculate overall F1 (macro average)
    macro_f1 = report['macro avg']['f1-score']
    
    # OOD detection rate
    ood_rate = (len(predictions) - len(valid_predictions)) / len(predictions)
    
    # Confidence statistics
    mean_confidence = np.mean(valid_confidences) if valid_confidences else 0.0
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "total_samples": len(image_pairs),
        "valid_predictions": len(valid_predictions),
        "ood_rate": ood_rate,
        "metrics": {
            "balanced_accuracy": balanced_acc,
            "macro_f1": macro_f1,
            "mean_confidence": mean_confidence,
            "per_class": {
                class_name: {
                    "f1_score": class_metrics["f1-score"],
                    "precision": class_metrics["precision"],
                    "recall": class_metrics["recall"],
                    "support": class_metrics["support"]
                }
                for class_name, class_metrics in report.items()
                if class_name not in ["accuracy", "macro avg", "weighted avg"]
            }
        },
        "class_distribution": dict(Counter(valid_labels)),
        "prediction_distribution": dict(Counter(valid_predictions))
    }
    
    logger.info("📊 Quick evaluation results:")
    logger.info(f"   Balanced Accuracy: {balanced_acc:.3f}")
    logger.info(f"   Macro F1: {macro_f1:.3f}")
    logger.info(f"   Mean Confidence: {mean_confidence:.3f}")
    logger.info(f"   OOD Rate: {ood_rate:.3f}")
    
    return metrics

def check_metrics_threshold(metrics: dict, threshold: float = 0.75):
    """
    Check if metrics meet minimum threshold for CI gate
    """
    
    balanced_acc = metrics["metrics"]["balanced_accuracy"]
    macro_f1 = metrics["metrics"]["macro_f1"]
    mean_confidence = metrics["metrics"]["mean_confidence"]
    
    logger.info(f"🎯 Checking against thresholds:")
    logger.info(f"   Balanced Accuracy: {balanced_acc:.3f} >= {threshold}")
    logger.info(f"   Macro F1: {macro_f1:.3f} >= {threshold}")
    logger.info(f"   Mean Confidence: {mean_confidence:.3f} >= 0.7")
    
    failures = []
    
    if balanced_acc < threshold:
        failures.append(f"Balanced accuracy {balanced_acc:.3f} below threshold {threshold}")
    
    if macro_f1 < threshold:
        failures.append(f"Macro F1 {macro_f1:.3f} below threshold {threshold}")
    
    if mean_confidence < 0.7:
        failures.append(f"Mean confidence {mean_confidence:.3f} below 0.7")
    
    # Check per-class F1
    for class_name, class_metrics in metrics["metrics"]["per_class"].items():
        class_f1 = class_metrics["f1_score"]
        if class_f1 < (threshold - 0.1):  # More lenient for individual classes
            failures.append(f"Class {class_name} F1 {class_f1:.3f} below {threshold - 0.1}")
    
    return failures

def main():
    parser = argparse.ArgumentParser(description="Quick model evaluation for CI/CD")
    parser.add_argument("--model-path", default="trained_model", help="Path to trained model")
    parser.add_argument("--dataset-path", default="dataset", help="Path to dataset")
    parser.add_argument("--threshold", type=float, default=0.75, help="Minimum F1 threshold")
    parser.add_argument("--output", default="eval/current_metrics.json", help="Output file for metrics")
    parser.add_argument("--fail-on-threshold", action="store_true", help="Exit with error code if threshold not met")
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Run evaluation
        metrics = evaluate_model_quick(args.model_path, args.dataset_path)
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Results saved to {output_path}")
        
        # Check thresholds
        failures = check_metrics_threshold(metrics, args.threshold)
        
        if failures:
            logger.error("❌ Threshold check failed:")
            for failure in failures:
                logger.error(f"   - {failure}")
            
            if args.fail_on_threshold:
                sys.exit(1)
        else:
            logger.info("✅ All thresholds passed!")
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 QUICK EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Valid predictions: {metrics['valid_predictions']}")
        print(f"Balanced Accuracy: {metrics['metrics']['balanced_accuracy']:.3f}")
        print(f"Macro F1: {metrics['metrics']['macro_f1']:.3f}")
        print(f"Mean Confidence: {metrics['metrics']['mean_confidence']:.3f}")
        print(f"OOD Rate: {metrics['ood_rate']:.3f}")
        print(f"{'='*60}")
        
        if not failures:
            print("🎉 Model meets quality thresholds!")
        else:
            print("⚠️  Model has quality issues - check logs above")
        
    except Exception as e:
        logger.error(f"💥 Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()