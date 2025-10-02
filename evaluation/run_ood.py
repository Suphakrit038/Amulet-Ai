#!/usr/bin/env python3
"""
🚨 OOD Detection Runner
======================

Command-line tool for Out-of-Distribution (OOD) detection evaluation.
Supports multiple OOD detection methods and comprehensive evaluation.

Features:
- Multiple OOD detection algorithms
- Comprehensive evaluation metrics
- Model integration
- Batch processing
- Automated reporting

Usage:
    python run_ood.py --model_path model.pth --in_dist_data train/ --ood_data test_ood/
    python run_ood.py --model_path model.pth --config config.json

Author: Amulet-AI Team
Date: October 2, 2025
"""

import argparse
import logging
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from tqdm.auto import tqdm

from .ood_detection import (
    IsolationForestDetector, MahalanobisDetector, 
    extract_features, compute_ood_auroc
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfidenceBasedDetector:
    """
    🎯 Confidence-based OOD Detection
    
    Simple baseline that uses prediction confidence for OOD detection.
    Lower confidence = more likely to be OOD.
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def fit(self, confidence_scores: np.ndarray):
        """Fit threshold based on validation confidence scores"""
        # Use percentile as threshold
        self.threshold = np.percentile(confidence_scores, 10)  # Bottom 10%
        logger.info(f"Confidence threshold set to: {self.threshold:.4f}")
    
    def score(self, confidence_scores: np.ndarray) -> np.ndarray:
        """Return OOD scores (higher = more OOD)"""
        return 1.0 - confidence_scores  # Invert confidence
    
    def predict(self, confidence_scores: np.ndarray) -> np.ndarray:
        """Predict OOD (1) or in-dist (0)"""
        return (confidence_scores < self.threshold).astype(int)


class EnsembleOODDetector:
    """
    🔗 Ensemble OOD Detector
    
    Combines multiple OOD detection methods for improved performance.
    """
    
    def __init__(self, detectors: List, weights: Optional[List[float]] = None):
        self.detectors = detectors
        self.weights = weights or [1.0] * len(detectors)
        self.weights = np.array(self.weights) / np.sum(self.weights)  # Normalize
    
    def fit(self, **kwargs):
        """Fit all detectors"""
        for detector in self.detectors:
            detector.fit(**kwargs)
    
    def score(self, **kwargs) -> np.ndarray:
        """Ensemble scoring"""
        scores = []
        for detector in self.detectors:
            try:
                score = detector.score(**kwargs)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Detector {detector.__class__.__name__} failed: {e}")
                scores.append(np.zeros(len(kwargs)))
        
        if len(scores) == 0:
            raise RuntimeError("All detectors failed")
        
        # Weighted ensemble
        ensemble_score = np.average(scores, axis=0, weights=self.weights[:len(scores)])
        return ensemble_score


class OODEvaluationRunner:
    """
    🔍 OOD Detection Evaluation Runner
    
    Comprehensive evaluation framework for OOD detection methods.
    
    Features:
    - Multiple detection algorithms
    - Comprehensive metrics (AUROC, AUPR, FPR@95)
    - Model integration
    - Visualization and reporting
    - Batch evaluation
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'auto',
        batch_size: int = 32,
        num_workers: int = 4
    ):
        """
        Initialize OOD Evaluation Runner
        
        Args:
            model_path: Path to trained model
            device: Device for computation
            batch_size: Batch size for data loading
            num_workers: Number of data loading workers
        """
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Load model if provided
        self.model = None
        if model_path:
            self.load_model(model_path)
        
        # Available detectors
        self.detector_classes = {
            'isolation_forest': IsolationForestDetector,
            'mahalanobis': MahalanobisDetector,
            'confidence': ConfidenceBasedDetector
        }
        
        logger.info(f"OODEvaluationRunner initialized on {self.device}")
    
    def load_model(self, model_path: str):
        """Load trained model"""
        logger.info(f"Loading model from: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model state dict if saved as checkpoint
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                config = checkpoint.get('config', {})
                # Model reconstruction would need to be implemented based on config
                # For now, assume model is saved directly
            else:
                state_dict = checkpoint
            
            # This would need to be adapted based on actual model architecture
            # For now, assume AmuletTransferModel
            from ..model_training.transfer_learning import AmuletTransferModel
            
            # Try to infer model config
            num_classes = config.get('num_classes', 6)
            backbone = config.get('backbone', 'resnet50')
            
            self.model = AmuletTransferModel(
                backbone_name=backbone,
                num_classes=num_classes
            )
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def extract_model_outputs(
        self,
        dataloader: DataLoader,
        extract_features: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract predictions, confidence, and features from model
        
        Args:
            dataloader: DataLoader
            extract_features: Whether to extract feature embeddings
            
        Returns:
            Dictionary with predictions, confidence, features, labels
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Use load_model() first.")
        
        self.model.eval()
        
        all_predictions = []
        all_confidence = []
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting model outputs"):
                images, labels = batch
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Predictions and confidence
                probs = F.softmax(outputs, dim=1)
                confidence = probs.max(dim=1)[0]  # Max probability as confidence
                predictions = outputs.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_confidence.extend(confidence.cpu().numpy())
                all_labels.extend(labels.numpy())
                
                # Features
                if extract_features and hasattr(self.model, 'get_features'):
                    features = self.model.get_features(images)
                    all_features.extend(features.cpu().numpy())
        
        result = {
            'predictions': np.array(all_predictions),
            'confidence': np.array(all_confidence),
            'labels': np.array(all_labels)
        }
        
        if extract_features and all_features:
            result['features'] = np.array(all_features)
        
        return result
    
    def evaluate_detector(
        self,
        detector,
        in_dist_data: Dict[str, np.ndarray],
        ood_data: Dict[str, np.ndarray],
        fit_data: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Evaluate single OOD detector
        
        Args:
            detector: OOD detector instance
            in_dist_data: In-distribution data
            ood_data: OOD data
            fit_data: Data to fit detector (if None, uses in_dist_data)
            
        Returns:
            Evaluation metrics
        """
        fit_data = fit_data or in_dist_data
        
        # Fit detector
        if hasattr(detector, 'fit'):
            if 'features' in fit_data:
                detector.fit(fit_data['features'], fit_data.get('labels'))
            elif 'confidence' in fit_data:
                detector.fit(fit_data['confidence'])
        
        # Get scores
        if 'features' in in_dist_data:
            in_dist_scores = detector.score(in_dist_data['features'])
            ood_scores = detector.score(ood_data['features'])
        elif 'confidence' in in_dist_data:
            in_dist_scores = detector.score(in_dist_data['confidence'])
            ood_scores = detector.score(ood_data['confidence'])
        else:
            raise ValueError("No suitable data for detector scoring")
        
        # Combine scores and labels
        all_scores = np.concatenate([in_dist_scores, ood_scores])
        all_labels = np.concatenate([
            np.zeros(len(in_dist_scores)),  # 0 = in-distribution
            np.ones(len(ood_scores))        # 1 = OOD
        ])
        
        # Compute metrics
        auroc = roc_auc_score(all_labels, all_scores)
        aupr = average_precision_score(all_labels, all_scores)
        
        # FPR@95 (False Positive Rate at 95% True Positive Rate)
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        fpr_at_95_tpr = fpr[np.argmax(tpr >= 0.95)]
        
        # Detection accuracy at optimal threshold
        optimal_threshold = self._find_optimal_threshold(all_labels, all_scores)
        predictions = (all_scores > optimal_threshold).astype(int)
        detection_acc = np.mean(predictions == all_labels)
        
        metrics = {
            'auroc': auroc,
            'aupr': aupr,
            'fpr_at_95_tpr': fpr_at_95_tpr,
            'detection_accuracy': detection_acc,
            'optimal_threshold': optimal_threshold,
            'num_in_dist': len(in_dist_scores),
            'num_ood': len(ood_scores)
        }
        
        return metrics
    
    def _find_optimal_threshold(self, labels: np.ndarray, scores: np.ndarray) -> float:
        """Find optimal threshold using Youden's J statistic"""
        fpr, tpr, thresholds = roc_curve(labels, scores)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx]
    
    def run_comprehensive_evaluation(
        self,
        in_dist_dataloader: DataLoader,
        ood_dataloader: DataLoader,
        train_dataloader: Optional[DataLoader] = None,
        detectors: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive OOD evaluation
        
        Args:
            in_dist_dataloader: In-distribution test data
            ood_dataloader: OOD test data
            train_dataloader: Training data for fitting detectors
            detectors: List of detector names to evaluate
            output_dir: Output directory for results
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("🚨 Running comprehensive OOD evaluation...")
        
        detectors = detectors or ['isolation_forest', 'mahalanobis', 'confidence']
        
        # Extract model outputs
        logger.info("Extracting in-distribution data...")
        in_dist_data = self.extract_model_outputs(in_dist_dataloader)
        
        logger.info("Extracting OOD data...")
        ood_data = self.extract_model_outputs(ood_dataloader)
        
        # Extract training data for fitting
        train_data = None
        if train_dataloader:
            logger.info("Extracting training data...")
            train_data = self.extract_model_outputs(train_dataloader)
        
        # Evaluate each detector
        results = {}
        
        for detector_name in detectors:
            logger.info(f"Evaluating {detector_name} detector...")
            
            try:
                # Create detector
                if detector_name in self.detector_classes:
                    detector = self.detector_classes[detector_name]()
                else:
                    logger.warning(f"Unknown detector: {detector_name}")
                    continue
                
                # Evaluate
                metrics = self.evaluate_detector(
                    detector=detector,
                    in_dist_data=in_dist_data,
                    ood_data=ood_data,
                    fit_data=train_data
                )
                
                results[detector_name] = metrics
                
                logger.info(f"{detector_name} AUROC: {metrics['auroc']:.4f}")
                
            except Exception as e:
                logger.error(f"Evaluation of {detector_name} failed: {e}")
                results[detector_name] = {'error': str(e)}
        
        # Create ensemble if multiple detectors
        if len([r for r in results.values() if 'error' not in r]) > 1:
            logger.info("Creating ensemble detector...")
            try:
                ensemble_metrics = self._evaluate_ensemble(
                    in_dist_data, ood_data, train_data, 
                    [k for k, v in results.items() if 'error' not in v]
                )
                results['ensemble'] = ensemble_metrics
            except Exception as e:
                logger.warning(f"Ensemble evaluation failed: {e}")
        
        # Summary
        summary = self._create_evaluation_summary(results)
        
        # Complete results
        evaluation_results = {
            'evaluation_type': 'comprehensive_ood',
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'detectors_evaluated': list(results.keys()),
            'dataset_sizes': {
                'in_distribution': len(in_dist_data['labels']),
                'ood': len(ood_data['labels']),
                'training': len(train_data['labels']) if train_data else 0
            },
            'results': results,
            'summary': summary
        }
        
        # Save results
        if output_dir:
            self._save_evaluation_results(evaluation_results, output_dir)
        
        logger.info("✅ Comprehensive OOD evaluation completed!")
        
        return evaluation_results
    
    def _evaluate_ensemble(
        self,
        in_dist_data: Dict,
        ood_data: Dict,
        train_data: Optional[Dict],
        detector_names: List[str]
    ) -> Dict[str, float]:
        """Evaluate ensemble of detectors"""
        
        # Create individual detectors
        detectors = []
        for name in detector_names:
            if name in self.detector_classes:
                detectors.append(self.detector_classes[name]())
        
        if len(detectors) < 2:
            raise ValueError("Need at least 2 detectors for ensemble")
        
        # Create ensemble
        ensemble = EnsembleOODDetector(detectors)
        
        # Evaluate ensemble
        return self.evaluate_detector(
            ensemble, in_dist_data, ood_data, train_data
        )
    
    def _create_evaluation_summary(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Create summary of evaluation results"""
        
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return {'error': 'No valid results'}
        
        # Best detector by AUROC
        best_auroc = max(valid_results.items(), key=lambda x: x[1]['auroc'])
        
        # Average metrics
        avg_auroc = np.mean([r['auroc'] for r in valid_results.values()])
        avg_aupr = np.mean([r['aupr'] for r in valid_results.values()])
        avg_fpr95 = np.mean([r['fpr_at_95_tpr'] for r in valid_results.values()])
        
        summary = {
            'num_detectors_evaluated': len(valid_results),
            'num_detectors_failed': len(results) - len(valid_results),
            'best_detector': {
                'name': best_auroc[0],
                'auroc': best_auroc[1]['auroc'],
                'aupr': best_auroc[1]['aupr']
            },
            'average_performance': {
                'auroc': avg_auroc,
                'aupr': avg_aupr,
                'fpr_at_95_tpr': avg_fpr95
            },
            'performance_ranking': sorted(
                valid_results.items(), 
                key=lambda x: x[1]['auroc'], 
                reverse=True
            )
        }
        
        return summary
    
    def _save_evaluation_results(self, results: Dict, output_dir: str):
        """Save evaluation results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report
        json_path = output_path / f"ood_evaluation_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save CSV summary
        csv_path = output_path / f"ood_summary_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Detector', 'AUROC', 'AUPR', 'FPR@95', 'Detection_Accuracy'])
            
            for detector_name, metrics in results['results'].items():
                if 'error' not in metrics:
                    writer.writerow([
                        detector_name,
                        f"{metrics['auroc']:.4f}",
                        f"{metrics['aupr']:.4f}",
                        f"{metrics['fpr_at_95_tpr']:.4f}",
                        f"{metrics['detection_accuracy']:.4f}"
                    ])
        
        logger.info(f"Evaluation results saved to: {output_path}")
    
    def create_visualizations(
        self,
        results: Dict,
        output_dir: str
    ):
        """Create evaluation visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Performance comparison plot
        detector_names = []
        auroc_scores = []
        aupr_scores = []
        
        for name, metrics in results['results'].items():
            if 'error' not in metrics:
                detector_names.append(name)
                auroc_scores.append(metrics['auroc'])
                aupr_scores.append(metrics['aupr'])
        
        if detector_names:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # AUROC comparison
            ax1.bar(detector_names, auroc_scores)
            ax1.set_title('AUROC Comparison')
            ax1.set_ylabel('AUROC')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)
            
            # AUPR comparison
            ax2.bar(detector_names, aupr_scores)
            ax2.set_title('AUPR Comparison')
            ax2.set_ylabel('AUPR')
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_path / 'ood_performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to: {output_path}")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='OOD Detection Evaluation Runner')
    
    # Model and data
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--in_dist_data', required=True, help='In-distribution test data directory')
    parser.add_argument('--ood_data', required=True, help='OOD test data directory')
    parser.add_argument('--train_data', help='Training data directory (for fitting detectors)')
    
    # Configuration
    parser.add_argument('--config', help='JSON configuration file')
    parser.add_argument('--detectors', nargs='+', default=['isolation_forest', 'mahalanobis', 'confidence'],
                       choices=['isolation_forest', 'mahalanobis', 'confidence'],
                       help='OOD detectors to evaluate')
    
    # Output
    parser.add_argument('--output_dir', default='ood_evaluation_results', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    # Parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--device', default='auto', help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Initialize runner
    runner = OODEvaluationRunner(
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create data loaders (implementation would depend on dataset structure)
    # For now, this is a placeholder
    logger.info("Creating data loaders...")
    # in_dist_loader = create_dataloader(args.in_dist_data, args.batch_size, args.num_workers)
    # ood_loader = create_dataloader(args.ood_data, args.batch_size, args.num_workers)
    # train_loader = create_dataloader(args.train_data, args.batch_size, args.num_workers) if args.train_data else None
    
    print("⚠️  Data loader creation needs to be implemented based on your dataset structure.")
    print("Please modify the main() function to create appropriate data loaders.")
    
    # Run evaluation
    # results = runner.run_comprehensive_evaluation(
    #     in_dist_dataloader=in_dist_loader,
    #     ood_dataloader=ood_loader,
    #     train_dataloader=train_loader,
    #     detectors=args.detectors,
    #     output_dir=args.output_dir
    # )
    
    # Create visualizations
    # if args.visualize:
    #     runner.create_visualizations(results, args.output_dir)
    
    # Print summary
    # print(f"\n✅ OOD evaluation completed!")
    # print(f"Best detector: {results['summary']['best_detector']['name']}")
    # print(f"Best AUROC: {results['summary']['best_detector']['auroc']:.4f}")
    # print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()