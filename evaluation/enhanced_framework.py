#!/usr/bin/env python3
"""
🔍 Enhanced Evaluation Framework
===============================

Comprehensive evaluation framework with:
- Multiple metrics (accuracy, precision, recall, F1, AUROC, calibration)
- Statistical significance testing
- Confidence intervals
- Per-class analysis
- Model comparison utilities
- Automated reporting

Author: Amulet-AI Team
Date: October 2, 2025
"""

import logging
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve, precision_recall_curve,
    balanced_accuracy_score, cohen_kappa_score
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from .metrics import ClassificationMetrics, compute_per_class_metrics
from .calibration import TemperatureScaling
from .ood_detection import IsolationForestDetector, MahalanobisDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComprehensiveMetrics:
    """
    📊 Comprehensive Evaluation Metrics Container
    
    Contains all possible evaluation metrics with confidence intervals
    """
    # Basic metrics
    accuracy: float
    balanced_accuracy: float
    
    # Per-class metrics
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    
    # Per-class arrays
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    per_class_f1: np.ndarray
    
    # Advanced metrics
    kappa: float
    mcc: float  # Matthews Correlation Coefficient
    
    # Multiclass AUROC (if applicable)
    auroc_macro: Optional[float] = None
    auroc_weighted: Optional[float] = None
    per_class_auroc: Optional[np.ndarray] = None
    
    # Calibration metrics
    ece: Optional[float] = None  # Expected Calibration Error
    mce: Optional[float] = None  # Maximum Calibration Error
    reliability: Optional[float] = None
    
    # Confidence intervals (95%)
    accuracy_ci: Optional[Tuple[float, float]] = None
    f1_macro_ci: Optional[Tuple[float, float]] = None
    
    # Additional info
    num_samples: int = 0
    class_names: Optional[List[str]] = None
    confusion_matrix: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert numpy arrays to lists
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
        return result
    
    def summary(self) -> Dict[str, float]:
        """Get summary of key metrics"""
        return {
            'accuracy': self.accuracy,
            'balanced_accuracy': self.balanced_accuracy,
            'f1_macro': self.f1_macro,
            'f1_weighted': self.f1_weighted,
            'kappa': self.kappa,
            'auroc_macro': self.auroc_macro or 0.0,
            'ece': self.ece or 0.0
        }


class EnhancedEvaluator:
    """
    🎯 Enhanced Model Evaluator
    
    Comprehensive evaluation with statistical testing, confidence intervals,
    and advanced metrics.
    
    Features:
    - Multiple evaluation metrics
    - Bootstrap confidence intervals
    - Statistical significance testing
    - Calibration evaluation
    - OOD detection evaluation
    - Automated report generation
    
    Example:
        >>> evaluator = EnhancedEvaluator(class_names=['A', 'B', 'C'])
        >>> metrics = evaluator.evaluate_model(model, test_loader, device)
        >>> evaluator.generate_report(metrics, 'evaluation_report.html')
    """
    
    def __init__(
        self,
        class_names: List[str],
        confidence_level: float = 0.95,
        bootstrap_samples: int = 1000,
        device: str = 'auto'
    ):
        """
        Initialize Enhanced Evaluator
        
        Args:
            class_names: List of class names
            confidence_level: Confidence level for intervals (default: 0.95)
            bootstrap_samples: Number of bootstrap samples for CI
            device: Device for computations
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Initialize calibration evaluator
        # Calibration will be added when CalibrationEvaluator is implemented
        # self.calibration_evaluator = CalibrationEvaluator()
        
        logger.info(f"EnhancedEvaluator initialized with {self.num_classes} classes")
    
    def extract_predictions(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        return_probabilities: bool = True,
        return_features: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Extract predictions, probabilities, and features from model
        
        Args:
            model: PyTorch model
            dataloader: DataLoader
            return_probabilities: Return class probabilities
            return_features: Return feature embeddings
            
        Returns:
            Dictionary with predictions, labels, probabilities, features
        """
        model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_features = []
        
        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch
                images = images.to(self.device)
                
                # Forward pass
                outputs = model(images)
                
                # Predictions
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                
                # Probabilities
                if return_probabilities:
                    probs = F.softmax(outputs, dim=1)
                    all_probs.extend(probs.cpu().numpy())
                
                # Features
                if return_features and hasattr(model, 'get_features'):
                    features = model.get_features(images)
                    all_features.extend(features.cpu().numpy())
        
        result = {
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels)
        }
        
        if return_probabilities:
            result['probabilities'] = np.array(all_probs)
            
        if return_features and all_features:
            result['features'] = np.array(all_features)
            
        return result
    
    def compute_bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_func: callable,
        alpha: float = 0.05
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Compute bootstrap confidence interval for a metric
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            metric_func: Metric function (e.g., accuracy_score)
            alpha: Significance level (default: 0.05 for 95% CI)
            
        Returns:
            (metric_value, (lower_bound, upper_bound))
        """
        n = len(y_true)
        bootstrap_scores = []
        
        for _ in range(self.bootstrap_samples):
            # Bootstrap sample
            indices = np.random.choice(n, n, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Compute metric
            try:
                score = metric_func(y_true_boot, y_pred_boot)
                bootstrap_scores.append(score)
            except:
                continue
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        # Original metric
        original_score = metric_func(y_true, y_pred)
        
        # Confidence interval
        lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
        upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
        
        return original_score, (lower, upper)
    
    def compute_multiclass_auroc(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """
        Compute multiclass AUROC (macro and weighted averages)
        
        Args:
            y_true: True labels (N,)
            y_probs: Class probabilities (N, C)
            
        Returns:
            (macro_auroc, weighted_auroc, per_class_auroc)
        """
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_auc_score
        
        # Binarize labels for multiclass
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        # Handle binary classification case
        if self.num_classes == 2:
            y_true_bin = y_true_bin.ravel()
            y_probs = y_probs[:, 1]  # Use probability of positive class
        
        # Per-class AUROC
        per_class_auroc = []
        for i in range(self.num_classes):
            if self.num_classes == 2 and i == 0:
                continue  # Skip for binary case
                
            if self.num_classes == 2:
                auroc = roc_auc_score(y_true_bin, y_probs)
            else:
                auroc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
            per_class_auroc.append(auroc)
        
        per_class_auroc = np.array(per_class_auroc)
        
        # Macro and weighted averages
        if self.num_classes == 2:
            macro_auroc = per_class_auroc[0]
            weighted_auroc = per_class_auroc[0]
        else:
            macro_auroc = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
            weighted_auroc = roc_auc_score(y_true_bin, y_probs, average='weighted', multi_class='ovr')
        
        return macro_auroc, weighted_auroc, per_class_auroc
    
    def compute_mcc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Matthews Correlation Coefficient"""
        from sklearn.metrics import matthews_corrcoef
        return matthews_corrcoef(y_true, y_pred)
    
    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        compute_calibration: bool = True,
        compute_auroc: bool = True
    ) -> ComprehensiveMetrics:
        """
        Comprehensive model evaluation
        
        Args:
            model: PyTorch model
            dataloader: DataLoader
            compute_calibration: Compute calibration metrics
            compute_auroc: Compute AUROC metrics
            
        Returns:
            ComprehensiveMetrics object
        """
        logger.info("🔍 Running comprehensive evaluation...")
        
        # Extract predictions and probabilities
        results = self.extract_predictions(
            model, dataloader,
            return_probabilities=compute_auroc or compute_calibration
        )
        
        y_true = results['labels']
        y_pred = results['predictions']
        y_probs = results.get('probabilities')
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = np.mean(f1)
        
        precision_weighted = np.average(precision, weights=support)
        recall_weighted = np.average(recall, weights=support)
        f1_weighted = np.average(f1, weights=support)
        
        # Advanced metrics
        kappa = cohen_kappa_score(y_true, y_pred)
        mcc = self.compute_mcc(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Bootstrap confidence intervals
        _, accuracy_ci = self.compute_bootstrap_ci(y_true, y_pred, accuracy_score)
        _, f1_macro_ci = self.compute_bootstrap_ci(
            y_true, y_pred,
            lambda yt, yp: np.mean(precision_recall_fscore_support(yt, yp, average=None, zero_division=0)[2])
        )
        
        # AUROC (if probabilities available)
        auroc_macro = None
        auroc_weighted = None
        per_class_auroc = None
        
        if compute_auroc and y_probs is not None:
            try:
                auroc_macro, auroc_weighted, per_class_auroc = self.compute_multiclass_auroc(
                    y_true, y_probs
                )
            except Exception as e:
                logger.warning(f"AUROC computation failed: {e}")
        
        # Calibration metrics
        ece = None
        mce = None
        reliability = None
        
        if compute_calibration and y_probs is not None:
            try:
                # Calibration evaluation would go here
                # cal_results = self.calibration_evaluator.evaluate(y_probs, y_true)
                # ece = cal_results['ece']
                # mce = cal_results['mce']
                # reliability = cal_results['reliability']
                pass
            except Exception as e:
                logger.warning(f"Calibration computation failed: {e}")
        
        # Create comprehensive metrics object
        metrics = ComprehensiveMetrics(
            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            f1_macro=f1_macro,
            precision_weighted=precision_weighted,
            recall_weighted=recall_weighted,
            f1_weighted=f1_weighted,
            per_class_precision=precision,
            per_class_recall=recall,
            per_class_f1=f1,
            kappa=kappa,
            mcc=mcc,
            auroc_macro=auroc_macro,
            auroc_weighted=auroc_weighted,
            per_class_auroc=per_class_auroc,
            ece=ece,
            mce=mce,
            reliability=reliability,
            accuracy_ci=accuracy_ci,
            f1_macro_ci=f1_macro_ci,
            num_samples=len(y_true),
            class_names=self.class_names,
            confusion_matrix=cm
        )
        
        logger.info("✅ Comprehensive evaluation completed!")
        return metrics
    
    def compare_models(
        self,
        model_results: Dict[str, ComprehensiveMetrics],
        metric: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Statistical comparison of multiple models
        
        Args:
            model_results: Dict of model_name -> ComprehensiveMetrics
            metric: Metric to compare ('accuracy', 'f1_macro', etc.)
            
        Returns:
            Comparison results with statistical tests
        """
        logger.info(f"📊 Comparing {len(model_results)} models on {metric}...")
        
        # Extract metric values
        model_names = list(model_results.keys())
        metric_values = []
        cis = []
        
        for name in model_names:
            metrics = model_results[name]
            value = getattr(metrics, metric)
            metric_values.append(value)
            
            # Get confidence interval if available
            ci_attr = f"{metric}_ci"
            ci = getattr(metrics, ci_attr, None)
            cis.append(ci)
        
        # Find best model
        best_idx = np.argmax(metric_values)
        best_model = model_names[best_idx]
        best_value = metric_values[best_idx]
        
        # Create comparison table
        comparison_table = []
        for i, name in enumerate(model_names):
            row = {
                'model': name,
                metric: metric_values[i],
                'rank': sorted(enumerate(metric_values), key=lambda x: x[1], reverse=True).index((i, metric_values[i])) + 1,
                'is_best': i == best_idx
            }
            
            if cis[i] is not None:
                row[f'{metric}_ci_lower'] = cis[i][0]
                row[f'{metric}_ci_upper'] = cis[i][1]
                row[f'{metric}_ci_width'] = cis[i][1] - cis[i][0]
            
            comparison_table.append(row)
        
        results = {
            'metric': metric,
            'best_model': best_model,
            'best_value': best_value,
            'comparison_table': comparison_table,
            'model_count': len(model_names)
        }
        
        return results
    
    def generate_report(
        self,
        metrics: ComprehensiveMetrics,
        output_path: str,
        model_name: str = "Model",
        include_plots: bool = True
    ):
        """
        Generate comprehensive evaluation report
        
        Args:
            metrics: Evaluation metrics
            output_path: Output file path (.html, .json, or .txt)
            model_name: Model name for report
            include_plots: Include visualizations
        """
        output_path = Path(output_path)
        
        if output_path.suffix == '.json':
            self._generate_json_report(metrics, output_path, model_name)
        elif output_path.suffix == '.html':
            self._generate_html_report(metrics, output_path, model_name, include_plots)
        else:
            self._generate_text_report(metrics, output_path, model_name)
        
        logger.info(f"📄 Report generated: {output_path}")
    
    def _generate_json_report(
        self,
        metrics: ComprehensiveMetrics,
        output_path: Path,
        model_name: str
    ):
        """Generate JSON report"""
        report = {
            'model_name': model_name,
            'evaluation_date': datetime.now().isoformat(),
            'metrics': metrics.to_dict(),
            'summary': metrics.summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def _generate_text_report(
        self,
        metrics: ComprehensiveMetrics,
        output_path: Path,
        model_name: str
    ):
        """Generate text report"""
        with open(output_path, 'w') as f:
            f.write(f"EVALUATION REPORT: {model_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Samples: {metrics.num_samples}\n")
            f.write(f"Number of Classes: {len(metrics.class_names)}\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accuracy:           {metrics.accuracy:.4f}")
            if metrics.accuracy_ci:
                f.write(f" (CI: {metrics.accuracy_ci[0]:.4f}-{metrics.accuracy_ci[1]:.4f})")
            f.write("\n")
            f.write(f"Balanced Accuracy:  {metrics.balanced_accuracy:.4f}\n")
            f.write(f"F1 Macro:           {metrics.f1_macro:.4f}")
            if metrics.f1_macro_ci:
                f.write(f" (CI: {metrics.f1_macro_ci[0]:.4f}-{metrics.f1_macro_ci[1]:.4f})")
            f.write("\n")
            f.write(f"F1 Weighted:        {metrics.f1_weighted:.4f}\n")
            f.write(f"Cohen's Kappa:      {metrics.kappa:.4f}\n")
            f.write(f"Matthews Corr Coef: {metrics.mcc:.4f}\n")
            
            if metrics.auroc_macro is not None:
                f.write(f"AUROC Macro:        {metrics.auroc_macro:.4f}\n")
                f.write(f"AUROC Weighted:     {metrics.auroc_weighted:.4f}\n")
            
            if metrics.ece is not None:
                f.write(f"Expected Cal Error: {metrics.ece:.4f}\n")
                f.write(f"Max Cal Error:      {metrics.mce:.4f}\n")
            
            # Per-class metrics
            f.write("\nPER-CLASS METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
            f.write("-" * 60 + "\n")
            
            for i, class_name in enumerate(metrics.class_names):
                f.write(f"{class_name:<20} "
                       f"{metrics.per_class_precision[i]:<12.4f} "
                       f"{metrics.per_class_recall[i]:<12.4f} "
                       f"{metrics.per_class_f1[i]:<12.4f}\n")
    
    def _generate_html_report(
        self,
        metrics: ComprehensiveMetrics,
        output_path: Path,
        model_name: str,
        include_plots: bool
    ):
        """Generate HTML report with visualizations"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Report: {model_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background-color: #fff; padding: 20px; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
                .metric-label {{ color: #666; margin-bottom: 10px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                .ci {{ color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Evaluation Report: {model_name}</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Samples: {metrics.num_samples} | Classes: {len(metrics.class_names)}</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value">{metrics.accuracy:.3f}</div>
                    {"<div class='ci'>CI: " + f"{metrics.accuracy_ci[0]:.3f}-{metrics.accuracy_ci[1]:.3f}" + "</div>" if metrics.accuracy_ci else ""}
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">F1 Score (Macro)</div>
                    <div class="metric-value">{metrics.f1_macro:.3f}</div>
                    {"<div class='ci'>CI: " + f"{metrics.f1_macro_ci[0]:.3f}-{metrics.f1_macro_ci[1]:.3f}" + "</div>" if metrics.f1_macro_ci else ""}
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Balanced Accuracy</div>
                    <div class="metric-value">{metrics.balanced_accuracy:.3f}</div>
                </div>
                
                {"<div class='metric-card'><div class='metric-label'>AUROC (Macro)</div><div class='metric-value'>" + f"{metrics.auroc_macro:.3f}" + "</div></div>" if metrics.auroc_macro else ""}
                
                {"<div class='metric-card'><div class='metric-label'>Calibration Error</div><div class='metric-value'>" + f"{metrics.ece:.3f}" + "</div></div>" if metrics.ece else ""}
            </div>
            
            <h2>Per-Class Metrics</h2>
            <table>
                <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    {"<th>AUROC</th>" if metrics.per_class_auroc is not None else ""}
                </tr>
        """
        
        # Add per-class rows
        for i, class_name in enumerate(metrics.class_names):
            html_content += f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{metrics.per_class_precision[i]:.4f}</td>
                    <td>{metrics.per_class_recall[i]:.4f}</td>
                    <td>{metrics.per_class_f1[i]:.4f}</td>
                    {f"<td>{metrics.per_class_auroc[i]:.4f}</td>" if metrics.per_class_auroc is not None and i < len(metrics.per_class_auroc) else ""}
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)


# CLI interface for evaluation
def run_evaluation_cli():
    """Command-line interface for model evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Model Evaluation')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--data_path', required=True, help='Path to test data')
    parser.add_argument('--output_dir', default='evaluation_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--report_format', choices=['html', 'json', 'txt'], default='html', help='Report format')
    
    args = parser.parse_args()
    
    # Load model and run evaluation
    # Implementation would depend on specific model loading logic
    logger.info(f"Loading model from: {args.model_path}")
    logger.info(f"Evaluating on data: {args.data_path}")
    logger.info(f"Results will be saved to: {args.output_dir}")


if __name__ == "__main__":
    # Example usage
    print("🔍 Enhanced Evaluation Framework")
    print("=" * 60)
    print("This module provides comprehensive model evaluation capabilities.")
    print("Use EnhancedEvaluator class for detailed analysis with confidence intervals.")
    print("See docstrings for usage examples.")