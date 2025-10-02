"""
📊 Classification Metrics
========================

Per-class and overall classification metrics.

Features:
- Per-class Precision, Recall, F1
- Balanced Accuracy
- Confusion Matrix
- Classification Report

Author: Amulet-AI Team
Date: October 2025
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix as sklearn_confusion_matrix,
    balanced_accuracy_score,
    classification_report
)
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """
    📈 Classification Metrics Container
    
    Attributes:
        per_class_precision: Precision per class
        per_class_recall: Recall per class
        per_class_f1: F1 score per class
        macro_avg_f1: Macro-averaged F1
        weighted_avg_f1: Weighted-averaged F1
        balanced_accuracy: Balanced accuracy score
        confusion_matrix: Confusion matrix (N x N)
        accuracy: Overall accuracy
        num_samples: Number of samples
    """
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    per_class_f1: np.ndarray
    macro_avg_f1: float
    weighted_avg_f1: float
    balanced_accuracy: float
    confusion_matrix: np.ndarray
    accuracy: float
    num_samples: int
    class_names: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        result = {
            'accuracy': self.accuracy,
            'balanced_accuracy': self.balanced_accuracy,
            'macro_f1': self.macro_avg_f1,
            'weighted_f1': self.weighted_avg_f1,
            'num_samples': self.num_samples,
        }
        
        # Per-class metrics
        for i in range(len(self.per_class_f1)):
            class_name = self.class_names[i] if self.class_names else f"class_{i}"
            result[f'{class_name}_precision'] = self.per_class_precision[i]
            result[f'{class_name}_recall'] = self.per_class_recall[i]
            result[f'{class_name}_f1'] = self.per_class_f1[i]
        
        return result
    
    def print_report(self):
        """Print formatted classification report"""
        print("\n" + "="*70)
        print("📊 Classification Metrics Report")
        print("="*70)
        print(f"Overall Accuracy:       {self.accuracy:.4f}")
        print(f"Balanced Accuracy:      {self.balanced_accuracy:.4f}")
        print(f"Macro F1 Score:         {self.macro_avg_f1:.4f}")
        print(f"Weighted F1 Score:      {self.weighted_avg_f1:.4f}")
        print(f"Total Samples:          {self.num_samples}")
        
        print("\n" + "-"*70)
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
        print("-"*70)
        
        # Per-class metrics
        support = self.confusion_matrix.sum(axis=1)
        for i in range(len(self.per_class_f1)):
            class_name = self.class_names[i] if self.class_names else f"Class {i}"
            print(f"{class_name:<20} "
                  f"{self.per_class_precision[i]:<12.4f} "
                  f"{self.per_class_recall[i]:<12.4f} "
                  f"{self.per_class_f1[i]:<12.4f} "
                  f"{support[i]:<.0f}")
        
        print("="*70)


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> ClassificationMetrics:
    """
    คำนวณ metrics ทั้งหมด
    
    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        class_names: Optional class names
        
    Returns:
        ClassificationMetrics object
        
    Example:
        >>> metrics = compute_per_class_metrics(y_true, y_pred)
        >>> metrics.print_report()
        >>> print(f"Macro F1: {metrics.macro_avg_f1:.4f}")
    """
    # Per-class precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro and weighted averages
    macro_f1 = np.mean(f1)
    weighted_f1 = np.average(f1, weights=support)
    
    # Convert to numpy arrays if needed
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    cm = sklearn_confusion_matrix(y_true, y_pred)
    
    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)
    
    metrics = ClassificationMetrics(
        per_class_precision=precision,
        per_class_recall=recall,
        per_class_f1=f1,
        macro_avg_f1=macro_f1,
        weighted_avg_f1=weighted_f1,
        balanced_accuracy=balanced_acc,
        confusion_matrix=cm,
        accuracy=accuracy,
        num_samples=len(y_true),
        class_names=class_names
    )
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    คำนวณ confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: 'true', 'pred', 'all', or None
        
    Returns:
        Confusion matrix
    """
    cm = sklearn_confusion_matrix(y_true, y_pred)
    
    if normalize == 'true':
        cm = cm / cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        cm = cm / cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        cm = cm / cm.sum()
    
    return cm


def compute_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """คำนวณ balanced accuracy"""
    return balanced_accuracy_score(y_true, y_pred)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None
) -> ClassificationMetrics:
    """
    ประเมินโมเดลบน dataset
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: Device
        class_names: Optional class names
        
    Returns:
        ClassificationMetrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    return compute_per_class_metrics(y_true, y_pred, class_names)


if __name__ == "__main__":
    # Quick test
    print("📊 Classification Metrics Module")
    print("=" * 60)
    
    # Create dummy predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 6, 1000)
    y_pred = y_true.copy()
    # Add some errors
    errors = np.random.choice(1000, 200, replace=False)
    y_pred[errors] = np.random.randint(0, 6, 200)
    
    # Compute metrics
    class_names = ['Class A', 'Class B', 'Class C', 'Class D', 'Class E', 'Class F']
    metrics = compute_per_class_metrics(y_true, y_pred, class_names)
    
    # Print report
    metrics.print_report()
    
    print("\n✅ Metrics module ready!")
