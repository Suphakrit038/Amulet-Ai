"""
📊 Evaluation Module
===================

Comprehensive evaluation metrics and calibration.

Author: Amulet-AI Team
Date: October 2025
"""

from .metrics import (
    compute_per_class_metrics,
    compute_confusion_matrix,
    compute_balanced_accuracy,
    ClassificationMetrics
)

from .calibration import (
    TemperatureScaling,
    compute_ece,
    compute_brier_score,
    calibrate_model
)

from .ood_detection import (
    OODDetector,
    IsolationForestDetector,
    MahalanobisDetector,
    compute_ood_auroc
)

from .fid_kid import (
    FIDCalculator,
    KIDCalculator,
    compute_fid,
    compute_kid
)

__all__ = [
    # Metrics
    'compute_per_class_metrics',
    'compute_confusion_matrix',
    'compute_balanced_accuracy',
    'ClassificationMetrics',
    
    # Calibration
    'TemperatureScaling',
    'compute_ece',
    'compute_brier_score',
    'calibrate_model',
    
    # OOD Detection
    'OODDetector',
    'IsolationForestDetector',
    'MahalanobisDetector',
    'compute_ood_auroc',
    
    # FID/KID
    'FIDCalculator',
    'KIDCalculator',
    'compute_fid',
    'compute_kid',
]
