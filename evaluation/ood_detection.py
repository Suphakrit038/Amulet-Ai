"""
🔍 Out-of-Distribution (OOD) Detection
=====================================

Detect inputs that are out-of-distribution.

Methods:
- IsolationForest (embedding space)
- Mahalanobis Distance
- Confidence-based thresholds

Author: Amulet-AI Team
Date: October 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score
import logging

logger = logging.getLogger(__name__)


class OODDetector:
    """
    🎯 Base OOD Detector
    
    Detect out-of-distribution samples based on features or confidence.
    """
    
    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None):
        """Fit detector on in-distribution data"""
        raise NotImplementedError
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict OOD (1) or in-dist (0)"""
        raise NotImplementedError
    
    def score(self, features: np.ndarray) -> np.ndarray:
        """Get OOD scores (higher = more OOD)"""
        raise NotImplementedError


class IsolationForestDetector(OODDetector):
    """
    🌲 Isolation Forest OOD Detector
    
    Detect OOD based on feature embeddings using Isolation Forest.
    
    Works well for:
    - High-dimensional embeddings
    - Unsupervised OOD detection
    - No assumptions about distribution
    
    Example:
        >>> detector = IsolationForestDetector(contamination=0.01)
        >>> detector.fit(train_embeddings)
        >>> scores = detector.score(test_embeddings)
        >>> is_ood = scores < 0  # Negative scores = OOD
    """
    
    def __init__(self, contamination: float = 0.01, random_state: int = 42):
        """
        Args:
            contamination: Expected proportion of outliers (0.01 = 1%)
            random_state: Random seed
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
    
    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None):
        """Fit Isolation Forest"""
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(features)
        logger.info(f"IsolationForest fitted on {len(features)} samples")
    
    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores
        
        Returns:
            Scores (negative = OOD, positive = in-dist)
        """
        return self.model.decision_function(features)
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict OOD (1) or in-dist (-1)
        
        Returns:
            Predictions (1 = in-dist, -1 = OOD)
        """
        return self.model.predict(features)


class MahalanobisDetector(OODDetector):
    """
    📏 Mahalanobis Distance OOD Detector
    
    Detect OOD based on Mahalanobis distance from class centroids.
    
    Better than Euclidean distance as it accounts for covariance.
    
    Example:
        >>> detector = MahalanobisDetector()
        >>> detector.fit(train_embeddings, train_labels)
        >>> distances = detector.score(test_embeddings)
    """
    
    def __init__(self):
        self.class_means = {}
        self.class_covs = {}
        self.global_cov = None
    
    def fit(self, features: np.ndarray, labels: np.ndarray):
        """Fit class-conditional Gaussians"""
        unique_labels = np.unique(labels)
        
        # Compute per-class statistics
        for label in unique_labels:
            class_features = features[labels == label]
            self.class_means[label] = class_features.mean(axis=0)
            
            # Use global covariance (more stable)
            if self.global_cov is None:
                cov_estimator = EmpiricalCovariance()
                cov_estimator.fit(features)
                self.global_cov = cov_estimator.covariance_
        
        logger.info(f"Mahalanobis detector fitted on {len(unique_labels)} classes")
    
    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Compute minimum Mahalanobis distance to any class
        
        Returns:
            Distances (higher = more OOD)
        """
        distances = []
        
        # Compute distance to each class centroid
        for mean in self.class_means.values():
            diff = features - mean
            # Mahalanobis distance: sqrt(d^T Σ^-1 d)
            inv_cov = np.linalg.pinv(self.global_cov)
            dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
            distances.append(dist)
        
        # Return minimum distance (closest class)
        min_distances = np.min(distances, axis=0)
        
        return min_distances
    
    def predict(self, features: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Predict OOD based on threshold
        
        Args:
            threshold: Distance threshold (3.0 ~ 3 std devs)
            
        Returns:
            Predictions (0 = in-dist, 1 = OOD)
        """
        distances = self.score(features)
        return (distances > threshold).astype(int)


def extract_features(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from model for OOD detection
    
    Args:
        model: Model with get_features() method
        dataloader: DataLoader
        device: Device
        
    Returns:
        (features, labels)
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Extract features (embeddings)
            if hasattr(model, 'get_features'):
                features = model.get_features(images)
            else:
                # Fallback: use second-to-last layer
                features = model(images)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    
    return features, labels


def compute_ood_auroc(
    in_dist_scores: np.ndarray,
    ood_scores: np.ndarray
) -> float:
    """
    Compute AUROC for OOD detection
    
    Args:
        in_dist_scores: Scores for in-distribution samples
        ood_scores: Scores for OOD samples
        
    Returns:
        AUROC (1.0 = perfect, 0.5 = random)
    """
    # Combine scores
    scores = np.concatenate([in_dist_scores, ood_scores])
    
    # Labels (0 = in-dist, 1 = OOD)
    labels = np.concatenate([
        np.zeros(len(in_dist_scores)),
        np.ones(len(ood_scores))
    ])
    
    # Compute AUROC
    auroc = roc_auc_score(labels, scores)
    
    return auroc


if __name__ == "__main__":
    # Quick test
    print("🔍 OOD Detection Module")
    print("=" * 60)
    
    # Create dummy data
    np.random.seed(42)
    
    # In-distribution (clustered)
    in_dist = np.random.randn(1000, 128) * 0.5
    
    # Out-of-distribution (far from cluster)
    ood = np.random.randn(100, 128) * 2.0 + 3
    
    # Test IsolationForest
    print("\n🌲 Testing Isolation Forest...")
    detector = IsolationForestDetector(contamination=0.05)
    detector.fit(in_dist)
    
    in_scores = detector.score(in_dist)
    ood_scores = detector.score(ood)
    
    auroc = compute_ood_auroc(-in_scores, -ood_scores)  # Negative for higher = more OOD
    
    print(f"  In-dist mean score: {in_scores.mean():.4f}")
    print(f"  OOD mean score: {ood_scores.mean():.4f}")
    print(f"  AUROC: {auroc:.4f}")
    
    print("\n✅ OOD detection module ready!")
