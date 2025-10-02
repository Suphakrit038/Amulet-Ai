"""
🎯 Model Calibration
===================

Temperature scaling and calibration metrics.

Features:
- Temperature Scaling (post-processing)
- Expected Calibration Error (ECE)
- Brier Score
- Reliability diagrams

References:
- Guo et al. (2017): "On Calibration of Modern Neural Networks"

Author: Amulet-AI Team
Date: October 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """
    🌡️ Temperature Scaling
    
    Post-training calibration by scaling logits with temperature T.
    
    Calibrated probabilities: softmax(logits / T)
    
    Process:
    1. Collect logits on calibration set
    2. Optimize T to minimize NLL
    3. Apply T during inference
    
    Reference: Guo et al. (2017)
    
    Example:
        >>> temp_scaler = TemperatureScaling()
        >>> temp_scaler.fit(model, calib_loader, device)
        >>> calibrated_probs = temp_scaler(logits)
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling
        
        Args:
            logits: Model logits (B, C)
            
        Returns:
            Calibrated probabilities (B, C)
        """
        return torch.softmax(logits / self.temperature, dim=1)
    
    def fit(
        self,
        model: nn.Module,
        calib_loader: torch.utils.data.DataLoader,
        device: torch.device,
        lr: float = 0.01,
        max_iter: int = 50
    ):
        """
        Fit temperature on calibration set
        
        Args:
            model: Trained model (frozen)
            calib_loader: Calibration DataLoader
            device: Device
            lr: Learning rate
            max_iter: Max iterations
        """
        model.eval()
        self.to(device)
        
        # Collect logits and labels
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for images, labels in calib_loader:
                images = images.to(device)
                logits = model(images)
                logits_list.append(logits)
                labels_list.append(labels.to(device))
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        # Optimize temperature
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        optimal_temp = self.temperature.item()
        logger.info(f"Temperature scaling fitted: T = {optimal_temp:.4f}")
        
        return optimal_temp


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    คำนวณ Expected Calibration Error (ECE)
    
    ECE measures the difference between predicted confidence
    and actual accuracy across confidence bins.
    
    Args:
        probs: Predicted probabilities (N, C)
        labels: True labels (N,)
        n_bins: Number of bins
        
    Returns:
        ECE score (lower is better)
        
    Example:
        >>> ece = compute_ece(probs, labels, n_bins=15)
        >>> print(f"ECE: {ece:.4f}")
    """
    # Get predicted class and confidence
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def compute_brier_score(
    probs: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    คำนวณ Brier Score
    
    Brier score measures the mean squared difference between
    predicted probabilities and true labels (one-hot).
    
    Args:
        probs: Predicted probabilities (N, C)
        labels: True labels (N,)
        
    Returns:
        Brier score (lower is better)
    """
    # Convert labels to one-hot
    num_classes = probs.shape[1]
    labels_onehot = np.eye(num_classes)[labels]
    
    # Compute Brier score
    brier = ((probs - labels_onehot) ** 2).sum(axis=1).mean()
    
    return brier


def calibrate_model(
    model: nn.Module,
    calib_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> TemperatureScaling:
    """
    Calibrate model using temperature scaling
    
    Args:
        model: Trained model
        calib_loader: Calibration DataLoader
        device: Device
        
    Returns:
        Fitted TemperatureScaling module
        
    Example:
        >>> temp_scaler = calibrate_model(model, calib_loader, device)
        >>> # At inference
        >>> logits = model(images)
        >>> calibrated_probs = temp_scaler(logits)
    """
    temp_scaler = TemperatureScaling()
    temp_scaler.fit(model, calib_loader, device)
    return temp_scaler


def evaluate_calibration(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    temp_scaler: Optional[TemperatureScaling] = None
) -> dict:
    """
    ประเมิน calibration ของโมเดล
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader
        device: Device
        temp_scaler: Optional temperature scaler
        
    Returns:
        Dict with ECE, Brier score, accuracy
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            
            # Apply temperature scaling if provided
            if temp_scaler is not None:
                probs = temp_scaler(logits).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    
    probs = np.vstack(all_probs)
    labels = np.concatenate(all_labels)
    
    # Compute metrics
    ece = compute_ece(probs, labels)
    brier = compute_brier_score(probs, labels)
    accuracy = (probs.argmax(axis=1) == labels).mean()
    
    results = {
        'ece': ece,
        'brier_score': brier,
        'accuracy': accuracy
    }
    
    logger.info(f"Calibration metrics: ECE={ece:.4f}, Brier={brier:.4f}, Acc={accuracy:.4f}")
    
    return results


if __name__ == "__main__":
    # Quick test
    print("🎯 Calibration Module")
    print("=" * 60)
    
    # Create dummy predictions
    np.random.seed(42)
    n_samples = 1000
    n_classes = 6
    
    # Simulate uncalibrated (overconfident) predictions
    probs = np.random.dirichlet(np.ones(n_classes) * 0.5, n_samples)
    # Make them overconfident
    probs = probs ** 0.5
    probs = probs / probs.sum(axis=1, keepdims=True)
    
    labels = np.random.randint(0, n_classes, n_samples)
    
    # Compute ECE
    ece = compute_ece(probs, labels)
    brier = compute_brier_score(probs, labels)
    
    print(f"\n📊 Calibration Metrics (Before):")
    print(f"  ECE: {ece:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    
    print("\n✅ Calibration module ready!")
