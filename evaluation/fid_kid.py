"""
📊 FID and KID Calculation
==========================

Fréchet Inception Distance (FID) and Kernel Inception Distance (KID)
for validating synthetic image quality.

References:
- FID: Heusel et al. (2017)
- KID: Bińkowski et al. (2018)

Author: Amulet-AI Team
Date: October 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from scipy import linalg
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    from torchvision.models import inception_v3, Inception_V3_Weights
    INCEPTION_AVAILABLE = True
except ImportError:
    INCEPTION_AVAILABLE = False
    logger.warning("InceptionV3 not available. FID/KID calculation limited.")


class FIDCalculator:
    """
    📈 FID (Fréchet Inception Distance) Calculator
    
    Measures distribution distance between real and synthetic images
    using InceptionV3 features.
    
    Lower FID = better quality/diversity
    
    Formula:
    FID = ||μ_r - μ_f||^2 + Tr(Σ_r + Σ_f - 2(Σ_r Σ_f)^{1/2})
    
    Example:
        >>> calculator = FIDCalculator()
        >>> fid = calculator.compute(real_images, fake_images)
        >>> print(f"FID: {fid:.2f}")
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize FID Calculator
        
        Args:
            device: 'cuda', 'cpu', or 'auto'
        """
        if not INCEPTION_AVAILABLE:
            raise ImportError("torchvision InceptionV3 required for FID")
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load InceptionV3
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception.fc = nn.Identity()  # Remove final FC
        self.inception.eval()
        self.inception.to(self.device)
        
        logger.info(f"FID Calculator initialized on {self.device}")
    
    def get_activations(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract InceptionV3 pool3 features
        
        Args:
            images: Batch of images (B, 3, H, W), normalized [-1, 1] or [0, 1]
            
        Returns:
            Features (B, 2048)
        """
        with torch.no_grad():
            images = images.to(self.device)
            
            # Inception expects 299x299
            if images.shape[-1] != 299:
                images = nn.functional.interpolate(
                    images, size=(299, 299), mode='bilinear', align_corners=False
                )
            
            # Get features
            features = self.inception(images)
            
        return features.cpu().numpy()
    
    def compute_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and covariance
        
        Args:
            features: Features (N, D)
            
        Returns:
            (mean, covariance)
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def compute_fid(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6
    ) -> float:
        """
        Compute FID between two distributions
        
        Args:
            mu1, sigma1: Real distribution statistics
            mu2, sigma2: Fake distribution statistics
            eps: Numerical stability
            
        Returns:
            FID score
        """
        # Difference in means
        diff = mu1 - mu2
        
        # Product of covariances
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Numerical error handling
        if not np.isfinite(covmean).all():
            logger.warning("FID calculation: adding eps to diagonal of cov")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Real values
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # FID formula
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return float(fid)
    
    def compute(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> float:
        """
        Compute FID between real and fake images
        
        Args:
            real_images: Real images (N, 3, H, W)
            fake_images: Fake/synthetic images (N, 3, H, W)
            
        Returns:
            FID score
        """
        logger.info("Computing FID...")
        
        # Extract features
        logger.info(f"Extracting features from {len(real_images)} real images...")
        real_features = self.get_activations(real_images)
        
        logger.info(f"Extracting features from {len(fake_images)} fake images...")
        fake_features = self.get_activations(fake_images)
        
        # Compute statistics
        mu_real, sigma_real = self.compute_statistics(real_features)
        mu_fake, sigma_fake = self.compute_statistics(fake_features)
        
        # Compute FID
        fid = self.compute_fid(mu_real, sigma_real, mu_fake, sigma_fake)
        
        logger.info(f"FID: {fid:.2f}")
        return fid


class KIDCalculator:
    """
    📉 KID (Kernel Inception Distance) Calculator
    
    Unbiased estimator using polynomial kernel.
    Better for small sample sizes than FID.
    
    Example:
        >>> calculator = KIDCalculator()
        >>> kid = calculator.compute(real_images, fake_images)
        >>> print(f"KID: {kid:.4f}")
    """
    
    def __init__(self, device: str = 'auto'):
        """Initialize KID Calculator"""
        if not INCEPTION_AVAILABLE:
            raise ImportError("torchvision InceptionV3 required for KID")
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Reuse InceptionV3 from FID
        self.fid_calc = FIDCalculator(device)
        
        logger.info(f"KID Calculator initialized on {self.device}")
    
    def polynomial_kernel(self, X: np.ndarray, Y: np.ndarray, degree: int = 3, gamma: float = None) -> np.ndarray:
        """
        Polynomial kernel: (γ·X·Y^T + 1)^d
        
        Args:
            X, Y: Feature matrices (N1, D) and (N2, D)
            degree: Polynomial degree
            gamma: Scaling (default: 1/D)
            
        Returns:
            Kernel matrix (N1, N2)
        """
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        
        return (gamma * X.dot(Y.T) + 1) ** degree
    
    def compute_kid(self, X: np.ndarray, Y: np.ndarray, degree: int = 3) -> float:
        """
        Compute KID between two feature sets
        
        Args:
            X: Real features (N1, D)
            Y: Fake features (N2, D)
            degree: Polynomial degree
            
        Returns:
            KID score (unbiased estimator)
        """
        m = X.shape[0]
        n = Y.shape[0]
        
        # Kernels
        K_XX = self.polynomial_kernel(X, X, degree)
        K_YY = self.polynomial_kernel(Y, Y, degree)
        K_XY = self.polynomial_kernel(X, Y, degree)
        
        # Unbiased estimator (remove diagonal)
        np.fill_diagonal(K_XX, 0)
        np.fill_diagonal(K_YY, 0)
        
        # KID formula
        kid = (K_XX.sum() / (m * (m - 1)) +
               K_YY.sum() / (n * (n - 1)) -
               2 * K_XY.sum() / (m * n))
        
        return float(kid)
    
    def compute(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> float:
        """
        Compute KID between real and fake images
        
        Args:
            real_images: Real images (N, 3, H, W)
            fake_images: Fake images (N, 3, H, W)
            
        Returns:
            KID score
        """
        logger.info("Computing KID...")
        
        # Extract features (reuse InceptionV3)
        real_features = self.fid_calc.get_activations(real_images)
        fake_features = self.fid_calc.get_activations(fake_images)
        
        # Compute KID
        kid = self.compute_kid(real_features, fake_features)
        
        logger.info(f"KID: {kid:.6f}")
        return kid


# ============================================================================
# Helper Functions
# ============================================================================

def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: str = 'auto'
) -> float:
    """
    Quick FID computation
    
    Args:
        real_images: Real images
        fake_images: Fake images
        device: Device
        
    Returns:
        FID score
    """
    calculator = FIDCalculator(device)
    return calculator.compute(real_images, fake_images)


def compute_kid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: str = 'auto'
) -> float:
    """
    Quick KID computation
    
    Args:
        real_images: Real images
        fake_images: Fake images
        device: Device
        
    Returns:
        KID score
    """
    calculator = KIDCalculator(device)
    return calculator.compute(real_images, fake_images)


if __name__ == "__main__":
    # Quick test
    print("📊 FID/KID Calculation Module")
    print("=" * 60)
    
    if not INCEPTION_AVAILABLE:
        print("⚠️  InceptionV3 not available")
    else:
        print("✅ InceptionV3 available")
        print("✅ FID/KID calculators ready!")
        print("\nNote: Run with actual images to test FID/KID computation")
