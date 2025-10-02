"""
Saliency Map Implementations

Provides gradient-based visualization methods:
- Vanilla Saliency: Simple gradient visualization
- SmoothGrad: Noise-reduced saliency maps
- Integrated Gradients: Path-based attribution method

Author: Amulet-AI Team
Date: October 2, 2025
"""

import logging
from typing import Optional, Union, List
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_saliency_map(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    normalize: bool = True,
    absolute: bool = True
) -> np.ndarray:
    """
    Generate vanilla saliency map.
    
    Shows which pixels have the most influence on the prediction
    by computing gradients of output with respect to input.
    
    Parameters:
    -----------
    model : nn.Module
        PyTorch model
    input_tensor : torch.Tensor
        Input image (1, C, H, W)
    target_class : int, optional
        Target class (uses predicted if None)
    normalize : bool
        Normalize to [0, 1]
    absolute : bool
        Use absolute values of gradients
        
    Returns:
    --------
    np.ndarray : Saliency map (H, W) or (H, W, C)
    
    Example:
    --------
    >>> saliency = generate_saliency_map(model, image_tensor)
    >>> plt.imshow(saliency, cmap='hot')
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Prepare input
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad = True
    
    # Forward pass
    output = model(input_tensor)
    
    # Get target class
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Zero gradients
    model.zero_grad()
    if input_tensor.grad is not None:
        input_tensor.grad.zero_()
    
    # Backward pass
    score = output[0, target_class]
    score.backward()
    
    # Get gradients
    saliency = input_tensor.grad[0].cpu().numpy()  # (C, H, W)
    
    # Take maximum across channels
    if saliency.shape[0] > 1:
        saliency = np.max(np.abs(saliency) if absolute else saliency, axis=0)
    else:
        saliency = saliency[0]
    
    # Normalize
    if normalize:
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
    
    return saliency


def generate_smoothgrad(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    n_samples: int = 50,
    noise_level: float = 0.15,
    normalize: bool = True
) -> np.ndarray:
    """
    Generate SmoothGrad saliency map.
    
    Reduces noise in saliency maps by averaging saliency maps
    computed on noisy versions of the input image.
    
    Reference:
    - Smilkov et al. (2017) "SmoothGrad: removing noise by adding noise"
    
    Parameters:
    -----------
    model : nn.Module
        Model
    input_tensor : torch.Tensor
        Input (1, C, H, W)
    target_class : int, optional
        Target class
    n_samples : int
        Number of noisy samples
    noise_level : float
        Standard deviation of Gaussian noise (relative to input range)
    normalize : bool
        Normalize output
        
    Returns:
    --------
    np.ndarray : Smoothed saliency map
    
    Example:
    --------
    >>> smoothgrad = generate_smoothgrad(model, image_tensor, n_samples=50)
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Compute noise std
    stdev = noise_level * (input_tensor.max() - input_tensor.min())
    
    # Accumulate gradients
    total_gradients = None
    
    for _ in range(n_samples):
        # Add noise
        noise = torch.randn_like(input_tensor) * stdev
        noisy_input = input_tensor + noise
        noisy_input.requires_grad = True
        
        # Forward pass
        output = model(noisy_input)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        model.zero_grad()
        if noisy_input.grad is not None:
            noisy_input.grad.zero_()
        
        score = output[0, target_class]
        score.backward()
        
        # Accumulate gradients
        if total_gradients is None:
            total_gradients = noisy_input.grad.clone()
        else:
            total_gradients += noisy_input.grad
    
    # Average gradients
    smooth_grad = total_gradients / n_samples
    smooth_grad = smooth_grad[0].cpu().numpy()  # (C, H, W)
    
    # Take maximum across channels
    if smooth_grad.shape[0] > 1:
        smooth_grad = np.max(np.abs(smooth_grad), axis=0)
    else:
        smooth_grad = smooth_grad[0]
    
    # Normalize
    if normalize:
        smooth_grad = (smooth_grad - smooth_grad.min()) / (smooth_grad.max() - smooth_grad.min() + 1e-10)
    
    return smooth_grad


def generate_integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    n_steps: int = 50,
    baseline: Optional[torch.Tensor] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Generate Integrated Gradients attribution.
    
    Computes path integral of gradients from baseline to input.
    Satisfies axioms like sensitivity and implementation invariance.
    
    Reference:
    - Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
    
    Parameters:
    -----------
    model : nn.Module
        Model
    input_tensor : torch.Tensor
        Input (1, C, H, W)
    target_class : int, optional
        Target class
    n_steps : int
        Number of interpolation steps
    baseline : torch.Tensor, optional
        Baseline input (default: zeros)
    normalize : bool
        Normalize output
        
    Returns:
    --------
    np.ndarray : Integrated gradients map
    
    Example:
    --------
    >>> ig = generate_integrated_gradients(model, image_tensor, n_steps=50)
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Create baseline (black image)
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    else:
        baseline = baseline.to(device)
    
    # Generate interpolated inputs
    alphas = torch.linspace(0, 1, n_steps).to(device)
    
    # Accumulate gradients
    integrated_grads = torch.zeros_like(input_tensor)
    
    for alpha in alphas:
        # Interpolate
        interpolated = baseline + alpha * (input_tensor - baseline)
        interpolated.requires_grad = True
        
        # Forward pass
        output = model(interpolated)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        model.zero_grad()
        if interpolated.grad is not None:
            interpolated.grad.zero_()
        
        score = output[0, target_class]
        score.backward()
        
        # Accumulate
        integrated_grads += interpolated.grad
    
    # Average and scale by input difference
    integrated_grads = integrated_grads / n_steps
    integrated_grads = integrated_grads * (input_tensor - baseline)
    
    # Convert to numpy
    ig = integrated_grads[0].cpu().numpy()  # (C, H, W)
    
    # Take maximum across channels
    if ig.shape[0] > 1:
        ig = np.max(np.abs(ig), axis=0)
    else:
        ig = ig[0]
    
    # Normalize
    if normalize:
        ig = (ig - ig.min()) / (ig.max() - ig.min() + 1e-10)
    
    return ig


def visualize_saliency(
    saliency_map: np.ndarray,
    original_image: Optional[Union[np.ndarray, Image.Image]] = None,
    alpha: float = 0.5,
    colormap: str = 'hot'
) -> np.ndarray:
    """
    Visualize saliency map.
    
    Parameters:
    -----------
    saliency_map : np.ndarray
        Saliency map (H, W)
    original_image : np.ndarray or PIL.Image, optional
        Original image for overlay
    alpha : float
        Blending factor
    colormap : str
        Matplotlib colormap name
        
    Returns:
    --------
    np.ndarray : Visualization
    
    Example:
    --------
    >>> import matplotlib.pyplot as plt
    >>> vis = visualize_saliency(saliency, original_image, alpha=0.5)
    >>> plt.imshow(vis)
    """
    # Convert PIL to numpy
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    # Resize saliency if needed
    if original_image is not None and saliency_map.shape != original_image.shape[:2]:
        from PIL import Image as PILImage
        saliency_map = np.array(
            PILImage.fromarray((saliency_map * 255).astype(np.uint8)).resize(
                (original_image.shape[1], original_image.shape[0])
            )
        ) / 255.0
    
    # Apply colormap
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    cmap = cm.get_cmap(colormap)
    saliency_colored = cmap(saliency_map)[..., :3]  # RGB
    saliency_colored = (saliency_colored * 255).astype(np.uint8)
    
    # Overlay if image provided
    if original_image is not None:
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        
        visualization = (alpha * saliency_colored + (1 - alpha) * original_image).astype(np.uint8)
    else:
        visualization = saliency_colored
    
    return visualization


def compare_saliency_methods(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    original_image: Optional[Union[np.ndarray, Image.Image]] = None
) -> dict:
    """
    Compare different saliency methods side-by-side.
    
    Generates vanilla saliency, SmoothGrad, and Integrated Gradients
    for the same input.
    
    Parameters:
    -----------
    model : nn.Module
        Model
    input_tensor : torch.Tensor
        Input
    target_class : int, optional
        Target class
    original_image : image, optional
        For visualization
        
    Returns:
    --------
    dict : {
        'vanilla': np.ndarray,
        'smoothgrad': np.ndarray,
        'integrated_gradients': np.ndarray,
        'vanilla_vis': np.ndarray,
        'smoothgrad_vis': np.ndarray,
        'ig_vis': np.ndarray
    }
    
    Example:
    --------
    >>> results = compare_saliency_methods(model, image_tensor, original_image=image)
    >>> 
    >>> fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    >>> axes[0].imshow(results['vanilla_vis'])
    >>> axes[0].set_title('Vanilla Saliency')
    >>> axes[1].imshow(results['smoothgrad_vis'])
    >>> axes[1].set_title('SmoothGrad')
    >>> axes[2].imshow(results['ig_vis'])
    >>> axes[2].set_title('Integrated Gradients')
    """
    # Generate saliency maps
    vanilla = generate_saliency_map(model, input_tensor, target_class)
    smoothgrad = generate_smoothgrad(model, input_tensor, target_class, n_samples=50)
    ig = generate_integrated_gradients(model, input_tensor, target_class, n_steps=50)
    
    results = {
        'vanilla': vanilla,
        'smoothgrad': smoothgrad,
        'integrated_gradients': ig
    }
    
    # Create visualizations if original image provided
    if original_image is not None:
        results['vanilla_vis'] = visualize_saliency(vanilla, original_image)
        results['smoothgrad_vis'] = visualize_saliency(smoothgrad, original_image)
        results['ig_vis'] = visualize_saliency(ig, original_image)
    
    return results
