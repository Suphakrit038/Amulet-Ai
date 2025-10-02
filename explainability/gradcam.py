"""
Grad-CAM Implementation for Visual Explanations

Implements:
- Grad-CAM: Original gradient-weighted class activation mapping
- Grad-CAM++: Improved version with better multi-instance localization
- Visualization utilities for UI integration

References:
- Grad-CAM: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks"
- Grad-CAM++: Chattopadhay et al. (2018) "Grad-CAM++: Improved Visual Explanations"

Author: Amulet-AI Team
Date: October 2, 2025
"""

import logging
from typing import Optional, List, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    
    Generates visual explanations by highlighting regions that are important
    for predictions by computing gradients of target class with respect to
    feature maps in the last convolutional layer.
    
    Parameters:
    -----------
    model : nn.Module
        PyTorch model
    target_layer : nn.Module
        Target convolutional layer (usually last conv layer)
    device : str
        Device to use
        
    Example:
    --------
    >>> # For ResNet
    >>> target_layer = model.layer4[-1]
    >>> gradcam = GradCAM(model, target_layer)
    >>> heatmap = gradcam.generate(image, target_class=2)
    >>> overlay = gradcam.overlay_heatmap(image, heatmap)
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.target_layer = target_layer
        self.device = device
        
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
        
        logger.info(f"Grad-CAM initialized with target layer: {target_layer.__class__.__name__}")
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Parameters:
        -----------
        input_tensor : torch.Tensor
            Input image tensor (1, C, H, W)
        target_class : int, optional
            Target class index. If None, uses predicted class
        normalize : bool
            Whether to normalize heatmap to [0, 1]
            
        Returns:
        --------
        np.ndarray : Heatmap (H, W) in range [0, 1] if normalized
        
        Example:
        --------
        >>> heatmap = gradcam.generate(image_tensor, target_class=2)
        """
        self.model.eval()
        
        # Forward pass
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        
        # One-hot encode target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        
        # Compute gradients
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        # Weights = global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # ReLU to keep only positive influences
        
        # Resize to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Convert to numpy
        heatmap = cam.squeeze().cpu().numpy()
        
        # Normalize
        if normalize:
            if heatmap.max() > 0:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            else:
                heatmap = np.zeros_like(heatmap)
        
        return heatmap
    
    def generate_batch(
        self,
        input_tensor: torch.Tensor,
        target_classes: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        Generate Grad-CAM for a batch of images.
        
        Parameters:
        -----------
        input_tensor : torch.Tensor
            Batch of images (B, C, H, W)
        target_classes : list, optional
            List of target classes for each image
            
        Returns:
        --------
        list : List of heatmaps
        """
        heatmaps = []
        
        for i in range(input_tensor.shape[0]):
            target_class = target_classes[i] if target_classes is not None else None
            heatmap = self.generate(input_tensor[i:i+1], target_class=target_class)
            heatmaps.append(heatmap)
        
        return heatmaps


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++: Improved version with better localization
    
    Uses weighted average of gradients instead of global average pooling,
    providing better localization for multiple instances of same class.
    
    Reference:
    - Chattopadhay et al. (2018) "Grad-CAM++: Improved Visual Explanations"
    
    Usage same as GradCAM
    """
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """Generate Grad-CAM++ heatmap"""
        self.model.eval()
        
        # Forward pass
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        
        # Compute gradients
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        # Grad-CAM++ weights
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Compute alpha (importance weights)
        # alpha_k^c = (∂²y^c / ∂A_k²) / (2 * ∂²y^c / ∂A_k² + Σ_ij A_k^ij * ∂³y^c / ∂A_k³)
        
        # Second derivative (approximated)
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        
        # Global sum
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        
        # Alpha calculation
        alpha_denom = 2.0 * grad_2 + sum_activations * grad_3 + 1e-10
        alpha = grad_2 / alpha_denom
        
        # Normalize alpha
        alpha = alpha / (alpha.sum(dim=(2, 3), keepdim=True) + 1e-10)
        
        # Weighted combination
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Resize to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Convert to numpy
        heatmap = cam.squeeze().cpu().numpy()
        
        # Normalize
        if normalize:
            if heatmap.max() > 0:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            else:
                heatmap = np.zeros_like(heatmap)
        
        return heatmap


def overlay_heatmap(
    image: Union[np.ndarray, Image.Image],
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on original image.
    
    Parameters:
    -----------
    image : np.ndarray or PIL.Image
        Original image (H, W, 3) in RGB
    heatmap : np.ndarray
        Heatmap (H, W) in range [0, 1]
    alpha : float
        Blending factor (0 = original image, 1 = pure heatmap)
    colormap : int
        OpenCV colormap (default: COLORMAP_JET)
        
    Returns:
    --------
    np.ndarray : Overlayed image (H, W, 3) in RGB
    
    Example:
    --------
    >>> overlay = overlay_heatmap(original_image, heatmap, alpha=0.5)
    >>> Image.fromarray(overlay).show()
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure image is RGB
    if image.shape[-1] == 4:  # RGBA
        image = image[..., :3]
    
    # Resize heatmap to match image
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlay = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)
    
    return overlay


def visualize_gradcam(
    model: nn.Module,
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    target_layer: nn.Module,
    target_class: Optional[int] = None,
    transform: Optional[callable] = None,
    class_names: Optional[List[str]] = None,
    method: str = 'gradcam',
    return_dict: bool = False
) -> Union[np.ndarray, dict]:
    """
    Complete Grad-CAM visualization pipeline.
    
    Parameters:
    -----------
    model : nn.Module
        PyTorch model
    image : torch.Tensor, np.ndarray, or PIL.Image
        Input image
    target_layer : nn.Module
        Target layer for Grad-CAM
    target_class : int, optional
        Target class (uses predicted if None)
    transform : callable, optional
        Image preprocessing transform
    class_names : list, optional
        List of class names
    method : str
        'gradcam' or 'gradcam++'
    return_dict : bool
        If True, return dict with additional info
        
    Returns:
    --------
    np.ndarray or dict : Overlayed image or dict with full results
    
    Example:
    --------
    >>> from torchvision import transforms
    >>> transform = transforms.Compose([
    ...     transforms.Resize((224, 224)),
    ...     transforms.ToTensor(),
    ...     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ... ])
    >>> 
    >>> overlay = visualize_gradcam(
    ...     model=model,
    ...     image='amulet.jpg',
    ...     target_layer=model.layer4[-1],
    ...     transform=transform,
    ...     class_names=['Class A', 'Class B', ...]
    ... )
    >>> Image.fromarray(overlay).show()
    """
    # Load image if path
    if isinstance(image, (str, Path)):
        original_image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        original_image = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        original_image = image
    else:
        # Already tensor
        original_image = None
    
    # Prepare input tensor
    if not isinstance(image, torch.Tensor):
        if transform is None:
            raise ValueError("Transform must be provided for non-tensor input")
        input_tensor = transform(original_image).unsqueeze(0)
    else:
        input_tensor = image
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
    
    # Create Grad-CAM
    if method == 'gradcam':
        gradcam = GradCAM(model, target_layer)
    elif method == 'gradcam++':
        gradcam = GradCAMPlusPlus(model, target_layer)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Generate heatmap
    heatmap = gradcam.generate(input_tensor, target_class=target_class)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor.to(gradcam.device))
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        pred_conf = probs[0, pred_class].item()
    
    # Create overlay
    if original_image is not None:
        overlay = overlay_heatmap(original_image, heatmap)
    else:
        # Convert tensor back to image
        img_np = input_tensor[0].cpu().permute(1, 2, 0).numpy()
        # Denormalize if needed
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        img_np = (img_np * 255).astype(np.uint8)
        overlay = overlay_heatmap(img_np, heatmap)
    
    if return_dict:
        result = {
            'overlay': overlay,
            'heatmap': heatmap,
            'predicted_class': pred_class,
            'confidence': pred_conf,
            'original_image': np.array(original_image) if original_image else None
        }
        
        if class_names is not None:
            result['class_name'] = class_names[pred_class]
        
        return result
    
    return overlay


def generate_explanation(
    model: nn.Module,
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    target_layer: nn.Module,
    transform: Optional[callable],
    class_names: List[str],
    top_k: int = 3,
    method: str = 'gradcam'
) -> dict:
    """
    Generate comprehensive explanation for UI display.
    
    Returns heatmaps for top-k predicted classes with confidence scores.
    
    Parameters:
    -----------
    model : nn.Module
        Model
    image : image input
        Input image
    target_layer : nn.Module
        Target layer
    transform : callable
        Preprocessing transform
    class_names : list
        Class names
    top_k : int
        Number of top predictions to explain
    method : str
        'gradcam' or 'gradcam++'
        
    Returns:
    --------
    dict : {
        'top_predictions': [
            {
                'class': 'Class A',
                'confidence': 0.85,
                'overlay': np.ndarray,
                'heatmap': np.ndarray
            },
            ...
        ],
        'original_image': np.ndarray
    }
    
    Example (for Streamlit UI):
    --------
    >>> result = generate_explanation(model, image, target_layer, transform, class_names, top_k=3)
    >>> 
    >>> st.write(f"Top prediction: {result['top_predictions'][0]['class']}")
    >>> st.write(f"Confidence: {result['top_predictions'][0]['confidence']:.2%}")
    >>> st.image(result['top_predictions'][0]['overlay'], caption='Grad-CAM')
    """
    # Load image
    if isinstance(image, (str, Path)):
        original_image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        original_image = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        original_image = image
    else:
        original_image = None
    
    # Prepare input
    if not isinstance(image, torch.Tensor):
        input_tensor = transform(original_image).unsqueeze(0)
    else:
        input_tensor = image
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
    
    # Get predictions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    
    with torch.no_grad():
        output = model(input_tensor.to(device))
        probs = F.softmax(output, dim=1)[0]
    
    # Get top-k predictions
    top_probs, top_indices = probs.topk(min(top_k, len(probs)))
    
    # Create Grad-CAM
    if method == 'gradcam':
        gradcam = GradCAM(model, target_layer, device=device)
    else:
        gradcam = GradCAMPlusPlus(model, target_layer, device=device)
    
    # Generate explanations for each top prediction
    explanations = []
    
    for prob, idx in zip(top_probs, top_indices):
        idx = idx.item()
        prob = prob.item()
        
        # Generate heatmap for this class
        heatmap = gradcam.generate(input_tensor, target_class=idx)
        
        # Create overlay
        if original_image is not None:
            overlay = overlay_heatmap(original_image, heatmap)
        else:
            img_np = input_tensor[0].cpu().permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            img_np = (img_np * 255).astype(np.uint8)
            overlay = overlay_heatmap(img_np, heatmap)
        
        explanations.append({
            'class': class_names[idx],
            'class_index': idx,
            'confidence': prob,
            'overlay': overlay,
            'heatmap': heatmap
        })
    
    result = {
        'top_predictions': explanations,
        'original_image': np.array(original_image) if original_image else None
    }
    
    return result


# Helper function to find target layer automatically
def get_target_layer(model: nn.Module, architecture: str = 'resnet') -> nn.Module:
    """
    Automatically find appropriate target layer for Grad-CAM.
    
    Parameters:
    -----------
    model : nn.Module
        Model
    architecture : str
        Architecture type ('resnet', 'efficientnet', 'mobilenet', 'vgg')
        
    Returns:
    --------
    nn.Module : Target layer
    
    Example:
    --------
    >>> target_layer = get_target_layer(model, architecture='resnet')
    >>> gradcam = GradCAM(model, target_layer)
    """
    if architecture.lower() in ['resnet', 'resnext']:
        # Handle wrapped models (AmuletTransferModel)
        base_model = model.backbone if hasattr(model, 'backbone') else model
        # Last block of layer4
        return base_model.layer4[-1]
    
    elif architecture.lower() == 'efficientnet':
        # Handle wrapped models
        base_model = model.backbone if hasattr(model, 'backbone') else model
        # Last conv layer
        try:
            return base_model.features[-1]
        except:
            return base_model.conv_head
    
    elif architecture.lower() == 'mobilenet':
        # Handle wrapped models
        base_model = model.backbone if hasattr(model, 'backbone') else model
        # Last conv layer
        return base_model.features[-1]
    
    elif architecture.lower() == 'vgg':
        # Handle wrapped models
        base_model = model.backbone if hasattr(model, 'backbone') else model
        # Last conv layer
        return base_model.features[-1]
    
    elif architecture.lower() == 'densenet':
        # Handle wrapped models
        base_model = model.backbone if hasattr(model, 'backbone') else model
        return base_model.features[-1]
    
    else:
        logger.warning(f"Unknown architecture: {architecture}. Trying to find last conv layer...")
        
        # Try to find last conv layer automatically
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        
        if last_conv is None:
            raise ValueError("Could not find convolutional layer. Please specify target_layer manually.")
        
        return last_conv
