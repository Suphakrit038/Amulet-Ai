"""
🖼️ Image Preprocessing Module
============================

Basic image preprocessing operations for Amulet-AI dataset.

Features:
- Resize & normalization
- Color space conversion
- Batch processing
- Memory-efficient operations

Author: Amulet-AI Team
Date: October 2025
"""

import numpy as np
from PIL import Image, ImageEnhance
from typing import Union, Tuple, List, Optional, Dict
import torch
from torchvision import transforms
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    🎯 Basic Image Processor
    
    ประมวลผลภาพพื้นฐานสำหรับ Amulet dataset
    
    Features:
    - Resize with aspect ratio preservation
    - Normalization (ImageNet stats or custom)
    - Color space conversion (RGB/Grayscale/HSV)
    - Batch processing support
    
    Example:
        >>> processor = ImageProcessor(target_size=(224, 224))
        >>> img = Image.open("amulet.jpg")
        >>> processed = processor.process(img)
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        interpolation: str = 'bilinear'
    ):
        """
        Initialize Image Processor
        
        Args:
            target_size: Output image size (H, W)
            normalize: Apply normalization
            mean: Normalization mean (ImageNet default)
            std: Normalization std (ImageNet default)
            interpolation: Resize method ('bilinear', 'bicubic', 'nearest')
        """
        self.target_size = target_size
        self.normalize = normalize
        self.mean = mean
        self.std = std
        
        # Interpolation mapping
        interp_map = {
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'nearest': Image.NEAREST,
            'lanczos': Image.LANCZOS
        }
        self.interpolation = interp_map.get(interpolation, Image.BILINEAR)
        
        # Build transform pipeline
        self.transform = self._build_transform()
        
        logger.info(f"ImageProcessor initialized: size={target_size}, normalize={normalize}")
    
    def _build_transform(self) -> transforms.Compose:
        """สร้าง transform pipeline"""
        transform_list = [
            transforms.Resize(self.target_size, interpolation=self.interpolation),
            transforms.ToTensor()
        ]
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        
        return transforms.Compose(transform_list)
    
    def process(
        self,
        image: Union[Image.Image, np.ndarray, str, Path]
    ) -> torch.Tensor:
        """
        ประมวลผลภาพเดี่ยว
        
        Args:
            image: Input image (PIL, numpy, or path)
            
        Returns:
            Processed tensor (C, H, W)
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        # Convert numpy to PIL
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transform
        tensor = self.transform(image)
        
        return tensor
    
    def process_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, str, Path]]
    ) -> torch.Tensor:
        """
        ประมวลผล batch ของภาพ
        
        Args:
            images: List of images
            
        Returns:
            Batch tensor (B, C, H, W)
        """
        processed = [self.process(img) for img in images]
        return torch.stack(processed)
    
    def denormalize(
        self,
        tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        แปลง normalized tensor กลับเป็นภาพ [0, 1]
        
        Args:
            tensor: Normalized tensor (C, H, W) or (B, C, H, W)
            
        Returns:
            Denormalized tensor
        """
        if not self.normalize:
            return tensor
        
        mean = torch.tensor(self.mean).view(-1, 1, 1)
        std = torch.tensor(self.std).view(-1, 1, 1)
        
        if tensor.dim() == 4:  # Batch
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        
        return tensor * std + mean
    
    def to_pil(
        self,
        tensor: torch.Tensor
    ) -> Image.Image:
        """
        แปลง tensor เป็น PIL Image
        
        Args:
            tensor: Image tensor (C, H, W)
            
        Returns:
            PIL Image
        """
        # Denormalize if needed
        if self.normalize:
            tensor = self.denormalize(tensor)
        
        # Clamp to [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy
        img_np = tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        return Image.fromarray(img_np)


class ColorSpaceConverter:
    """
    🎨 Color Space Converter
    
    แปลงภาพระหว่าง color spaces ต่างๆ
    
    Supported:
    - RGB ↔ Grayscale
    - RGB ↔ HSV
    - RGB ↔ LAB
    """
    
    @staticmethod
    def rgb_to_grayscale(image: Image.Image) -> Image.Image:
        """แปลง RGB เป็น Grayscale"""
        return image.convert('L')
    
    @staticmethod
    def rgb_to_hsv(image: Image.Image) -> np.ndarray:
        """แปลง RGB เป็น HSV (numpy array)"""
        import cv2
        img_np = np.array(image)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    
    @staticmethod
    def hsv_to_rgb(hsv_array: np.ndarray) -> Image.Image:
        """แปลง HSV เป็น RGB"""
        import cv2
        rgb_array = cv2.cvtColor(hsv_array, cv2.COLOR_HSV2RGB)
        return Image.fromarray(rgb_array)
    
    @staticmethod
    def rgb_to_lab(image: Image.Image) -> np.ndarray:
        """แปลง RGB เป็น LAB color space"""
        import cv2
        img_np = np.array(image)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    
    @staticmethod
    def lab_to_rgb(lab_array: np.ndarray) -> Image.Image:
        """แปลง LAB เป็น RGB"""
        import cv2
        rgb_array = cv2.cvtColor(lab_array, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb_array)


class BasicPreprocessor:
    """
    ⚙️ Basic Preprocessor
    
    Pipeline สำหรับ preprocessing พื้นฐาน
    
    Steps:
    1. Resize to target size
    2. Optional: Color space conversion
    3. Optional: Brightness/Contrast adjustment
    4. Normalization
    
    Example:
        >>> preprocessor = BasicPreprocessor(
        ...     target_size=(224, 224),
        ...     adjust_brightness=True
        ... )
        >>> img = Image.open("amulet.jpg")
        >>> processed = preprocessor(img)
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        grayscale: bool = False,
        adjust_brightness: bool = False,
        brightness_factor: float = 1.0,
        adjust_contrast: bool = False,
        contrast_factor: float = 1.0,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize Basic Preprocessor
        
        Args:
            target_size: Output image size
            grayscale: Convert to grayscale
            adjust_brightness: Apply brightness adjustment
            brightness_factor: Brightness factor (1.0 = no change)
            adjust_contrast: Apply contrast adjustment
            contrast_factor: Contrast factor (1.0 = no change)
            normalize: Apply normalization
            mean: Normalization mean
            std: Normalization std
        """
        self.target_size = target_size
        self.grayscale = grayscale
        self.adjust_brightness = adjust_brightness
        self.brightness_factor = brightness_factor
        self.adjust_contrast = adjust_contrast
        self.contrast_factor = contrast_factor
        self.normalize = normalize
        self.mean = mean
        self.std = std
        
        self.processor = ImageProcessor(
            target_size=target_size,
            normalize=normalize,
            mean=mean,
            std=std
        )
    
    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str, Path]
    ) -> torch.Tensor:
        """
        ประมวลผลภาพด้วย pipeline
        
        Args:
            image: Input image
            
        Returns:
            Processed tensor
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to grayscale
        if self.grayscale:
            image = image.convert('L').convert('RGB')
        
        # Adjust brightness
        if self.adjust_brightness and self.brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(self.brightness_factor)
        
        # Adjust contrast
        if self.adjust_contrast and self.contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(self.contrast_factor)
        
        # Apply standard processing
        tensor = self.processor.process(image)
        
        return tensor
    
    def get_config(self) -> Dict:
        """Get preprocessor configuration"""
        return {
            'target_size': self.target_size,
            'grayscale': self.grayscale,
            'adjust_brightness': self.adjust_brightness,
            'brightness_factor': self.brightness_factor,
            'adjust_contrast': self.adjust_contrast,
            'contrast_factor': self.contrast_factor,
            'normalize': self.normalize,
            'mean': self.mean,
            'std': self.std
        }


# ============================================================================
# Helper Functions
# ============================================================================

def create_standard_processor(
    image_size: int = 224,
    normalize: bool = True
) -> ImageProcessor:
    """
    สร้าง standard processor สำหรับ Amulet-AI
    
    Args:
        image_size: Target image size (square)
        normalize: Apply ImageNet normalization
        
    Returns:
        ImageProcessor instance
    """
    return ImageProcessor(
        target_size=(image_size, image_size),
        normalize=normalize,
        interpolation='bilinear'
    )


def create_basic_preprocessor(
    image_size: int = 224,
    enhance_brightness: bool = False,
    enhance_contrast: bool = False
) -> BasicPreprocessor:
    """
    สร้าง basic preprocessor พร้อม enhancements
    
    Args:
        image_size: Target size
        enhance_brightness: Enable brightness adjustment
        enhance_contrast: Enable contrast adjustment
        
    Returns:
        BasicPreprocessor instance
    """
    return BasicPreprocessor(
        target_size=(image_size, image_size),
        adjust_brightness=enhance_brightness,
        brightness_factor=1.1 if enhance_brightness else 1.0,
        adjust_contrast=enhance_contrast,
        contrast_factor=1.1 if enhance_contrast else 1.0
    )


if __name__ == "__main__":
    # Quick test
    print("🖼️ Image Processor Module")
    print("=" * 50)
    
    # Create processor
    processor = create_standard_processor()
    print(f"✅ Created processor: {processor.target_size}")
    
    # Create preprocessor
    preprocessor = create_basic_preprocessor(enhance_brightness=True)
    print(f"✅ Created preprocessor with config:")
    for k, v in preprocessor.get_config().items():
        print(f"   • {k}: {v}")
