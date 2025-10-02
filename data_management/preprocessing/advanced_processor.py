"""
🔬 Advanced Image Preprocessing
==============================

Advanced preprocessing techniques for medical/cultural artifact images.

Features:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Denoising (Non-local means, Bilateral filter)
- Edge enhancement
- Morphological operations
- Unsharp masking

Author: Amulet-AI Team
Date: October 2025
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Union, Tuple, Optional, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. Some features will be limited.")


class CLAHEProcessor:
    """
    🔆 CLAHE Processor
    
    Contrast Limited Adaptive Histogram Equalization
    ปรับ contrast แบบ adaptive สำหรับรายละเอียดที่มืด/สว่างเกินไป
    
    ดีสำหรับ:
    - ภาพที่มีแสงไม่สม่ำเสมอ
    - รายละเอียดที่ซ่อนอยู่ในเงา/แสงจ้า
    - ภาพวัตถุโบราณที่มีความคมชัดต่ำ
    
    Example:
        >>> clahe = CLAHEProcessor(clip_limit=2.0, tile_size=8)
        >>> enhanced = clahe.process(image)
    """
    
    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_size: int = 8,
        apply_to_lab: bool = True
    ):
        """
        Initialize CLAHE Processor
        
        Args:
            clip_limit: Contrast limiting (1.0-4.0 recommended)
                       Higher = more contrast but risk of noise amplification
            tile_size: Size of grid for histogram equalization (8x8 recommended)
            apply_to_lab: Apply to L channel in LAB space (better for color)
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for CLAHE. Install: pip install opencv-python")
        
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.apply_to_lab = apply_to_lab
        
        # Create CLAHE object
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size)
        )
        
        logger.info(f"CLAHE initialized: clip={clip_limit}, tile={tile_size}")
    
    def process(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Apply CLAHE to image
        
        Args:
            image: Input image (PIL or numpy)
            
        Returns:
            Enhanced PIL Image
        """
        # Convert to numpy
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image.copy()
        
        if self.apply_to_lab and len(img_np.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale or direct application
            if len(img_np.shape) == 3:
                # Convert to grayscale for CLAHE
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                enhanced = self.clahe.apply(gray)
                # Convert back to RGB
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            else:
                enhanced = self.clahe.apply(img_np)
        
        return Image.fromarray(enhanced)


class DenoisingProcessor:
    """
    🧹 Denoising Processor
    
    Remove noise while preserving details
    
    Methods:
    - Non-local means denoising (best for Gaussian noise)
    - Bilateral filter (edge-preserving smoothing)
    - Gaussian blur (simple but effective)
    
    Example:
        >>> denoiser = DenoisingProcessor(method='nlm', strength=10)
        >>> clean = denoiser.process(noisy_image)
    """
    
    def __init__(
        self,
        method: str = 'nlm',
        strength: int = 10,
        search_window: int = 21,
        block_size: int = 7
    ):
        """
        Initialize Denoising Processor
        
        Args:
            method: Denoising method ('nlm', 'bilateral', 'gaussian')
            strength: Denoising strength (higher = more smoothing)
                     NLM: 10-20 recommended
                     Bilateral: 50-100 recommended
            search_window: Search window size for NLM (21 recommended)
            block_size: Block size for NLM (7 recommended)
        """
        self.method = method.lower()
        self.strength = strength
        self.search_window = search_window
        self.block_size = block_size
        
        if method in ['nlm', 'bilateral'] and not CV2_AVAILABLE:
            logger.warning(f"OpenCV not available. Falling back to gaussian blur.")
            self.method = 'gaussian'
        
        logger.info(f"Denoising initialized: method={method}, strength={strength}")
    
    def process(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Apply denoising to image
        
        Args:
            image: Input image
            
        Returns:
            Denoised PIL Image
        """
        # Convert to numpy
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image.copy()
        
        if self.method == 'nlm' and CV2_AVAILABLE:
            # Non-local means denoising
            if len(img_np.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(
                    img_np,
                    None,
                    h=self.strength,
                    hColor=self.strength,
                    templateWindowSize=self.block_size,
                    searchWindowSize=self.search_window
                )
            else:
                denoised = cv2.fastNlMeansDenoising(
                    img_np,
                    None,
                    h=self.strength,
                    templateWindowSize=self.block_size,
                    searchWindowSize=self.search_window
                )
        
        elif self.method == 'bilateral' and CV2_AVAILABLE:
            # Bilateral filter
            denoised = cv2.bilateralFilter(
                img_np,
                d=9,
                sigmaColor=self.strength,
                sigmaSpace=self.strength
            )
        
        else:
            # Gaussian blur fallback
            pil_img = Image.fromarray(img_np) if isinstance(image, np.ndarray) else image
            radius = max(1, self.strength // 10)
            denoised_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
            denoised = np.array(denoised_pil)
        
        return Image.fromarray(denoised)


class EdgeEnhancer:
    """
    ✨ Edge Enhancement Processor
    
    Enhance edges and details in images
    
    Methods:
    - Unsharp masking (traditional)
    - Laplacian sharpening
    - High-pass filter
    
    Example:
        >>> enhancer = EdgeEnhancer(method='unsharp', amount=1.5)
        >>> sharp = enhancer.process(blurry_image)
    """
    
    def __init__(
        self,
        method: str = 'unsharp',
        amount: float = 1.5,
        radius: float = 1.0,
        threshold: int = 0
    ):
        """
        Initialize Edge Enhancer
        
        Args:
            method: Enhancement method ('unsharp', 'laplacian', 'detail')
            amount: Enhancement strength (1.0-2.0 recommended)
            radius: Blur radius for unsharp mask
            threshold: Minimum difference for edge detection
        """
        self.method = method.lower()
        self.amount = amount
        self.radius = radius
        self.threshold = threshold
        
        logger.info(f"Edge enhancer initialized: method={method}, amount={amount}")
    
    def process(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Apply edge enhancement
        
        Args:
            image: Input image
            
        Returns:
            Enhanced PIL Image
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if self.method == 'unsharp':
            # Unsharp masking
            from PIL import ImageFilter
            
            # Create blurred version
            blurred = image.filter(ImageFilter.GaussianBlur(radius=self.radius))
            
            # Unsharp mask = original + amount * (original - blurred)
            img_np = np.array(image).astype(np.float32)
            blur_np = np.array(blurred).astype(np.float32)
            
            mask = img_np - blur_np
            enhanced = img_np + self.amount * mask
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            return Image.fromarray(enhanced)
        
        elif self.method == 'laplacian' and CV2_AVAILABLE:
            # Laplacian sharpening
            img_np = np.array(image)
            
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            
            if len(img_np.shape) == 3:
                laplacian = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
            
            enhanced = cv2.addWeighted(img_np, 1.0, laplacian, self.amount - 1.0, 0)
            return Image.fromarray(enhanced)
        
        else:
            # Simple detail enhancement
            enhancer = ImageEnhance.Sharpness(image)
            return enhancer.enhance(self.amount)


class AdvancedPreprocessor:
    """
    🚀 Advanced Preprocessing Pipeline
    
    รวม preprocessing techniques ขั้นสูงทั้งหมด
    
    Pipeline:
    1. Denoising (optional)
    2. CLAHE (optional)
    3. Edge enhancement (optional)
    
    Example:
        >>> preprocessor = AdvancedPreprocessor(
        ...     enable_clahe=True,
        ...     enable_denoise=True,
        ...     enable_edge_enhance=True
        ... )
        >>> enhanced = preprocessor(image)
    """
    
    def __init__(
        self,
        enable_denoise: bool = False,
        denoise_method: str = 'nlm',
        denoise_strength: int = 10,
        enable_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
        enable_edge_enhance: bool = False,
        edge_method: str = 'unsharp',
        edge_amount: float = 1.5
    ):
        """
        Initialize Advanced Preprocessor
        
        Args:
            enable_denoise: Apply denoising
            denoise_method: Denoising method ('nlm', 'bilateral', 'gaussian')
            denoise_strength: Denoising strength
            enable_clahe: Apply CLAHE
            clahe_clip_limit: CLAHE clip limit
            clahe_tile_size: CLAHE tile size
            enable_edge_enhance: Apply edge enhancement
            edge_method: Edge enhancement method
            edge_amount: Edge enhancement amount
        """
        self.enable_denoise = enable_denoise
        self.enable_clahe = enable_clahe
        self.enable_edge_enhance = enable_edge_enhance
        
        # Initialize processors
        if enable_denoise:
            self.denoiser = DenoisingProcessor(
                method=denoise_method,
                strength=denoise_strength
            )
        
        if enable_clahe:
            try:
                self.clahe = CLAHEProcessor(
                    clip_limit=clahe_clip_limit,
                    tile_size=clahe_tile_size
                )
            except ImportError:
                logger.warning("CLAHE disabled: OpenCV not available")
                self.enable_clahe = False
        
        if enable_edge_enhance:
            self.edge_enhancer = EdgeEnhancer(
                method=edge_method,
                amount=edge_amount
            )
        
        logger.info(f"Advanced preprocessor initialized: "
                   f"denoise={enable_denoise}, clahe={enable_clahe}, "
                   f"edge={enable_edge_enhance}")
    
    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, str, Path]
    ) -> Image.Image:
        """
        Apply full preprocessing pipeline
        
        Args:
            image: Input image (PIL, numpy, or path)
            
        Returns:
            Processed PIL Image
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Step 1: Denoising
        if self.enable_denoise:
            image = self.denoiser.process(image)
        
        # Step 2: CLAHE
        if self.enable_clahe:
            image = self.clahe.process(image)
        
        # Step 3: Edge enhancement
        if self.enable_edge_enhance:
            image = self.edge_enhancer.process(image)
        
        return image
    
    def get_config(self) -> Dict:
        """Get preprocessor configuration"""
        return {
            'enable_denoise': self.enable_denoise,
            'enable_clahe': self.enable_clahe,
            'enable_edge_enhance': self.enable_edge_enhance
        }


# ============================================================================
# Helper Functions
# ============================================================================

def create_medical_preprocessor() -> AdvancedPreprocessor:
    """
    สร้าง preprocessor สำหรับภาพแบบ medical/artifact
    
    Config:
    - CLAHE enabled (enhance contrast)
    - Mild denoising
    - No edge enhancement (preserve original details)
    
    Returns:
        AdvancedPreprocessor instance
    """
    return AdvancedPreprocessor(
        enable_denoise=True,
        denoise_method='nlm',
        denoise_strength=8,
        enable_clahe=True,
        clahe_clip_limit=2.0,
        enable_edge_enhance=False
    )


def create_artifact_preprocessor() -> AdvancedPreprocessor:
    """
    สร้าง preprocessor สำหรับวัตถุโบราณ
    
    Config:
    - CLAHE enabled (enhance low contrast details)
    - No denoising (preserve texture)
    - Edge enhancement (sharpen inscriptions/patterns)
    
    Returns:
        AdvancedPreprocessor instance
    """
    return AdvancedPreprocessor(
        enable_denoise=False,
        enable_clahe=True,
        clahe_clip_limit=3.0,
        clahe_tile_size=8,
        enable_edge_enhance=True,
        edge_method='unsharp',
        edge_amount=1.3
    )


if __name__ == "__main__":
    # Quick test
    print("🔬 Advanced Preprocessing Module")
    print("=" * 50)
    
    # Test CLAHE
    if CV2_AVAILABLE:
        clahe = CLAHEProcessor(clip_limit=2.0)
        print("✅ CLAHE processor created")
    else:
        print("⚠️  OpenCV not available, CLAHE skipped")
    
    # Test preprocessor
    preprocessor = create_artifact_preprocessor()
    print(f"✅ Artifact preprocessor created:")
    for k, v in preprocessor.get_config().items():
        print(f"   • {k}: {v}")
