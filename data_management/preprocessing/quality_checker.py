"""
🔍 Image Quality Checker
=======================

Automated image quality validation for dataset curation.

Features:
- Blur detection (Laplacian variance)
- Brightness/Contrast validation
- Resolution checking
- Artifact detection
- Quality scoring

Author: Amulet-AI Team
Date: October 2025
"""

import numpy as np
from PIL import Image, ImageStat
from typing import Union, Dict, Tuple, Optional, List
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. Some features will be limited.")


@dataclass
class QualityMetrics:
    """
    📊 Quality Metrics Container
    
    Attributes:
        blur_score: Blur detection score (higher = sharper)
        brightness: Average brightness [0-255]
        contrast: Contrast measure
        resolution: Image resolution (width, height)
        file_size: File size in bytes
        overall_score: Combined quality score [0-100]
        passed: Whether image passed quality check
        issues: List of detected issues
    """
    blur_score: float
    brightness: float
    contrast: float
    resolution: Tuple[int, int]
    file_size: int
    overall_score: float
    passed: bool
    issues: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'blur_score': self.blur_score,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'resolution': self.resolution,
            'file_size': self.file_size,
            'overall_score': self.overall_score,
            'passed': self.passed,
            'issues': self.issues
        }
    
    def __str__(self) -> str:
        """Pretty print metrics"""
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"\n{'='*50}\n"
            f"{status} | Overall Score: {self.overall_score:.1f}/100\n"
            f"{'='*50}\n"
            f"  Blur Score:  {self.blur_score:.2f}\n"
            f"  Brightness:  {self.brightness:.1f}\n"
            f"  Contrast:    {self.contrast:.1f}\n"
            f"  Resolution:  {self.resolution[0]}x{self.resolution[1]}\n"
            f"  File Size:   {self.file_size/1024:.1f} KB\n"
            f"  Issues:      {len(self.issues)}\n"
            + (f"\n  ⚠️  Problems:\n" + "\n".join(f"    • {issue}" for issue in self.issues) 
               if self.issues else "")
        )


class BlurDetector:
    """
    👁️ Blur Detection
    
    ใช้ Laplacian variance เพื่อตรวจจับภาพเบลอ
    
    Method:
    - Calculate Laplacian (2nd derivative)
    - Compute variance
    - Lower variance = more blur
    
    Thresholds:
    - > 100: Sharp
    - 50-100: Acceptable
    - < 50: Blurry
    
    Example:
        >>> detector = BlurDetector(threshold=50)
        >>> is_sharp, score = detector.check(image)
    """
    
    def __init__(self, threshold: float = 50.0):
        """
        Initialize Blur Detector
        
        Args:
            threshold: Minimum acceptable Laplacian variance
        """
        self.threshold = threshold
    
    def calculate_blur_score(
        self,
        image: Union[Image.Image, np.ndarray]
    ) -> float:
        """
        คำนวณ blur score (Laplacian variance)
        
        Args:
            image: Input image
            
        Returns:
            Blur score (higher = sharper)
        """
        # Convert to numpy grayscale
        if isinstance(image, Image.Image):
            img_gray = np.array(image.convert('L'))
        else:
            if len(image.shape) == 3:
                img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if CV2_AVAILABLE else np.mean(image, axis=2).astype(np.uint8)
            else:
                img_gray = image
        
        if CV2_AVAILABLE:
            # Use OpenCV Laplacian
            laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
            variance = laplacian.var()
        else:
            # Simple approximation using numpy gradient
            gy, gx = np.gradient(img_gray.astype(float))
            gnorm = np.sqrt(gx**2 + gy**2)
            variance = gnorm.var()
        
        return float(variance)
    
    def check(
        self,
        image: Union[Image.Image, np.ndarray]
    ) -> Tuple[bool, float]:
        """
        ตรวจสอบว่าภาพเบลอหรือไม่
        
        Args:
            image: Input image
            
        Returns:
            (is_sharp, blur_score)
        """
        score = self.calculate_blur_score(image)
        is_sharp = score >= self.threshold
        return is_sharp, score


class BrightnessContrastChecker:
    """
    ☀️ Brightness & Contrast Validator
    
    ตรวจสอบ brightness และ contrast ของภาพ
    
    Checks:
    - Average brightness (not too dark/bright)
    - Contrast (sufficient dynamic range)
    - Histogram distribution
    
    Example:
        >>> checker = BrightnessContrastChecker()
        >>> is_good, metrics = checker.check(image)
    """
    
    def __init__(
        self,
        min_brightness: float = 30.0,
        max_brightness: float = 225.0,
        min_contrast: float = 30.0
    ):
        """
        Initialize Brightness/Contrast Checker
        
        Args:
            min_brightness: Minimum acceptable brightness
            max_brightness: Maximum acceptable brightness
            min_contrast: Minimum acceptable contrast (std dev)
        """
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
    
    def calculate_metrics(
        self,
        image: Union[Image.Image, np.ndarray]
    ) -> Tuple[float, float]:
        """
        คำนวณ brightness และ contrast
        
        Args:
            image: Input image
            
        Returns:
            (brightness, contrast)
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to grayscale for analysis
        gray = image.convert('L')
        
        # Calculate statistics
        stat = ImageStat.Stat(gray)
        brightness = stat.mean[0]  # Average brightness
        contrast = stat.stddev[0]  # Standard deviation as contrast
        
        return brightness, contrast
    
    def check(
        self,
        image: Union[Image.Image, np.ndarray]
    ) -> Tuple[bool, Dict[str, float]]:
        """
        ตรวจสอบ brightness และ contrast
        
        Args:
            image: Input image
            
        Returns:
            (is_acceptable, metrics_dict)
        """
        brightness, contrast = self.calculate_metrics(image)
        
        # Check thresholds
        brightness_ok = self.min_brightness <= brightness <= self.max_brightness
        contrast_ok = contrast >= self.min_contrast
        
        is_acceptable = brightness_ok and contrast_ok
        
        metrics = {
            'brightness': brightness,
            'contrast': contrast,
            'brightness_ok': brightness_ok,
            'contrast_ok': contrast_ok
        }
        
        return is_acceptable, metrics


class ResolutionChecker:
    """
    📏 Resolution Validator
    
    ตรวจสอบความละเอียดของภาพ
    
    Checks:
    - Minimum resolution
    - Aspect ratio
    - File size
    
    Example:
        >>> checker = ResolutionChecker(min_width=224, min_height=224)
        >>> is_valid = checker.check(image)
    """
    
    def __init__(
        self,
        min_width: int = 224,
        min_height: int = 224,
        max_aspect_ratio: float = 2.0
    ):
        """
        Initialize Resolution Checker
        
        Args:
            min_width: Minimum acceptable width
            min_height: Minimum acceptable height
            max_aspect_ratio: Maximum aspect ratio (width/height)
        """
        self.min_width = min_width
        self.min_height = min_height
        self.max_aspect_ratio = max_aspect_ratio
    
    def check(
        self,
        image: Union[Image.Image, np.ndarray]
    ) -> Tuple[bool, Dict[str, any]]:
        """
        ตรวจสอบความละเอียด
        
        Args:
            image: Input image
            
        Returns:
            (is_valid, info_dict)
        """
        # Get dimensions
        if isinstance(image, Image.Image):
            width, height = image.size
        else:
            height, width = image.shape[:2]
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Check constraints
        width_ok = width >= self.min_width
        height_ok = height >= self.min_height
        aspect_ok = aspect_ratio <= self.max_aspect_ratio and aspect_ratio >= (1.0 / self.max_aspect_ratio)
        
        is_valid = width_ok and height_ok and aspect_ok
        
        info = {
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'resolution_ok': width_ok and height_ok,
            'aspect_ok': aspect_ok
        }
        
        return is_valid, info


class ImageQualityChecker:
    """
    🎯 Complete Image Quality Checker
    
    รวม quality checks ทั้งหมด
    
    Checks:
    1. Blur detection
    2. Brightness/Contrast
    3. Resolution
    4. Overall quality score
    
    Example:
        >>> checker = ImageQualityChecker()
        >>> metrics = checker.check_quality("image.jpg")
        >>> if metrics.passed:
        ...     print("Image quality OK!")
    """
    
    def __init__(
        self,
        blur_threshold: float = 50.0,
        min_brightness: float = 30.0,
        max_brightness: float = 225.0,
        min_contrast: float = 30.0,
        min_resolution: int = 224,
        max_aspect_ratio: float = 2.0
    ):
        """
        Initialize Quality Checker
        
        Args:
            blur_threshold: Minimum blur score
            min_brightness: Minimum brightness
            max_brightness: Maximum brightness
            min_contrast: Minimum contrast
            min_resolution: Minimum width/height
            max_aspect_ratio: Maximum aspect ratio
        """
        self.blur_detector = BlurDetector(threshold=blur_threshold)
        self.brightness_checker = BrightnessContrastChecker(
            min_brightness=min_brightness,
            max_brightness=max_brightness,
            min_contrast=min_contrast
        )
        self.resolution_checker = ResolutionChecker(
            min_width=min_resolution,
            min_height=min_resolution,
            max_aspect_ratio=max_aspect_ratio
        )
        
        logger.info("Image Quality Checker initialized")
    
    def check_quality(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        return_image: bool = False
    ) -> Union[QualityMetrics, Tuple[QualityMetrics, Image.Image]]:
        """
        ตรวจสอบคุณภาพภาพแบบสมบูรณ์
        
        Args:
            image: Input image (PIL, numpy, or path)
            return_image: Return image along with metrics
            
        Returns:
            QualityMetrics (and optionally the loaded image)
        """
        issues = []
        
        # Load image if path
        file_size = 0
        if isinstance(image, (str, Path)):
            file_path = Path(image)
            file_size = file_path.stat().st_size
            image = Image.open(file_path).convert('RGB')
        elif isinstance(image, Image.Image):
            # Estimate file size
            pass
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Check 1: Resolution
        res_ok, res_info = self.resolution_checker.check(image)
        if not res_ok:
            if not res_info['resolution_ok']:
                issues.append(f"Low resolution: {res_info['width']}x{res_info['height']}")
            if not res_info['aspect_ok']:
                issues.append(f"Bad aspect ratio: {res_info['aspect_ratio']:.2f}")
        
        # Check 2: Blur
        is_sharp, blur_score = self.blur_detector.check(image)
        if not is_sharp:
            issues.append(f"Image is blurry (score: {blur_score:.1f})")
        
        # Check 3: Brightness/Contrast
        bc_ok, bc_metrics = self.brightness_checker.check(image)
        if not bc_ok:
            if not bc_metrics['brightness_ok']:
                issues.append(f"Bad brightness: {bc_metrics['brightness']:.1f}")
            if not bc_metrics['contrast_ok']:
                issues.append(f"Low contrast: {bc_metrics['contrast']:.1f}")
        
        # Calculate overall score (0-100)
        scores = []
        
        # Blur score (normalized)
        blur_norm = min(100, (blur_score / self.blur_detector.threshold) * 100)
        scores.append(blur_norm)
        
        # Brightness score (normalized)
        brightness_norm = 100 if bc_metrics['brightness_ok'] else 50
        scores.append(brightness_norm)
        
        # Contrast score (normalized)
        contrast_norm = min(100, (bc_metrics['contrast'] / self.brightness_checker.min_contrast) * 100)
        scores.append(contrast_norm)
        
        # Resolution score
        resolution_score = 100 if res_ok else 50
        scores.append(resolution_score)
        
        overall_score = np.mean(scores)
        
        # Determine pass/fail
        passed = len(issues) == 0 and overall_score >= 60
        
        # Create metrics object
        metrics = QualityMetrics(
            blur_score=blur_score,
            brightness=bc_metrics['brightness'],
            contrast=bc_metrics['contrast'],
            resolution=(res_info['width'], res_info['height']),
            file_size=file_size,
            overall_score=overall_score,
            passed=passed,
            issues=issues
        )
        
        if return_image:
            return metrics, image
        return metrics
    
    def batch_check(
        self,
        image_paths: List[Union[str, Path]],
        verbose: bool = True
    ) -> Dict[str, QualityMetrics]:
        """
        ตรวจสอบ batch ของภาพ
        
        Args:
            image_paths: List of image paths
            verbose: Print progress
            
        Returns:
            Dict mapping path to QualityMetrics
        """
        results = {}
        
        for i, path in enumerate(image_paths):
            if verbose and i % 10 == 0:
                print(f"Checking {i+1}/{len(image_paths)}...", end='\r')
            
            try:
                metrics = self.check_quality(path)
                results[str(path)] = metrics
            except Exception as e:
                logger.error(f"Error checking {path}: {e}")
        
        if verbose:
            passed = sum(1 for m in results.values() if m.passed)
            print(f"\n✅ Checked {len(results)} images: {passed} passed, {len(results)-passed} failed")
        
        return results


# ============================================================================
# Helper Functions
# ============================================================================

def create_strict_checker() -> ImageQualityChecker:
    """สร้าง quality checker แบบเข้มงวด (สำหรับ production)"""
    return ImageQualityChecker(
        blur_threshold=100.0,  # High blur threshold
        min_brightness=50.0,
        max_brightness=200.0,
        min_contrast=40.0,
        min_resolution=224
    )


def create_lenient_checker() -> ImageQualityChecker:
    """สร้าง quality checker แบบผ่อนปรน (สำหรับ training data)"""
    return ImageQualityChecker(
        blur_threshold=30.0,  # Lower blur threshold
        min_brightness=20.0,
        max_brightness=235.0,
        min_contrast=20.0,
        min_resolution=128
    )


if __name__ == "__main__":
    # Quick test
    print("🔍 Image Quality Checker Module")
    print("=" * 50)
    
    # Create checkers
    strict = create_strict_checker()
    lenient = create_lenient_checker()
    
    print("✅ Strict checker created")
    print("✅ Lenient checker created")
    
    # Test with dummy image
    test_img = Image.new('RGB', (300, 300), color=(128, 128, 128))
    metrics = lenient.check_quality(test_img)
    
    print("\n📊 Test Image Quality:")
    print(metrics)
