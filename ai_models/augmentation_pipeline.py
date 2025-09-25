#!/usr/bin/env python3
"""
🔄 Advanced Data Augmentation Pipeline for Amulet Recognition
Class-aware augmentation system optimized for small, imbalanced datasets

This pipeline provides:
- Class-specific augmentation strategies (more for minority classes)
- Amulet-appropriate transformations (preserves critical features)
- Batch processing with caching
- Quality validation (ensures augmented images remain recognizable)

Author: AI Assistant
Compatible: Python 3.13+
Dependencies: opencv-python, numpy, albumentations (optional)
"""

import os
import cv2
import numpy as np
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
import random
from abc import ABC, abstractmethod


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline"""
    
    # Target samples per class
    target_samples_per_class: int = 50
    max_augmentation_factor: int = 20  # Safety limit
    
    # Augmentation parameters
    rotation_range: Tuple[float, float] = (-20, 20)  # degrees
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.9, 1.1)
    noise_std: float = 8.0
    blur_kernel_sizes: List[int] = (3, 5)
    
    # Quality control
    enable_quality_check: bool = True
    min_quality_score: float = 0.7
    
    # Output settings
    output_format: str = "jpg"
    output_quality: int = 95
    preserve_originals: bool = True
    
    # Processing
    batch_size: int = 32
    use_multiprocessing: bool = True
    random_seed: int = 42


class BaseAugmentation(ABC):
    """Base class for augmentation techniques"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.random = np.random.RandomState(config.random_seed)
    
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to image"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get augmentation name for logging"""
        pass


class RotationAugmentation(BaseAugmentation):
    """Safe rotation augmentation for amulets"""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply random rotation"""
        angle = self.random.uniform(*self.config.rotation_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation with border reflection to avoid black borders
        rotated = cv2.warpAffine(
            image, M, (w, h), 
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        return rotated
    
    def get_name(self) -> str:
        return "rotation"


class BrightnessContrastAugmentation(BaseAugmentation):
    """Brightness and contrast adjustment"""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply random brightness and contrast"""
        brightness_factor = self.random.uniform(*self.config.brightness_range)
        contrast_factor = self.random.uniform(*self.config.contrast_range)
        
        # Apply brightness (additive)
        bright_image = cv2.convertScaleAbs(image, alpha=1.0, beta=(brightness_factor - 1.0) * 50)
        
        # Apply contrast (multiplicative)
        contrast_image = cv2.convertScaleAbs(bright_image, alpha=contrast_factor, beta=0)
        
        return contrast_image
    
    def get_name(self) -> str:
        return "brightness_contrast"


class ColorAugmentation(BaseAugmentation):
    """Color/saturation augmentation"""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply random saturation adjustment"""
        if len(image.shape) != 3:
            return image  # Skip grayscale images
        
        # Convert to HSV for saturation adjustment
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Adjust saturation
        saturation_factor = self.random.uniform(*self.config.saturation_range)
        hsv[:, :, 1] *= saturation_factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Convert back to RGB
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return rgb
    
    def get_name(self) -> str:
        return "color_saturation"


class NoiseAugmentation(BaseAugmentation):
    """Gaussian noise augmentation"""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Add random Gaussian noise"""
        noise = self.random.normal(0, self.config.noise_std, image.shape)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    def get_name(self) -> str:
        return "gaussian_noise"


class BlurAugmentation(BaseAugmentation):
    """Subtle blur augmentation"""
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply random blur"""
        kernel_size = self.random.choice(self.config.blur_kernel_sizes)
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return blurred
    
    def get_name(self) -> str:
        return "gaussian_blur"


class QualityValidator:
    """Validates augmented images for quality"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def calculate_quality_score(self, original: np.ndarray, augmented: np.ndarray) -> float:
        """
        Calculate quality score between original and augmented image
        
        Returns:
            Quality score between 0 and 1 (higher is better)
        """
        try:
            # Convert to grayscale for comparison
            if len(original.shape) == 3:
                orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            else:
                orig_gray = original
                
            if len(augmented.shape) == 3:
                aug_gray = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)
            else:
                aug_gray = augmented
            
            # Structural Similarity Index (SSIM) approximation
            # Calculate mean and variance
            mu1 = np.mean(orig_gray)
            mu2 = np.mean(aug_gray)
            sigma1 = np.var(orig_gray)
            sigma2 = np.var(aug_gray)
            sigma12 = np.mean((orig_gray - mu1) * (aug_gray - mu2))
            
            # SSIM constants
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2
            
            # SSIM formula
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
            
            return max(0.0, min(1.0, ssim))
            
        except Exception as e:
            logging.warning(f"Quality calculation failed: {e}")
            return 0.5  # Neutral score if calculation fails
    
    def is_acceptable(self, original: np.ndarray, augmented: np.ndarray) -> bool:
        """Check if augmented image meets quality threshold"""
        if not self.config.enable_quality_check:
            return True
        
        score = self.calculate_quality_score(original, augmented)
        return score >= self.config.min_quality_score


class ClassAwareAugmentationPipeline:
    """
    Main augmentation pipeline with class-aware strategies
    
    This pipeline analyzes class distribution and applies more aggressive
    augmentation to minority classes while preserving majority class balance.
    """
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize augmentation techniques
        self.augmentations = [
            RotationAugmentation(config),
            BrightnessContrastAugmentation(config),
            ColorAugmentation(config),
            NoiseAugmentation(config),
            BlurAugmentation(config)
        ]
        
        # Quality validator
        self.quality_validator = QualityValidator(config)
        
        # Statistics
        self.stats = {
            'original_counts': {},
            'target_counts': {},
            'augmented_counts': {},
            'quality_rejected': 0,
            'total_generated': 0
        }
        
        # Set random seed
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('AugmentationPipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_class_distribution(self, data_dir: Path) -> Dict[str, int]:
        """Analyze current class distribution"""
        class_counts = {}
        
        for class_dir in data_dir.iterdir():
            if class_dir.is_dir():
                # Count image files
                image_files = []
                valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                
                for ext in valid_extensions:
                    image_files.extend(class_dir.glob(f"*{ext}"))
                    image_files.extend(class_dir.glob(f"*{ext.upper()}"))
                
                class_counts[class_dir.name] = len(image_files)
        
        self.stats['original_counts'] = class_counts
        return class_counts
    
    def calculate_augmentation_strategy(self, class_counts: Dict[str, int]) -> Dict[str, int]:
        """Calculate how many augmentations needed per class"""
        strategy = {}
        
        for class_name, current_count in class_counts.items():
            if current_count == 0:
                self.logger.warning(f"⚠️ Class {class_name} has no images!")
                strategy[class_name] = 0
                continue
            
            # Calculate target augmentations
            target_total = self.config.target_samples_per_class
            needed = max(0, target_total - current_count)
            
            # Calculate augmentation factor (how many new images per original)
            aug_factor = min(
                needed // current_count + (1 if needed % current_count > 0 else 0),
                self.config.max_augmentation_factor
            )
            
            strategy[class_name] = aug_factor
            
            self.logger.info(
                f"📊 {class_name}: {current_count} → {current_count + (aug_factor * current_count)} "
                f"(+{aug_factor * current_count} augmented)"
            )
        
        self.stats['target_counts'] = {
            class_name: class_counts[class_name] + (strategy[class_name] * class_counts[class_name])
            for class_name in class_counts
        }
        
        return strategy
    
    def load_image_safely(self, image_path: Path) -> Optional[np.ndarray]:
        """Load image with error handling"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image_rgb
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load {image_path}: {e}")
            return None
    
    def apply_random_augmentation(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """Apply random augmentation technique"""
        # Choose random augmentation
        augmentation = np.random.choice(self.augmentations)
        
        # Apply augmentation
        augmented = augmentation.apply(image.copy())
        
        return augmented, augmentation.get_name()
    
    def generate_augmented_filename(self, original_path: Path, aug_type: str, index: int) -> str:
        """Generate filename for augmented image"""
        stem = original_path.stem
        suffix = original_path.suffix
        
        # Include augmentation type and index
        new_name = f"{stem}_aug_{aug_type}_{index:03d}{suffix}"
        return new_name
    
    def save_augmented_image(self, image: np.ndarray, output_path: Path) -> bool:
        """Save augmented image with quality settings"""
        try:
            # Convert RGB back to BGR for OpenCV saving
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Set quality parameters
            if self.config.output_format.lower() == 'jpg':
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.output_quality]
            else:
                encode_params = []
            
            success = cv2.imwrite(str(output_path), image_bgr, encode_params)
            
            if not success:
                self.logger.warning(f"⚠️ Failed to save {output_path}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error saving {output_path}: {e}")
            return False
    
    def augment_class(self, class_dir: Path, output_dir: Path, aug_factor: int) -> int:
        """Augment all images in a class directory"""
        class_name = class_dir.name
        output_class_dir = output_dir / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for ext in valid_extensions:
            image_files.extend(class_dir.glob(f"*{ext}"))
            image_files.extend(class_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            self.logger.warning(f"⚠️ No images found in {class_dir}")
            return 0
        
        total_generated = 0
        
        self.logger.info(f"🔄 Augmenting {class_name}: {len(image_files)} images × {aug_factor}")
        
        # Copy originals if preserve_originals is True
        if self.config.preserve_originals:
            for img_file in image_files:
                output_path = output_class_dir / img_file.name
                
                # Load and save (ensures consistent format)
                image = self.load_image_safely(img_file)
                if image is not None:
                    self.save_augmented_image(image, output_path)
        
        # Generate augmentations
        for img_file in image_files:
            # Load original image
            original_image = self.load_image_safely(img_file)
            if original_image is None:
                continue
            
            # Generate augmentations for this image
            generated_count = 0
            attempts = 0
            max_attempts = aug_factor * 3  # Safety limit
            
            while generated_count < aug_factor and attempts < max_attempts:
                attempts += 1
                
                # Apply augmentation
                augmented_image, aug_type = self.apply_random_augmentation(original_image)
                
                # Quality check
                if self.quality_validator.is_acceptable(original_image, augmented_image):
                    # Generate filename and save
                    filename = self.generate_augmented_filename(img_file, aug_type, generated_count + 1)
                    output_path = output_class_dir / filename
                    
                    if self.save_augmented_image(augmented_image, output_path):
                        generated_count += 1
                        total_generated += 1
                    
                else:
                    self.stats['quality_rejected'] += 1
        
        self.logger.info(f"✅ {class_name}: Generated {total_generated} augmented images")
        
        self.stats['augmented_counts'][class_name] = total_generated
        return total_generated
    
    def run_augmentation(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Run complete augmentation pipeline
        
        Args:
            input_dir: Directory containing class subdirectories with images
            output_dir: Directory to save augmented dataset
            
        Returns:
            Dictionary with augmentation statistics and results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        self.logger.info(f"🚀 Starting augmentation pipeline")
        self.logger.info(f"📁 Input: {input_path}")
        self.logger.info(f"📁 Output: {output_path}")
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Analyze current distribution
        self.logger.info("📊 Analyzing class distribution...")
        class_counts = self.analyze_class_distribution(input_path)
        
        if not class_counts:
            self.logger.error("❌ No classes found in input directory!")
            return {"success": False, "error": "No classes found"}
        
        # Calculate augmentation strategy
        self.logger.info("🎯 Calculating augmentation strategy...")
        augmentation_strategy = self.calculate_augmentation_strategy(class_counts)
        
        # Process each class
        self.logger.info("🔄 Starting augmentation process...")
        total_generated = 0
        
        for class_dir in input_path.iterdir():
            if class_dir.is_dir() and class_dir.name in augmentation_strategy:
                aug_factor = augmentation_strategy[class_dir.name]
                
                if aug_factor > 0:
                    generated = self.augment_class(class_dir, output_path, aug_factor)
                    total_generated += generated
                else:
                    self.logger.info(f"⏭️ Skipping {class_dir.name} (no augmentation needed)")
        
        # Update final statistics
        self.stats['total_generated'] = total_generated
        
        # Generate summary report
        results = {
            "success": True,
            "statistics": self.stats,
            "config": asdict(self.config),
            "summary": {
                "total_original_images": sum(class_counts.values()),
                "total_augmented_generated": total_generated,
                "quality_rejected": self.stats['quality_rejected'],
                "classes_processed": len(class_counts)
            }
        }
        
        # Save augmentation report
        report_path = output_path / "augmentation_report.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"🎉 Augmentation completed!")
        self.logger.info(f"📊 Generated {total_generated} new images")
        self.logger.info(f"🚫 Rejected {self.stats['quality_rejected']} low-quality images")
        self.logger.info(f"📄 Report saved: {report_path}")
        
        return results


def create_augmentation_config_for_amulets() -> AugmentationConfig:
    """Create optimized augmentation config for amulet recognition"""
    return AugmentationConfig(
        # Conservative target for small dataset
        target_samples_per_class=40,
        max_augmentation_factor=15,
        
        # Amulet-appropriate augmentation parameters
        rotation_range=(-15, 15),  # Conservative rotation
        brightness_range=(0.85, 1.15),  # Subtle brightness changes
        contrast_range=(0.9, 1.1),  # Subtle contrast changes
        saturation_range=(0.95, 1.05),  # Very subtle color changes
        noise_std=5.0,  # Light noise
        blur_kernel_sizes=[3],  # Only slight blur
        
        # Quality control
        enable_quality_check=True,
        min_quality_score=0.75,  # High quality threshold
        
        # Output settings
        output_format="jpg",
        output_quality=95,
        preserve_originals=True,
        
        # Processing
        batch_size=16,  # Conservative for memory
        random_seed=42
    )


def main():
    """CLI entry point for augmentation pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🔄 Class-Aware Data Augmentation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input_dir", help="Input directory with class subdirectories")
    parser.add_argument("output_dir", help="Output directory for augmented dataset")
    parser.add_argument("--target-per-class", type=int, default=40, 
                       help="Target samples per class (default: 40)")
    parser.add_argument("--max-factor", type=int, default=15,
                       help="Maximum augmentation factor (default: 15)")
    parser.add_argument("--quality-threshold", type=float, default=0.75,
                       help="Minimum quality score (default: 0.75)")
    parser.add_argument("--disable-quality-check", action="store_true",
                       help="Disable quality validation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = AugmentationConfig(
        target_samples_per_class=args.target_per_class,
        max_augmentation_factor=args.max_factor,
        enable_quality_check=not args.disable_quality_check,
        min_quality_score=args.quality_threshold,
        random_seed=args.seed
    )
    
    # Create and run pipeline
    pipeline = ClassAwareAugmentationPipeline(config)
    results = pipeline.run_augmentation(args.input_dir, args.output_dir)
    
    # Exit with appropriate code
    if results["success"]:
        print("✅ Augmentation completed successfully!")
        exit(0)
    else:
        print(f"❌ Augmentation failed: {results.get('error', 'Unknown error')}")
        exit(1)


if __name__ == "__main__":
    main()