#!/usr/bin/env python3
"""
🎯 Modular Feature Extraction System for Hybrid ML Pipeline
Combines CNN features (PyTorch) with Classical CV features (OpenCV)

This system provides:
- Abstract FeatureExtractor interface for modularity
- PyTorch-based CNN feature extraction (CPU-optimized)
- Classical computer vision features (HOG, LBP, Color, Texture)
- Feature caching and batch processing
- Automatic fallback mechanisms

Author: AI Assistant
Compatible: Python 3.13+
Dependencies: torch, torchvision, opencv-python, scikit-image, numpy
"""

import os
import sys
import cv2
import numpy as np
import json
import joblib
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports with fallback
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    PYTORCH_AVAILABLE = True
except ImportError as e:
    PYTORCH_AVAILABLE = False
    print(f"⚠️ PyTorch not available: {e}")

# scikit-image imports with fallback
try:
    from skimage.feature import local_binary_pattern, hog
    from skimage import filters, measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️ scikit-image not available, using OpenCV alternatives")


@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    
    # CNN Features (PyTorch)
    cnn_backbone: str = "resnet18"  # resnet18, resnet50, mobilenet_v2, efficientnet_b0
    cnn_pretrained: bool = True
    cnn_freeze_weights: bool = True
    cnn_device: str = "cpu"  # Force CPU for compatibility
    cnn_batch_size: int = 8
    
    # Classical Features
    enable_hog: bool = True
    enable_lbp: bool = True
    enable_color_hist: bool = True
    enable_texture: bool = True
    enable_edge_features: bool = True
    
    # HOG parameters
    hog_orientations: int = 9
    hog_pixels_per_cell: Tuple[int, int] = (8, 8)
    hog_cells_per_block: Tuple[int, int] = (2, 2)
    
    # LBP parameters
    lbp_radius: int = 3
    lbp_n_points: int = 24
    
    # Color histogram parameters
    color_bins: int = 32
    color_ranges: List[Tuple[int, int]] = ((0, 256), (0, 256), (0, 256))
    
    # Image preprocessing
    target_size: Tuple[int, int] = (224, 224)
    normalize_method: str = "imagenet"  # imagenet, zero_one, standard
    
    # Caching
    enable_caching: bool = True
    cache_dir: str = "feature_cache"
    
    # Processing
    use_multiprocessing: bool = False  # Disabled for stability
    verbose: bool = True


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(f'{self.__class__.__name__}')
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from a single image"""
        pass
    
    @abstractmethod
    def get_feature_dimension(self) -> int:
        """Get the dimension of extracted features"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get names/descriptions of features for interpretability"""
        pass
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract features from batch of images"""
        features = []
        for image in images:
            feat = self.extract_features(image)
            features.append(feat)
        return np.array(features)
    
    def _compute_image_hash(self, image: np.ndarray) -> str:
        """Compute hash of image for caching"""
        return hashlib.md5(image.tobytes()).hexdigest()[:16]
    
    def _get_cache_path(self, image_hash: str) -> Path:
        """Get cache file path for image hash"""
        return self.cache_dir / f"{self.__class__.__name__}_{image_hash}.npy"
    
    def _load_from_cache(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Load features from cache if available"""
        if not self.config.enable_caching:
            return None
        
        image_hash = self._compute_image_hash(image)
        cache_path = self._get_cache_path(image_hash)
        
        if cache_path.exists():
            try:
                features = np.load(cache_path)
                return features
            except Exception as e:
                self.logger.warning(f"Cache load failed for {cache_path}: {e}")
        
        return None
    
    def _save_to_cache(self, image: np.ndarray, features: np.ndarray) -> None:
        """Save features to cache"""
        if not self.config.enable_caching:
            return
        
        try:
            image_hash = self._compute_image_hash(image)
            cache_path = self._get_cache_path(image_hash)
            np.save(cache_path, features)
        except Exception as e:
            self.logger.warning(f"Cache save failed: {e}")


class CNNFeatureExtractor(FeatureExtractor):
    """CNN feature extractor using PyTorch pre-trained models"""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available for CNN feature extraction")
        
        self.device = torch.device(config.cnn_device)
        self.model = None
        self.transform = None
        self.feature_dim = 0
        
        self._setup_model()
        self._setup_transform()
    
    def _setup_model(self):
        """Setup pre-trained CNN model"""
        self.logger.info(f"🧠 Loading {self.config.cnn_backbone} (CPU-only)...")
        
        try:
            if self.config.cnn_backbone == "resnet18":
                if self.config.cnn_pretrained:
                    model = models.resnet18(weights='IMAGENET1K_V1')
                else:
                    model = models.resnet18(weights=None)
                # Remove final classification layer
                self.model = nn.Sequential(*list(model.children())[:-1])
                self.feature_dim = 512
                
            elif self.config.cnn_backbone == "resnet50":
                if self.config.cnn_pretrained:
                    model = models.resnet50(weights='IMAGENET1K_V2')
                else:
                    model = models.resnet50(weights=None)
                self.model = nn.Sequential(*list(model.children())[:-1])
                self.feature_dim = 2048
                
            elif self.config.cnn_backbone == "mobilenet_v2":
                if self.config.cnn_pretrained:
                    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
                else:
                    model = models.mobilenet_v2(weights=None)
                # Remove classifier
                self.model = model.features
                self.model.add_module('global_pool', nn.AdaptiveAvgPool2d(1))
                self.feature_dim = 1280
                
            elif self.config.cnn_backbone == "efficientnet_b0":
                if self.config.cnn_pretrained:
                    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
                else:
                    model = models.efficientnet_b0(weights=None)
                self.model = model.features
                self.model.add_module('global_pool', nn.AdaptiveAvgPool2d(1))
                self.feature_dim = 1280
                
            else:
                raise ValueError(f"Unsupported backbone: {self.config.cnn_backbone}")
            
            # Freeze weights if specified
            if self.config.cnn_freeze_weights:
                for param in self.model.parameters():
                    param.requires_grad = False
            
            # Set to evaluation mode and move to device
            self.model.eval()
            self.model.to(self.device)
            
            self.logger.info(f"✅ Model loaded: {self.config.cnn_backbone}, "
                           f"feature dim: {self.feature_dim}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _setup_transform(self):
        """Setup image preprocessing transform"""
        if self.config.normalize_method == "imagenet":
            # ImageNet normalization
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        elif self.config.normalize_method == "zero_one":
            # Simple 0-1 normalization
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            # No normalization
            normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.config.target_size),
            transforms.CenterCrop(self.config.target_size),
            transforms.ToTensor(),
            normalize
        ])
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract CNN features from image"""
        # Check cache first
        cached_features = self._load_from_cache(image)
        if cached_features is not None:
            return cached_features
        
        try:
            # Ensure image is in correct format (RGB)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Already RGB
                image_rgb = image
            elif len(image.shape) == 3 and image.shape[2] == 1:
                # Grayscale to RGB
                image_rgb = np.repeat(image, 3, axis=2)
            elif len(image.shape) == 2:
                # Grayscale to RGB
                image_rgb = np.stack([image, image, image], axis=2)
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
            
            # Ensure uint8 format
            if image_rgb.dtype != np.uint8:
                image_rgb = (image_rgb * 255).astype(np.uint8)
            
            # Apply transform
            image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
                
                # Flatten features
                if features.dim() > 2:
                    features = features.view(features.size(0), -1)
                
                # Convert to numpy
                features_np = features.cpu().numpy().flatten()
                
                # Ensure correct dimension
                if len(features_np) != self.feature_dim:
                    self.logger.warning(f"Feature dimension mismatch: {len(features_np)} vs {self.feature_dim}")
                    # Pad or truncate as needed
                    if len(features_np) < self.feature_dim:
                        features_np = np.pad(features_np, (0, self.feature_dim - len(features_np)))
                    else:
                        features_np = features_np[:self.feature_dim]
                
                # Cache features
                self._save_to_cache(image, features_np)
                
                return features_np
                
        except Exception as e:
            self.logger.error(f"❌ CNN feature extraction failed: {e}")
            # Return zero vector as fallback
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def get_feature_dimension(self) -> int:
        return self.feature_dim
    
    def get_feature_names(self) -> List[str]:
        return [f"cnn_feat_{i}" for i in range(self.feature_dim)]
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Optimized batch processing for CNN features"""
        batch_features = []
        
        # Process in smaller batches to avoid memory issues
        batch_size = self.config.cnn_batch_size
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Check for cached features first
            batch_tensors = []
            batch_indices = []
            
            for j, image in enumerate(batch):
                cached = self._load_from_cache(image)
                if cached is not None:
                    batch_features.append(cached)
                else:
                    batch_tensors.append(image)
                    batch_indices.append(i + j)
            
            # Process uncached images
            if batch_tensors:
                try:
                    # Prepare batch tensor
                    processed_tensors = []
                    for image in batch_tensors:
                        # Process image (same as single image processing)
                        if len(image.shape) == 3 and image.shape[2] == 3:
                            image_rgb = image
                        elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                            image_rgb = np.stack([image.squeeze()] * 3, axis=2) if len(image.shape) == 3 else np.stack([image] * 3, axis=2)
                        else:
                            raise ValueError(f"Unsupported image shape: {image.shape}")
                        
                        if image_rgb.dtype != np.uint8:
                            image_rgb = (image_rgb * 255).astype(np.uint8)
                        
                        tensor = self.transform(image_rgb)
                        processed_tensors.append(tensor)
                    
                    # Stack into batch
                    batch_tensor = torch.stack(processed_tensors).to(self.device)
                    
                    # Extract features
                    with torch.no_grad():
                        batch_feats = self.model(batch_tensor)
                        
                        if batch_feats.dim() > 2:
                            batch_feats = batch_feats.view(batch_feats.size(0), -1)
                        
                        batch_feats_np = batch_feats.cpu().numpy()
                    
                    # Add to results and cache
                    for j, feat in enumerate(batch_feats_np):
                        if len(feat) != self.feature_dim:
                            if len(feat) < self.feature_dim:
                                feat = np.pad(feat, (0, self.feature_dim - len(feat)))
                            else:
                                feat = feat[:self.feature_dim]
                        
                        batch_features.append(feat)
                        # Cache individual features
                        self._save_to_cache(batch_tensors[j], feat)
                        
                except Exception as e:
                    self.logger.error(f"❌ Batch CNN extraction failed: {e}")
                    # Add zero vectors for failed batch
                    for _ in batch_tensors:
                        batch_features.append(np.zeros(self.feature_dim, dtype=np.float32))
        
        return np.array(batch_features[:len(images)])  # Ensure correct length


class ClassicalFeatureExtractor(FeatureExtractor):
    """Classical computer vision feature extractor using OpenCV"""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.feature_dim = self._calculate_feature_dimension()
        self.logger.info(f"🔧 Classical feature extractor initialized, dim: {self.feature_dim}")
    
    def _calculate_feature_dimension(self) -> int:
        """Calculate total feature dimension"""
        dim = 0
        
        if self.config.enable_hog:
            # HOG features dimension calculation
            # Simplified: assume standard image size and parameters
            hog_dim = 3780  # Approximate for 128x128 image
            dim += hog_dim
        
        if self.config.enable_lbp:
            # LBP histogram (uniform patterns)
            lbp_dim = self.config.lbp_n_points + 2  # +2 for non-uniform patterns
            dim += lbp_dim
        
        if self.config.enable_color_hist:
            # Color histograms (3 channels)
            color_dim = self.config.color_bins * 3
            dim += color_dim
        
        if self.config.enable_texture:
            # Basic texture features
            texture_dim = 8  # Mean, std, contrast, etc.
            dim += texture_dim
        
        if self.config.enable_edge_features:
            # Edge statistics and orientation histogram
            edge_dim = 39  # 3 stats + 36-bin orientation histogram
            dim += edge_dim
        
        return dim
    
    def _extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG (Histogram of Oriented Gradients) features"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Resize to standard size for consistent HOG features
            target_size = (128, 128)  # Standard size for HOG
            gray_resized = cv2.resize(gray, target_size)
            
            if SKIMAGE_AVAILABLE:
                # Use scikit-image HOG (more features)
                from skimage.feature import hog
                features = hog(
                    gray_resized,
                    orientations=self.config.hog_orientations,
                    pixels_per_cell=self.config.hog_pixels_per_cell,
                    cells_per_block=self.config.hog_cells_per_block,
                    block_norm='L2-Hys',
                    visualize=False,
                    feature_vector=True
                )
            else:
                # Use OpenCV HOG
                hog = cv2.HOGDescriptor(
                    _winSize=(128, 128),
                    _blockSize=(16, 16),
                    _blockStride=(8, 8),
                    _cellSize=(8, 8),
                    _nbins=self.config.hog_orientations
                )
                features = hog.compute(gray_resized)
                features = features.flatten()
            
            return features.astype(np.float32)
            
        except Exception as e:
            self.logger.warning(f"HOG extraction failed: {e}")
            return np.zeros(3780, dtype=np.float32)  # Default HOG size
    
    def _extract_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """Extract LBP (Local Binary Pattern) features"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            if SKIMAGE_AVAILABLE:
                # Use scikit-image LBP
                from skimage.feature import local_binary_pattern
                lbp = local_binary_pattern(
                    gray,
                    self.config.lbp_n_points,
                    self.config.lbp_radius,
                    method='uniform'
                )
                
                # Create histogram
                n_bins = self.config.lbp_n_points + 2
                hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
                
            else:
                # Simple LBP implementation
                lbp = self._compute_lbp_opencv(gray)
                hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            
            # Normalize histogram
            hist = hist.astype(np.float32)
            hist = hist / (np.sum(hist) + 1e-8)
            
            return hist
            
        except Exception as e:
            self.logger.warning(f"LBP extraction failed: {e}")
            return np.zeros(self.config.lbp_n_points + 2, dtype=np.float32)
    
    def _compute_lbp_opencv(self, gray: np.ndarray) -> np.ndarray:
        """Simple LBP computation using OpenCV"""
        height, width = gray.shape
        lbp = np.zeros_like(gray)
        
        # Simple 8-connectivity LBP
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                center = gray[i, j]
                code = 0
                
                # 8 neighbors
                neighbors = [
                    gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                    gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                    gray[i+1, j-1], gray[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i, j] = code
        
        return lbp
    
    def _extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """Extract color histogram features"""
        try:
            if len(image.shape) == 3:
                # RGB histograms
                features = []
                for channel in range(3):
                    hist = cv2.calcHist(
                        [image], [channel], None,
                        [self.config.color_bins],
                        [0, 256]
                    )
                    hist = hist.flatten().astype(np.float32)
                    hist = hist / (np.sum(hist) + 1e-8)  # Normalize
                    features.extend(hist)
                
                return np.array(features, dtype=np.float32)
            else:
                # Grayscale histogram
                hist = cv2.calcHist([image], [0], None, [self.config.color_bins], [0, 256])
                hist = hist.flatten().astype(np.float32)
                hist = hist / (np.sum(hist) + 1e-8)
                
                # Repeat for 3 channels to maintain consistency
                return np.tile(hist, 3)
                
        except Exception as e:
            self.logger.warning(f"Color feature extraction failed: {e}")
            return np.zeros(self.config.color_bins * 3, dtype=np.float32)
    
    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """Extract basic texture features"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
            else:
                gray = image.astype(np.float32)
            
            # Basic texture statistics
            features = []
            
            # First-order statistics
            features.append(np.mean(gray))  # Mean intensity
            features.append(np.std(gray))   # Standard deviation
            features.append(np.min(gray))   # Minimum
            features.append(np.max(gray))   # Maximum
            
            # Gradient-based features
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            
            features.append(np.mean(np.abs(grad_x)))  # Mean gradient X
            features.append(np.mean(np.abs(grad_y)))  # Mean gradient Y
            
            # Laplacian (edge response)
            laplacian = cv2.Laplacian(gray, cv2.CV_32F)
            features.append(np.mean(np.abs(laplacian)))  # Mean Laplacian
            features.append(np.std(laplacian))           # Std Laplacian
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.warning(f"Texture feature extraction failed: {e}")
            return np.zeros(8, dtype=np.float32)
    
    def _extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """Extract edge-based features"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Edge statistics
            edge_density = np.sum(edges > 0) / edges.size
            edge_mean = np.mean(edges)
            edge_std = np.std(edges)
            
            # Gradient direction analysis
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient directions
            angles = np.arctan2(grad_y, grad_x)
            
            # Create orientation histogram (36 bins for 10-degree intervals)
            hist, _ = np.histogram(angles.ravel(), bins=36, range=(-np.pi, np.pi))
            hist = hist.astype(np.float32)
            hist = hist / (np.sum(hist) + 1e-8)  # Normalize
            
            # Combine edge statistics and orientation histogram
            features = [edge_density, edge_mean, edge_std]
            features.extend(hist)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.warning(f"Edge feature extraction failed: {e}")
            return np.zeros(39, dtype=np.float32)  # 3 stats + 36 histogram bins
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract all classical features from image"""
        # Check cache first
        cached_features = self._load_from_cache(image)
        if cached_features is not None:
            return cached_features
        
        try:
            # Resize image to standard size
            image_resized = cv2.resize(image, self.config.target_size)
            
            all_features = []
            
            # Extract individual feature types
            if self.config.enable_hog:
                hog_features = self._extract_hog_features(image_resized)
                all_features.append(hog_features)
            
            if self.config.enable_lbp:
                lbp_features = self._extract_lbp_features(image_resized)
                all_features.append(lbp_features)
            
            if self.config.enable_color_hist:
                color_features = self._extract_color_features(image_resized)
                all_features.append(color_features)
            
            if self.config.enable_texture:
                texture_features = self._extract_texture_features(image_resized)
                all_features.append(texture_features)
            
            if self.config.enable_edge_features:
                edge_features = self._extract_edge_features(image_resized)
                all_features.append(edge_features)
            
            # Concatenate all features
            if all_features:
                combined_features = np.concatenate(all_features)
            else:
                combined_features = np.array([], dtype=np.float32)
            
            # Ensure correct dimension
            if len(combined_features) != self.feature_dim:
                self.logger.warning(f"Feature dimension mismatch: {len(combined_features)} vs {self.feature_dim}")
                if len(combined_features) < self.feature_dim:
                    combined_features = np.pad(combined_features, (0, self.feature_dim - len(combined_features)))
                else:
                    combined_features = combined_features[:self.feature_dim]
            
            # Cache features
            self._save_to_cache(image, combined_features)
            
            return combined_features
            
        except Exception as e:
            self.logger.error(f"❌ Classical feature extraction failed: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def get_feature_dimension(self) -> int:
        return self.feature_dim
    
    def get_feature_names(self) -> List[str]:
        """Get names of classical features"""
        names = []
        
        if self.config.enable_hog:
            names.extend([f"hog_{i}" for i in range(3780)])  # Approximate HOG size
        
        if self.config.enable_lbp:
            names.extend([f"lbp_{i}" for i in range(self.config.lbp_n_points + 2)])
        
        if self.config.enable_color_hist:
            for channel in ['r', 'g', 'b']:
                names.extend([f"color_{channel}_{i}" for i in range(self.config.color_bins)])
        
        if self.config.enable_texture:
            names.extend(['texture_mean', 'texture_std', 'texture_min', 'texture_max',
                         'grad_x_mean', 'grad_y_mean', 'laplacian_mean', 'laplacian_std'])
        
        if self.config.enable_edge_features:
            names.extend(['edge_density', 'edge_mean', 'edge_std'])
            names.extend([f"edge_orient_{i}" for i in range(36)])
        
        return names[:self.feature_dim]  # Ensure correct length


class HybridFeatureExtractor(FeatureExtractor):
    """Hybrid feature extractor combining CNN and Classical features"""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        
        # Initialize sub-extractors
        self.cnn_extractor = None
        self.classical_extractor = None
        
        # Try to initialize CNN extractor
        if PYTORCH_AVAILABLE:
            try:
                self.cnn_extractor = CNNFeatureExtractor(config)
                self.logger.info("✅ CNN feature extractor initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ CNN extractor failed, falling back to classical only: {e}")
        else:
            self.logger.warning("⚠️ PyTorch not available, using classical features only")
        
        # Initialize classical extractor
        self.classical_extractor = ClassicalFeatureExtractor(config)
        
        # Calculate total feature dimension
        self.feature_dim = self._calculate_total_dimension()
        
        self.logger.info(f"🔗 Hybrid extractor initialized: {self.feature_dim} total features")
    
    def _calculate_total_dimension(self) -> int:
        """Calculate total feature dimension"""
        total_dim = 0
        
        if self.cnn_extractor:
            total_dim += self.cnn_extractor.get_feature_dimension()
        
        if self.classical_extractor:
            total_dim += self.classical_extractor.get_feature_dimension()
        
        return total_dim
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract hybrid features from image"""
        # Check cache first
        cached_features = self._load_from_cache(image)
        if cached_features is not None:
            return cached_features
        
        feature_parts = []
        
        # Extract CNN features
        if self.cnn_extractor:
            try:
                cnn_features = self.cnn_extractor.extract_features(image)
                feature_parts.append(cnn_features)
            except Exception as e:
                self.logger.warning(f"CNN feature extraction failed: {e}")
                # Add zero vector as fallback
                feature_parts.append(np.zeros(self.cnn_extractor.get_feature_dimension()))
        
        # Extract classical features
        if self.classical_extractor:
            try:
                classical_features = self.classical_extractor.extract_features(image)
                feature_parts.append(classical_features)
            except Exception as e:
                self.logger.warning(f"Classical feature extraction failed: {e}")
                # Add zero vector as fallback
                feature_parts.append(np.zeros(self.classical_extractor.get_feature_dimension()))
        
        # Combine features
        if feature_parts:
            combined_features = np.concatenate(feature_parts)
        else:
            combined_features = np.zeros(self.feature_dim, dtype=np.float32)
        
        # Cache combined features
        self._save_to_cache(image, combined_features)
        
        return combined_features
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Optimized batch processing for hybrid features"""
        batch_features = []
        
        # Extract CNN features in batch (if available)
        cnn_batch_features = None
        if self.cnn_extractor:
            try:
                cnn_batch_features = self.cnn_extractor.extract_batch(images)
            except Exception as e:
                self.logger.warning(f"Batch CNN extraction failed: {e}")
                cnn_batch_features = np.zeros((len(images), self.cnn_extractor.get_feature_dimension()))
        
        # Extract classical features in batch
        classical_batch_features = None
        if self.classical_extractor:
            try:
                classical_batch_features = self.classical_extractor.extract_batch(images)
            except Exception as e:
                self.logger.warning(f"Batch classical extraction failed: {e}")
                classical_batch_features = np.zeros((len(images), self.classical_extractor.get_feature_dimension()))
        
        # Combine features
        for i in range(len(images)):
            feature_parts = []
            
            if cnn_batch_features is not None:
                feature_parts.append(cnn_batch_features[i])
            
            if classical_batch_features is not None:
                feature_parts.append(classical_batch_features[i])
            
            if feature_parts:
                combined = np.concatenate(feature_parts)
            else:
                combined = np.zeros(self.feature_dim, dtype=np.float32)
            
            batch_features.append(combined)
        
        return np.array(batch_features)
    
    def get_feature_dimension(self) -> int:
        return self.feature_dim
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features"""
        names = []
        
        if self.cnn_extractor:
            names.extend(self.cnn_extractor.get_feature_names())
        
        if self.classical_extractor:
            names.extend(self.classical_extractor.get_feature_names())
        
        return names


def create_optimal_config() -> FeatureConfig:
    """Create optimized configuration for amulet recognition"""
    return FeatureConfig(
        # CNN settings (CPU-optimized)
        cnn_backbone="resnet18",  # Lighter model for CPU
        cnn_pretrained=True,
        cnn_freeze_weights=True,
        cnn_device="cpu",
        cnn_batch_size=4,  # Small batch for memory efficiency
        
        # Classical features (all enabled for completeness)
        enable_hog=True,
        enable_lbp=True,
        enable_color_hist=True,
        enable_texture=True,
        enable_edge_features=True,
        
        # Preprocessing
        target_size=(224, 224),
        normalize_method="imagenet",
        
        # Caching for performance
        enable_caching=True,
        cache_dir="ai_models/feature_cache",
        
        # Processing
        use_multiprocessing=False,  # Disabled for stability
        verbose=True
    )


def main():
    """Test feature extraction system"""
    print("🎯 Feature Extraction System Test")
    print("=" * 50)
    
    # Create test configuration
    config = create_optimal_config()
    
    # Test CNN extractor (if available)
    if PYTORCH_AVAILABLE:
        print("\n🧠 Testing CNN Feature Extractor...")
        try:
            cnn_extractor = CNNFeatureExtractor(config)
            print(f"✅ CNN extractor ready, dimension: {cnn_extractor.get_feature_dimension()}")
        except Exception as e:
            print(f"❌ CNN extractor failed: {e}")
    else:
        print("⚠️ PyTorch not available, skipping CNN test")
    
    # Test classical extractor
    print("\n🔧 Testing Classical Feature Extractor...")
    try:
        classical_extractor = ClassicalFeatureExtractor(config)
        print(f"✅ Classical extractor ready, dimension: {classical_extractor.get_feature_dimension()}")
    except Exception as e:
        print(f"❌ Classical extractor failed: {e}")
    
    # Test hybrid extractor
    print("\n🔗 Testing Hybrid Feature Extractor...")
    try:
        hybrid_extractor = HybridFeatureExtractor(config)
        print(f"✅ Hybrid extractor ready, dimension: {hybrid_extractor.get_feature_dimension()}")
        
        # Test with dummy image
        print("\n📸 Testing with dummy image...")
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        features = hybrid_extractor.extract_features(dummy_image)
        print(f"✅ Extracted {len(features)} features")
        
    except Exception as e:
        print(f"❌ Hybrid extractor failed: {e}")
    
    print("\n🎉 Feature extraction system test completed!")


if __name__ == "__main__":
    main()