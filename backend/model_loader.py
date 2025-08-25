"""
🏺 Amulet-AI Advanced Model Loader
AI Model Loader with Advanced Features and Performance Optimization
จัดการโมเดล AI และการทำนายแบบ optimized พร้อมระบบ caching และ feature extraction
"""
import json
import logging
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import random
import time

# Enhanced imports with comprehensive fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: numpy not available. Install with: pip install numpy")
    # Mock numpy for basic functionality
    class MockNumpy:
        @staticmethod
        def mean(data, axis=None): return sum(data) / len(data) if hasattr(data, '__iter__') else data
        @staticmethod
        def std(data, axis=None): return 1.0
        @staticmethod
        def sqrt(data): return data ** 0.5 if isinstance(data, (int, float)) else data
        @staticmethod
        def array(data): return data
    np = MockNumpy()
    NUMPY_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: PIL not available. Install with: pip install Pillow")
    # Mock PIL for basic functionality
    class MockImage:
        @staticmethod
        def open(data): return MockImageObject()
        class Resampling:
            LANCZOS = "lanczos"
    
    class MockImageObject:
        def convert(self, mode): return self
        def resize(self, size, resample=None): return self
        @property
        def mode(self): return "RGB"
    
    Image = MockImage()
    ImageEnhance = None
    ImageFilter = None
    PIL_AVAILABLE = False

# Enhanced imports with fallbacks
try:
    from .config import model_config, get_config
except ImportError:
    # Fallback configuration
    class MockConfig:
        debug = False
        use_advanced_simulation = True
        labels_file = "labels.json"
    
    model_config = MockConfig()
    get_config = lambda: MockConfig()

# Logging with fallbacks
try:
    from ..development.utils.logger import get_logger, performance_monitor
    logger = get_logger("model_loader")
except ImportError:
    logger = logging.getLogger(__name__)
    performance_monitor = lambda name: lambda func: func  # No-op decorator

class AmuletModelLoader:
    """
    🚀 Advanced AI Model Loader with caching, performance tracking, and intelligent analysis
    """
    
    def __init__(self):
        """Initialize the advanced model loader with comprehensive features"""
        self.config = get_config()
        self.model = None
        self.labels = self._load_labels()
        self.cache = {}  # In-memory prediction cache
        self.feature_cache = {}  # Feature extraction cache
        
        # Performance statistics
        self.stats = {
            "predictions_count": 0,
            "cache_hits": 0,
            "avg_processing_time": 0.0,
            "total_processing_time": 0.0,
            "errors_count": 0,
            "successful_predictions": 0
        }
        
        # AI simulation configuration
        self.use_advanced_simulation = model_config.use_advanced_simulation
        
        logger.info(f"🚀 AmuletModelLoader initialized successfully")
        logger.info(f"📊 Classes loaded: {len(self.labels)}")
        logger.info(f"🤖 AI Mode: {'Advanced Simulation' if self.use_advanced_simulation else 'Simple Mock'}")
        
        if self.use_advanced_simulation:
            self._initialize_advanced_features()

    @performance_monitor("load_labels")
    def _load_labels(self) -> Dict[str, str]:
        """Load class labels with comprehensive fallback system"""
        try:
            # Try multiple label file locations
            possible_paths = [
                Path(model_config.labels_file),
                Path("labels.json"),
                Path(__file__).parent.parent / "labels.json",
                Path(__file__).parent / "labels.json"
            ]
            
            for labels_path in possible_paths:
                if labels_path.exists():
                    with open(labels_path, "r", encoding="utf-8") as f:
                        labels = json.load(f)
                    logger.info(f"✅ Labels loaded from {labels_path}")
                    return labels
                    
        except Exception as e:
            logger.warning(f"Failed to load labels from file: {e}")
        
        # Enhanced default labels with Thai Buddhist amulet categories
        default_labels = {
            "0": "หลวงพ่อกวยแหวกม่าน",
            "1": "โพธิ์ฐานบัว", 
            "2": "ฐานสิงห์",
            "3": "สีวลี",
            "4": "พระสมเด็จ",
            "5": "พระพิมพ์ปรก",
            "6": "พระนาคปรก"
        }
        
        logger.info("📋 Using enhanced default labels (7 classes)")
        return default_labels

    @performance_monitor("initialize_advanced_features")
    def _initialize_advanced_features(self):
        """Initialize advanced AI simulation features with detailed pattern recognition"""
        try:
            # Comprehensive color and pattern analysis for Thai amulets
            self.amulet_patterns = {
                "หลวงพ่อกวยแหวกม่าน": {
                    "color_profile": {
                        "primary_colors": [(139, 69, 19), (184, 134, 11), (160, 82, 45)],  # Browns, Gold
                        "brightness_range": (0.3, 0.7),
                        "contrast_preference": 0.6,
                        "golden_ratio": 0.4
                    },
                    "texture_features": {
                        "roughness": 0.7,
                        "detail_level": 0.8,
                        "edge_complexity": 0.6
                    },
                    "shape_characteristics": {
                        "roundness": 0.4,
                        "symmetry": 0.7,
                        "aspect_ratio_preference": 1.2
                    }
                },
                "โพธิ์ฐานบัว": {
                    "color_profile": {
                        "primary_colors": [(160, 82, 45), (205, 133, 63), (101, 67, 33)],
                        "brightness_range": (0.3, 0.6),
                        "contrast_preference": 0.5,
                        "golden_ratio": 0.3
                    },
                    "texture_features": {
                        "roughness": 0.6,
                        "detail_level": 0.7,
                        "edge_complexity": 0.5
                    },
                    "shape_characteristics": {
                        "roundness": 0.6,
                        "symmetry": 0.8,
                        "aspect_ratio_preference": 1.0
                    }
                },
                "ฐานสิงห์": {
                    "color_profile": {
                        "primary_colors": [(101, 67, 33), (139, 69, 19), (92, 51, 23)],
                        "brightness_range": (0.2, 0.5),
                        "contrast_preference": 0.7,
                        "golden_ratio": 0.2
                    },
                    "texture_features": {
                        "roughness": 0.8,
                        "detail_level": 0.9,
                        "edge_complexity": 0.8
                    },
                    "shape_characteristics": {
                        "roundness": 0.3,
                        "symmetry": 0.6,
                        "aspect_ratio_preference": 1.4
                    }
                },
                "สีวลี": {
                    "color_profile": {
                        "primary_colors": [(160, 82, 45), (222, 184, 135), (139, 69, 19)],
                        "brightness_range": (0.4, 0.8),
                        "contrast_preference": 0.4,
                        "golden_ratio": 0.5
                    },
                    "texture_features": {
                        "roughness": 0.5,
                        "detail_level": 0.6,
                        "edge_complexity": 0.4
                    },
                    "shape_characteristics": {
                        "roundness": 0.7,
                        "symmetry": 0.5,
                        "aspect_ratio_preference": 0.9
                    }
                }
            }
            
            # Advanced feature weights for intelligent analysis
            self.analysis_weights = {
                "color_similarity": 0.30,
                "brightness_match": 0.20,
                "contrast_level": 0.15,
                "texture_complexity": 0.15,
                "golden_tone_analysis": 0.10,
                "edge_density": 0.05,
                "shape_features": 0.05
            }
            
            logger.info("🎨 Advanced AI simulation features initialized with 7 analysis components")
            logger.info(f"📊 Pattern database: {len(self.amulet_patterns)} amulet types")
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced features: {e}")
            self.use_advanced_simulation = False
            
    def _generate_cache_key(self, image_data: bytes, analysis_type: str = "prediction") -> str:
        """Generate optimized cache key for different analysis types"""
        data_hash = hash(image_data) % 1000000
        return f"{analysis_type}_{data_hash}_{len(image_data)}"

    @performance_monitor("analyze_image_advanced")
    def _analyze_image_advanced(self, image_bytes: bytes) -> Dict:
        """
        Advanced image analysis with multiple feature extraction techniques
        """
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for consistent analysis
            analysis_size = (128, 128)
            image_resized = image.resize(analysis_size, Image.Resampling.LANCZOS)
            img_array = np.array(image_resized)
            
            # Extract comprehensive features
            features = self._extract_comprehensive_features(img_array)
            
            processing_time = time.time() - start_time
            logger.debug(f"🔍 Image analysis completed in {processing_time:.3f}s")
            
            return features
            
        except Exception as e:
            logger.error(f"Advanced image analysis failed: {e}")
            return self._get_fallback_features()
    
    def _extract_comprehensive_features(self, img_array) -> Dict:
        """Extract comprehensive image features from image array"""
        features = {}
        
        # Color analysis with fallback for mock numpy
        if NUMPY_AVAILABLE:
            avg_color = np.mean(img_array, axis=(0, 1))
            color_std = np.std(img_array, axis=(0, 1))
        else:
            # Mock analysis for non-numpy environment
            avg_color = [127, 127, 127]  # Neutral gray
            color_std = [50, 50, 50]     # Standard deviation
        
        features.update({
            "avg_color": avg_color.tolist() if hasattr(avg_color, 'tolist') else avg_color,
            "color_std": color_std.tolist() if hasattr(color_std, 'tolist') else color_std,
            "brightness": (avg_color[0] + avg_color[1] + avg_color[2]) / (3 * 255.0),
            "contrast": (color_std[0] + color_std[1] + color_std[2]) / (3 * 255.0),
            "red_ratio": avg_color[0] / 255.0,
            "green_ratio": avg_color[1] / 255.0,
            "blue_ratio": avg_color[2] / 255.0,
            "golden_tone": (avg_color[0] + avg_color[1]) / (2 * 255.0)
        })
        
        # Texture analysis with fallback
        if NUMPY_AVAILABLE and hasattr(img_array, 'shape') and len(img_array.shape) >= 2:
            try:
                gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
                
                # Edge detection (simplified Sobel)
                sobel_x = np.abs(np.diff(gray, axis=1))
                sobel_y = np.abs(np.diff(gray, axis=0))
                edge_density = (np.mean(sobel_x) + np.mean(sobel_y)) / 2
                
                # Texture complexity
                texture_variance = np.var(gray)
                
                features.update({
                    "edge_density": edge_density / 255.0,
                    "texture_complexity": texture_variance / 65025.0,  # 255^2
                    "darkness_level": 1.0 - features["brightness"]
                })
            except Exception:
                # Fallback texture features
                features.update({
                    "edge_density": 0.3,
                    "texture_complexity": 0.4,
                    "darkness_level": 0.5
                })
        else:
            # Mock texture analysis for non-numpy environment
            features.update({
                "edge_density": 0.3,
                "texture_complexity": 0.4,
                "darkness_level": 1.0 - features["brightness"]
            })
        
        return features
    
    def _get_fallback_features(self) -> Dict:
        """Get fallback features when analysis fails"""
        return {
            "brightness": 0.5,
            "contrast": 0.3,
            "golden_tone": 0.4,
            "edge_density": 0.3,
            "texture_complexity": 0.4,
            "avg_color": [127, 127, 127],
            "color_std": [50, 50, 50]
        }
    
    def _calculate_class_scores(self, features: Dict) -> Dict[str, float]:
        """Calculate prediction scores for each class"""
        scores = {}
        
        for class_id, class_name in self.labels.items():
            if class_name in self.color_patterns:
                pattern = self.color_patterns[class_name]
                score = self._calculate_similarity_score(features, pattern)
            else:
                # Fallback scoring
                score = random.uniform(0.1, 0.4)
            
            scores[class_id] = max(0.05, min(0.95, score))
        
        return scores
    
    def _calculate_similarity_score(self, features: Dict, pattern: Dict) -> float:
        """Calculate similarity score based on pattern matching"""
        score = 0.0
        
        # Color similarity
        if "avg_color" in features and "preferred_colors" in pattern:
            color_score = self._calculate_color_similarity(
                features["avg_color"], pattern["preferred_colors"]
            )
            score += color_score * self.feature_weights["color_similarity"]
        
        # Brightness matching
        if "brightness" in features and "brightness_range" in pattern:
            brightness_score = self._calculate_range_score(
                features["brightness"], pattern["brightness_range"]
            )
            score += brightness_score * self.feature_weights["brightness_match"]
        
        # Contrast matching
        if "contrast" in features and "contrast_preference" in pattern:
            contrast_score = 1.0 - abs(features["contrast"] - pattern["contrast_preference"])
            score += max(0, contrast_score) * self.feature_weights["contrast_level"]
        
        # Additional features
        score += features.get("texture_complexity", 0.3) * self.feature_weights["texture_complexity"]
        score += features.get("edge_density", 0.3) * self.feature_weights["edge_density"]
        
        # Add controlled randomness
        score += random.uniform(-0.05, 0.05)
        
        return score
    
    def _calculate_color_similarity(self, avg_color: List[float], preferred_colors: List[Tuple]) -> float:
        """Calculate color similarity score"""
        try:
            if len(avg_color) != 3:
                return 0.3
            
            similarities = []
            for preferred_rgb in preferred_colors:
                # Calculate Euclidean distance in RGB space
                distance = np.sqrt(sum([(avg_color[i] - preferred_rgb[i]) ** 2 for i in range(3)]))
                # Convert to similarity (0-1 scale)
                similarity = max(0, 1.0 - (distance / 441.67))  # 441.67 ≈ sqrt(3*255^2)
                similarities.append(similarity)
            
            return max(similarities) if similarities else 0.3
            
        except Exception as e:
            logger.debug(f"Color similarity calculation failed: {e}")
            return 0.3
    
    def _calculate_range_score(self, value: float, range_tuple: Tuple[float, float]) -> float:
        """Calculate score based on range matching"""
        min_val, max_val = range_tuple
        if min_val <= value <= max_val:
            return 1.0
        elif value < min_val:
            return max(0, 1.0 - (min_val - value) / min_val)
        else:
            return max(0, 1.0 - (value - max_val) / (1.0 - max_val))
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to sum to 1.0"""
        total_score = sum(scores.values())
        if total_score > 0:
            return {k: v / total_score for k, v in scores.items()}
        else:
            # Equal distribution fallback
            equal_score = 1.0 / len(scores)
            return {k: equal_score for k in scores.keys()}
    
    def predict(self, image_file) -> Dict:
        """
        Main prediction method with caching and optimization
        """
        start_time = time.time()
        
        try:
            # Read image data
            if hasattr(image_file, 'read'):
                image_file.seek(0)
                image_bytes = image_file.read()
            else:
                with open(image_file, 'rb') as f:
                    image_bytes = f.read()
            
            # Check cache
            cache_key = self._generate_cache_key(image_bytes)
            if cache_key in self.cache and not self.config.debug:
                self.stats["cache_hits"] += 1
                logger.debug(f"🎯 Cache hit for image {cache_key}")
                return self.cache[cache_key]
            
            # Perform analysis
            if self.use_advanced_simulation:
                features = self._analyze_image_advanced(image_bytes)
                scores = self._calculate_class_scores(features)
                scores = self._normalize_scores(scores)
                
                # Get top prediction
                sorted_classes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top_class_id, top_confidence = sorted_classes[0]
                top_class_name = self.labels[top_class_id]
                
                analysis_mode = "advanced_simulation"
                
            else:
                # Simple fallback
                top_class_name = random.choice(list(self.labels.values()))
                top_confidence = random.uniform(0.7, 0.9)
                analysis_mode = "simple_mock"
            
            # Prepare result
            result = {
                "class": top_class_name,
                "confidence": float(top_confidence),
                "analysis_mode": analysis_mode,
                "processing_time": time.time() - start_time
            }
            
            # Cache result
            self.cache[cache_key] = result
            
            # Update statistics
            self.stats["predictions_count"] += 1
            self.stats["avg_processing_time"] = (
                (self.stats["avg_processing_time"] * (self.stats["predictions_count"] - 1) + 
                 result["processing_time"]) / self.stats["predictions_count"]
            )
            
            logger.info(f"🔮 Prediction: {top_class_name} ({top_confidence:.3f}) in {result['processing_time']:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "class": list(self.labels.values())[0],
                "confidence": 0.5,
                "analysis_mode": "error_fallback",
                "error": str(e)
            }
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            **self.stats,
            "cache_size": len(self.cache),
            "cache_hit_rate": (
                self.stats["cache_hits"] / max(1, self.stats["predictions_count"])
            ),
            "labels_count": len(self.labels),
            "mode": "advanced_simulation" if self.use_advanced_simulation else "simple_mock"
        }
    
    def clear_cache(self) -> None:
        """Clear prediction cache"""
        cache_size = len(self.cache)
        self.cache.clear()
        logger.info(f"🧹 Cache cleared ({cache_size} items removed)")

# Global instance (singleton pattern)
_model_loader_instance = None

def get_model_loader() -> AmuletModelLoader:
    """Get singleton model loader instance"""
    global _model_loader_instance
    if _model_loader_instance is None:
        _model_loader_instance = AmuletModelLoader()
    return _model_loader_instance
