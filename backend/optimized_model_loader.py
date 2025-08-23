"""
Optimized AI Model Loader with Advanced Features
จัดการโมเดล AI และการทำนายแบบ optimized
"""
import json
import logging
import numpy as np
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import random
from PIL import Image, ImageEnhance, ImageFilter
import time

# Import configuration
from .config import model_config, get_config

# Setup logging
logger = logging.getLogger(__name__)

class OptimizedModelLoader:
    """
    Optimized AI Model Loader with caching and advanced features
    """
    
    def __init__(self):
        """Initialize the optimized model loader"""
        self.config = get_config()
        self.model = None
        self.labels = self._load_labels()
        self.cache = {}  # Simple in-memory cache
        self.stats = {
            "predictions_count": 0,
            "cache_hits": 0,
            "avg_processing_time": 0.0
        }
        
        # Advanced simulation features
        self.use_advanced_simulation = model_config.use_advanced_simulation
        
        logger.info(f"🚀 OptimizedModelLoader initialized")
        logger.info(f"📊 Classes loaded: {len(self.labels)}")
        logger.info(f"🤖 AI Mode: {'Advanced Simulation' if self.use_advanced_simulation else 'Simple Mock'}")
        
        if self.use_advanced_simulation:
            self._initialize_advanced_features()
    
    def _load_labels(self) -> Dict[str, str]:
        """Load class labels with fallback"""
        try:
            labels_path = Path(model_config.labels_file)
            if labels_path.exists():
                with open(labels_path, "r", encoding="utf-8") as f:
                    labels = json.load(f)
                logger.info(f"✅ Labels loaded from {labels_path}")
                return labels
        except Exception as e:
            logger.warning(f"Failed to load labels: {e}")
        
        # Default labels
        default_labels = {
            "0": "หลวงพ่อกวยแหวกม่าน",
            "1": "โพธิ์ฐานบัว",
            "2": "ฐานสิงห์", 
            "3": "สีวลี"
        }
        logger.info("📋 Using default labels")
        return default_labels
    
    def _initialize_advanced_features(self):
        """Initialize advanced AI simulation features"""
        try:
            # Color analysis patterns
            self.color_patterns = {
                "หลวงพ่อกวยแหวกม่าน": {
                    "preferred_colors": [(139, 69, 19), (184, 134, 11)],  # Brown, Gold
                    "brightness_range": (0.4, 0.7),
                    "contrast_preference": 0.6
                },
                "โพธิ์ฐานบัว": {
                    "preferred_colors": [(160, 82, 45), (205, 133, 63)],  # Saddle brown
                    "brightness_range": (0.3, 0.6),
                    "contrast_preference": 0.5
                },
                "ฐานสิงห์": {
                    "preferred_colors": [(101, 67, 33), (139, 69, 19)],   # Dark brown
                    "brightness_range": (0.2, 0.5),
                    "contrast_preference": 0.7
                },
                "สีวลี": {
                    "preferred_colors": [(160, 82, 45), (222, 184, 135)], # Burlywood
                    "brightness_range": (0.4, 0.8),
                    "contrast_preference": 0.4
                }
            }
            
            # Feature extractors
            self.feature_weights = {
                "color_similarity": 0.35,
                "brightness_match": 0.25,
                "contrast_level": 0.20,
                "texture_complexity": 0.15,
                "edge_density": 0.05
            }
            
            logger.info("🎨 Advanced AI simulation features initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced features: {e}")
            self.use_advanced_simulation = False
    
    def _generate_cache_key(self, image_data: bytes) -> str:
        """Generate cache key for image"""
        return f"img_{hash(image_data) % 1000000}"
    
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
    
    def _extract_comprehensive_features(self, img_array: np.ndarray) -> Dict:
        """Extract comprehensive image features"""
        features = {}
        
        # Color analysis
        avg_color = np.mean(img_array, axis=(0, 1))
        color_std = np.std(img_array, axis=(0, 1))
        
        features.update({
            "avg_color": avg_color.tolist(),
            "color_std": color_std.tolist(),
            "brightness": np.mean(avg_color) / 255.0,
            "contrast": np.mean(color_std) / 255.0,
            "red_ratio": avg_color[0] / 255.0,
            "green_ratio": avg_color[1] / 255.0,
            "blue_ratio": avg_color[2] / 255.0,
            "golden_tone": (avg_color[0] + avg_color[1]) / (2 * 255.0)
        })
        
        # Texture analysis
        gray = np.mean(img_array, axis=2)
        
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

def get_model_loader() -> OptimizedModelLoader:
    """Get singleton model loader instance"""
    global _model_loader_instance
    if _model_loader_instance is None:
        _model_loader_instance = OptimizedModelLoader()
    return _model_loader_instance
