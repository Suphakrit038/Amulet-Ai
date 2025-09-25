"""
🔧 Compatible Data Pipeline for Python 3.13
ระบบการประมวลผลข้อมูลที่ทำงานได้จริงกับ Python 3.13
แทนที่ PyTorch/TensorFlow ด้วย OpenCV + scikit-learn
"""
import os
import numpy as np
import cv2
from PIL import Image
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import concurrent.futures
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class CompatibleDataPipelineConfig:
    """Configuration for compatible data pipeline"""
    data_path: str = "ai_models/dataset_split"
    batch_size: int = 16
    image_size: Tuple[int, int] = (224, 224)
    use_augmentation: bool = True
    output_dir: str = "ai_models/compatible_output"
    
    # Feature extraction parameters
    use_hog: bool = True
    use_lbp: bool = True  
    use_color_histogram: bool = True
    use_edge_features: bool = True
    
    # Processing parameters
    max_workers: int = 4
    cache_features: bool = True

class CompatibleDataPipeline:
    """
    🚀 Data Pipeline ที่ทำงานได้จริงกับ Python 3.13
    ใช้ OpenCV + scikit-learn แทนที่ PyTorch/TensorFlow
    """
    
    def __init__(self, config: CompatibleDataPipelineConfig):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_cache = {}
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        logger.info(f"🔧 Compatible Data Pipeline initialized")
        logger.info(f"📁 Data path: {config.data_path}")
        logger.info(f"🖼️ Image size: {config.image_size}")
    
    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG features from image"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # HOG parameters
        hog = cv2.HOGDescriptor()
        features = hog.compute(gray)
        return features.flatten()
    
    def extract_lbp_features(self, image: np.ndarray, radius: int = 3, n_points: int = 24) -> np.ndarray:
        """Extract Local Binary Pattern features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Simple LBP implementation
        lbp = np.zeros_like(gray)
        for i in range(radius, gray.shape[0] - radius):
            for j in range(radius, gray.shape[1] - radius):
                center = gray[i, j]
                pattern = 0
                for k in range(8):  # 8-neighbor
                    y = i + int(radius * np.sin(2 * np.pi * k / 8))
                    x = j + int(radius * np.cos(2 * np.pi * k / 8))
                    if gray[y, x] >= center:
                        pattern |= (1 << k)
                lbp[i, j] = pattern
        
        # Histogram of LBP
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        return hist.astype(np.float32)
    
    def extract_color_histogram(self, image: np.ndarray, bins: int = 32) -> np.ndarray:
        """Extract color histogram features"""
        if len(image.shape) == 3:
            # RGB histograms
            hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
            hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])
            return np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
        else:
            # Grayscale histogram
            hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
            return hist.flatten()
    
    def extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """Extract edge-based features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Edge statistics
        edge_density = np.sum(edges > 0) / edges.size
        edge_mean = np.mean(edges)
        edge_std = np.std(edges)
        
        # Edge direction histogram
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        angles = np.arctan2(sobely, sobelx)
        hist, _ = np.histogram(angles.ravel(), bins=36, range=(-np.pi, np.pi))
        
        return np.concatenate([[edge_density, edge_mean, edge_std], hist.astype(np.float32)])
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract comprehensive features from image"""
        # Check cache first
        if self.config.cache_features and image_path in self.feature_cache:
            return self.feature_cache[image_path]
        
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"⚠️ Could not load image: {image_path}")
                return np.array([])
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.config.image_size)
            
            features = []
            
            # Extract different types of features
            if self.config.use_hog:
                hog_features = self.extract_hog_features(image)
                features.append(hog_features)
            
            if self.config.use_lbp:
                lbp_features = self.extract_lbp_features(image)
                features.append(lbp_features)
            
            if self.config.use_color_histogram:
                color_features = self.extract_color_histogram(image)
                features.append(color_features)
            
            if self.config.use_edge_features:
                edge_features = self.extract_edge_features(image)
                features.append(edge_features)
            
            # Combine all features
            combined_features = np.concatenate(features) if features else np.array([])
            
            # Cache features
            if self.config.cache_features:
                self.feature_cache[image_path] = combined_features
            
            return combined_features
            
        except Exception as e:
            logger.error(f"❌ Error extracting features from {image_path}: {e}")
            return np.array([])
    
    def load_dataset(self) -> Tuple[List[str], List[str], List[str]]:
        """Load dataset paths and labels"""
        train_paths = []
        val_paths = []
        train_labels = []
        val_labels = []
        
        # Load training data
        train_dir = Path(self.config.data_path) / "train"
        if train_dir.exists():
            for class_dir in train_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    for img_file in class_dir.glob("*"):
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                            train_paths.append(str(img_file))
                            train_labels.append(class_name)
        
        # Load validation data
        val_dir = Path(self.config.data_path) / "validation"
        if val_dir.exists():
            for class_dir in val_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    for img_file in class_dir.glob("*"):
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                            val_paths.append(str(img_file))
                            val_labels.append(class_name)
        
        logger.info(f"📊 Loaded {len(train_paths)} training images")
        logger.info(f"📊 Loaded {len(val_paths)} validation images")
        
        return train_paths, val_paths, train_labels, val_labels
    
    def process_images_batch(self, image_paths: List[str]) -> np.ndarray:
        """Process a batch of images in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            features_list = list(tqdm(
                executor.map(self.extract_features, image_paths),
                total=len(image_paths),
                desc="🖼️ Processing images"
            ))
        
        # Filter out empty features and pad if needed
        valid_features = [f for f in features_list if len(f) > 0]
        
        if not valid_features:
            logger.error("❌ No valid features extracted!")
            return np.array([])
        
        # Ensure all feature vectors have the same length
        max_length = max(len(f) for f in valid_features)
        padded_features = []
        
        for i, features in enumerate(features_list):
            if len(features) == 0:
                # Use mean features for missing data
                if padded_features:
                    padded_features.append(np.mean(padded_features, axis=0))
                else:
                    padded_features.append(np.zeros(max_length))
            elif len(features) < max_length:
                # Pad with zeros
                padded = np.pad(features, (0, max_length - len(features)), 'constant')
                padded_features.append(padded)
            else:
                padded_features.append(features[:max_length])
        
        return np.array(padded_features)
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and validation data"""
        logger.info("🚀 Starting data preparation...")
        
        # Load dataset
        train_paths, val_paths, train_labels, val_labels = self.load_dataset()
        
        if not train_paths:
            raise ValueError("❌ No training data found!")
        
        # Encode labels
        all_labels = train_labels + val_labels
        self.label_encoder.fit(all_labels)
        
        encoded_train_labels = self.label_encoder.transform(train_labels)
        encoded_val_labels = self.label_encoder.transform(val_labels) if val_labels else np.array([])
        
        # Extract features
        logger.info("🔧 Extracting training features...")
        X_train = self.process_images_batch(train_paths)
        
        X_val = np.array([])
        if val_paths:
            logger.info("🔧 Extracting validation features...")
            X_val = self.process_images_batch(val_paths)
        
        # Scale features
        if len(X_train) > 0:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val) if len(X_val) > 0 else np.array([])
        else:
            raise ValueError("❌ No features extracted from training data!")
        
        # Save preprocessing objects
        joblib.dump(self.label_encoder, os.path.join(self.config.output_dir, 'label_encoder.joblib'))
        joblib.dump(self.scaler, os.path.join(self.config.output_dir, 'scaler.joblib'))
        
        logger.info(f"✅ Data preparation completed!")
        logger.info(f"📊 Training shape: {X_train_scaled.shape}")
        logger.info(f"📊 Validation shape: {X_val_scaled.shape}")
        logger.info(f"🏷️ Number of classes: {len(self.label_encoder.classes_)}")
        
        return X_train_scaled, X_val_scaled, encoded_train_labels, encoded_val_labels, train_paths, val_paths
    
    def save_dataset(self, X_train: np.ndarray, X_val: np.ndarray, 
                    y_train: np.ndarray, y_val: np.ndarray):
        """Save processed dataset"""
        np.save(os.path.join(self.config.output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(self.config.output_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(self.config.output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(self.config.output_dir, 'y_val.npy'), y_val)
        
        logger.info(f"💾 Dataset saved to {self.config.output_dir}")

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = CompatibleDataPipelineConfig()
    pipeline = CompatibleDataPipeline(config)
    
    try:
        X_train, X_val, y_train, y_val, train_paths, val_paths = pipeline.prepare_data()
        pipeline.save_dataset(X_train, X_val, y_train, y_val)
        print("✅ Compatible data pipeline completed successfully!")
    except Exception as e:
        print(f"❌ Error in data pipeline: {e}")