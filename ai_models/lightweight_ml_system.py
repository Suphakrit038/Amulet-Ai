"""
Lightweight Machine Learning System for Amulet-AI
ระบบ ML ที่เบาและมีประสิทธิภาพสำหรับการจดจำพระเครื่อง
ใช้ scikit-learn + OpenCV แทน PyTorch/TensorFlow
"""
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class LightweightMLConfig:
    """Configuration for lightweight ML system"""
    # Data settings
    data_path: str = "ai_models/dataset_split"
    image_size: Tuple[int, int] = (224, 224)
    
    # Feature extraction
    use_hog: bool = True
    use_lbp: bool = True
    use_color_histogram: bool = True
    use_hu_moments: bool = True
    
    # Model settings
    model_type: str = "random_forest"  # random_forest, gradient_boost, svm
    n_estimators: int = 100
    max_depth: int = 10
    random_state: int = 42
    
    # PCA settings  
    use_pca: bool = True
    pca_components: int = 100
    
    # Output settings
    output_dir: str = "ai_models/lightweight_output"
    model_filename: str = "lightweight_amulet_model.joblib"

class FeatureExtractor:
    """Extract various image features for traditional ML"""
    
    def __init__(self, config: LightweightMLConfig):
        self.config = config
        
    def extract_hog_features(self, image):
        """Extract HOG (Histogram of Oriented Gradients) features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # HOG parameters
        hog = cv2.HOGDescriptor(
            _winSize=(64, 64),
            _blockSize=(16, 16), 
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
        
        # Resize image to fit HOG window
        resized = cv2.resize(gray, (64, 64))
        features = hog.compute(resized)
        
        return features.flatten() if features is not None else np.zeros(1764)
    
    def extract_lbp_features(self, image):
        """Extract LBP (Local Binary Pattern) features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Simple LBP implementation
        radius = 3
        n_points = 8 * radius
        
        # Resize for consistency
        resized = cv2.resize(gray, self.config.image_size)
        
        # Create LBP histogram
        hist, _ = np.histogram(resized.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)  # Normalize
        
        return hist
    
    def extract_color_histogram(self, image):
        """Extract color histogram features"""
        if len(image.shape) == 3:
            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms for each channel
            hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])
            
            # Concatenate and normalize
            hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-7)
            
            return hist
        else:
            # Grayscale histogram
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist = hist.flatten().astype(np.float32)
            hist /= (hist.sum() + 1e-7)
            return hist
    
    def extract_hu_moments(self, image):
        """Extract Hu moments for shape description"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Calculate moments
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments)
        
        # Log transform for better numerical stability
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        return hu_moments.flatten()
    
    def extract_all_features(self, image):
        """Extract all configured features from an image"""
        features = []
        
        if self.config.use_hog:
            hog_feat = self.extract_hog_features(image)
            features.append(hog_feat)
            
        if self.config.use_lbp:
            lbp_feat = self.extract_lbp_features(image)
            features.append(lbp_feat)
            
        if self.config.use_color_histogram:
            color_feat = self.extract_color_histogram(image)
            features.append(color_feat)
            
        if self.config.use_hu_moments:
            hu_feat = self.extract_hu_moments(image)
            features.append(hu_feat)
        
        # Concatenate all features
        if features:
            return np.concatenate(features)
        else:
            return np.array([])

class LightweightAmuletClassifier:
    """Lightweight classifier for amulet recognition"""
    
    def __init__(self, config: LightweightMLConfig):
        self.config = config
        self.feature_extractor = FeatureExtractor(config)
        self.model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=config.pca_components) if config.use_pca else None
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
        # Setup output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def _create_model(self):
        """Create the ML model based on configuration"""
        if self.config.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        elif self.config.model_type == "gradient_boost":
            return GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state
            )
        elif self.config.model_type == "svm":
            return SVC(
                kernel='rbf',
                random_state=self.config.random_state,
                probability=True  # Enable probability estimates
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def load_dataset(self):
        """Load dataset from directory structure"""
        data_path = Path(self.config.data_path)
        
        X, y, class_names = [], [], []
        
        # Load training data
        train_path = data_path / "train"
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        # Get class directories
        class_dirs = [d for d in train_path.iterdir() if d.is_dir()]
        class_dirs.sort()
        self.class_names = [d.name for d in class_dirs]
        
        logger.info(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        # Load images and extract features
        for class_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            logger.info(f"Processing class: {class_name}")
            
            # Get all image files
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            for img_path in image_files:
                try:
                    # Load image
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue
                        
                    # Resize image
                    image = cv2.resize(image, self.config.image_size)
                    
                    # Extract features
                    features = self.feature_extractor.extract_all_features(image)
                    
                    if len(features) > 0:
                        X.append(features)
                        y.append(class_idx)
                        
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features each")
        return X, y
    
    def train(self):
        """Train the lightweight model"""
        logger.info("Starting lightweight model training...")
        start_time = time.time()
        
        # Load data
        X, y = self.load_dataset()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.config.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply PCA if configured
        if self.pca:
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            X_test_scaled = self.pca.transform(X_test_scaled)
            logger.info(f"Applied PCA: {X.shape[1]} -> {X_train_scaled.shape[1]} features")
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test_scaled)
        
        training_time = time.time() - start_time
        
        # Create results
        results = {
            "model_type": self.config.model_type,
            "training_time": training_time,
            "train_accuracy": train_score,
            "test_accuracy": test_score,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "num_features": X_train_scaled.shape[1],
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Train accuracy: {train_score:.4f}")
        logger.info(f"Test accuracy: {test_score:.4f}")
        
        # Save model and results
        self.save_model()
        self.save_results(results)
        
        return results
    
    def save_model(self):
        """Save the trained model"""
        model_path = Path(self.config.output_dir) / self.config.model_filename
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'class_names': self.class_names,
            'config': self.config
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to: {model_path}")
    
    def save_results(self, results):
        """Save training results"""
        results_path = Path(self.config.output_dir) / "training_results.json"
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {results_path}")

def create_lightweight_classifier(config: Optional[LightweightMLConfig] = None) -> LightweightAmuletClassifier:
    """Create lightweight classifier with configuration"""
    if config is None:
        config = LightweightMLConfig()
    
    return LightweightAmuletClassifier(config)

def main():
    """Main training function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and train classifier
    classifier = create_lightweight_classifier()
    
    try:
        results = classifier.train()
        
        print("\n" + "="*50)
        print("🎯 LIGHTWEIGHT TRAINING COMPLETED!")
        print("="*50)
        print(f"Model: {results['model_type']}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        print(f"Test accuracy: {results['test_accuracy']:.4f}")
        print(f"Classes: {len(results['class_names'])}")
        print(f"Features: {results['num_features']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()