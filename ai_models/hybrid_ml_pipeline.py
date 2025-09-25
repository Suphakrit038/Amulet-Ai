"""
🧠 Optimal Hybrid ML Pipeline for Amulet Recognition
Combines CNN features (PyTorch) + Classical features (OpenCV) + Ensemble ML (scikit-learn)
Designed for Python 3.13 compatibility and small dataset optimization
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

@dataclass
class HybridPipelineConfig:
    """Optimized configuration for hybrid ML pipeline"""
    
    # Dataset settings
    data_path: str = "dataset"
    image_size: Tuple[int, int] = (224, 224)
    test_size: float = 0.2
    random_state: int = 42
    
    # CNN Feature Extraction
    use_cnn_features: bool = True
    cnn_backbone: str = "resnet50"  # resnet50, mobilenet_v2, efficientnet_b0
    cnn_freeze_layers: bool = True
    cnn_feature_dim: int = 2048  # ResNet50 output
    
    # Classical Feature Extraction  
    use_classical_features: bool = True
    use_hog: bool = True
    use_lbp: bool = True
    use_color_hist: bool = True
    use_edge_features: bool = True
    
    # Feature Processing
    use_pca: bool = True
    pca_components: int = 500  # Reduce from 1M+ to 500
    normalize_features: bool = True
    
    # Model Ensemble
    ensemble_models: List[str] = None
    ensemble_voting: str = "soft"
    
    # Training Strategy (for small dataset)
    use_stratified_kfold: bool = True
    cv_folds: int = 5
    use_class_weights: bool = True  # Handle imbalanced classes
    
    # Data Augmentation (critical for small dataset)
    use_augmentation: bool = True
    augmentation_factor: int = 5  # Multiply dataset by 5x
    
    # Performance
    n_jobs: int = -1  # Use all CPU cores
    verbose: int = 1
    
    def __post_init__(self):
        if self.ensemble_models is None:
            self.ensemble_models = ['random_forest', 'svm', 'gradient_boost']

class CNNFeatureExtractor:
    """CNN feature extractor using PyTorch pre-trained models"""
    
    def __init__(self, backbone: str = "resnet50", device: str = "cpu"):
        self.backbone = backbone
        self.device = torch.device(device)
        self.model = None
        self.transform = None
        self.feature_dim = 0
        self._setup_model()
    
    def _setup_model(self):
        """Setup pre-trained CNN model for feature extraction"""
        print(f"🔧 Loading {self.backbone} backbone...")
        
        if self.backbone == "resnet50":
            model = models.resnet50(weights='IMAGENET1K_V2')
            # Remove final FC layer
            self.model = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 2048
            
        elif self.backbone == "mobilenet_v2":
            model = models.mobilenet_v2(weights='IMAGENET1K_V2')
            # Remove classifier
            self.model = model.features
            self.model.add_module('global_pool', nn.AdaptiveAvgPool2d(1))
            self.feature_dim = 1280
            
        elif self.backbone == "efficientnet_b0":
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            # Remove classifier
            self.model = model.features
            self.model.add_module('global_pool', nn.AdaptiveAvgPool2d(1))
            self.feature_dim = 1280
        
        self.model.eval()
        self.model.to(self.device)
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ {self.backbone} loaded, feature dim: {self.feature_dim}")
    
    def extract_features_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract CNN features from batch of images"""
        batch_features = []
        
        with torch.no_grad():
            for image in tqdm(images, desc="🖼️ CNN feature extraction"):
                try:
                    # Preprocess
                    if len(image.shape) == 3:
                        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    else:
                        continue
                        
                    # Extract features
                    features = self.model(image_tensor)
                    features = features.squeeze().cpu().numpy()
                    
                    # Ensure correct shape
                    if features.ndim > 1:
                        features = features.flatten()
                        
                    batch_features.append(features)
                    
                except Exception as e:
                    print(f"⚠️ CNN extraction error: {e}")
                    # Use zero vector as fallback
                    batch_features.append(np.zeros(self.feature_dim))
        
        return np.array(batch_features)

class ClassicalFeatureExtractor:
    """Classical computer vision features using OpenCV"""
    
    def __init__(self, config: HybridPipelineConfig):
        self.config = config
        self.feature_dim = 0
        self._calculate_feature_dim()
    
    def _calculate_feature_dim(self):
        """Calculate total classical feature dimension"""
        dim = 0
        if self.config.use_hog:
            dim += 3780  # HOG features
        if self.config.use_lbp:
            dim += 256   # LBP histogram
        if self.config.use_color_hist:
            dim += 96    # Color histograms (32 bins × 3 channels)
        if self.config.use_edge_features:
            dim += 39    # Edge statistics + direction histogram
        
        self.feature_dim = dim
        print(f"📊 Classical features dimension: {self.feature_dim}")
    
    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Optimized HOG parameters for amulets
        hog = cv2.HOGDescriptor(
            _winSize=(64,128),
            _blockSize=(16,16), 
            _blockStride=(8,8),
            _cellSize=(8,8),
            _nbins=9
        )
        
        # Resize to standard size for HOG
        gray_resized = cv2.resize(gray, (64, 128))
        features = hog.compute(gray_resized)
        return features.flatten()
    
    def extract_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Local Binary Pattern features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Simple LBP implementation
        h, w = gray.shape
        lbp = np.zeros_like(gray)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray[i, j]
                code = 0
                
                # 8-connectivity
                neighbors = [
                    gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                    gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                    gray[i+1, j-1], gray[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                        
                lbp[i, j] = code
        
        # Histogram of LBP codes
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        return hist.astype(np.float32)
    
    def extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """Extract color histogram features"""
        if len(image.shape) == 3:
            # RGB histograms
            hist_r = cv2.calcHist([image], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
            hist_b = cv2.calcHist([image], [2], None, [32], [0, 256])
            return np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
        else:
            # Grayscale histogram
            hist = cv2.calcHist([image], [0], None, [32], [0, 256])
            return hist.flatten()
    
    def extract_edge_features(self, image: np.ndarray) -> np.ndarray:
        """Extract edge-based features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Canny edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Edge statistics
        edge_density = np.sum(edges > 0) / edges.size
        edge_mean = np.mean(edges)
        edge_std = np.std(edges)
        
        # Gradient directions
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        angles = np.arctan2(sobely, sobelx)
        
        # Direction histogram
        hist, _ = np.histogram(angles.ravel(), bins=36, range=(-np.pi, np.pi))
        
        return np.concatenate([[edge_density, edge_mean, edge_std], hist.astype(np.float32)])
    
    def extract_features_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract classical features from batch of images"""
        batch_features = []
        
        for image in tqdm(images, desc="🔧 Classical feature extraction"):
            try:
                features = []
                
                # Resize image to standard size
                image_resized = cv2.resize(image, self.config.image_size)
                
                if self.config.use_hog:
                    hog_feat = self.extract_hog_features(image_resized)
                    features.append(hog_feat)
                
                if self.config.use_lbp:
                    lbp_feat = self.extract_lbp_features(image_resized)
                    features.append(lbp_feat)
                
                if self.config.use_color_hist:
                    color_feat = self.extract_color_features(image_resized)
                    features.append(color_feat)
                
                if self.config.use_edge_features:
                    edge_feat = self.extract_edge_features(image_resized)
                    features.append(edge_feat)
                
                # Combine all features
                combined = np.concatenate(features) if features else np.array([])
                batch_features.append(combined)
                
            except Exception as e:
                print(f"⚠️ Classical extraction error: {e}")
                batch_features.append(np.zeros(self.feature_dim))
        
        return np.array(batch_features)

class DataAugmenter:
    """Data augmentation for small dataset"""
    
    def __init__(self, factor: int = 5):
        self.factor = factor
        
        # Augmentation transforms for amulets
        self.transforms = [
            self._rotate,
            self._brightness,
            self._contrast,
            self._noise,
            self._blur
        ]
    
    def _rotate(self, image: np.ndarray) -> np.ndarray:
        """Random rotation (-15 to +15 degrees)"""
        angle = np.random.uniform(-15, 15)
        h, w = image.shape[:2]
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    
    def _brightness(self, image: np.ndarray) -> np.ndarray:
        """Random brightness adjustment"""
        factor = np.random.uniform(0.8, 1.2)
        return np.clip(image * factor, 0, 255).astype(np.uint8)
    
    def _contrast(self, image: np.ndarray) -> np.ndarray:
        """Random contrast adjustment"""
        factor = np.random.uniform(0.8, 1.2)
        mean = np.mean(image)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    def _noise(self, image: np.ndarray) -> np.ndarray:
        """Add random noise"""
        noise = np.random.normal(0, 5, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)
    
    def _blur(self, image: np.ndarray) -> np.ndarray:
        """Random blur"""
        ksize = np.random.choice([3, 5])
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    
    def augment_dataset(self, images: List[np.ndarray], labels: List[int]) -> Tuple[List[np.ndarray], List[int]]:
        """Augment dataset by specified factor"""
        augmented_images = images.copy()
        augmented_labels = labels.copy()
        
        print(f"🔄 Augmenting dataset by {self.factor}x...")
        
        for _ in range(self.factor - 1):  # -1 because we already have original
            for img, label in tqdm(zip(images, labels), desc="Augmenting"):
                # Apply random transform
                transform = np.random.choice(self.transforms)
                aug_img = transform(img)
                augmented_images.append(aug_img)
                augmented_labels.append(label)
        
        print(f"✅ Dataset augmented: {len(images)} → {len(augmented_images)} images")
        return augmented_images, augmented_labels

class HybridMLPipeline:
    """Complete hybrid ML pipeline combining CNN + Classical features"""
    
    def __init__(self, config: HybridPipelineConfig):
        self.config = config
        self.cnn_extractor = None
        self.classical_extractor = None
        self.augmenter = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=config.pca_components) if config.use_pca else None
        self.label_encoder = LabelEncoder()
        self.ensemble_model = None
        
        # Setup components
        self._setup_extractors()
        self._setup_augmenter()
        self._setup_ensemble()
    
    def _setup_extractors(self):
        """Setup feature extractors"""
        if self.config.use_cnn_features:
            self.cnn_extractor = CNNFeatureExtractor(
                backbone=self.config.cnn_backbone,
                device="cpu"  # Force CPU for compatibility
            )
        
        if self.config.use_classical_features:
            self.classical_extractor = ClassicalFeatureExtractor(self.config)
    
    def _setup_augmenter(self):
        """Setup data augmenter"""
        if self.config.use_augmentation:
            self.augmenter = DataAugmenter(factor=self.config.augmentation_factor)
    
    def _setup_ensemble(self):
        """Setup ensemble classifier"""
        estimators = []
        
        if 'random_forest' in self.config.ensemble_models:
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                class_weight='balanced' if self.config.use_class_weights else None
            )
            estimators.append(('rf', rf))
        
        if 'svm' in self.config.ensemble_models:
            svm = SVC(
                kernel='rbf',
                probability=True,  # For soft voting
                random_state=self.config.random_state,
                class_weight='balanced' if self.config.use_class_weights else None
            )
            estimators.append(('svm', svm))
        
        if 'gradient_boost' in self.config.ensemble_models:
            gb = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=self.config.random_state
            )
            estimators.append(('gb', gb))
        
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting=self.config.ensemble_voting,
            n_jobs=self.config.n_jobs
        )
    
    def extract_features(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract hybrid features from images"""
        feature_groups = []
        
        # CNN features
        if self.config.use_cnn_features and self.cnn_extractor:
            print("🧠 Extracting CNN features...")
            cnn_features = self.cnn_extractor.extract_features_batch(images)
            feature_groups.append(cnn_features)
            print(f"✅ CNN features shape: {cnn_features.shape}")
        
        # Classical features  
        if self.config.use_classical_features and self.classical_extractor:
            print("🔧 Extracting classical features...")
            classical_features = self.classical_extractor.extract_features_batch(images)
            feature_groups.append(classical_features)
            print(f"✅ Classical features shape: {classical_features.shape}")
        
        # Combine all features
        if len(feature_groups) > 0:
            combined_features = np.hstack(feature_groups)
            print(f"🔗 Combined features shape: {combined_features.shape}")
            return combined_features
        else:
            raise ValueError("No features extracted!")
    
    def fit(self, images: List[np.ndarray], labels: List[int]):
        """Train the hybrid pipeline"""
        print("🚀 Starting hybrid pipeline training...")
        
        # Data augmentation for small dataset
        if self.config.use_augmentation and self.augmenter:
            images, labels = self.augmenter.augment_dataset(images, labels)
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Extract features
        features = self.extract_features(images)
        
        # Feature preprocessing
        print("⚙️ Preprocessing features...")
        features_scaled = self.scaler.fit_transform(features)
        
        if self.config.use_pca and self.pca:
            print(f"📉 Applying PCA: {features_scaled.shape[1]} → {self.config.pca_components}")
            features_final = self.pca.fit_transform(features_scaled)
        else:
            features_final = features_scaled
        
        print(f"✅ Final features shape: {features_final.shape}")
        
        # Train ensemble model
        print("🏋️ Training ensemble model...")
        
        if self.config.use_stratified_kfold:
            # Cross-validation for small dataset
            skf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                                random_state=self.config.random_state)
            
            cv_scores = cross_val_score(
                self.ensemble_model, features_final, labels_encoded, 
                cv=skf, scoring='accuracy', n_jobs=self.config.n_jobs
            )
            
            print(f"📊 Cross-validation scores: {cv_scores}")
            print(f"📊 Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Fit final model
        self.ensemble_model.fit(features_final, labels_encoded)
        
        # Training accuracy
        train_pred = self.ensemble_model.predict(features_final)
        train_acc = accuracy_score(labels_encoded, train_pred)
        print(f"✅ Training accuracy: {train_acc:.4f}")
        
        # Feature importance (if available)
        self._analyze_feature_importance()
        
        print("🎉 Hybrid pipeline training completed!")
    
    def predict(self, images: List[np.ndarray]) -> Tuple[List[str], np.ndarray]:
        """Predict on new images"""
        # Extract features
        features = self.extract_features(images)
        
        # Preprocess
        features_scaled = self.scaler.transform(features)
        if self.config.use_pca and self.pca:
            features_final = self.pca.transform(features_scaled)
        else:
            features_final = features_scaled
        
        # Predict
        predictions = self.ensemble_model.predict(features_final)
        probabilities = self.ensemble_model.predict_proba(features_final)
        
        # Decode labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels.tolist(), probabilities
    
    def _analyze_feature_importance(self):
        """Analyze feature importance from ensemble models"""
        print("📊 Analyzing feature importance...")
        
        for name, estimator in self.ensemble_model.named_estimators_.items():
            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
                print(f"{name} - Top 5 features: {np.argsort(importances)[-5:]}")
    
    def save(self, save_dir: str):
        """Save trained pipeline"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save models and preprocessors
        joblib.dump(self.ensemble_model, save_path / "ensemble_model.joblib")
        joblib.dump(self.scaler, save_path / "scaler.joblib")
        if self.pca:
            joblib.dump(self.pca, save_path / "pca.joblib")
        joblib.dump(self.label_encoder, save_path / "label_encoder.joblib")
        
        # Save config
        with open(save_path / "config.json", "w") as f:
            # Convert dataclass to dict for JSON serialization
            config_dict = {
                'data_path': self.config.data_path,
                'image_size': self.config.image_size,
                'use_cnn_features': self.config.use_cnn_features,
                'use_classical_features': self.config.use_classical_features,
                'cnn_backbone': self.config.cnn_backbone,
                'ensemble_models': self.config.ensemble_models,
                'pca_components': self.config.pca_components if self.config.use_pca else None
            }
            json.dump(config_dict, f, indent=2)
        
        print(f"💾 Pipeline saved to {save_path}")
    
    def load(self, save_dir: str):
        """Load trained pipeline"""
        save_path = Path(save_dir)
        
        # Load models and preprocessors
        self.ensemble_model = joblib.load(save_path / "ensemble_model.joblib")
        self.scaler = joblib.load(save_path / "scaler.joblib")
        if (save_path / "pca.joblib").exists():
            self.pca = joblib.load(save_path / "pca.joblib")
        self.label_encoder = joblib.load(save_path / "label_encoder.joblib")
        
        print(f"📂 Pipeline loaded from {save_path}")

# Example usage and testing
if __name__ == "__main__":
    # Configuration optimized for small amulet dataset
    config = HybridPipelineConfig(
        # Dataset settings
        data_path="dataset",
        image_size=(224, 224),
        
        # Feature extraction
        use_cnn_features=True,
        cnn_backbone="resnet50",
        use_classical_features=True,
        
        # Dimensionality reduction (critical for small dataset)
        use_pca=True,
        pca_components=300,  # Reduced for small dataset
        
        # Ensemble configuration
        ensemble_models=['random_forest', 'svm'],
        ensemble_voting="soft",
        
        # Small dataset optimization
        use_augmentation=True,
        augmentation_factor=5,
        use_class_weights=True,
        use_stratified_kfold=True,
        cv_folds=5
    )
    
    # Create pipeline
    pipeline = HybridMLPipeline(config)
    
    print("✅ Hybrid ML Pipeline initialized!")
    print("🎯 Optimized for small, imbalanced amulet dataset")
    print("🔧 Features: CNN (ResNet50) + Classical (HOG+LBP+Color+Edge)")
    print("🏗️  Ensemble: RandomForest + SVM + Gradient Boosting")
    print("📈 Augmentation: 5x dataset expansion")
    print("⚡ Python 3.13 compatible!")