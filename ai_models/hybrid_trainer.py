#!/usr/bin/env python3
"""
🏗️ Hybrid ML Training System for Amulet Recognition
Complete training pipeline combining CNN + Classical features with Ensemble ML

This system provides:
- End-to-end training pipeline from raw images to trained model
- Automatic data loading and preprocessing
- Feature extraction with caching
- Dimensionality reduction (PCA/SelectKBest)
- Ensemble classification with hyperparameter tuning
- Comprehensive evaluation and validation
- Model persistence and deployment preparation

Author: AI Assistant
Compatible: Python 3.13+
Dependencies: scikit-learn, opencv-python, torch (optional), joblib
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core ML imports
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, 
    RandomizedSearchCV, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, 
    GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Metrics
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

# Our modules
from feature_extractors import HybridFeatureExtractor, FeatureConfig, create_optimal_config
from augmentation_pipeline import ClassAwareAugmentationPipeline, AugmentationConfig
from dataset_inspector import DatasetInspector

import cv2


@dataclass
class TrainingConfig:
    """Configuration for hybrid training pipeline"""
    
    # Data settings
    data_dir: str = "dataset"
    validation_split: float = 0.2
    test_split: float = 0.15
    random_state: int = 42
    
    # Feature extraction
    feature_config: FeatureConfig = None
    enable_feature_caching: bool = True
    
    # Data preprocessing
    apply_augmentation: bool = True
    augmentation_config: AugmentationConfig = None
    
    # Feature engineering
    apply_scaling: bool = True
    scaler_type: str = "standard"  # standard, robust, minmax
    
    apply_dimensionality_reduction: bool = True
    dr_method: str = "pca"  # pca, truncated_svd, select_k_best
    dr_components: Optional[int] = None  # Auto-select if None
    
    # Model configuration
    ensemble_models: List[str] = None
    voting_strategy: str = "soft"  # soft, hard
    enable_calibration: bool = True
    
    # Training strategy
    use_stratified_cv: bool = True
    cv_folds: int = 5
    enable_hyperparameter_tuning: bool = True
    tuning_method: str = "randomized"  # randomized, grid
    tuning_iterations: int = 50
    
    # Class balancing
    handle_imbalance: bool = True
    class_weight_strategy: str = "balanced"  # balanced, balanced_subsample, None
    
    # Performance
    n_jobs: int = -1
    verbose: int = 1
    
    # Output
    model_save_dir: str = "ai_models/saved_models/hybrid_pipeline"
    save_intermediate_results: bool = True
    
    def __post_init__(self):
        if self.feature_config is None:
            self.feature_config = create_optimal_config()
        
        if self.augmentation_config is None:
            from augmentation_pipeline import create_augmentation_config_for_amulets
            self.augmentation_config = create_augmentation_config_for_amulets()
        
        if self.ensemble_models is None:
            self.ensemble_models = ['random_forest', 'svm', 'gradient_boost']


class DataLoader:
    """Handles data loading and organization"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('DataLoader')
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_dataset(self) -> Tuple[List[np.ndarray], List[str], Dict[str, int]]:
        """
        Load dataset from directory structure
        
        Returns:
            images: List of image arrays
            labels: List of string labels
            class_counts: Dictionary of class counts
        """
        data_path = Path(self.config.data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_path}")
        
        images = []
        labels = []
        class_counts = {}
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        self.logger.info(f"📂 Loading dataset from {data_path}")
        
        for class_dir in data_path.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            class_images = []
            
            # Find all image files
            for ext in valid_extensions:
                class_images.extend(class_dir.glob(f"*{ext}"))
                class_images.extend(class_dir.glob(f"*{ext.upper()}"))
            
            # Load images
            loaded_count = 0
            for img_path in class_images:
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # Convert BGR to RGB
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img_rgb)
                        labels.append(class_name)
                        loaded_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to load {img_path}: {e}")
            
            class_counts[class_name] = loaded_count
            self.logger.info(f"  {class_name}: {loaded_count} images")
        
        self.logger.info(f"✅ Loaded {len(images)} images from {len(class_counts)} classes")
        
        return images, labels, class_counts
    
    def split_dataset(self, images: List[np.ndarray], labels: List[str]) -> Dict[str, Any]:
        """
        Split dataset into train/validation/test sets with stratification
        
        Returns:
            Dictionary containing split datasets
        """
        self.logger.info("🔄 Splitting dataset...")
        
        # Convert to numpy arrays for easier handling
        X = np.array(images, dtype=object)
        y = np.array(labels)
        
        # First split: train+val vs test
        test_size = self.config.test_split
        val_size = self.config.validation_split / (1 - test_size)  # Adjust for remaining data
        
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            stratify=y,
            random_state=self.config.random_state
        )
        
        # Second split: train vs validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_size,
            stratify=y_trainval,
            random_state=self.config.random_state
        )
        
        splits = {
            'X_train': X_train.tolist(),
            'y_train': y_train.tolist(),
            'X_val': X_val.tolist(),
            'y_val': y_val.tolist(),
            'X_test': X_test.tolist(),
            'y_test': y_test.tolist()
        }
        
        self.logger.info(f"📊 Dataset split:")
        self.logger.info(f"  Train: {len(X_train)} samples")
        self.logger.info(f"  Validation: {len(X_val)} samples")
        self.logger.info(f"  Test: {len(X_test)} samples")
        
        return splits


class HybridTrainer:
    """Main training system for hybrid ML pipeline"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.data_loader = DataLoader(config)
        self.feature_extractor = None
        self.scaler = None
        self.dimensionality_reducer = None
        self.label_encoder = LabelEncoder()
        self.ensemble_model = None
        
        # Training data
        self.dataset_splits = None
        self.features_train = None
        self.features_val = None
        self.features_test = None
        self.y_train_encoded = None
        self.y_val_encoded = None
        self.y_test_encoded = None
        
        # Results storage
        self.training_results = {}
        self.evaluation_results = {}
        
        # Create save directory
        self.save_dir = Path(self.config.model_save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('HybridTrainer')
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def prepare_data(self) -> None:
        """Load and prepare dataset"""
        self.logger.info("🚀 Starting data preparation...")
        
        # Load dataset
        images, labels, class_counts = self.data_loader.load_dataset()
        
        if not images:
            raise ValueError("No images loaded from dataset!")
        
        # Apply augmentation if enabled
        if self.config.apply_augmentation:
            self.logger.info("🔄 Applying data augmentation...")
            
            # Create temporary directory for augmented data
            temp_aug_dir = Path("temp_augmented_data")
            
            try:
                # Use augmentation pipeline
                aug_pipeline = ClassAwareAugmentationPipeline(self.config.augmentation_config)
                aug_results = aug_pipeline.run_augmentation(self.config.data_dir, str(temp_aug_dir))
                
                if aug_results["success"]:
                    # Reload augmented dataset
                    original_data_dir = self.config.data_dir
                    self.config.data_dir = str(temp_aug_dir)
                    images, labels, class_counts = self.data_loader.load_dataset()
                    self.config.data_dir = original_data_dir  # Restore original
                    
                    self.logger.info(f"✅ Augmentation completed: {len(images)} total images")
                else:
                    self.logger.warning("⚠️ Augmentation failed, using original data")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Augmentation error: {e}, using original data")
        
        # Split dataset
        self.dataset_splits = self.data_loader.split_dataset(images, labels)
        
        # Store class information
        self.training_results['original_class_counts'] = class_counts
        self.training_results['total_samples'] = len(images)
        self.training_results['n_classes'] = len(set(labels))
    
    def extract_features(self) -> None:
        """Extract features from all dataset splits"""
        self.logger.info("🎯 Starting feature extraction...")
        
        # Initialize feature extractor
        self.feature_extractor = HybridFeatureExtractor(self.config.feature_config)
        
        feature_dim = self.feature_extractor.get_feature_dimension()
        self.logger.info(f"📊 Feature dimension: {feature_dim}")
        
        # Extract features for each split
        self.logger.info("  Extracting training features...")
        self.features_train = self.feature_extractor.extract_batch(self.dataset_splits['X_train'])
        
        self.logger.info("  Extracting validation features...")
        self.features_val = self.feature_extractor.extract_batch(self.dataset_splits['X_val'])
        
        self.logger.info("  Extracting test features...")
        self.features_test = self.feature_extractor.extract_batch(self.dataset_splits['X_test'])
        
        # Encode labels
        self.label_encoder.fit(self.dataset_splits['y_train'])
        self.y_train_encoded = self.label_encoder.transform(self.dataset_splits['y_train'])
        self.y_val_encoded = self.label_encoder.transform(self.dataset_splits['y_val'])
        self.y_test_encoded = self.label_encoder.transform(self.dataset_splits['y_test'])
        
        self.logger.info(f"✅ Feature extraction completed")
        self.logger.info(f"   Train features: {self.features_train.shape}")
        self.logger.info(f"   Val features: {self.features_val.shape}")
        self.logger.info(f"   Test features: {self.features_test.shape}")
        
        # Store feature information
        self.training_results['feature_dimension'] = feature_dim
        self.training_results['feature_names'] = self.feature_extractor.get_feature_names()
    
    def preprocess_features(self) -> None:
        """Apply scaling and dimensionality reduction"""
        self.logger.info("⚙️ Preprocessing features...")
        
        # Feature scaling
        if self.config.apply_scaling:
            self.logger.info(f"  Applying {self.config.scaler_type} scaling...")
            
            if self.config.scaler_type == "standard":
                self.scaler = StandardScaler()
            elif self.config.scaler_type == "robust":
                self.scaler = RobustScaler()
            else:
                from sklearn.preprocessing import MinMaxScaler
                self.scaler = MinMaxScaler()
            
            # Fit on training data and transform all splits
            self.features_train = self.scaler.fit_transform(self.features_train)
            self.features_val = self.scaler.transform(self.features_val)
            self.features_test = self.scaler.transform(self.features_test)
        
        # Dimensionality reduction
        if self.config.apply_dimensionality_reduction:
            self.logger.info(f"  Applying {self.config.dr_method} dimensionality reduction...")
            
            # Determine number of components
            n_components = self.config.dr_components
            if n_components is None:
                # Auto-select based on data size and features
                n_samples = len(self.features_train)
                n_features = self.features_train.shape[1]
                
                # Conservative selection for small datasets
                max_components = min(n_samples // 2, n_features // 2, 500)
                n_components = max(50, max_components)
            
            # Create dimensionality reducer
            if self.config.dr_method == "pca":
                self.dimensionality_reducer = PCA(n_components=n_components, random_state=self.config.random_state)
            elif self.config.dr_method == "truncated_svd":
                self.dimensionality_reducer = TruncatedSVD(n_components=n_components, random_state=self.config.random_state)
            else:  # select_k_best
                self.dimensionality_reducer = SelectKBest(score_func=f_classif, k=n_components)
            
            # Fit and transform
            self.features_train = self.dimensionality_reducer.fit_transform(self.features_train, self.y_train_encoded)
            self.features_val = self.dimensionality_reducer.transform(self.features_val)
            self.features_test = self.dimensionality_reducer.transform(self.features_test)
            
            self.logger.info(f"  Reduced to {self.features_train.shape[1]} dimensions")
            self.training_results['reduced_dimension'] = self.features_train.shape[1]
        
        self.logger.info("✅ Feature preprocessing completed")
    
    def create_ensemble_model(self) -> VotingClassifier:
        """Create ensemble model with specified estimators"""
        estimators = []
        
        # Random Forest
        if 'random_forest' in self.config.ensemble_models:
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                class_weight=self.config.class_weight_strategy if self.config.handle_imbalance else None
            )
            estimators.append(('rf', rf))
        
        # Support Vector Machine
        if 'svm' in self.config.ensemble_models:
            svm = SVC(
                kernel='rbf',
                probability=True,  # Required for soft voting
                random_state=self.config.random_state,
                class_weight=self.config.class_weight_strategy if self.config.handle_imbalance else None
            )
            estimators.append(('svm', svm))
        
        # Gradient Boosting
        if 'gradient_boost' in self.config.ensemble_models:
            gb = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config.random_state
            )
            estimators.append(('gb', gb))
        
        # Extra Trees (additional diversity)
        if 'extra_trees' in self.config.ensemble_models:
            et = ExtraTreesClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                class_weight=self.config.class_weight_strategy if self.config.handle_imbalance else None
            )
            estimators.append(('et', et))
        
        # Logistic Regression (linear model for diversity)
        if 'logistic' in self.config.ensemble_models:
            lr = LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000,
                n_jobs=self.config.n_jobs,
                class_weight=self.config.class_weight_strategy if self.config.handle_imbalance else None
            )
            estimators.append(('lr', lr))
        
        if not estimators:
            raise ValueError("No valid estimators specified for ensemble!")
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=self.config.voting_strategy,
            n_jobs=self.config.n_jobs
        )
        
        return ensemble
    
    def hyperparameter_tuning(self, model: VotingClassifier) -> VotingClassifier:
        """Perform hyperparameter tuning"""
        if not self.config.enable_hyperparameter_tuning:
            return model
        
        self.logger.info("🔧 Starting hyperparameter tuning...")
        
        # Define parameter grid for each estimator
        param_grid = {}
        
        # Random Forest parameters
        if 'rf' in [name for name, _ in model.named_estimators_.items()]:
            param_grid.update({
                'rf__n_estimators': [50, 100, 200],
                'rf__max_depth': [None, 10, 20, 30],
                'rf__min_samples_split': [2, 5, 10],
                'rf__min_samples_leaf': [1, 2, 4]
            })
        
        # SVM parameters
        if 'svm' in [name for name, _ in model.named_estimators_.items()]:
            param_grid.update({
                'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'svm__kernel': ['rbf', 'poly']
            })
        
        # Gradient Boosting parameters
        if 'gb' in [name for name, _ in model.named_estimators_.items()]:
            param_grid.update({
                'gb__n_estimators': [50, 100, 200],
                'gb__max_depth': [3, 6, 9],
                'gb__learning_rate': [0.01, 0.1, 0.2]
            })
        
        # Choose search method
        if self.config.tuning_method == "grid":
            search = GridSearchCV(
                model,
                param_grid,
                cv=StratifiedKFold(n_splits=min(self.config.cv_folds, 3), shuffle=True, random_state=self.config.random_state),
                scoring='balanced_accuracy',
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose
            )
        else:  # randomized
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=self.config.tuning_iterations,
                cv=StratifiedKFold(n_splits=min(self.config.cv_folds, 3), shuffle=True, random_state=self.config.random_state),
                scoring='balanced_accuracy',
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
                verbose=self.config.verbose
            )
        
        # Perform search
        search.fit(self.features_train, self.y_train_encoded)
        
        self.logger.info(f"✅ Best cross-validation score: {search.best_score_:.4f}")
        self.logger.info(f"✅ Best parameters: {search.best_params_}")
        
        # Store tuning results
        self.training_results['best_cv_score'] = search.best_score_
        self.training_results['best_params'] = search.best_params_
        
        return search.best_estimator_
    
    def train_model(self) -> None:
        """Train the ensemble model"""
        self.logger.info("🏋️ Training ensemble model...")
        
        # Create ensemble model
        ensemble = self.create_ensemble_model()
        
        # Hyperparameter tuning
        if self.config.enable_hyperparameter_tuning:
            ensemble = self.hyperparameter_tuning(ensemble)
        
        # Apply calibration if enabled
        if self.config.enable_calibration:
            self.logger.info("📊 Applying probability calibration...")
            ensemble = CalibratedClassifierCV(
                ensemble,
                method='sigmoid',
                cv=3,
                n_jobs=self.config.n_jobs
            )
        
        # Cross-validation evaluation
        if self.config.use_stratified_cv:
            self.logger.info("📈 Performing cross-validation...")
            
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
            
            cv_scores = cross_val_score(
                ensemble, self.features_train, self.y_train_encoded,
                cv=cv, scoring='balanced_accuracy', n_jobs=self.config.n_jobs
            )
            
            self.logger.info(f"📊 CV Scores: {cv_scores}")
            self.logger.info(f"📊 Mean CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            self.training_results['cv_scores'] = cv_scores.tolist()
            self.training_results['mean_cv_score'] = cv_scores.mean()
            self.training_results['std_cv_score'] = cv_scores.std()
        
        # Final training on full training set
        self.logger.info("🎯 Training final model...")
        ensemble.fit(self.features_train, self.y_train_encoded)
        
        self.ensemble_model = ensemble
        self.logger.info("✅ Model training completed")
    
    def evaluate_model(self) -> None:
        """Comprehensive model evaluation"""
        self.logger.info("📊 Starting model evaluation...")
        
        if self.ensemble_model is None:
            raise ValueError("Model not trained yet!")
        
        # Predictions on all splits
        train_pred = self.ensemble_model.predict(self.features_train)
        train_pred_proba = self.ensemble_model.predict_proba(self.features_train)
        
        val_pred = self.ensemble_model.predict(self.features_val)
        val_pred_proba = self.ensemble_model.predict_proba(self.features_val)
        
        test_pred = self.ensemble_model.predict(self.features_test)
        test_pred_proba = self.ensemble_model.predict_proba(self.features_test)
        
        # Calculate metrics for each split
        def calculate_metrics(y_true, y_pred, y_pred_proba, split_name):
            metrics = {}
            
            # Basic metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
            
            # Precision, Recall, F1 per class
            precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
            
            class_names = self.label_encoder.classes_
            for i, class_name in enumerate(class_names):
                metrics[f'precision_{class_name}'] = precision[i]
                metrics[f'recall_{class_name}'] = recall[i]
                metrics[f'f1_{class_name}'] = f1[i]
                metrics[f'support_{class_name}'] = support[i]
            
            # Classification report
            metrics['classification_report'] = classification_report(
                y_true, y_pred, 
                target_names=class_names,
                output_dict=True
            )
            
            # Confusion matrix
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
            
            # ROC AUC (if multiclass)
            try:
                if len(np.unique(y_true)) > 2:
                    metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                    metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='macro')
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                pass  # Skip if can't calculate ROC AUC
            
            return metrics
        
        # Evaluate each split
        self.evaluation_results['train'] = calculate_metrics(
            self.y_train_encoded, train_pred, train_pred_proba, 'train'
        )
        
        self.evaluation_results['validation'] = calculate_metrics(
            self.y_val_encoded, val_pred, val_pred_proba, 'validation'
        )
        
        self.evaluation_results['test'] = calculate_metrics(
            self.y_test_encoded, test_pred, test_pred_proba, 'test'
        )
        
        # Print summary
        self.logger.info("📊 Evaluation Results:")
        for split in ['train', 'validation', 'test']:
            metrics = self.evaluation_results[split]
            self.logger.info(f"  {split.capitalize()}:")
            self.logger.info(f"    Accuracy: {metrics['accuracy']:.4f}")
            self.logger.info(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            self.logger.info(f"    F1 (macro): {metrics['f1_macro']:.4f}")
        
        self.logger.info("✅ Evaluation completed")
    
    def save_model(self) -> None:
        """Save trained model and all components"""
        self.logger.info(f"💾 Saving model to {self.save_dir}")
        
        # Save main ensemble model
        joblib.dump(self.ensemble_model, self.save_dir / "ensemble_model.joblib")
        
        # Save preprocessing components
        if self.scaler:
            joblib.dump(self.scaler, self.save_dir / "scaler.joblib")
        
        if self.dimensionality_reducer:
            joblib.dump(self.dimensionality_reducer, self.save_dir / "dimensionality_reducer.joblib")
        
        joblib.dump(self.label_encoder, self.save_dir / "label_encoder.joblib")
        
        # Save configurations
        config_dict = asdict(self.config)
        # Convert FeatureConfig and AugmentationConfig to dict
        config_dict['feature_config'] = asdict(self.config.feature_config)
        config_dict['augmentation_config'] = asdict(self.config.augmentation_config)
        
        with open(self.save_dir / "training_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save training results
        with open(self.save_dir / "training_results.json", 'w') as f:
            json.dump(self.training_results, f, indent=2)
        
        # Save evaluation results
        with open(self.save_dir / "evaluation_results.json", 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        # Save feature extractor (just the config, model will be recreated)
        feature_config_dict = asdict(self.config.feature_config)
        with open(self.save_dir / "feature_config.json", 'w') as f:
            json.dump(feature_config_dict, f, indent=2)
        
        self.logger.info("✅ Model saved successfully")
    
    def run_complete_training(self) -> Dict[str, Any]:
        """Run complete training pipeline"""
        start_time = datetime.now()
        
        try:
            self.logger.info("🚀 Starting hybrid ML training pipeline")
            self.logger.info("=" * 60)
            
            # Step 1: Prepare data
            self.prepare_data()
            
            # Step 2: Extract features
            self.extract_features()
            
            # Step 3: Preprocess features
            self.preprocess_features()
            
            # Step 4: Train model
            self.train_model()
            
            # Step 5: Evaluate model
            self.evaluate_model()
            
            # Step 6: Save model
            self.save_model()
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            self.logger.info("=" * 60)
            self.logger.info(f"🎉 Training completed successfully in {training_time:.2f} seconds")
            
            # Final summary
            test_metrics = self.evaluation_results['test']
            self.logger.info(f"📊 Final Test Results:")
            self.logger.info(f"   Accuracy: {test_metrics['accuracy']:.4f}")
            self.logger.info(f"   Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
            self.logger.info(f"   F1 (macro): {test_metrics['f1_macro']:.4f}")
            
            return {
                'success': True,
                'training_time': training_time,
                'test_accuracy': test_metrics['accuracy'],
                'test_balanced_accuracy': test_metrics['balanced_accuracy'],
                'test_f1_macro': test_metrics['f1_macro'],
                'model_save_path': str(self.save_dir)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'training_time': (datetime.now() - start_time).total_seconds()
            }


def create_training_config_for_amulets() -> TrainingConfig:
    """Create optimized training configuration for amulet recognition"""
    
    # Feature extraction config
    feature_config = create_optimal_config()
    
    # Augmentation config
    from augmentation_pipeline import create_augmentation_config_for_amulets
    aug_config = create_augmentation_config_for_amulets()
    
    return TrainingConfig(
        # Data settings
        data_dir="dataset",
        validation_split=0.2,
        test_split=0.15,
        random_state=42,
        
        # Feature settings
        feature_config=feature_config,
        enable_feature_caching=True,
        
        # Preprocessing
        apply_augmentation=True,
        augmentation_config=aug_config,
        
        # Feature engineering
        apply_scaling=True,
        scaler_type="standard",
        apply_dimensionality_reduction=True,
        dr_method="pca",
        dr_components=200,  # Conservative for small dataset
        
        # Model settings
        ensemble_models=['random_forest', 'svm', 'gradient_boost'],
        voting_strategy="soft",
        enable_calibration=True,
        
        # Training strategy
        use_stratified_cv=True,
        cv_folds=5,
        enable_hyperparameter_tuning=True,
        tuning_method="randomized",
        tuning_iterations=30,  # Reduced for faster training
        
        # Class balancing
        handle_imbalance=True,
        class_weight_strategy="balanced",
        
        # Performance
        n_jobs=-1,
        verbose=1,
        
        # Output
        model_save_dir="ai_models/saved_models/hybrid_amulet_classifier",
        save_intermediate_results=True
    )


def main():
    """CLI entry point for hybrid training"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🏗️ Hybrid ML Training System for Amulet Recognition"
    )
    
    parser.add_argument("--data-dir", default="dataset", help="Dataset directory")
    parser.add_argument("--output-dir", default="ai_models/saved_models/hybrid_amulet_classifier", 
                       help="Output directory for trained model")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable data augmentation")
    parser.add_argument("--no-tuning", action="store_true", help="Disable hyperparameter tuning")
    parser.add_argument("--quick", action="store_true", help="Quick training (reduced parameters)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_training_config_for_amulets()
    config.data_dir = args.data_dir
    config.model_save_dir = args.output_dir
    
    if args.no_augmentation:
        config.apply_augmentation = False
    
    if args.no_tuning:
        config.enable_hyperparameter_tuning = False
    
    if args.quick:
        config.tuning_iterations = 10
        config.cv_folds = 3
    
    # Create trainer and run
    trainer = HybridTrainer(config)
    results = trainer.run_complete_training()
    
    # Exit with appropriate code
    if results['success']:
        print(f"✅ Training successful! Model saved to: {results['model_save_path']}")
        print(f"📊 Test Accuracy: {results['test_accuracy']:.4f}")
        exit(0)
    else:
        print(f"❌ Training failed: {results['error']}")
        exit(1)


if __name__ == "__main__":
    main()