#!/usr/bin/env python3
"""
📊 Comprehensive Evaluation Suite for Hybrid ML Pipeline
Advanced model evaluation, visualization, and performance analysis

This system provides:
- Detailed performance metrics and analysis
- Confusion matrix visualization
- Feature importance analysis
- Error case analysis and visualization
- Model comparison and benchmarking
- Performance profiling and optimization insights

Author: AI Assistant
Compatible: Python 3.13+
Dependencies: scikit-learn, plotly, opencv-python, numpy, pandas
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

# Visualization imports (with fallbacks)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available, skipping advanced visualizations")

# ML imports
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve, validation_curve

# Our modules
from hybrid_trainer import HybridTrainer, TrainingConfig
from feature_extractors import HybridFeatureExtractor, FeatureConfig
import cv2


@dataclass
class EvaluationConfig:
    """Configuration for evaluation suite"""
    
    # Model paths
    model_dir: str = "ai_models/saved_models/hybrid_amulet_classifier"
    test_data_dir: Optional[str] = None  # If None, use holdout test set
    
    # Evaluation settings
    calculate_feature_importance: bool = True
    perform_error_analysis: bool = True
    generate_visualizations: bool = True
    create_detailed_report: bool = True
    
    # Visualization settings
    save_plots: bool = True
    plot_format: str = "html"  # html, png, pdf
    plot_dir: str = "evaluation_plots"
    
    # Error analysis
    max_error_examples: int = 10
    save_error_images: bool = True
    
    # Performance profiling
    profile_inference_speed: bool = True
    n_speed_test_samples: int = 100
    
    # Output
    report_dir: str = "evaluation_reports"
    verbose: bool = True


class ModelLoader:
    """Handles loading of trained models and components"""
    
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.logger = self._setup_logging()
        
        # Model components
        self.ensemble_model = None
        self.scaler = None
        self.dimensionality_reducer = None
        self.label_encoder = None
        self.feature_extractor = None
        
        # Configurations
        self.training_config = None
        self.feature_config = None
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('ModelLoader')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_model(self) -> bool:
        """Load all model components"""
        try:
            self.logger.info(f"📂 Loading model from {self.model_dir}")
            
            # Load main ensemble model
            model_path = self.model_dir / "ensemble_model.joblib"
            if model_path.exists():
                self.ensemble_model = joblib.load(model_path)
                self.logger.info("✅ Ensemble model loaded")
            else:
                raise FileNotFoundError(f"Ensemble model not found: {model_path}")
            
            # Load preprocessing components
            scaler_path = self.model_dir / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                self.logger.info("✅ Scaler loaded")
            
            dr_path = self.model_dir / "dimensionality_reducer.joblib"
            if dr_path.exists():
                self.dimensionality_reducer = joblib.load(dr_path)
                self.logger.info("✅ Dimensionality reducer loaded")
            
            encoder_path = self.model_dir / "label_encoder.joblib"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                self.logger.info("✅ Label encoder loaded")
            
            # Load configurations
            config_path = self.model_dir / "training_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.training_config = json.load(f)
                self.logger.info("✅ Training config loaded")
            
            feature_config_path = self.model_dir / "feature_config.json"
            if feature_config_path.exists():
                with open(feature_config_path, 'r') as f:
                    feature_config_dict = json.load(f)
                    # Reconstruct FeatureConfig object
                    self.feature_config = FeatureConfig(**feature_config_dict)
                self.logger.info("✅ Feature config loaded")
            
            # Initialize feature extractor
            if self.feature_config:
                self.feature_extractor = HybridFeatureExtractor(self.feature_config)
                self.logger.info("✅ Feature extractor initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def predict(self, images: List[np.ndarray]) -> Tuple[List[str], np.ndarray]:
        """Predict on new images using loaded model"""
        if not all([self.ensemble_model, self.feature_extractor, self.label_encoder]):
            raise ValueError("Model not properly loaded!")
        
        # Extract features
        features = self.feature_extractor.extract_batch(images)
        
        # Apply preprocessing
        if self.scaler:
            features = self.scaler.transform(features)
        
        if self.dimensionality_reducer:
            features = self.dimensionality_reducer.transform(features)
        
        # Make predictions
        predictions = self.ensemble_model.predict(features)
        probabilities = self.ensemble_model.predict_proba(features)
        
        # Decode labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels.tolist(), probabilities


class PerformanceAnalyzer:
    """Analyzes model performance in detail"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Results storage
        self.detailed_metrics = {}
        self.feature_importance_results = {}
        self.error_analysis_results = {}
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('PerformanceAnalyzer')
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def calculate_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_pred_proba: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        self.logger.info("📊 Calculating detailed metrics...")
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Per-class and averaged metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        metrics['precision_macro'] = precision_recall_fscore_support(y_true, y_pred, average='macro')[0]
        metrics['recall_macro'] = precision_recall_fscore_support(y_true, y_pred, average='macro')[1]
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        
        metrics['precision_weighted'] = precision_recall_fscore_support(y_true, y_pred, average='weighted')[0]
        metrics['recall_weighted'] = precision_recall_fscore_support(y_true, y_pred, average='weighted')[1]
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        metrics['per_class'] = {}
        for i, class_name in enumerate(class_names):
            metrics['per_class'][class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': int(support[i])
            }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # ROC AUC and Average Precision (for multiclass)
        try:
            n_classes = len(class_names)
            if n_classes > 2:
                # Multiclass ROC AUC
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='macro')
                
                # Per-class ROC AUC (one-vs-rest)
                metrics['per_class_roc_auc'] = {}
                for i, class_name in enumerate(class_names):
                    y_true_binary = (y_true == i).astype(int)
                    y_score = y_pred_proba[:, i]
                    try:
                        auc = roc_auc_score(y_true_binary, y_score)
                        metrics['per_class_roc_auc'][class_name] = auc
                    except ValueError:
                        metrics['per_class_roc_auc'][class_name] = 0.5  # If only one class present
            else:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except Exception as e:
            self.logger.warning(f"Could not calculate ROC AUC: {e}")
        
        # Classification report (detailed)
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )
        
        return metrics
    
    def analyze_feature_importance(self, model, X: np.ndarray, y: np.ndarray, 
                                 feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature importance using various methods"""
        if not self.config.calculate_feature_importance:
            return {}
        
        self.logger.info("🔍 Analyzing feature importance...")
        
        importance_results = {}
        
        try:
            # Method 1: Built-in feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                importance_results['tree_importance'] = {
                    'values': model.feature_importances_.tolist(),
                    'feature_names': feature_names[:len(model.feature_importances_)]
                }
            elif hasattr(model, 'named_estimators_'):
                # For ensemble models, try to get importance from tree-based estimators
                tree_importances = []
                for name, estimator in model.named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        tree_importances.append((name, estimator.feature_importances_))
                
                if tree_importances:
                    importance_results['ensemble_tree_importance'] = tree_importances
            
            # Method 2: Permutation importance (more robust but slower)
            self.logger.info("  Computing permutation importance...")
            
            # Use a subset of data for speed
            if len(X) > 500:
                indices = np.random.choice(len(X), 500, replace=False)
                X_subset = X[indices]
                y_subset = y[indices]
            else:
                X_subset = X
                y_subset = y
            
            perm_importance = permutation_importance(
                model, X_subset, y_subset,
                n_repeats=5,
                random_state=42,
                scoring='balanced_accuracy',
                n_jobs=-1
            )
            
            importance_results['permutation_importance'] = {
                'importances_mean': perm_importance.importances_mean.tolist(),
                'importances_std': perm_importance.importances_std.tolist(),
                'feature_names': feature_names[:len(perm_importance.importances_mean)]
            }
            
        except Exception as e:
            self.logger.warning(f"Feature importance analysis failed: {e}")
        
        return importance_results
    
    def analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_pred_proba: np.ndarray, images: List[np.ndarray],
                      class_names: List[str]) -> Dict[str, Any]:
        """Analyze prediction errors in detail"""
        if not self.config.perform_error_analysis:
            return {}
        
        self.logger.info("🔍 Analyzing prediction errors...")
        
        error_analysis = {}
        
        # Find misclassified samples
        misclassified_indices = np.where(y_true != y_pred)[0]
        
        if len(misclassified_indices) == 0:
            self.logger.info("🎉 No misclassifications found!")
            return {'no_errors': True}
        
        error_analysis['total_errors'] = len(misclassified_indices)
        error_analysis['error_rate'] = len(misclassified_indices) / len(y_true)
        
        # Analyze error patterns
        error_patterns = {}
        for idx in misclassified_indices:
            true_class = class_names[y_true[idx]]
            pred_class = class_names[y_pred[idx]]
            pattern = f"{true_class} -> {pred_class}"
            
            if pattern not in error_patterns:
                error_patterns[pattern] = []
            
            error_patterns[pattern].append({
                'index': int(idx),
                'true_class': true_class,
                'predicted_class': pred_class,
                'confidence': float(np.max(y_pred_proba[idx])),
                'true_class_prob': float(y_pred_proba[idx, y_true[idx]])
            })
        
        # Sort error patterns by frequency
        error_analysis['error_patterns'] = {
            pattern: {
                'count': len(errors),
                'examples': errors[:self.config.max_error_examples]
            }
            for pattern, errors in sorted(error_patterns.items(), 
                                        key=lambda x: len(x[1]), reverse=True)
        }
        
        # Analyze confidence of errors
        error_confidences = [np.max(y_pred_proba[idx]) for idx in misclassified_indices]
        
        error_analysis['confidence_analysis'] = {
            'mean_error_confidence': float(np.mean(error_confidences)),
            'median_error_confidence': float(np.median(error_confidences)),
            'high_confidence_errors': sum(1 for conf in error_confidences if conf > 0.8),
            'low_confidence_errors': sum(1 for conf in error_confidences if conf < 0.5)
        }
        
        # Find most confident errors (potentially systematic issues)
        most_confident_errors = []
        for idx in misclassified_indices:
            confidence = np.max(y_pred_proba[idx])
            if confidence > 0.7:  # High confidence but wrong
                most_confident_errors.append({
                    'index': int(idx),
                    'true_class': class_names[y_true[idx]],
                    'predicted_class': class_names[y_pred[idx]],
                    'confidence': float(confidence)
                })
        
        error_analysis['high_confidence_errors'] = sorted(
            most_confident_errors, key=lambda x: x['confidence'], reverse=True
        )[:self.config.max_error_examples]
        
        return error_analysis
    
    def profile_inference_speed(self, model_loader: ModelLoader) -> Dict[str, Any]:
        """Profile model inference speed"""
        if not self.config.profile_inference_speed:
            return {}
        
        self.logger.info("⚡ Profiling inference speed...")
        
        # Create test images
        test_images = []
        for _ in range(self.config.n_speed_test_samples):
            # Random image
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            test_images.append(img)
        
        # Time feature extraction
        import time
        
        start_time = time.time()
        features = model_loader.feature_extractor.extract_batch(test_images)
        feature_extraction_time = time.time() - start_time
        
        # Time preprocessing
        start_time = time.time()
        if model_loader.scaler:
            features_scaled = model_loader.scaler.transform(features)
        else:
            features_scaled = features
        
        if model_loader.dimensionality_reducer:
            features_final = model_loader.dimensionality_reducer.transform(features_scaled)
        else:
            features_final = features_scaled
        preprocessing_time = time.time() - start_time
        
        # Time prediction
        start_time = time.time()
        predictions = model_loader.ensemble_model.predict(features_final)
        probabilities = model_loader.ensemble_model.predict_proba(features_final)
        prediction_time = time.time() - start_time
        
        # Calculate speeds
        total_time = feature_extraction_time + preprocessing_time + prediction_time
        
        speed_profile = {
            'n_samples': self.config.n_speed_test_samples,
            'feature_extraction_time': feature_extraction_time,
            'preprocessing_time': preprocessing_time,
            'prediction_time': prediction_time,
            'total_time': total_time,
            'images_per_second': self.config.n_speed_test_samples / total_time,
            'ms_per_image': (total_time * 1000) / self.config.n_speed_test_samples,
            'breakdown': {
                'feature_extraction_pct': (feature_extraction_time / total_time) * 100,
                'preprocessing_pct': (preprocessing_time / total_time) * 100,
                'prediction_pct': (prediction_time / total_time) * 100
            }
        }
        
        self.logger.info(f"  Speed: {speed_profile['images_per_second']:.2f} images/sec")
        self.logger.info(f"  Time per image: {speed_profile['ms_per_image']:.2f} ms")
        
        return speed_profile


class VisualizationGenerator:
    """Generates visualizations for model evaluation"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.plot_dir = Path(config.plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('VisualizationGenerator')
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_confusion_matrix_plot(self, cm: np.ndarray, class_names: List[str]) -> Optional[str]:
        """Create confusion matrix visualization"""
        if not PLOTLY_AVAILABLE:
            return None
        
        self.logger.info("📊 Creating confusion matrix plot...")
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=class_names,
            y=class_names,
            text=cm,  # Show actual counts
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title="Confusion Matrix (Normalized)",
            xaxis_title="Predicted Class",
            yaxis_title="True Class",
            width=800,
            height=600
        )
        
        # Save plot
        plot_path = self.plot_dir / f"confusion_matrix.{self.config.plot_format}"
        if self.config.plot_format == "html":
            fig.write_html(plot_path)
        else:
            fig.write_image(plot_path)
        
        return str(plot_path)
    
    def create_feature_importance_plot(self, importance_data: Dict[str, Any]) -> Optional[str]:
        """Create feature importance visualization"""
        if not PLOTLY_AVAILABLE or not importance_data:
            return None
        
        self.logger.info("📊 Creating feature importance plot...")
        
        # Use permutation importance if available, otherwise tree importance
        if 'permutation_importance' in importance_data:
            importances = importance_data['permutation_importance']['importances_mean']
            feature_names = importance_data['permutation_importance']['feature_names']
            errors = importance_data['permutation_importance']['importances_std']
            title = "Permutation Feature Importance"
        elif 'tree_importance' in importance_data:
            importances = importance_data['tree_importance']['values']
            feature_names = importance_data['tree_importance']['feature_names']
            errors = None
            title = "Tree-based Feature Importance"
        else:
            return None
        
        # Sort by importance
        sorted_indices = np.argsort(importances)[::-1][:20]  # Top 20 features
        
        top_importances = [importances[i] for i in sorted_indices]
        top_features = [feature_names[i] for i in sorted_indices]
        top_errors = [errors[i] for i in sorted_indices] if errors else None
        
        # Create bar plot
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_features,
            y=top_importances,
            error_y=dict(array=top_errors) if top_errors else None,
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Features",
            yaxis_title="Importance",
            xaxis_tickangle=-45,
            width=1000,
            height=600
        )
        
        # Save plot
        plot_path = self.plot_dir / f"feature_importance.{self.config.plot_format}"
        if self.config.plot_format == "html":
            fig.write_html(plot_path)
        else:
            fig.write_image(plot_path)
        
        return str(plot_path)
    
    def create_class_performance_plot(self, metrics: Dict[str, Any]) -> Optional[str]:
        """Create per-class performance visualization"""
        if not PLOTLY_AVAILABLE:
            return None
        
        self.logger.info("📊 Creating class performance plot...")
        
        per_class = metrics['per_class']
        class_names = list(per_class.keys())
        
        precisions = [per_class[cls]['precision'] for cls in class_names]
        recalls = [per_class[cls]['recall'] for cls in class_names]
        f1_scores = [per_class[cls]['f1_score'] for cls in class_names]
        supports = [per_class[cls]['support'] for cls in class_names]
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Precision', 'Recall', 'F1-Score', 'Support'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Precision
        fig.add_trace(
            go.Bar(x=class_names, y=precisions, name='Precision', marker_color='blue'),
            row=1, col=1
        )
        
        # Recall
        fig.add_trace(
            go.Bar(x=class_names, y=recalls, name='Recall', marker_color='green'),
            row=1, col=2
        )
        
        # F1-Score
        fig.add_trace(
            go.Bar(x=class_names, y=f1_scores, name='F1-Score', marker_color='red'),
            row=2, col=1
        )
        
        # Support
        fig.add_trace(
            go.Bar(x=class_names, y=supports, name='Support', marker_color='orange'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Per-Class Performance Metrics",
            showlegend=False,
            width=1200,
            height=800
        )
        
        # Rotate x-axis labels
        fig.update_xaxes(tickangle=-45)
        
        # Save plot
        plot_path = self.plot_dir / f"class_performance.{self.config.plot_format}"
        if self.config.plot_format == "html":
            fig.write_html(plot_path)
        else:
            fig.write_image(plot_path)
        
        return str(plot_path)


class EvaluationSuite:
    """Main evaluation suite orchestrating all evaluation components"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.model_loader = ModelLoader(config.model_dir)
        self.performance_analyzer = PerformanceAnalyzer(config)
        self.visualization_generator = VisualizationGenerator(config) if config.generate_visualizations else None
        
        # Results storage
        self.evaluation_results = {}
        
        # Create output directories
        self.report_dir = Path(config.report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('EvaluationSuite')
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_test_data(self) -> Tuple[List[np.ndarray], List[str]]:
        """Load test data for evaluation"""
        if self.config.test_data_dir:
            # Load from separate test directory
            self.logger.info(f"📂 Loading test data from {self.config.test_data_dir}")
            
            test_images = []
            test_labels = []
            
            test_path = Path(self.config.test_data_dir)
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            
            for class_dir in test_path.iterdir():
                if not class_dir.is_dir():
                    continue
                
                class_name = class_dir.name
                
                for ext in valid_extensions:
                    for img_path in class_dir.glob(f"*{ext}"):
                        try:
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                test_images.append(img_rgb)
                                test_labels.append(class_name)
                        except Exception as e:
                            self.logger.warning(f"Failed to load {img_path}: {e}")
            
            return test_images, test_labels
        
        else:
            # Use holdout test set from training
            self.logger.info("📂 Using holdout test set from training")
            
            # Load evaluation results from training
            eval_results_path = Path(self.config.model_dir) / "evaluation_results.json"
            if eval_results_path.exists():
                with open(eval_results_path, 'r') as f:
                    training_eval = json.load(f)
                
                # This is a limitation - we can't recreate the actual test images
                # In a real scenario, test data would be saved separately
                self.logger.warning("⚠️ Cannot recreate test images from saved results")
                self.logger.info("💡 Consider using --test-data-dir option with separate test data")
                
                return [], []
            else:
                raise FileNotFoundError("No evaluation results found and no test data directory specified")
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation suite"""
        start_time = datetime.now()
        
        try:
            self.logger.info("🚀 Starting comprehensive model evaluation")
            self.logger.info("=" * 60)
            
            # Load model
            if not self.model_loader.load_model():
                raise ValueError("Failed to load model")
            
            # Load test data
            test_images, test_labels = self.load_test_data()
            
            if not test_images:
                self.logger.warning("⚠️ No test images available for evaluation")
                # Use saved evaluation results instead
                eval_results_path = Path(self.config.model_dir) / "evaluation_results.json"
                if eval_results_path.exists():
                    with open(eval_results_path, 'r') as f:
                        saved_results = json.load(f)
                    
                    self.evaluation_results = {
                        'using_saved_results': True,
                        'saved_evaluation': saved_results,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Still perform speed profiling
                    if self.config.profile_inference_speed:
                        speed_profile = self.performance_analyzer.profile_inference_speed(self.model_loader)
                        self.evaluation_results['speed_profile'] = speed_profile
                    
                    return self.evaluation_results
            
            # Make predictions
            self.logger.info("🎯 Making predictions on test data...")
            predicted_labels, predicted_probabilities = self.model_loader.predict(test_images)
            
            # Encode true labels for analysis
            true_labels_encoded = self.model_loader.label_encoder.transform(test_labels)
            pred_labels_encoded = self.model_loader.label_encoder.transform(predicted_labels)
            
            class_names = self.model_loader.label_encoder.classes_.tolist()
            
            # Calculate detailed metrics
            detailed_metrics = self.performance_analyzer.calculate_detailed_metrics(
                true_labels_encoded, pred_labels_encoded, predicted_probabilities, class_names
            )
            
            # Feature importance analysis
            if self.config.calculate_feature_importance and hasattr(self.model_loader, 'feature_extractor'):
                # Extract features for importance analysis (subset for speed)
                subset_size = min(100, len(test_images))
                subset_indices = np.random.choice(len(test_images), subset_size, replace=False)
                subset_images = [test_images[i] for i in subset_indices]
                subset_labels = true_labels_encoded[subset_indices]
                
                features = self.model_loader.feature_extractor.extract_batch(subset_images)
                if self.model_loader.scaler:
                    features = self.model_loader.scaler.transform(features)
                if self.model_loader.dimensionality_reducer:
                    features = self.model_loader.dimensionality_reducer.transform(features)
                
                feature_names = self.model_loader.feature_extractor.get_feature_names()
                if self.model_loader.dimensionality_reducer:
                    feature_names = [f"component_{i}" for i in range(features.shape[1])]
                
                feature_importance = self.performance_analyzer.analyze_feature_importance(
                    self.model_loader.ensemble_model, features, subset_labels, feature_names
                )
            else:
                feature_importance = {}
            
            # Error analysis
            error_analysis = self.performance_analyzer.analyze_errors(
                true_labels_encoded, pred_labels_encoded, predicted_probabilities,
                test_images, class_names
            )
            
            # Speed profiling
            speed_profile = self.performance_analyzer.profile_inference_speed(self.model_loader)
            
            # Generate visualizations
            visualization_paths = {}
            if self.visualization_generator:
                self.logger.info("📊 Generating visualizations...")
                
                # Confusion matrix
                cm_path = self.visualization_generator.create_confusion_matrix_plot(
                    np.array(detailed_metrics['confusion_matrix']), class_names
                )
                if cm_path:
                    visualization_paths['confusion_matrix'] = cm_path
                
                # Feature importance
                fi_path = self.visualization_generator.create_feature_importance_plot(feature_importance)
                if fi_path:
                    visualization_paths['feature_importance'] = fi_path
                
                # Class performance
                cp_path = self.visualization_generator.create_class_performance_plot(detailed_metrics)
                if cp_path:
                    visualization_paths['class_performance'] = cp_path
            
            # Compile results
            self.evaluation_results = {
                'timestamp': datetime.now().isoformat(),
                'model_info': {
                    'model_path': str(self.config.model_dir),
                    'n_classes': len(class_names),
                    'class_names': class_names
                },
                'test_data_info': {
                    'n_samples': len(test_images),
                    'data_source': self.config.test_data_dir or 'holdout_test_set'
                },
                'detailed_metrics': detailed_metrics,
                'feature_importance': feature_importance,
                'error_analysis': error_analysis,
                'speed_profile': speed_profile,
                'visualization_paths': visualization_paths
            }
            
            # Save results
            self.save_evaluation_report()
            
            end_time = datetime.now()
            evaluation_time = (end_time - start_time).total_seconds()
            
            self.logger.info("=" * 60)
            self.logger.info(f"🎉 Evaluation completed in {evaluation_time:.2f} seconds")
            self.logger.info(f"📊 Test Accuracy: {detailed_metrics['accuracy']:.4f}")
            self.logger.info(f"📊 Balanced Accuracy: {detailed_metrics['balanced_accuracy']:.4f}")
            self.logger.info(f"📊 F1 (macro): {detailed_metrics['f1_macro']:.4f}")
            
            return self.evaluation_results
            
        except Exception as e:
            self.logger.error(f"❌ Evaluation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def save_evaluation_report(self) -> None:
        """Save comprehensive evaluation report"""
        # Save JSON report
        json_path = self.report_dir / "evaluation_report.json"
        with open(json_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        self.logger.info(f"💾 Evaluation report saved: {json_path}")
        
        # Save human-readable report
        if self.config.create_detailed_report:
            self.create_detailed_report()
    
    def create_detailed_report(self) -> None:
        """Create detailed human-readable evaluation report"""
        report_content = []
        
        # Header
        report_content.append("# 🔍 Comprehensive Model Evaluation Report")
        report_content.append(f"**Generated:** {self.evaluation_results['timestamp']}")
        report_content.append(f"**Model:** {self.evaluation_results['model_info']['model_path']}")
        report_content.append("")
        
        # Model Information
        model_info = self.evaluation_results['model_info']
        report_content.append("## 🏗️ Model Information")
        report_content.append(f"- **Classes:** {model_info['n_classes']}")
        report_content.append(f"- **Class Names:** {', '.join(model_info['class_names'])}")
        report_content.append("")
        
        # Test Data Information
        test_info = self.evaluation_results['test_data_info']
        report_content.append("## 📊 Test Data Information")
        report_content.append(f"- **Samples:** {test_info['n_samples']}")
        report_content.append(f"- **Data Source:** {test_info['data_source']}")
        report_content.append("")
        
        # Performance Metrics
        metrics = self.evaluation_results['detailed_metrics']
        report_content.append("## 📈 Performance Metrics")
        report_content.append(f"- **Accuracy:** {metrics['accuracy']:.4f}")
        report_content.append(f"- **Balanced Accuracy:** {metrics['balanced_accuracy']:.4f}")
        report_content.append(f"- **F1-Score (Macro):** {metrics['f1_macro']:.4f}")
        report_content.append(f"- **F1-Score (Weighted):** {metrics['f1_weighted']:.4f}")
        report_content.append("")
        
        # Per-class performance
        report_content.append("### Per-Class Performance")
        report_content.append("| Class | Precision | Recall | F1-Score | Support |")
        report_content.append("|-------|-----------|--------|----------|---------|")
        
        for class_name, class_metrics in metrics['per_class'].items():
            report_content.append(
                f"| {class_name} | {class_metrics['precision']:.3f} | "
                f"{class_metrics['recall']:.3f} | {class_metrics['f1_score']:.3f} | "
                f"{class_metrics['support']} |"
            )
        
        report_content.append("")
        
        # Speed Performance
        if 'speed_profile' in self.evaluation_results:
            speed = self.evaluation_results['speed_profile']
            report_content.append("## ⚡ Speed Performance")
            report_content.append(f"- **Images per Second:** {speed['images_per_second']:.2f}")
            report_content.append(f"- **Time per Image:** {speed['ms_per_image']:.2f} ms")
            report_content.append(f"- **Feature Extraction:** {speed['breakdown']['feature_extraction_pct']:.1f}%")
            report_content.append(f"- **Preprocessing:** {speed['breakdown']['preprocessing_pct']:.1f}%")
            report_content.append(f"- **Prediction:** {speed['breakdown']['prediction_pct']:.1f}%")
            report_content.append("")
        
        # Error Analysis
        if 'error_analysis' in self.evaluation_results and self.evaluation_results['error_analysis']:
            error_analysis = self.evaluation_results['error_analysis']
            if not error_analysis.get('no_errors', False):
                report_content.append("## 🔍 Error Analysis")
                report_content.append(f"- **Total Errors:** {error_analysis['total_errors']}")
                report_content.append(f"- **Error Rate:** {error_analysis['error_rate']:.4f}")
                
                if 'error_patterns' in error_analysis:
                    report_content.append("\n### Most Common Error Patterns")
                    for pattern, data in list(error_analysis['error_patterns'].items())[:5]:
                        report_content.append(f"- **{pattern}:** {data['count']} cases")
                
                report_content.append("")
        
        # Save markdown report
        md_path = self.report_dir / "evaluation_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        self.logger.info(f"📝 Detailed report saved: {md_path}")


def main():
    """CLI entry point for evaluation suite"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="📊 Comprehensive Model Evaluation Suite"
    )
    
    parser.add_argument(
        "--model-dir", 
        default="ai_models/saved_models/hybrid_amulet_classifier",
        help="Directory containing trained model"
    )
    
    parser.add_argument(
        "--test-data-dir",
        help="Directory containing test data (if different from training holdout)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="evaluation_reports",
        help="Output directory for evaluation reports"
    )
    
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip generating visualizations"
    )
    
    parser.add_argument(
        "--no-feature-importance",
        action="store_true", 
        help="Skip feature importance analysis"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick evaluation (reduced analysis)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = EvaluationConfig(
        model_dir=args.model_dir,
        test_data_dir=args.test_data_dir,
        report_dir=args.output_dir,
        generate_visualizations=not args.no_visualizations,
        calculate_feature_importance=not args.no_feature_importance
    )
    
    if args.quick:
        config.n_speed_test_samples = 20
        config.max_error_examples = 3
    
    # Run evaluation
    evaluator = EvaluationSuite(config)
    results = evaluator.run_comprehensive_evaluation()
    
    # Print summary
    if results.get('success', True):
        print(f"✅ Evaluation completed! Reports saved to: {args.output_dir}")
        if 'detailed_metrics' in results:
            metrics = results['detailed_metrics']
            print(f"📊 Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"📊 Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            print(f"📊 F1 (macro): {metrics['f1_macro']:.4f}")
        exit(0)
    else:
        print(f"❌ Evaluation failed: {results.get('error', 'Unknown error')}")
        exit(1)


if __name__ == "__main__":
    main()