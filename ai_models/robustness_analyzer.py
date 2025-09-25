#!/usr/bin/env python3
"""
🔍 Robustness & Out-of-Distribution Detection System
ระบบตรวจจับและจัดการ:
1. ความแตกต่างภายใน class เดียวกัน (Intra-class Variation)
2. การส่งภาพที่ไม่ใช่ target object (Out-of-Distribution Detection)

Author: AI Assistant
Compatible: Python 3.13+
"""

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope

# Our modules
from feature_extractors import HybridFeatureExtractor, FeatureConfig


@dataclass
class RobustnessConfig:
    """Configuration for robustness analysis"""
    
    # Intra-class variation analysis
    analyze_intra_class_variation: bool = True
    variation_threshold: float = 0.3  # Threshold for acceptable variation
    
    # Out-of-distribution detection (Fast mode settings)
    enable_ood_detection: bool = True
    ood_method: str = "isolation_forest"  # Using fastest method only
    ood_contamination: float = 0.1  # Expected proportion of outliers
    ood_confidence_threshold: float = 0.5  # Minimum confidence for in-distribution
    
    # Robustness testing (Simplified for speed)
    enable_augmentation_testing: bool = False  # Disabled for fast mode
    test_lighting_changes: bool = False
    test_rotation_changes: bool = False
    test_blur_changes: bool = False
    test_noise_changes: bool = False
    
    # Output settings
    save_analysis_plots: bool = True
    verbose: bool = True


class IntraClassVariationAnalyzer:
    """วิเคราะห์ความแตกต่างภายใน class เดียวกัน"""
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('IntraClassVariationAnalyzer')
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_class_variations(self, data_dir: str, feature_extractor: HybridFeatureExtractor) -> Dict[str, Any]:
        """วิเคราะห์ความแตกต่างในแต่ละ class"""
        self.logger.info("🔍 Analyzing intra-class variations...")
        
        data_path = Path(data_dir)
        results = {}
        
        for class_dir in data_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            self.logger.info(f"  📂 Processing class: {class_name}")
            
            # โหลดภาพทั้งหมดในคลาส
            images = []
            image_paths = []
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            
            for ext in valid_extensions:
                for img_path in class_dir.glob(f"*{ext}"):
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            images.append(img_rgb)
                            image_paths.append(str(img_path))
                    except Exception as e:
                        self.logger.warning(f"Failed to load {img_path}: {e}")
            
            if len(images) < 2:
                self.logger.warning(f"  ⚠️ Insufficient images in {class_name}: {len(images)}")
                continue
            
            # สกัดฟีเจอร์
            self.logger.info(f"  🧠 Extracting features for {len(images)} images...")
            features = feature_extractor.extract_batch(images)
            
            # วิเคราะห์ความแตกต่าง
            class_analysis = self._analyze_feature_variations(features, class_name, image_paths)
            results[class_name] = class_analysis
        
        return results
    
    def _analyze_feature_variations(self, features: np.ndarray, class_name: str, 
                                  image_paths: List[str]) -> Dict[str, Any]:
        """วิเคราะห์ความแตกต่างของฟีเจอร์ในคลาส"""
        
        # คำนวณสถิติพื้นฐาน
        feature_mean = np.mean(features, axis=0)
        feature_std = np.std(features, axis=0)
        feature_var = np.var(features, axis=0)
        
        # คำนวณระยะห่างระหว่างตัวอย่าง
        distances = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                dist = np.linalg.norm(features[i] - features[j])
                distances.append(dist)
        
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Silhouette score (ถ้ามีมากกว่า 1 คลาส)
        try:
            # สร้าง dummy labels สำหรับ silhouette score
            labels = np.zeros(len(features))
            silhouette_avg = silhouette_score(features, labels) if len(features) > 1 else 0.0
        except:
            silhouette_avg = 0.0
        
        # ค้นหา outliers ภายในคลาส
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = isolation_forest.fit_predict(features)
        outlier_indices = np.where(outlier_labels == -1)[0]
        
        # คำนวณ variation score
        variation_score = np.mean(feature_std) / (np.mean(feature_mean) + 1e-8)
        
        analysis = {
            'n_samples': len(features),
            'feature_dim': features.shape[1],
            'variation_metrics': {
                'avg_distance': float(avg_distance),
                'std_distance': float(std_distance),
                'variation_score': float(variation_score),
                'silhouette_score': float(silhouette_avg),
                'mean_feature_std': float(np.mean(feature_std)),
                'max_feature_std': float(np.max(feature_std))
            },
            'outliers': {
                'n_outliers': len(outlier_indices),
                'outlier_ratio': len(outlier_indices) / len(features),
                'outlier_paths': [image_paths[i] for i in outlier_indices] if len(outlier_indices) > 0 else []
            },
            'robustness_assessment': {
                'is_robust': variation_score < self.config.variation_threshold,
                'variation_level': 'Low' if variation_score < 0.2 else 'Medium' if variation_score < 0.4 else 'High',
                'recommendations': self._get_robustness_recommendations(variation_score, len(outlier_indices), len(features))
            }
        }
        
        self.logger.info(f"    📊 {class_name}: variation={variation_score:.3f}, outliers={len(outlier_indices)}/{len(features)}")
        
        return analysis
    
    def _get_robustness_recommendations(self, variation_score: float, n_outliers: int, n_samples: int) -> List[str]:
        """ให้คำแนะนำสำหรับการปรับปรุง robustness"""
        recommendations = []
        
        if variation_score > self.config.variation_threshold:
            recommendations.append("High intra-class variation detected. Consider data cleaning or feature engineering.")
        
        if n_outliers / n_samples > 0.2:
            recommendations.append("High number of outliers detected. Review outlier samples for data quality.")
        
        if n_samples < 50:
            recommendations.append("Insufficient training samples. Consider data augmentation.")
        
        if variation_score < 0.1:
            recommendations.append("Very low variation. May indicate duplicate or overly similar samples.")
        
        return recommendations


class OutOfDistributionDetector:
    """ตรวจจับ Out-of-Distribution samples"""
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # OOD detectors
        self.ood_detectors = {}
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('OutOfDistributionDetector')
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def fit(self, in_distribution_features: np.ndarray, class_labels: np.ndarray) -> None:
        """ฝึก OOD detector ด้วยข้อมูล in-distribution (Optimized version)"""
        self.logger.info("🎯 Training Out-of-Distribution detectors (Fast mode)...")
        
        # Sample data for faster training if dataset is large
        n_samples = len(in_distribution_features)
        if n_samples > 500:
            self.logger.info(f"⚡ Using subset of {min(500, n_samples)} samples for faster training")
            indices = np.random.choice(n_samples, min(500, n_samples), replace=False)
            features_subset = in_distribution_features[indices]
        else:
            features_subset = in_distribution_features
        
        # Normalize features
        features_scaled = self.feature_scaler.fit_transform(features_subset)
        
        # Use only the fastest method - Isolation Forest
        self.logger.info("🚀 Using Isolation Forest (fastest method)")
        self.ood_detectors['isolation_forest'] = IsolationForest(
            n_estimators=50,  # Reduced from default 100
            contamination=self.config.ood_contamination,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        self.ood_detectors['isolation_forest'].fit(features_scaled)
        
        self.is_fitted = True
        self.logger.info(f"✅ OOD detector trained on {len(features_subset)} samples (Fast mode)")
    
    def _create_ensemble_detector(self, features: np.ndarray) -> None:
        """สร้าง ensemble ของ OOD detectors (Simplified for speed)"""
        # Skip ensemble creation in fast mode - use only primary detector
        self.logger.info("⚡ Skipping ensemble creation for faster execution")
        pass
    
    def detect_ood(self, test_features: np.ndarray) -> Dict[str, Any]:
        """ตรวจจับ OOD samples"""
        if not self.is_fitted:
            raise ValueError("OOD detector not fitted. Call fit() first.")
        
        # Normalize test features
        test_features_scaled = self.feature_scaler.transform(test_features)
        
        results = {
            'n_samples': len(test_features),
            'detectors': {},
            'ensemble_results': {}
        }
        
        # รัน detector แต่ละตัว
        for detector_name, detector in self.ood_detectors.items():
            if detector_name == 'ensemble':
                continue
                
            if detector_name == 'dbscan':
                # สำหรับ DBSCAN ใช้ distance-based approach
                predictions = self._dbscan_predict(test_features_scaled, detector)
            else:
                predictions = detector.predict(test_features_scaled)
                
            # แปลงเป็น binary labels (1 = in-distribution, -1 = out-of-distribution)
            ood_labels = predictions == -1
            
            results['detectors'][detector_name] = {
                'predictions': predictions.tolist(),
                'ood_count': int(np.sum(ood_labels)),
                'ood_ratio': float(np.mean(ood_labels))
            }
        
        # Skip ensemble voting in fast mode
        self.logger.info("⚡ Skipping ensemble voting for faster execution")
        
        return results
    
    def _dbscan_predict(self, test_features: np.ndarray, dbscan_info: Dict) -> np.ndarray:
        """Predict using DBSCAN-based approach"""
        core_samples = dbscan_info['core_samples']
        eps = dbscan_info['eps']
        
        predictions = []
        for test_sample in test_features:
            # คำนวณระยะห่างไปยัง core samples
            distances = np.linalg.norm(core_samples - test_sample, axis=1)
            min_distance = np.min(distances)
            
            # ถ้าระยะห่างน้อยกว่า eps ถือว่า in-distribution
            if min_distance <= eps:
                predictions.append(1)
            else:
                predictions.append(-1)
        
        return np.array(predictions)
    
    def _calculate_confidence_scores(self, test_features: np.ndarray) -> List[float]:
        """คำนวณคะแนนความเชื่อมั่น"""
        if 'isolation_forest' in self.ood_detectors:
            # ใช้ anomaly score จาก Isolation Forest
            scores = self.ood_detectors['isolation_forest'].decision_function(test_features)
            # แปลงเป็น confidence (0-1)
            confidence = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            return confidence.tolist()
        else:
            # Default confidence
            return [0.5] * len(test_features)


class RobustnessTestSuite:
    """ทดสอบความแข็งแกร่งของโมเดลต่อการเปลี่ยนแปลงต่าง ๆ"""
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('RobustnessTestSuite')
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def test_model_robustness(self, images: List[np.ndarray], 
                            model_predict_func, class_names: List[str]) -> Dict[str, Any]:
        """ทดสอบความแข็งแกร่งของโมเดล"""
        self.logger.info("🧪 Testing model robustness...")
        
        results = {
            'n_test_images': len(images),
            'test_results': {}
        }
        
        # ทดสอบแต่ละประเภท
        if self.config.test_lighting_changes:
            results['test_results']['lighting'] = self._test_lighting_robustness(
                images, model_predict_func, class_names
            )
        
        if self.config.test_rotation_changes:
            results['test_results']['rotation'] = self._test_rotation_robustness(
                images, model_predict_func, class_names
            )
        
        if self.config.test_blur_changes:
            results['test_results']['blur'] = self._test_blur_robustness(
                images, model_predict_func, class_names
            )
        
        if self.config.test_noise_changes:
            results['test_results']['noise'] = self._test_noise_robustness(
                images, model_predict_func, class_names
            )
        
        # สรุปผลรวม
        results['overall_assessment'] = self._assess_overall_robustness(results['test_results'])
        
        return results
    
    def _test_lighting_robustness(self, images: List[np.ndarray], 
                                model_predict_func, class_names: List[str]) -> Dict[str, Any]:
        """ทดสอบความแข็งแกร่งต่อการเปลี่ยนแปลงแสง"""
        self.logger.info("  💡 Testing lighting robustness...")
        
        brightness_factors = [0.5, 0.7, 1.0, 1.3, 1.5]  # 1.0 = original
        results = {}
        
        for factor in brightness_factors:
            modified_images = []
            for img in images:
                # ปรับความสว่าง
                bright_img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
                modified_images.append(bright_img)
            
            # ทำนาย
            predictions = model_predict_func(modified_images)
            
            # เปรียบเทียบกับ original
            original_predictions = model_predict_func(images)
            consistency = np.mean([
                orig == pred for orig, pred in zip(original_predictions, predictions)
            ])
            
            results[f'brightness_{factor}'] = {
                'consistency': float(consistency),
                'predictions': predictions
            }
        
        return results
    
    def _test_rotation_robustness(self, images: List[np.ndarray], 
                                model_predict_func, class_names: List[str]) -> Dict[str, Any]:
        """ทดสอบความแข็งแกร่งต่อการหมุน"""
        self.logger.info("  🔄 Testing rotation robustness...")
        
        rotation_angles = [-15, -5, 0, 5, 15]  # degrees
        results = {}
        
        for angle in rotation_angles:
            modified_images = []
            for img in images:
                # หมุนภาพ
                h, w = img.shape[:2]
                center = (w//2, h//2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_img = cv2.warpAffine(img, matrix, (w, h))
                modified_images.append(rotated_img)
            
            # ทำนาย
            predictions = model_predict_func(modified_images)
            
            # เปรียบเทียบกับ original
            original_predictions = model_predict_func(images)
            consistency = np.mean([
                orig == pred for orig, pred in zip(original_predictions, predictions)
            ])
            
            results[f'rotation_{angle}'] = {
                'consistency': float(consistency),
                'predictions': predictions
            }
        
        return results
    
    def _test_blur_robustness(self, images: List[np.ndarray], 
                            model_predict_func, class_names: List[str]) -> Dict[str, Any]:
        """ทดสอบความแข็งแกร่งต่อความเบลอ"""
        self.logger.info("  🌫️ Testing blur robustness...")
        
        blur_kernels = [1, 3, 5, 7]  # kernel sizes
        results = {}
        
        for kernel_size in blur_kernels:
            modified_images = []
            for img in images:
                # เบลอภาพ
                if kernel_size == 1:
                    blurred_img = img  # No blur
                else:
                    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
                modified_images.append(blurred_img)
            
            # ทำนาย
            predictions = model_predict_func(modified_images)
            
            # เปรียบเทียบกับ original
            original_predictions = model_predict_func(images)
            consistency = np.mean([
                orig == pred for orig, pred in zip(original_predictions, predictions)
            ])
            
            results[f'blur_{kernel_size}'] = {
                'consistency': float(consistency),
                'predictions': predictions
            }
        
        return results
    
    def _test_noise_robustness(self, images: List[np.ndarray], 
                             model_predict_func, class_names: List[str]) -> Dict[str, Any]:
        """ทดสอบความแข็งแกร่งต่อ noise"""
        self.logger.info("  📡 Testing noise robustness...")
        
        noise_levels = [0, 10, 20, 30]  # noise standard deviation
        results = {}
        
        for noise_std in noise_levels:
            modified_images = []
            for img in images:
                if noise_std == 0:
                    noisy_img = img  # No noise
                else:
                    # เพิ่ม Gaussian noise
                    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
                    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                modified_images.append(noisy_img)
            
            # ทำนาย
            predictions = model_predict_func(modified_images)
            
            # เปรียบเทียบกับ original
            original_predictions = model_predict_func(images)
            consistency = np.mean([
                orig == pred for orig, pred in zip(original_predictions, predictions)
            ])
            
            results[f'noise_{noise_std}'] = {
                'consistency': float(consistency),
                'predictions': predictions
            }
        
        return results
    
    def _assess_overall_robustness(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """ประเมินความแข็งแกร่งโดยรวม"""
        all_consistencies = []
        
        for test_type, test_data in test_results.items():
            for condition, result in test_data.items():
                if 'consistency' in result:
                    all_consistencies.append(result['consistency'])
        
        if not all_consistencies:
            return {'overall_robustness': 0.0, 'robustness_level': 'Unknown'}
        
        avg_consistency = np.mean(all_consistencies)
        min_consistency = np.min(all_consistencies)
        
        # ประเมินระดับ
        if avg_consistency >= 0.9:
            level = 'Excellent'
        elif avg_consistency >= 0.8:
            level = 'Good'
        elif avg_consistency >= 0.7:
            level = 'Fair'
        else:
            level = 'Poor'
        
        return {
            'overall_robustness': float(avg_consistency),
            'min_robustness': float(min_consistency),
            'robustness_level': level,
            'n_tests': len(all_consistencies)
        }


class ComprehensiveRobustnessAnalyzer:
    """ระบบวิเคราะห์ความแข็งแกร่งแบบครอบคลุม"""
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.variation_analyzer = IntraClassVariationAnalyzer(config)
        self.ood_detector = OutOfDistributionDetector(config)
        self.robustness_tester = RobustnessTestSuite(config)
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('ComprehensiveRobustnessAnalyzer')
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_dataset_robustness(self, data_dir: str, feature_extractor: HybridFeatureExtractor,
                                 model_predict_func=None) -> Dict[str, Any]:
        """วิเคราะห์ความแข็งแกร่งของ dataset และโมเดล"""
        self.logger.info("🚀 Starting comprehensive robustness analysis...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'data_dir': data_dir,
                'variation_threshold': self.config.variation_threshold,
                'ood_contamination': self.config.ood_contamination
            }
        }
        
        # 1. วิเคราะห์ intra-class variation
        if self.config.analyze_intra_class_variation:
            self.logger.info("📊 Step 1: Analyzing intra-class variations...")
            variation_results = self.variation_analyzer.analyze_class_variations(
                data_dir, feature_extractor
            )
            results['intra_class_analysis'] = variation_results
        
        # 2. ฝึกและทดสอบ OOD detection
        if self.config.enable_ood_detection:
            self.logger.info("🎯 Step 2: Setting up OOD detection...")
            
            # โหลดข้อมูลทั้งหมดเพื่อฝึก OOD detector
            all_features, all_labels, all_images = self._load_all_data(data_dir, feature_extractor)
            
            if len(all_features) > 0:
                # ฝึก OOD detector
                self.ood_detector.fit(all_features, all_labels)
                
                # ทดสอบ OOD detection กับข้อมูลชุดเดียวกัน (สำหรับ baseline)
                ood_results = self.ood_detector.detect_ood(all_features)
                results['ood_detection'] = ood_results
                
                # ทดสอบกับ synthetic out-of-distribution samples
                synthetic_ood_results = self._test_synthetic_ood(feature_extractor)
                results['synthetic_ood_test'] = synthetic_ood_results
        
        # 3. ทดสอบ robustness ของโมเดล (ถ้ามี model_predict_func)
        if model_predict_func and self.config.enable_augmentation_testing:
            self.logger.info("🧪 Step 3: Testing model robustness...")
            
            if 'all_images' in locals():
                # ใช้ subset ของภาพสำหรับทดสอบ
                test_images = all_images[:min(20, len(all_images))]  # จำกัดจำนวนเพื่อความเร็ว
                class_names = list(set(all_labels))
                
                robustness_results = self.robustness_tester.test_model_robustness(
                    test_images, model_predict_func, class_names
                )
                results['robustness_testing'] = robustness_results
        
        # 4. สรุปและคำแนะนำ
        results['summary'] = self._generate_summary_recommendations(results)
        
        self.logger.info("✅ Comprehensive robustness analysis completed!")
        return results
    
    def _load_all_data(self, data_dir: str, feature_extractor: HybridFeatureExtractor) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """โหลดข้อมูลทั้งหมดสำหรับการวิเคราะห์ (Fast mode - limited samples)"""
        data_path = Path(data_dir)
        all_features = []
        all_labels = []
        all_images = []
        
        label_to_idx = {}
        current_idx = 0
        max_samples_per_class = 30  # จำกัดจำนวนตัวอย่างต่อ class เพื่อความเร็ว
        
        self.logger.info(f"⚡ Fast mode: Loading max {max_samples_per_class} samples per class")
        
        for class_dir in data_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            if class_name not in label_to_idx:
                label_to_idx[class_name] = current_idx
                current_idx += 1
            
            # โหลดภาพ (จำกัดจำนวน)
            images = []
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            sample_count = 0
            
            for ext in valid_extensions:
                if sample_count >= max_samples_per_class:
                    break
                for img_path in class_dir.glob(f"*{ext}"):
                    if sample_count >= max_samples_per_class:
                        break
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            images.append(img_rgb)
                            all_images.append(img_rgb)
                            all_labels.append(label_to_idx[class_name])
                            sample_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to load {img_path}: {e}")
            
            if len(images) > 0:
                # สกัดฟีเจอร์
                self.logger.info(f"Extracting features for {len(images)} images from {class_name}")
                features = feature_extractor.extract_batch(images)
                all_features.extend(features)
        
        self.logger.info(f"✅ Loaded {len(all_features)} samples for robustness analysis")
        return np.array(all_features), np.array(all_labels), all_images
    
    def _test_synthetic_ood(self, feature_extractor: HybridFeatureExtractor) -> Dict[str, Any]:
        """ทดสอบ OOD detection ด้วย synthetic samples"""
        self.logger.info("  🧪 Testing with synthetic OOD samples...")
        
        # สร้าง synthetic images ที่แตกต่างจาก training data
        synthetic_images = []
        
        # ภาพสีเดียว
        for color in [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)]:
            img = np.full((224, 224, 3), color, dtype=np.uint8)
            synthetic_images.append(img)
        
        # ภาพ noise
        for _ in range(5):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            synthetic_images.append(img)
        
        # ภาพ gradient
        for direction in ['horizontal', 'vertical']:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            if direction == 'horizontal':
                for i in range(224):
                    img[:, i, :] = i
            else:
                for i in range(224):
                    img[i, :, :] = i
            synthetic_images.append(img)
        
        # สกัดฟีเจอร์
        synthetic_features = feature_extractor.extract_batch(synthetic_images)
        
        # ทดสอบ OOD detection
        ood_results = self.ood_detector.detect_ood(synthetic_features)
        
        # Get OOD ratio from available detector (since ensemble is disabled)
        available_detectors = list(ood_results['detectors'].keys())
        if available_detectors:
            detector_name = available_detectors[0]
            ood_ratio = ood_results['detectors'][detector_name]['ood_ratio']
            self.logger.info(f"    📊 Synthetic OOD detection ({detector_name}): {ood_ratio:.2f} detected as OOD")
        else:
            self.logger.warning("    ⚠️ No OOD detection results available")
        
        return {
            'n_synthetic_samples': len(synthetic_images),
            'ood_detection_results': ood_results,
            'expected_ood_ratio': 1.0  # All synthetic samples should be detected as OOD
        }
    
    def _generate_summary_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """สร้างสรุปและคำแนะนำ"""
        recommendations = []
        overall_score = 0.0
        scores = []
        
        # วิเคราะห์ intra-class variation
        if 'intra_class_analysis' in results:
            variation_scores = []
            high_variation_classes = []
            
            for class_name, analysis in results['intra_class_analysis'].items():
                var_score = analysis['variation_metrics']['variation_score']
                variation_scores.append(var_score)
                
                if not analysis['robustness_assessment']['is_robust']:
                    high_variation_classes.append(class_name)
            
            if variation_scores:
                avg_variation = np.mean(variation_scores)
                scores.append(max(0, 1 - avg_variation))  # Lower variation = higher score
                
                if high_variation_classes:
                    recommendations.append(f"High intra-class variation detected in: {', '.join(high_variation_classes)}")
                    recommendations.append("Consider data cleaning, feature engineering, or collecting more diverse samples")
        
        # วิเคราะห์ OOD detection
        if 'ood_detection' in results:
            ood_ratio = results['ood_detection'].get('ensemble_results', {}).get('ood_ratio', 0)
            scores.append(max(0, 1 - ood_ratio * 2))  # Lower OOD ratio in training = higher score
            
            if ood_ratio > 0.2:
                recommendations.append(f"High OOD ratio ({ood_ratio:.1%}) in training data detected")
                recommendations.append("Review and clean training data for quality issues")
        
        # วิเคราะห์ robustness testing
        if 'robustness_testing' in results:
            overall_robustness = results['robustness_testing'].get('overall_assessment', {}).get('overall_robustness', 0)
            scores.append(overall_robustness)
            
            level = results['robustness_testing'].get('overall_assessment', {}).get('robustness_level', 'Unknown')
            if level in ['Poor', 'Fair']:
                recommendations.append(f"Model robustness is {level.lower()}")
                recommendations.append("Consider data augmentation or robust training techniques")
        
        # คำนวณคะแนนรวม
        if scores:
            overall_score = np.mean(scores)
        
        # ประเมินระดับโดยรวม
        if overall_score >= 0.8:
            overall_level = 'Excellent'
        elif overall_score >= 0.7:
            overall_level = 'Good'
        elif overall_score >= 0.6:
            overall_level = 'Fair'
        else:
            overall_level = 'Needs Improvement'
        
        return {
            'overall_score': float(overall_score),
            'overall_level': overall_level,
            'recommendations': recommendations,
            'component_scores': {
                'variation_robustness': scores[0] if len(scores) > 0 else None,
                'ood_detection': scores[1] if len(scores) > 1 else None,
                'model_robustness': scores[2] if len(scores) > 2 else None
            }
        }


def main():
    """CLI entry point สำหรับ robustness analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🔍 Comprehensive Robustness & OOD Detection Analysis"
    )
    
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing dataset"
    )
    
    parser.add_argument(
        "--output-dir",
        default="robustness_analysis",
        help="Output directory for analysis results"
    )
    
    parser.add_argument(
        "--model-dir",
        help="Directory containing trained model (optional)"
    )
    
    parser.add_argument(
        "--variation-threshold",
        type=float,
        default=0.3,
        help="Threshold for acceptable intra-class variation"
    )
    
    parser.add_argument(
        "--ood-contamination",
        type=float,
        default=0.1,
        help="Expected proportion of outliers for OOD detection"
    )
    
    parser.add_argument(
        "--test-robustness",
        action="store_true",
        help="Enable model robustness testing (requires trained model)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = RobustnessConfig(
        variation_threshold=args.variation_threshold,
        ood_contamination=args.ood_contamination,
        enable_augmentation_testing=args.test_robustness
    )
    
    # Initialize analyzer
    analyzer = ComprehensiveRobustnessAnalyzer(config)
    
    # Initialize feature extractor
    feature_config = FeatureConfig(
        enable_hog=True,
        enable_lbp=True,
        enable_color_hist=True,
        enable_caching=True
    )
    feature_extractor = HybridFeatureExtractor(feature_config)
    
    # Load model if provided
    model_predict_func = None
    if args.model_dir:
        try:
            from evaluation_suite import ModelLoader
            model_loader = ModelLoader(args.model_dir)
            if model_loader.load_model():
                def predict_function(images):
                    predicted_labels, _ = model_loader.predict(images)
                    return predicted_labels
                model_predict_func = predict_function
                print("✅ Model loaded successfully for robustness testing")
            else:
                print("⚠️ Failed to load model, skipping robustness testing")
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
    
    # Run analysis
    print("🚀 Starting comprehensive robustness analysis...")
    results = analyzer.analyze_dataset_robustness(
        args.data_dir, feature_extractor, model_predict_func
    )
    
    # Save results with proper JSON serialization
    results_file = output_dir / "robustness_analysis.json"
    
    def convert_to_serializable(obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 ROBUSTNESS ANALYSIS SUMMARY")
    print("="*60)
    
    summary = results.get('summary', {})
    print(f"Overall Score: {summary.get('overall_score', 0):.2f}/1.0")
    print(f"Overall Level: {summary.get('overall_level', 'Unknown')}")
    
    if summary.get('recommendations'):
        print("\n🎯 Recommendations:")
        for i, rec in enumerate(summary['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    print("✅ Analysis completed!")


if __name__ == "__main__":
    main()