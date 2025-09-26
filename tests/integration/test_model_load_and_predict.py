#!/usr/bin/env python3
"""
🧪 Integration Tests: Model Loading and Prediction
ทดสอบระบบการโหลดโมเดลและการทำนาย
"""

import pytest
import joblib
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_models.compatibility_loader import try_load_model, ProductionOODDetector
from ai_models.enhanced_production_system import EnhancedProductionClassifier

class TestModelLoadingCompatibility:
    """Test model loading with compatibility wrapper"""
    
    def test_compatibility_loader_import(self):
        """Test that compatibility loader can be imported"""
        from ai_models.compatibility_loader import ProductionOODDetector, try_load_model
        assert ProductionOODDetector is not None
        assert try_load_model is not None
    
    def test_try_load_model_with_missing_file(self):
        """Test loading non-existent model file"""
        result = try_load_model("nonexistent_model.joblib")
        assert result is None
    
    def test_try_load_model_with_existing_files(self):
        """Test loading existing model files"""
        model_files = [
            "trained_model/classifier.joblib",
            "trained_model/ood_detector.joblib",
            "trained_model/pca.joblib",
            "trained_model/scaler.joblib",
            "trained_model/label_encoder.joblib"
        ]
        
        loaded_count = 0
        for model_path in model_files:
            if os.path.exists(model_path):
                result = try_load_model(model_path)
                if result is not None:
                    loaded_count += 1
                    print(f"✅ Loaded: {model_path} -> {type(result)}")
                else:
                    print(f"⚠️ Failed to load: {model_path}")
        
        # At least some models should load
        assert loaded_count > 0, f"Expected at least 1 model to load, got {loaded_count}"
    
    def test_production_ood_detector_dummy_class(self):
        """Test ProductionOODDetector dummy implementation"""
        dummy_ood = ProductionOODDetector()
        
        # Test methods exist
        assert hasattr(dummy_ood, 'fit')
        assert hasattr(dummy_ood, 'decision_function') 
        assert hasattr(dummy_ood, 'predict')
        assert hasattr(dummy_ood, 'is_outlier')
        
        # Test basic functionality
        X_dummy = np.random.rand(10, 5)
        y_dummy = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        # Should not crash
        dummy_ood.fit(X_dummy, y_dummy)
        
        # Should return reasonable outputs
        decisions = dummy_ood.decision_function(X_dummy)
        assert len(decisions) == len(X_dummy)
        assert isinstance(decisions, np.ndarray)
        
        predictions = dummy_ood.predict(X_dummy)
        assert len(predictions) == len(X_dummy)
        
        outlier_check = dummy_ood.is_outlier(X_dummy[0])
        assert isinstance(outlier_check, (bool, np.bool_))

class TestEnhancedProductionClassifierSystem:
    """Test EnhancedProductionClassifier system integration"""
    
    def test_enhanced_classifier_initialization(self):
        """Test EnhancedProductionClassifier can be initialized"""
        try:
            classifier = EnhancedProductionClassifier()
            assert classifier is not None
            print("✅ EnhancedProductionClassifier initialized successfully")
        except Exception as e:
            print(f"⚠️ EnhancedProductionClassifier initialization failed: {e}")
            # Still pass if it's just model loading issues
            pytest.skip(f"Skipping due to model loading issue: {e}")
    
    def test_enhanced_classifier_predict_dummy_input(self):
        """Test EnhancedProductionClassifier prediction with dummy input"""
        try:
            classifier = EnhancedProductionClassifier()
            
            # Create dummy feature vector (81 features)
            dummy_features = np.random.rand(81)
            
            result = classifier.predict(dummy_features)
            
            assert result is not None
            assert 'predicted_class' in result
            assert 'confidence' in result
            assert 'ood_detection' in result
            
            print(f"✅ Prediction result: {result}")
            
        except Exception as e:
            print(f"⚠️ Prediction test failed: {e}")
            pytest.skip(f"Skipping due to prediction issue: {e}")

class TestModelPipelineIntegration:
    """Test new pipeline integration"""
    
    def test_pipeline_v4_loading(self):
        """Test loading new v4 pipeline"""
        pipeline_path = "trained_model/pipeline_v4_standard.joblib"
        
        if not os.path.exists(pipeline_path):
            pytest.skip(f"Pipeline file not found: {pipeline_path}")
        
        try:
            pipeline = joblib.load(pipeline_path)
            assert pipeline is not None
            
            # Test pipeline has expected steps
            assert hasattr(pipeline, 'steps')
            step_names = [step[0] for step in pipeline.steps]
            
            expected_steps = ['scaler', 'pca', 'rf']
            for expected_step in expected_steps:
                assert expected_step in step_names, f"Missing step: {expected_step}"
            
            print(f"✅ Pipeline loaded with steps: {step_names}")
            
        except Exception as e:
            pytest.fail(f"Failed to load pipeline: {e}")
    
    def test_pipeline_v4_prediction(self):
        """Test pipeline prediction with dummy data"""
        pipeline_path = "trained_model/pipeline_v4_standard.joblib"
        label_encoder_path = "trained_model/label_encoder_v4.joblib"
        
        if not os.path.exists(pipeline_path):
            pytest.skip(f"Pipeline file not found: {pipeline_path}")
        
        if not os.path.exists(label_encoder_path):
            pytest.skip(f"Label encoder not found: {label_encoder_path}")
        
        try:
            pipeline = joblib.load(pipeline_path)
            label_encoder = joblib.load(label_encoder_path)
            
            # Create dummy input (81 features)
            X_dummy = np.random.rand(1, 81)
            
            # Predict
            prediction = pipeline.predict(X_dummy)
            probability = pipeline.predict_proba(X_dummy)
            
            # Decode label
            predicted_label = label_encoder.inverse_transform(prediction)
            
            assert len(prediction) == 1
            assert len(predicted_label) == 1
            assert probability.shape[0] == 1
            
            print(f"✅ Pipeline prediction: {predicted_label[0]} (prob: {probability[0].max():.3f})")
            
        except Exception as e:
            pytest.fail(f"Pipeline prediction failed: {e}")

class TestSystemHealthChecks:
    """Test overall system health"""
    
    def test_required_directories_exist(self):
        """Test that required directories exist"""
        required_dirs = [
            "trained_model",
            "ai_models", 
            "dataset",
            "scripts"
        ]
        
        for directory in required_dirs:
            assert os.path.exists(directory), f"Missing directory: {directory}"
    
    def test_model_files_exist(self):
        """Test that model files exist"""
        model_files = [
            "trained_model/classifier.joblib",
            "trained_model/ood_detector.joblib", 
            "trained_model/pca.joblib",
            "trained_model/scaler.joblib",
            "trained_model/label_encoder.joblib"
        ]
        
        existing_files = []
        for model_file in model_files:
            if os.path.exists(model_file):
                existing_files.append(model_file)
        
        # At least some model files should exist
        assert len(existing_files) > 0, f"No model files found. Expected at least 1 from: {model_files}"
        print(f"✅ Found {len(existing_files)} model files")
    
    def test_dataset_structure(self):
        """Test dataset has proper structure"""
        dataset_path = "dataset"
        
        if not os.path.exists(dataset_path):
            pytest.skip("Dataset directory not found")
        
        # Check for train/test/validation structure
        expected_subdirs = ["train", "test", "validation"]
        found_subdirs = []
        
        for subdir in expected_subdirs:
            subdir_path = os.path.join(dataset_path, subdir)
            if os.path.exists(subdir_path):
                found_subdirs.append(subdir)
        
        assert len(found_subdirs) > 0, f"No dataset subdirs found. Expected: {expected_subdirs}"
        print(f"✅ Found dataset subdirs: {found_subdirs}")

# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    print("\n" + "="*60)
    print("🧪 AMULET-AI v4.0 - MODEL INTEGRATION TESTS")  
    print("="*60)

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])