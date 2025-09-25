"""
🔍 Comprehensive AI Project Performance Diagnosis
การวินิจฉัยประสิทธิภาพโปรเจค AI อย่างละเอียด
"""
import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# Add project path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "ai_models"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIProjectDiagnostics:
    """Comprehensive diagnostics for AI project"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.ai_models_path = self.project_root / "ai_models"
        self.results = {}
        
    def check_environment(self):
        """Check Python environment and basic dependencies"""
        logger.info("🐍 Checking Python Environment...")
        
        env_info = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": sys.platform,
            "project_root": str(self.project_root),
            "ai_models_path": str(self.ai_models_path)
        }
        
        # Check basic imports
        basic_packages = [
            'numpy', 'pandas', 'matplotlib', 'cv2', 'PIL', 
            'sklearn', 'joblib', 'requests', 'json', 'pathlib'
        ]
        
        package_status = {}
        for package in basic_packages:
            try:
                if package == 'cv2':
                    import cv2
                    package_status[package] = f"✅ {cv2.__version__}"
                elif package == 'PIL':
                    from PIL import Image
                    package_status[package] = "✅ Available"
                elif package == 'sklearn':
                    import sklearn
                    package_status[package] = f"✅ {sklearn.__version__}"
                else:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'Unknown')
                    package_status[package] = f"✅ {version}"
            except ImportError as e:
                package_status[package] = f"❌ Missing: {e}"
        
        self.results['environment'] = {
            'info': env_info,
            'packages': package_status
        }
        
        return package_status
    
    def check_ml_frameworks(self):
        """Check ML frameworks (PyTorch, TensorFlow, etc.)"""
        logger.info("🤖 Checking ML Frameworks...")
        
        frameworks = {}
        
        # PyTorch
        try:
            import torch
            frameworks['pytorch'] = {
                'status': '✅ Available',
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            
            # Test basic operations
            try:
                x = torch.randn(2, 2)
                frameworks['pytorch']['basic_ops'] = '✅ Working'
            except Exception as e:
                frameworks['pytorch']['basic_ops'] = f'❌ Error: {e}'
                
        except ImportError as e:
            frameworks['pytorch'] = {
                'status': f'❌ Not available: {e}',
                'error': str(e)
            }
        
        # TensorFlow
        try:
            import tensorflow as tf
            frameworks['tensorflow'] = {
                'status': '✅ Available',
                'version': tf.__version__,
                'gpu_devices': len(tf.config.list_physical_devices('GPU'))
            }
            
            # Test basic operations
            try:
                x = tf.constant([[1.0, 2.0]])
                frameworks['tensorflow']['basic_ops'] = '✅ Working'
            except Exception as e:
                frameworks['tensorflow']['basic_ops'] = f'❌ Error: {e}'
                
        except ImportError as e:
            frameworks['tensorflow'] = {
                'status': f'❌ Not available: {e}',
                'error': str(e)
            }
        
        self.results['ml_frameworks'] = frameworks
        return frameworks
    
    def analyze_dataset_structure(self):
        """Analyze dataset structure and quality"""
        logger.info("📊 Analyzing Dataset Structure...")
        
        dataset_path = self.ai_models_path / "dataset_split"
        analysis = {
            'path_exists': dataset_path.exists(),
            'splits': {},
            'classes': [],
            'total_samples': 0,
            'issues': []
        }
        
        if not dataset_path.exists():
            analysis['issues'].append("Dataset path does not exist")
            self.results['dataset'] = analysis
            return analysis
        
        # Check splits
        for split in ['train', 'validation', 'test']:
            split_path = dataset_path / split
            if split_path.exists():
                classes = [d.name for d in split_path.iterdir() if d.is_dir()]
                class_counts = {}
                
                for class_name in classes:
                    class_path = split_path / class_name
                    image_files = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
                    class_counts[class_name] = len(image_files)
                
                analysis['splits'][split] = {
                    'exists': True,
                    'classes': classes,
                    'class_counts': class_counts,
                    'total_samples': sum(class_counts.values())
                }
                
                analysis['total_samples'] += sum(class_counts.values())
                
                # Check for small classes
                small_classes = [cls for cls, count in class_counts.items() if count < 10]
                if small_classes:
                    analysis['issues'].append(f"{split}: Small classes detected: {small_classes}")
            else:
                analysis['splits'][split] = {'exists': False}
        
        # Get unique classes across all splits
        all_classes = set()
        for split_info in analysis['splits'].values():
            if split_info.get('exists', False):
                all_classes.update(split_info.get('classes', []))
        
        analysis['classes'] = sorted(list(all_classes))
        analysis['num_classes'] = len(all_classes)
        
        # Check labels.json
        labels_path = dataset_path / "labels.json"
        if labels_path.exists():
            try:
                with open(labels_path, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
                analysis['labels_file'] = {
                    'exists': True,
                    'labels': labels,
                    'count': len(labels)
                }
            except Exception as e:
                analysis['labels_file'] = {
                    'exists': True,
                    'error': f"Error reading labels: {e}"
                }
        else:
            analysis['labels_file'] = {'exists': False}
            analysis['issues'].append("labels.json file missing")
        
        self.results['dataset'] = analysis
        return analysis
    
    def analyze_existing_models(self):
        """Analyze existing model files and their status"""
        logger.info("🧠 Analyzing Existing Models...")
        
        models_analysis = {
            'traditional_models': {},
            'deep_learning_models': {},
            'model_files': {},
            'issues': []
        }
        
        # Check core models
        core_path = self.ai_models_path / "core"
        if core_path.exists():
            model_files = ['amulet_model.h5', 'amulet_model.tflite']
            for model_file in model_files:
                model_path = core_path / model_file
                if model_path.exists():
                    file_size = model_path.stat().st_size / (1024 * 1024)  # MB
                    models_analysis['model_files'][model_file] = {
                        'exists': True,
                        'size_mb': round(file_size, 2),
                        'path': str(model_path)
                    }
                else:
                    models_analysis['model_files'][model_file] = {'exists': False}
        
        # Check saved_models directory
        saved_models_path = self.ai_models_path / "saved_models"
        if saved_models_path.exists():
            saved_files = list(saved_models_path.glob("*"))
            models_analysis['saved_models'] = {
                'path_exists': True,
                'files': [f.name for f in saved_files if f.is_file()],
                'count': len([f for f in saved_files if f.is_file()])
            }
        
        # Check training output
        training_output_path = self.ai_models_path / "training_output"
        if training_output_path.exists():
            models_path = training_output_path / "models"
            if models_path.exists():
                model_files = list(models_path.glob("*.pth")) + list(models_path.glob("*.joblib"))
                models_analysis['training_output'] = {
                    'path_exists': True,
                    'model_files': [f.name for f in model_files],
                    'count': len(model_files)
                }
        
        self.results['models'] = models_analysis
        return models_analysis
    
    def test_ai_modules(self):
        """Test AI modules for functionality"""
        logger.info("🔧 Testing AI Modules...")
        
        modules_test = {}
        
        # Test lightweight ML system
        try:
            from lightweight_ml_system import LightweightAmuletClassifier, LightweightMLConfig
            
            # Try to create instance
            config = LightweightMLConfig()
            classifier = LightweightAmuletClassifier(config)
            
            modules_test['lightweight_ml'] = {
                'import': '✅ Success',
                'instantiation': '✅ Success',
                'config': str(config)
            }
        except Exception as e:
            modules_test['lightweight_ml'] = {
                'import': f'❌ Failed: {e}',
                'error': traceback.format_exc()
            }
        
        # Test data pipeline
        try:
            from advanced_data_pipeline import AdvancedDataPipeline, DataPipelineConfig
            
            config = DataPipelineConfig()
            pipeline = AdvancedDataPipeline(config)
            
            modules_test['data_pipeline'] = {
                'import': '✅ Success',
                'instantiation': '✅ Success'
            }
        except Exception as e:
            modules_test['data_pipeline'] = {
                'import': f'❌ Failed: {e}',
                'error': traceback.format_exc()
            }
        
        # Test other modules
        other_modules = [
            'dataset_organizer',
            'self_supervised_learning',
            'modern_model'
        ]
        
        for module_name in other_modules:
            try:
                module = __import__(module_name)
                modules_test[module_name] = {'import': '✅ Success'}
            except Exception as e:
                modules_test[module_name] = {
                    'import': f'❌ Failed: {e}'
                }
        
        self.results['modules'] = modules_test
        return modules_test
    
    def performance_benchmark(self):
        """Run basic performance benchmarks"""
        logger.info("⚡ Running Performance Benchmarks...")
        
        benchmarks = {}
        
        # Test image processing performance
        try:
            import cv2
            import numpy as np
            from PIL import Image
            
            # Create test image
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # OpenCV operations
            start_time = time.time()
            for _ in range(100):
                resized = cv2.resize(test_image, (224, 224))
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            cv2_time = time.time() - start_time
            
            # PIL operations
            pil_image = Image.fromarray(test_image)
            start_time = time.time()
            for _ in range(100):
                resized = pil_image.resize((224, 224))
                gray = resized.convert('L')
            pil_time = time.time() - start_time
            
            benchmarks['image_processing'] = {
                'opencv_time_100ops': round(cv2_time, 4),
                'pil_time_100ops': round(pil_time, 4),
                'opencv_ops_per_sec': round(100 / cv2_time, 2),
                'pil_ops_per_sec': round(100 / pil_time, 2)
            }
            
        except Exception as e:
            benchmarks['image_processing'] = {'error': str(e)}
        
        # Test ML operations
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            
            # Generate test data
            X, y = make_classification(n_samples=1000, n_features=100, n_classes=10)
            
            # Train small model
            start_time = time.time()
            rf = RandomForestClassifier(n_estimators=10, random_state=42)
            rf.fit(X, y)
            training_time = time.time() - start_time
            
            # Prediction time
            start_time = time.time()
            for _ in range(100):
                pred = rf.predict(X[:10])
            prediction_time = time.time() - start_time
            
            benchmarks['ml_operations'] = {
                'training_time_1000samples': round(training_time, 4),
                'prediction_time_100x10samples': round(prediction_time, 4),
                'predictions_per_sec': round(1000 / prediction_time, 2)
            }
            
        except Exception as e:
            benchmarks['ml_operations'] = {'error': str(e)}
        
        self.results['benchmarks'] = benchmarks
        return benchmarks
    
    def generate_recommendations(self):
        """Generate recommendations based on analysis"""
        logger.info("💡 Generating Recommendations...")
        
        recommendations = {
            'critical': [],
            'important': [],
            'suggested': [],
            'optimizations': []
        }
        
        # Check for critical issues
        if not self.results.get('dataset', {}).get('path_exists', False):
            recommendations['critical'].append(
                "Dataset path missing - create proper dataset structure"
            )
        
        if 'pytorch' in self.results.get('ml_frameworks', {}):
            pytorch_status = self.results['ml_frameworks']['pytorch'].get('status', '')
            if '❌' in pytorch_status:
                recommendations['important'].append(
                    "PyTorch not working - consider using lightweight ML or fix installation"
                )
        
        # Dataset recommendations
        dataset_info = self.results.get('dataset', {})
        if dataset_info.get('issues'):
            for issue in dataset_info['issues']:
                recommendations['important'].append(f"Dataset issue: {issue}")
        
        # Performance recommendations
        benchmarks = self.results.get('benchmarks', {})
        if 'image_processing' in benchmarks:
            opencv_ops = benchmarks['image_processing'].get('opencv_ops_per_sec', 0)
            if opencv_ops < 100:
                recommendations['optimizations'].append(
                    "Image processing performance low - consider optimization"
                )
        
        # Module recommendations
        modules = self.results.get('modules', {})
        for module_name, module_info in modules.items():
            if '❌' in module_info.get('import', ''):
                recommendations['important'].append(
                    f"Module {module_name} not working - check dependencies"
                )
        
        # General suggestions
        if dataset_info.get('total_samples', 0) < 1000:
            recommendations['suggested'].append(
                "Dataset size is small - consider data augmentation"
            )
        
        if not self.results.get('models', {}).get('training_output', {}).get('model_files'):
            recommendations['suggested'].append(
                "No trained models found - train new models"
            )
        
        self.results['recommendations'] = recommendations
        return recommendations
    
    def run_full_diagnosis(self):
        """Run complete diagnosis"""
        logger.info("🚀 Starting Full AI Project Diagnosis...")
        
        start_time = time.time()
        
        try:
            # Run all diagnostic tests
            self.check_environment()
            self.check_ml_frameworks()
            self.analyze_dataset_structure()
            self.analyze_existing_models()
            self.test_ai_modules()
            self.performance_benchmark()
            self.generate_recommendations()
            
            total_time = time.time() - start_time
            self.results['diagnosis_info'] = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_time_seconds': round(total_time, 2),
                'status': 'completed'
            }
            
        except Exception as e:
            self.results['diagnosis_info'] = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        
        return self.results
    
    def save_results(self, filename="ai_diagnosis_results.json"):
        """Save diagnosis results to file"""
        results_path = self.project_root / filename
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 Results saved to: {results_path}")
        return results_path
    
    def print_summary(self):
        """Print diagnosis summary"""
        print("\n" + "="*80)
        print("🎯 AI PROJECT DIAGNOSIS SUMMARY")
        print("="*80)
        
        # Environment status
        env_packages = self.results.get('environment', {}).get('packages', {})
        working_packages = len([p for p in env_packages.values() if '✅' in p])
        total_packages = len(env_packages)
        
        print(f"🐍 Environment: {working_packages}/{total_packages} packages working")
        
        # ML Frameworks
        frameworks = self.results.get('ml_frameworks', {})
        for fw_name, fw_info in frameworks.items():
            status = fw_info.get('status', 'Unknown')
            print(f"🤖 {fw_name.title()}: {status}")
        
        # Dataset
        dataset = self.results.get('dataset', {})
        if dataset.get('path_exists'):
            print(f"📊 Dataset: {dataset.get('total_samples', 0)} samples, {dataset.get('num_classes', 0)} classes")
        else:
            print("📊 Dataset: ❌ Not found")
        
        # Modules
        modules = self.results.get('modules', {})
        working_modules = len([m for m in modules.values() if '✅' in m.get('import', '')])
        total_modules = len(modules)
        print(f"🔧 Modules: {working_modules}/{total_modules} working")
        
        # Recommendations
        recs = self.results.get('recommendations', {})
        critical_count = len(recs.get('critical', []))
        important_count = len(recs.get('important', []))
        
        print(f"\n💡 Issues Found:")
        print(f"   🔴 Critical: {critical_count}")
        print(f"   🟡 Important: {important_count}")
        
        if critical_count > 0:
            print("\n🔴 Critical Issues:")
            for issue in recs.get('critical', []):
                print(f"   - {issue}")
        
        if important_count > 0:
            print("\n🟡 Important Issues:")
            for issue in recs.get('important', []):
                print(f"   - {issue}")
        
        print("\n" + "="*80)
        
        diagnosis_info = self.results.get('diagnosis_info', {})
        if diagnosis_info.get('status') == 'completed':
            print(f"✅ Diagnosis completed in {diagnosis_info.get('total_time_seconds', 0)} seconds")
        else:
            print("❌ Diagnosis failed")
        
        print("="*80)

def main():
    """Main diagnosis function"""
    diagnostics = AIProjectDiagnostics()
    
    # Run full diagnosis
    results = diagnostics.run_full_diagnosis()
    
    # Print summary
    diagnostics.print_summary()
    
    # Save results
    results_file = diagnostics.save_results()
    
    print(f"\n📋 Detailed results saved to: {results_file}")
    print("🚀 Ready for AI model improvements!")

if __name__ == "__main__":
    main()