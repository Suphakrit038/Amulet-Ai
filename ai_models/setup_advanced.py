#!/usr/bin/env python3
"""
🚀 Advanced Amulet AI Setup Script
สคริปต์ติดตั้งและเตรียมระบบ AI พระเครื่องขั้นสูง
"""
import os
import sys
import subprocess
import platform
import json
from pathlib import Path
import logging
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedAmuletAISetup:
    """Setup manager for Advanced Amulet AI system"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.python_version = sys.version_info
        self.base_dir = Path(__file__).parent
        self.setup_complete = False
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        logger.info(f"🐍 Python version: {sys.version}")
        
        if self.python_version < (3, 8):
            logger.error("❌ Python 3.8+ is required")
            return False
        
        if self.python_version >= (3, 12):
            logger.warning("⚠️ Python 3.12+ may have compatibility issues with some packages")
        
        logger.info("✅ Python version compatible")
        return True
    
    def check_system_requirements(self) -> Dict[str, bool]:
        """Check system requirements"""
        requirements = {
            'python_compatible': self.check_python_version(),
            'pip_available': self._check_pip(),
            'git_available': self._check_git(),
            'sufficient_disk_space': self._check_disk_space(),
            'sufficient_memory': self._check_memory()
        }
        
        return requirements
    
    def _check_pip(self) -> bool:
        """Check if pip is available"""
        try:
            subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                         capture_output=True, check=True)
            logger.info("✅ pip is available")
            return True
        except subprocess.CalledProcessError:
            logger.error("❌ pip is not available")
            return False
    
    def _check_git(self) -> bool:
        """Check if git is available"""
        try:
            subprocess.run(['git', '--version'], 
                         capture_output=True, check=True)
            logger.info("✅ git is available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("⚠️ git is not available (optional)")
            return False
    
    def _check_disk_space(self, min_gb: float = 5.0) -> bool:
        """Check available disk space"""
        try:
            import shutil
            free_bytes = shutil.disk_usage(self.base_dir).free
            free_gb = free_bytes / (1024**3)
            
            if free_gb >= min_gb:
                logger.info(f"✅ Sufficient disk space: {free_gb:.1f} GB available")
                return True
            else:
                logger.error(f"❌ Insufficient disk space: {free_gb:.1f} GB available, {min_gb} GB required")
                return False
        except Exception as e:
            logger.warning(f"⚠️ Could not check disk space: {e}")
            return True
    
    def _check_memory(self, min_gb: float = 4.0) -> bool:
        """Check available memory"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            if memory_gb >= min_gb:
                logger.info(f"✅ Sufficient memory: {memory_gb:.1f} GB total")
                return True
            else:
                logger.warning(f"⚠️ Limited memory: {memory_gb:.1f} GB total, {min_gb} GB recommended")
                return True  # Don't fail, just warn
        except ImportError:
            logger.info("ℹ️ psutil not available, skipping memory check")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Could not check memory: {e}")
            return True
    
    def detect_gpu(self) -> Dict[str, any]:
        """Detect GPU capabilities"""
        gpu_info = {
            'cuda_available': False,
            'cuda_version': None,
            'gpu_count': 0,
            'gpu_names': [],
            'gpu_memory': []
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                gpu_info['cuda_version'] = torch.version.cuda
                gpu_info['gpu_count'] = torch.cuda.device_count()
                
                for i in range(gpu_info['gpu_count']):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    gpu_info['gpu_names'].append(gpu_name)
                    gpu_info['gpu_memory'].append(gpu_memory)
                
                logger.info(f"🔥 CUDA available: {gpu_info['gpu_count']} GPU(s)")
                for i, (name, memory) in enumerate(zip(gpu_info['gpu_names'], gpu_info['gpu_memory'])):
                    logger.info(f"   GPU {i}: {name} ({memory:.1f} GB)")
            else:
                logger.info("🖥️ CUDA not available, will use CPU")
                
        except ImportError:
            logger.info("ℹ️ PyTorch not yet installed, will check GPU later")
        
        return gpu_info
    
    def install_requirements(self, requirements_file: str = "requirements_advanced.txt") -> bool:
        """Install Python requirements"""
        requirements_path = self.base_dir / requirements_file
        
        if not requirements_path.exists():
            logger.error(f"❌ Requirements file not found: {requirements_path}")
            return False
        
        logger.info(f"📦 Installing requirements from {requirements_file}...")
        
        try:
            # Upgrade pip first
            logger.info("⬆️ Upgrading pip...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                         check=True)
            
            # Install requirements
            cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_path)]
            
            # Add GPU-specific packages if CUDA is available
            gpu_info = self.detect_gpu()
            if gpu_info['cuda_available']:
                logger.info("🔥 Installing GPU-optimized packages...")
                # Install GPU versions
                subprocess.run([sys.executable, '-m', 'pip', 'install', 
                               'torch', 'torchvision', 'torchaudio', '--index-url', 
                               'https://download.pytorch.org/whl/cu118'], check=True)
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'faiss-gpu'], 
                             check=False)  # Optional
            
            # Install main requirements
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            logger.info("✅ Requirements installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install requirements: {e}")
            if e.stdout:
                logger.error(f"stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"stderr: {e.stderr}")
            return False
    
    def create_directory_structure(self) -> bool:
        """Create necessary directories"""
        directories = [
            'dataset',
            'dataset_organized',
            'dataset_split',
            'training_output',
            'training_output/models',
            'training_output/logs', 
            'training_output/visualizations',
            'training_output/embeddings',
            'training_output/reports',
            'logs'
        ]
        
        logger.info("📁 Creating directory structure...")
        
        try:
            for directory in directories:
                dir_path = self.base_dir.parent / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {dir_path}")
            
            logger.info("✅ Directory structure created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create directories: {e}")
            return False
    
    def validate_installation(self) -> Dict[str, bool]:
        """Validate that key packages are installed correctly"""
        packages_to_check = [
            'torch',
            'torchvision', 
            'numpy',
            'PIL',
            'cv2',
            'sklearn',
            'faiss',
            'matplotlib',
            'pandas',
            'tqdm'
        ]
        
        validation_results = {}
        
        logger.info("🔍 Validating installation...")
        
        for package in packages_to_check:
            try:
                if package == 'cv2':
                    import cv2
                elif package == 'PIL':
                    from PIL import Image
                else:
                    __import__(package)
                
                validation_results[package] = True
                logger.debug(f"✅ {package} imported successfully")
                
            except ImportError as e:
                validation_results[package] = False
                logger.warning(f"⚠️ Failed to import {package}: {e}")
        
        # Summary
        successful = sum(validation_results.values())
        total = len(validation_results)
        
        logger.info(f"📊 Validation results: {successful}/{total} packages OK")
        
        if successful == total:
            logger.info("✅ All packages validated successfully")
        else:
            logger.warning("⚠️ Some packages failed validation")
        
        return validation_results
    
    def create_sample_config(self) -> bool:
        """Create a sample configuration file"""
        config_path = self.base_dir.parent / 'config.json'
        
        if config_path.exists():
            logger.info(f"ℹ️ Config file already exists: {config_path}")
            return True
        
        # Load default config
        default_config_path = self.base_dir / 'config_advanced.json'
        
        try:
            if default_config_path.exists():
                # Copy advanced config
                import shutil
                shutil.copy2(default_config_path, config_path)
                logger.info(f"✅ Created config file: {config_path}")
            else:
                # Create basic config
                basic_config = {
                    "dataset_path": "dataset",
                    "output_dir": "training_output",
                    "model_name": "efficientnet-b4",
                    "batch_size": 16,
                    "num_epochs": 50,
                    "learning_rate": 0.0001
                }
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(basic_config, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✅ Created basic config file: {config_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create config file: {e}")
            return False
    
    def run_setup(self) -> bool:
        """Run complete setup process"""
        logger.info("🚀 Starting Advanced Amulet AI Setup")
        logger.info("="*60)
        
        setup_steps = [
            ("Checking system requirements", self.check_system_requirements),
            ("Creating directory structure", self.create_directory_structure),
            ("Installing requirements", self.install_requirements),
            ("Validating installation", self.validate_installation),
            ("Creating sample configuration", self.create_sample_config),
        ]
        
        results = {}
        
        for step_name, step_func in setup_steps:
            logger.info(f"🔄 {step_name}...")
            
            try:
                result = step_func()
                results[step_name] = result
                
                if isinstance(result, bool) and not result:
                    logger.error(f"❌ Setup failed at: {step_name}")
                    return False
                elif isinstance(result, dict):
                    # Check if any critical failures
                    if step_name == "Checking system requirements":
                        if not result.get('python_compatible', False):
                            logger.error("❌ Python compatibility check failed")
                            return False
                
            except Exception as e:
                logger.error(f"❌ Error in {step_name}: {e}")
                return False
        
        # Detect GPU after installation
        logger.info("🔍 Detecting GPU capabilities...")
        gpu_info = self.detect_gpu()
        results['gpu_info'] = gpu_info
        
        # Setup complete
        self.setup_complete = True
        logger.info("="*60)
        logger.info("🎉 Setup completed successfully!")
        
        # Print summary
        self._print_setup_summary(results)
        
        return True
    
    def _print_setup_summary(self, results: Dict):
        """Print setup summary"""
        print("\n" + "="*60)
        print("📋 SETUP SUMMARY")
        print("="*60)
        
        # System info
        print(f"🖥️ System: {platform.system()} {platform.release()}")
        print(f"🐍 Python: {sys.version.split()[0]}")
        
        # GPU info
        gpu_info = results.get('gpu_info', {})
        if gpu_info.get('cuda_available'):
            print(f"🔥 CUDA: Available ({gpu_info['gpu_count']} GPU(s))")
        else:
            print("🖥️ CUDA: Not available (using CPU)")
        
        # Installation status
        validation = results.get('validating_installation', {})
        if isinstance(validation, dict):
            successful_packages = sum(1 for v in validation.values() if v)
            total_packages = len(validation)
            print(f"📦 Packages: {successful_packages}/{total_packages} validated")
        
        print("\n🚀 Ready to start training!")
        print("Run: python ai_models/train_advanced_amulet_ai.py --quick-start")
        print("="*60)

def main():
    """Main setup function"""
    setup = AdvancedAmuletAISetup()
    
    try:
        success = setup.run_setup()
        
        if success:
            print("\n🎯 Next steps:")
            print("1. Place your amulet images in the 'dataset' folder")
            print("2. Run: python ai_models/train_advanced_amulet_ai.py --quick-start")
            print("3. Check results in 'training_output' folder")
            return 0
        else:
            print("\n❌ Setup failed. Please check the logs above.")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
