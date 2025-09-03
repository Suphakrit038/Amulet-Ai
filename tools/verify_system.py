"""
Amulet-AI System Verification Tool
ระบบทดสอบและตรวจสอบความถูกต้องของระบบ Amulet-AI
"""

import os
import sys
import json
import time
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('verify')

# Define the project root
PROJECT_ROOT = Path(__file__).parent.absolute()

# Define color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    ENDC = '\033[0m'

def print_colored(color, message):
    """Print colored text to terminal"""
    print(f"{color}{message}{Colors.ENDC}")

def print_success(message):
    print_colored(Colors.GREEN, f"✅ {message}")

def print_warning(message):
    print_colored(Colors.YELLOW, f"⚠️ {message}")

def print_error(message):
    print_colored(Colors.RED, f"❌ {message}")

def print_info(message):
    print_colored(Colors.BLUE, f"ℹ️ {message}")

def print_header(message, color=Colors.CYAN):
    print("\n" + "="*80)
    print_colored(color, message)
    print("="*80)

def get_file_hash(file_path):
    """Get MD5 hash of a file"""
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5()
            chunk = f.read(8192)
            while chunk:
                file_hash.update(chunk)
                chunk = f.read(8192)
            return file_hash.hexdigest()
    except Exception as e:
        print_error(f"Error calculating hash for {file_path}: {e}")
        return None

class SystemVerifier:
    """System verification class"""
    
    def __init__(self):
        self.start_time = time.time()
        self.tests_passed = 0
        self.tests_failed = 0
        self.tests_skipped = 0
        self.critical_issues = 0
    
    def verify_file_exists(self, file_path, critical=False):
        """Verify that a file exists"""
        file_path = Path(file_path)
        
        if file_path.exists():
            print_success(f"File exists: {file_path}")
            self.tests_passed += 1
            return True
        else:
            if critical:
                print_error(f"CRITICAL: Missing required file: {file_path}")
                self.critical_issues += 1
            else:
                print_warning(f"File does not exist: {file_path}")
            self.tests_failed += 1
            return False
    
    def verify_directory_exists(self, dir_path, critical=False):
        """Verify that a directory exists"""
        dir_path = Path(dir_path)
        
        if dir_path.exists() and dir_path.is_dir():
            print_success(f"Directory exists: {dir_path}")
            self.tests_passed += 1
            return True
        else:
            if critical:
                print_error(f"CRITICAL: Missing required directory: {dir_path}")
                self.critical_issues += 1
            else:
                print_warning(f"Directory does not exist: {dir_path}")
            self.tests_failed += 1
            return False
    
    def verify_json_file(self, file_path, required_keys=None):
        """Verify that a JSON file is valid and contains required keys"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            print_warning(f"JSON file does not exist: {file_path}")
            self.tests_failed += 1
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print_success(f"JSON file is valid: {file_path}")
            self.tests_passed += 1
            
            if required_keys:
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    print_warning(f"Missing required keys in {file_path}: {', '.join(missing_keys)}")
                    self.tests_failed += 1
                    return False
                else:
                    print_success(f"All required keys present in {file_path}")
                    self.tests_passed += 1
            
            return True
        except json.JSONDecodeError as e:
            print_error(f"Invalid JSON in {file_path}: {e}")
            self.tests_failed += 1
            return False
        except Exception as e:
            print_error(f"Error verifying JSON file {file_path}: {e}")
            self.tests_failed += 1
            return False
    
    def verify_file_readable(self, file_path, encoding='utf-8'):
        """Verify that a file is readable"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            print_warning(f"File does not exist: {file_path}")
            self.tests_failed += 1
            return False
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read(1024)  # Read first 1KB
            
            print_success(f"File is readable: {file_path}")
            self.tests_passed += 1
            return True
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read(1024)
                
                print_warning(f"File is readable with latin-1 encoding: {file_path}")
                self.tests_passed += 1
                return True
            except Exception as e:
                print_error(f"Error reading file with latin-1: {e}")
                self.tests_failed += 1
                return False
        except Exception as e:
            print_error(f"Error reading file: {e}")
            self.tests_failed += 1
            return False
    
    def verify_config(self):
        """Verify the configuration system"""
        print_header("⚙️ VERIFYING CONFIGURATION SYSTEM")
        
        # 1. Check if config.json exists
        if not self.verify_file_exists('config.json', critical=True):
            return False
        
        # 2. Check if config.json is valid JSON
        if not self.verify_json_file('config.json'):
            return False
        
        # 3. Try to import the config manager
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from utils.config_manager import Config, get_config, set_config
            
            print_success("Successfully imported config manager")
            self.tests_passed += 1
            
            # 4. Test the config manager with a temporary file
            temp_config_path = "temp_verify_config.json"
            config = Config(temp_config_path)
            
            # Set a test value
            test_value = f"test_value_{int(time.time())}"
            config.set("test_key", test_value)
            config.save_config()
            
            # Reload and verify
            config2 = Config(temp_config_path)
            if config2.get("test_key") == test_value:
                print_success("Config manager is working correctly")
                self.tests_passed += 1
            else:
                print_error("Config manager is not working correctly")
                self.tests_failed += 1
            
            # Clean up
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            
            return True
        except ImportError as e:
            print_error(f"Failed to import config manager: {e}")
            self.tests_failed += 1
            return False
        except Exception as e:
            print_error(f"Error testing config manager: {e}")
            self.tests_failed += 1
            return False
    
    def verify_critical_files(self):
        """Verify that all critical files exist"""
        print_header("📋 VERIFYING CRITICAL FILES")
        
        critical_files = [
            'config.json',
            'requirements.txt',
            'README.md',
            'docs/SYSTEM_GUIDE.md',
            'backend/api.py',
            'frontend/app_streamlit.py',
            'utils/__init__.py',
            'utils/config_manager.py',
            'utils/logger.py',
            'utils/image_utils.py'
        ]
        
        for file_path in critical_files:
            self.verify_file_exists(file_path, critical=True)
        
        critical_dirs = [
            'backend',
            'frontend',
            'utils',
            'docs',
            'ai_models'
        ]
        
        for dir_path in critical_dirs:
            self.verify_directory_exists(dir_path, critical=True)
    
    def verify_python_modules(self):
        """Verify that all required Python modules are available"""
        print_header("🐍 VERIFYING PYTHON MODULES")
        
        required_modules = [
            ('fastapi', 'Web API framework'),
            ('streamlit', 'Web UI framework'),
            ('uvicorn', 'ASGI server'),
            ('numpy', 'Numerical computing'),
            ('pandas', 'Data analysis'),
            ('tensorflow', 'Machine learning (optional)'),
            ('torch', 'Machine learning (optional)'),
            ('requests', 'HTTP client')
        ]
        
        # Check for special cases
        special_cases = [
            # (module_name, import_name, description)
            ('pillow', 'PIL', 'Image processing')
        ]
        
        # Regular modules
        for module, description in required_modules:
            try:
                __import__(module)
                print_success(f"Module '{module}' is available - {description}")
                self.tests_passed += 1
            except ImportError:
                if '(optional)' in description:
                    print_warning(f"Optional module '{module}' is not available - {description}")
                    self.tests_skipped += 1
                else:
                    print_error(f"Required module '{module}' is not available - {description}")
                    self.tests_failed += 1
        
        # Special case modules with different import names
        for module_name, import_name, description in special_cases:
            try:
                module = __import__(import_name)
                if hasattr(module, '__version__'):
                    print_success(f"Module '{module_name}' is available (as {import_name}) - {description} - version {module.__version__}")
                else:
                    print_success(f"Module '{module_name}' is available (as {import_name}) - {description}")
                self.tests_passed += 1
            except ImportError:
                print_error(f"Required module '{module_name}' is not available - {description}")
                self.tests_failed += 1
    
    def verify_model_files(self):
        """Verify AI model files"""
        print_header("🤖 VERIFYING AI MODEL FILES")
        
        model_files = [
            ('ai_models/labels.json', True),  # (path, is_critical)
            ('ai_models/amulet_model.h5', False),
            ('ai_models/amulet_model.tflite', False)
        ]
        
        for file_path, is_critical in model_files:
            self.verify_file_exists(file_path, critical=is_critical)
    
    def verify_file_structure(self):
        """Verify the overall file structure"""
        print_header("📁 VERIFYING FILE STRUCTURE")
        
        # Check for common problematic patterns
        problems = []
        
        # 1. Check for .pyc files in the root
        pyc_files = list(PROJECT_ROOT.glob("*.pyc"))
        if pyc_files:
            problems.append(f"Found {len(pyc_files)} .pyc files in the root directory")
        
        # 2. Check for empty directories (excluding expected ones)
        empty_dirs = []
        excluded_dirs = {
            '.venv', '__pycache__', '.git', 
            '.ipynb_checkpoints', 'node_modules'
        }
        
        for path in PROJECT_ROOT.glob("**"):
            if path.is_dir():
                # Skip excluded directories
                if any(excluded in str(path) for excluded in excluded_dirs):
                    continue
                    
                # Check if empty
                if not any(path.iterdir()):
                    empty_dirs.append(path)
        
        if empty_dirs:
            problems.append(f"Found {len(empty_dirs)} empty directories")
            for dir_path in empty_dirs[:5]:  # Show first 5
                print_warning(f"Empty directory: {dir_path.relative_to(PROJECT_ROOT)}")
            if len(empty_dirs) > 5:
                print_warning(f"... and {len(empty_dirs) - 5} more")
        
        # 3. Check for large files (excluding virtual environment)
        large_files = []
        excluded_paths = {'.venv', 'node_modules', '.git'}
        
        for path in PROJECT_ROOT.glob("**/*"):
            if path.is_file():
                # Skip files in excluded directories
                if any(excluded in str(path) for excluded in excluded_paths):
                    continue
                    
                # Check file size
                if path.stat().st_size > 100 * 1024 * 1024:  # 100MB
                    large_files.append((path, path.stat().st_size))
        
        if large_files:
            problems.append(f"Found {len(large_files)} very large files (>100MB)")
            for file_path, size in large_files:
                print_warning(f"Large file: {file_path.relative_to(PROJECT_ROOT)} - {size / (1024*1024):.2f} MB")
        
        # 4. Check for common temporary files
        temp_patterns = ["*.tmp", "*.bak", "*.swp", "*~", "temp_*"]
        temp_files = []
        for pattern in temp_patterns:
            temp_files.extend(PROJECT_ROOT.glob(f"**/{pattern}"))
        
        if temp_files:
            problems.append(f"Found {len(temp_files)} temporary files")
            for file_path in temp_files[:5]:  # Show first 5
                print_warning(f"Temporary file: {file_path.relative_to(PROJECT_ROOT)}")
            if len(temp_files) > 5:
                print_warning(f"... and {len(temp_files) - 5} more")
        
        # 5. Check if directory structure is as expected
        expected_dirs = [
            'backend', 'frontend', 'ai_models', 'utils', 'docs', 'tests'
        ]
        
        missing_dirs = [dir_name for dir_name in expected_dirs if not (PROJECT_ROOT / dir_name).is_dir()]
        
        if missing_dirs:
            problems.append(f"Missing expected directories: {', '.join(missing_dirs)}")
        
        # Report overall structure status
        if problems:
            print_warning("File structure has issues:")
            for problem in problems:
                print_warning(f"- {problem}")
            self.tests_failed += 1
        else:
            print_success("File structure looks good")
            self.tests_passed += 1
    
    def run_all_verifications(self):
        """Run all verification tests"""
        print_header("🔍 AMULET-AI SYSTEM VERIFICATION", Colors.MAGENTA)
        
        # Run all verifications
        self.verify_critical_files()
        self.verify_config()
        self.verify_model_files()
        self.verify_file_structure()
        
        try:
            self.verify_python_modules()
        except Exception as e:
            print_error(f"Error verifying Python modules: {e}")
        
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        
        # Print summary
        print_header("📊 VERIFICATION SUMMARY", Colors.MAGENTA)
        print_info(f"Tests passed: {self.tests_passed}")
        print_info(f"Tests failed: {self.tests_failed}")
        print_info(f"Tests skipped: {self.tests_skipped}")
        print_info(f"Critical issues: {self.critical_issues}")
        print_info(f"Verification completed in {elapsed:.2f} seconds")
        
        if self.tests_failed == 0 and self.critical_issues == 0:
            print_success("System verification completed successfully! ✨")
            return True
        elif self.critical_issues > 0:
            print_error(f"System verification failed with {self.critical_issues} critical issues! 🚨")
            return False
        else:
            print_warning(f"System verification completed with {self.tests_failed} non-critical issues. 🔔")
            return True

def main():
    """Main function"""
    try:
        verifier = SystemVerifier()
        success = verifier.run_all_verifications()
        
        # Return exit code
        sys.exit(0 if success else 1)
    except Exception as e:
        print_error(f"Verification failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
