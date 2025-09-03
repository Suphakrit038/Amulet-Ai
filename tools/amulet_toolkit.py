"""
Amulet-AI Toolkit
เครื่องมือรวมอเนกประสงค์สำหรับระบบ Amulet-AI

รวมเครื่องมือทั้งหมดเข้าด้วยกันเพื่อให้สามารถใช้งานได้ง่ายผ่านเมนูเดียว:
- การตรวจสอบระบบ (System Verification)
- การซ่อมแซมระบบ (System Repair)
- การบำรุงรักษาระบบ (System Maintenance)
- การทดสอบไฟล์ (File Testing)
- การจัดการการตั้งค่า (Configuration Management)
"""

import os
import sys
import json
import time
import shutil
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import importlib.util

# =====================================================================
# กำหนดค่าเริ่มต้น (Configuration)
# =====================================================================

# Define the project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('amulet_toolkit')

# Define important directories
DIRS = {
    'backend': PROJECT_ROOT / 'backend',
    'frontend': PROJECT_ROOT / 'frontend',
    'ai_models': PROJECT_ROOT / 'ai_models',
    'utils': PROJECT_ROOT / 'utils',
    'docs': PROJECT_ROOT / 'docs',
    'tests': PROJECT_ROOT / 'tests',
    'tools': PROJECT_ROOT / 'tools',
}

# Define critical files that must be present
CRITICAL_FILES = [
    'amulet_launcher.py',
    'setup_models.py',
    'config.json',
    'requirements.txt',
    'backend/api.py',
    'backend/model_loader.py',
    'frontend/app_streamlit.py',
    'ai_models/labels.json',
]

# Define temporary file patterns to clean
TEMP_FILE_PATTERNS = [
    '*.pyc',
    '*.pyo',
    '*.pyd',
    '*__pycache__*',
    '*.log',
    '*.tmp',
    'temp_*',
    '*~',
]

# Define backup directory
BACKUP_DIR = PROJECT_ROOT / 'backups'

# =====================================================================
# Utility Functions (ฟังก์ชันสนับสนุน)
# =====================================================================

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

def try_import(module_name):
    """Try to import a module and return whether it succeeded"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def read_file(file_path, encoding='utf-8', read_binary=False, max_lines=None):
    """Read a file and return its contents"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print_error(f"File does not exist: {file_path}")
        return None
    
    print_info(f"Reading file: {file_path}")
    
    try:
        if read_binary:
            with open(file_path, 'rb') as f:
                data = f.read(1024)  # Read first 1KB for binary files
                print_success(f"Successfully read binary file (first 1KB)")
                print_info(f"File size: {file_path.stat().st_size:,} bytes")
                return data
        else:
            with open(file_path, 'r', encoding=encoding) as f:
                if max_lines:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line)
                    print_success(f"Successfully read file (first {max_lines} lines)")
                    return ''.join(lines)
                else:
                    content = f.read()
                    print_success(f"Successfully read file")
                    return content
    except UnicodeDecodeError:
        print_warning(f"Encoding error with {encoding}. Trying with latin-1...")
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                if max_lines:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line)
                    print_success(f"Successfully read file with latin-1 encoding (first {max_lines} lines)")
                    return ''.join(lines)
                else:
                    content = f.read()
                    print_success(f"Successfully read file with latin-1 encoding")
                    return content
        except Exception as e:
            print_error(f"Error reading file with latin-1: {e}")
            return None
    except Exception as e:
        print_error(f"Error reading file: {e}")
        return None

def read_json_file(file_path, encoding='utf-8'):
    """Read a JSON file and return its contents"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print_error(f"File does not exist: {file_path}")
        return None
    
    print_info(f"Reading JSON file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)
            print_success(f"Successfully parsed JSON file")
            return data
    except json.JSONDecodeError as e:
        print_error(f"JSON parsing error: {e}")
        # Try to read as plain text to see the issue
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read(1000)  # Read first 1000 chars
                print_info(f"File content (first 1000 chars):\n{content}")
        except:
            pass
        return None
    except UnicodeDecodeError:
        print_warning(f"Encoding error with {encoding}. Trying with latin-1...")
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                data = json.load(f)
                print_success(f"Successfully parsed JSON file with latin-1 encoding")
                return data
        except Exception as e:
            print_error(f"Error reading JSON file with latin-1: {e}")
            return None
    except Exception as e:
        print_error(f"Error reading JSON file: {e}")
        return None

# =====================================================================
# System Verification Class (คลาสสำหรับตรวจสอบระบบ)
# =====================================================================

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
        
        critical_files = CRITICAL_FILES
        
        for file_path in critical_files:
            self.verify_file_exists(file_path, critical=True)
        
        critical_dirs = list(DIRS.values())
        
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
        temp_files = []
        for pattern in TEMP_FILE_PATTERNS:
            temp_files.extend(PROJECT_ROOT.glob(f"**/{pattern}"))
        
        if temp_files:
            problems.append(f"Found {len(temp_files)} temporary files")
            for file_path in temp_files[:5]:  # Show first 5
                print_warning(f"Temporary file: {file_path.relative_to(PROJECT_ROOT)}")
            if len(temp_files) > 5:
                print_warning(f"... and {len(temp_files) - 5} more")
        
        # 5. Check if directory structure is as expected
        expected_dirs = list(DIRS.keys())
        
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

# =====================================================================
# System Repair Class (คลาสสำหรับซ่อมแซมระบบ)
# =====================================================================

class SystemRepair:
    """System repair class"""
    
    def __init__(self):
        self.start_time = time.time()
        self.fixed_issues = 0
        self.remaining_issues = 0
    
    def fix_temporary_files(self):
        """Remove temporary files"""
        print_header("🧹 REMOVING TEMPORARY FILES")
        
        temp_patterns = TEMP_FILE_PATTERNS
        temp_files = []
        
        for pattern in temp_patterns:
            temp_files.extend(list(PROJECT_ROOT.glob(f"**/{pattern}")))
        
        if not temp_files:
            print_info("No temporary files found")
            return
        
        for file_path in temp_files:
            try:
                if file_path.is_file():
                    file_path.unlink()
                    print_success(f"Removed temporary file: {file_path.relative_to(PROJECT_ROOT)}")
                    self.fixed_issues += 1
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    print_success(f"Removed temporary directory: {file_path.relative_to(PROJECT_ROOT)}")
                    self.fixed_issues += 1
            except Exception as e:
                print_error(f"Failed to remove {file_path.relative_to(PROJECT_ROOT)}: {e}")
                self.remaining_issues += 1
    
    def fix_empty_directories(self):
        """Fix empty directories by adding .gitkeep files"""
        print_header("📁 FIXING EMPTY DIRECTORIES")
        
        excluded_dirs = {
            '.venv', '__pycache__', '.git', 
            '.ipynb_checkpoints', 'node_modules'
        }
        
        empty_dirs = []
        for path in PROJECT_ROOT.glob("**"):
            if path.is_dir():
                # Skip excluded directories
                if any(excluded in str(path) for excluded in excluded_dirs):
                    continue
                    
                # Check if empty
                if not any(path.iterdir()):
                    empty_dirs.append(path)
        
        if not empty_dirs:
            print_info("No empty directories found")
            return
        
        print_info(f"Found {len(empty_dirs)} empty directories")
        
        for dir_path in empty_dirs:
            try:
                # Add .gitkeep file to empty directory
                gitkeep_path = dir_path / ".gitkeep"
                with open(gitkeep_path, 'w') as f:
                    f.write("# This file ensures the directory is not empty\n")
                
                print_success(f"Added .gitkeep to empty directory: {dir_path.relative_to(PROJECT_ROOT)}")
                self.fixed_issues += 1
            except Exception as e:
                print_error(f"Failed to fix empty directory {dir_path.relative_to(PROJECT_ROOT)}: {e}")
                self.remaining_issues += 1
    
    def ensure_directory_structure(self):
        """Ensure the directory structure is correct"""
        print_header("🏗️ ENSURING DIRECTORY STRUCTURE")
        
        for dir_name, dir_path in DIRS.items():
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True)
                    print_success(f"Created directory: {dir_name}")
                    self.fixed_issues += 1
                    
                    # Add .gitkeep to newly created directory
                    gitkeep_path = dir_path / ".gitkeep"
                    with open(gitkeep_path, 'w') as f:
                        f.write(f"# This file ensures the {dir_name} directory is not empty\n")
                except Exception as e:
                    print_error(f"Failed to create directory {dir_name}: {e}")
                    self.remaining_issues += 1
            else:
                print_info(f"Directory already exists: {dir_name}")
    
    def fix_requirements(self):
        """Fix requirements.txt file if needed"""
        print_header("📦 FIXING REQUIREMENTS")
        
        requirements_path = PROJECT_ROOT / "requirements.txt"
        
        if not requirements_path.exists():
            print_error("requirements.txt does not exist")
            return
        
        try:
            with open(requirements_path, 'r') as f:
                requirements = f.read()
            
            # Check for Pillow requirement
            if 'pillow' not in requirements.lower() and 'pil' not in requirements.lower():
                print_warning("Pillow not found in requirements.txt")
                
                with open(requirements_path, 'a') as f:
                    f.write("\n# Image processing\nPillow>=9.0.0\n")
                
                print_success("Added Pillow to requirements.txt")
                self.fixed_issues += 1
            else:
                print_info("Pillow already in requirements.txt")
        except Exception as e:
            print_error(f"Failed to fix requirements.txt: {e}")
            self.remaining_issues += 1
    
    def verify_pillow_import(self):
        """Verify that PIL (Pillow) can be imported"""
        print_header("🖼️ VERIFYING PILLOW")
        
        try:
            import PIL
            from PIL import Image
            print_success(f"Pillow (PIL) is installed (version {PIL.__version__})")
        except ImportError:
            print_warning("Pillow (PIL) is not installed")
            
            try:
                print_info("Attempting to install Pillow...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow>=9.0.0"])
                
                # Try import again
                import PIL
                from PIL import Image
                print_success(f"Successfully installed Pillow (version {PIL.__version__})")
                self.fixed_issues += 1
            except Exception as e:
                print_error(f"Failed to install Pillow: {e}")
                print_info("Please install Pillow manually: pip install Pillow>=9.0.0")
                self.remaining_issues += 1
    
    def create_backup(self):
        """Create a backup of critical files"""
        print_header("💾 CREATING BACKUP")
        
        if not BACKUP_DIR.exists():
            BACKUP_DIR.mkdir()
        
        # Create timestamped backup directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_subdir = BACKUP_DIR / f"backup_{timestamp}"
        backup_subdir.mkdir()
        
        # Files to backup
        backup_files = CRITICAL_FILES + ['README.md', 'docs/SYSTEM_GUIDE.md']
        
        backed_up = 0
        for file_path in backup_files:
            src_path = PROJECT_ROOT / file_path
            if src_path.exists() and src_path.is_file():
                # Create directory structure in backup
                dst_path = backup_subdir / file_path
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                try:
                    shutil.copy2(src_path, dst_path)
                    backed_up += 1
                except Exception as e:
                    print_error(f"Failed to backup {file_path}: {e}")
                    self.remaining_issues += 1
        
        if backed_up > 0:
            print_success(f"Backed up {backed_up} files to {backup_subdir}")
            self.fixed_issues += 1
        else:
            print_warning("No files were backed up")
    
    def run_all_repairs(self):
        """Run all repair operations"""
        print_header("🔧 AMULET-AI SYSTEM REPAIR", Colors.MAGENTA)
        
        self.fix_temporary_files()
        self.fix_empty_directories()
        self.ensure_directory_structure()
        self.fix_requirements()
        self.verify_pillow_import()
        self.create_backup()
        
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        
        # Print summary
        print_header("📊 REPAIR SUMMARY", Colors.MAGENTA)
        print_info(f"Issues fixed: {self.fixed_issues}")
        print_info(f"Issues remaining: {self.remaining_issues}")
        print_info(f"Repair completed in {elapsed:.2f} seconds")
        
        if self.remaining_issues == 0:
            print_success("System repair completed successfully! ✨")
            print_info("You can now run a system verification to check the system")
            return True
        else:
            print_warning(f"System repair completed with {self.remaining_issues} remaining issues. 🔔")
            print_info("Please check the output for details")
            return False

# =====================================================================
# System Maintenance Class (คลาสสำหรับบำรุงรักษาระบบ)
# =====================================================================

class SystemMaintainer:
    def __init__(self):
        self.start_time = time.time()
        self.files_cleaned = 0
        self.dirs_cleaned = 0
        self.errors = 0
        self.warnings = 0
        self.fixed_issues = 0

    def check_critical_files(self):
        """Check if all critical files exist"""
        print_header("⚙️ CHECKING CRITICAL SYSTEM FILES")
        missing_files = []
        
        for file_path in CRITICAL_FILES:
            full_path = PROJECT_ROOT / file_path
            if not full_path.exists():
                missing_files.append(file_path)
                logger.warning(f"Missing critical file: {file_path}")
                self.warnings += 1
                
        if missing_files:
            print_warning(f"Missing {len(missing_files)} critical files:")
            for file in missing_files:
                print_warning(f"  - {file}")
            return False
        else:
            print_success("All critical files present")
            return True

    def clean_temp_files(self):
        """Clean temporary files"""
        print_header("🧹 CLEANING TEMPORARY FILES")
        
        for pattern in TEMP_FILE_PATTERNS:
            for path in PROJECT_ROOT.glob(f"**/{pattern}"):
                try:
                    if path.is_file():
                        path.unlink()
                        logger.info(f"Removed file: {path.relative_to(PROJECT_ROOT)}")
                        self.files_cleaned += 1
                    elif path.is_dir():
                        shutil.rmtree(path)
                        logger.info(f"Removed directory: {path.relative_to(PROJECT_ROOT)}")
                        self.dirs_cleaned += 1
                except Exception as e:
                    logger.error(f"Failed to remove {path}: {str(e)}")
                    self.errors += 1
        
        print_success(f"Removed {self.files_cleaned} files and {self.dirs_cleaned} directories")

    def verify_config(self):
        """Verify config.json file integrity"""
        print_header("⚙️ VERIFYING CONFIGURATION FILE")
        config_path = PROJECT_ROOT / 'config.json'
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check for essential config keys
            required_keys = ['api', 'models', 'system']
            missing_keys = [key for key in required_keys if key not in config]
            
            if missing_keys:
                print_warning(f"Missing configuration sections: {', '.join(missing_keys)}")
                self.warnings += 1
                return False
            else:
                print_success("Configuration file is valid")
                return True
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON in config.json")
            self.errors += 1
            return False
        except Exception as e:
            logger.error(f"Error checking config: {str(e)}")
            self.errors += 1
            return False

    def cleanup_redundant_files(self):
        """Clean up redundant and unnecessary files"""
        print_header("🧹 CLEANING UP REDUNDANT FILES")
        
        # Files that can be safely removed if they exist
        redundant_files = [
            'README_updated.md',
            'docs/SYSTEM_GUIDE_updated.md',
            'cleanup_files_phase2.py',
            'cleanup_unused_files.py',
            'test_file_operations.py',
            'comprehensive_file_test.py',
            'test_config_manager.py',
            'file_access_test.py'
        ]
        
        removed_count = 0
        for file_path in redundant_files:
            full_path = PROJECT_ROOT / file_path
            if full_path.exists() and full_path.is_file():
                try:
                    full_path.unlink()
                    logger.info(f"Removed redundant file: {file_path}")
                    removed_count += 1
                    self.fixed_issues += 1
                except Exception as e:
                    logger.error(f"Failed to remove {file_path}: {str(e)}")
                    self.errors += 1
        
        print_success(f"Removed {removed_count} redundant files")

    def check_model_files(self):
        """Check for AI model files"""
        print_header("🤖 CHECKING AI MODEL FILES")
        
        model_files = [
            'ai_models/amulet_model.h5',
            'ai_models/amulet_model.tflite',
        ]
        
        missing_models = [file for file in model_files if not (PROJECT_ROOT / file).exists()]
        
        if missing_models:
            print_warning(f"Missing model files: {len(missing_models)}")
            for file in missing_models:
                print_warning(f"  - {file}")
            print_info("\n💡 Run 'python setup_models.py' to download missing models")
            self.warnings += 1
            return False
        else:
            print_success("All model files are present")
            return True

    def run_maintenance(self):
        """Run all maintenance tasks"""
        print_header("🛠️ AMULET-AI SYSTEM MAINTENANCE", Colors.MAGENTA)
        
        try:
            self.check_critical_files()
            self.clean_temp_files()
            self.verify_config()
            self.check_model_files()
            self.cleanup_redundant_files()
            
            # Calculate elapsed time
            elapsed = time.time() - self.start_time
            
            # Print summary
            print_header("📊 MAINTENANCE SUMMARY", Colors.MAGENTA)
            print_info(f"Files cleaned: {self.files_cleaned}")
            print_info(f"Directories cleaned: {self.dirs_cleaned}")
            print_info(f"Issues fixed: {self.fixed_issues}")
            print_info(f"Warnings: {self.warnings}")
            print_info(f"Errors: {self.errors}")
            print_info(f"Maintenance completed in {elapsed:.2f} seconds")
            
            if self.warnings > 0 or self.errors > 0:
                print_warning("\nSome issues were detected. Please check the logs for details.")
            else:
                print_success("\nSystem is in good condition!")
                
        except Exception as e:
            print_error(f"\nMaintenance failed: {str(e)}")
            logger.error(f"Maintenance failed: {str(e)}")
            return False
            
        return True

# =====================================================================
# File Testing Functions (ฟังก์ชันสำหรับทดสอบไฟล์)
# =====================================================================

def print_file_info(file_path):
    """Print detailed information about a file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print_error(f"File does not exist: {file_path}")
        return
    
    print_header("🔍 FILE INFORMATION")
    print_info(f"File: {file_path}")
    
    # Basic file stats
    stats = file_path.stat()
    print_info(f"Size: {stats.st_size:,} bytes")
    print_info(f"Last modified: {time.ctime(stats.st_mtime)}")
    print_info(f"Created: {time.ctime(stats.st_ctime)}")
    
    # Determine file type
    extension = file_path.suffix.lower()
    
    if extension in ['.py', '.txt', '.md', '.json', '.yml', '.yaml', '.html', '.css', '.js']:
        content = read_file(file_path, max_lines=10)
        if content:
            print_header("FILE CONTENT (FIRST 10 LINES)")
            print(content)
    elif extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
        print_info("Image file - binary content")
        try:
            from PIL import Image
            img = Image.open(file_path)
            print_info(f"Image format: {img.format}")
            print_info(f"Image size: {img.width}x{img.height}")
            print_info(f"Image mode: {img.mode}")
        except ImportError:
            print_warning("PIL not available for image information")
        except Exception as e:
            print_error(f"Error getting image info: {e}")
    elif extension in ['.json']:
        data = read_json_file(file_path)
        if data:
            print_header("JSON STRUCTURE")
            if isinstance(data, dict):
                print_info(f"Top-level keys: {', '.join(list(data.keys())[:10])}" + 
                           ("..." if len(data.keys()) > 10 else ""))
            elif isinstance(data, list):
                print_info(f"Array with {len(data)} items")
    else:
        print_info("Binary or unknown file format")
        read_file(file_path, read_binary=True)

# =====================================================================
# Main Menu Function (ฟังก์ชันเมนูหลัก)
# =====================================================================

def show_menu():
    """Show the main menu"""
    print("\n" + "="*80)
    print_colored(Colors.CYAN, "🔧 AMULET-AI TOOLKIT 🔧")
    print("="*80)
    print("")
    print_colored(Colors.GREEN, "1. Verify System (ตรวจสอบระบบ)")
    print_colored(Colors.YELLOW, "2. Repair System (ซ่อมแซมระบบ)")
    print_colored(Colors.BLUE, "3. Maintain System (บำรุงรักษาระบบ)")
    print_colored(Colors.MAGENTA, "4. Test File (ทดสอบไฟล์)")
    print_colored(Colors.CYAN, "5. Show Documentation (แสดงเอกสาร)")
    print_colored(Colors.RED, "0. Exit (ออก)")
    print("")
    
    choice = input("เลือกตัวเลือก (0-5): ")
    return choice

def main_menu():
    """Main menu function"""
    while True:
        choice = show_menu()
        
        if choice == '1':
            verifier = SystemVerifier()
            verifier.run_all_verifications()
        elif choice == '2':
            repair = SystemRepair()
            repair.run_all_repairs()
        elif choice == '3':
            maintainer = SystemMaintainer()
            maintainer.run_maintenance()
        elif choice == '4':
            file_path = input("ระบุเส้นทางไฟล์ที่ต้องการทดสอบ: ")
            print_file_info(file_path)
        elif choice == '5':
            print_header("📚 AMULET-AI DOCUMENTATION")
            docs_path = PROJECT_ROOT / "docs" / "SYSTEM_GUIDE.md"
            if docs_path.exists():
                content = read_file(docs_path, max_lines=30)
                print(content if content else "Error reading documentation")
                print_info("\nFull documentation available in docs/SYSTEM_GUIDE.md")
            else:
                print_error("Documentation not found")
        elif choice == '0':
            print_info("ขอบคุณที่ใช้ Amulet-AI Toolkit!")
            break
        else:
            print_error("ตัวเลือกไม่ถูกต้อง กรุณาลองใหม่")
        
        input("\nกด Enter เพื่อดำเนินการต่อ...")

# =====================================================================
# Command Line Interface (อินเตอร์เฟซคำสั่ง)
# =====================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Amulet-AI Toolkit')
    parser.add_argument('--verify', action='store_true', help='Verify system integrity')
    parser.add_argument('--repair', action='store_true', help='Repair system issues')
    parser.add_argument('--maintain', action='store_true', help='Perform system maintenance')
    parser.add_argument('--test-file', type=str, help='Test a specific file')
    parser.add_argument('--menu', action='store_true', help='Show interactive menu')
    parser.add_argument('--clean', action='store_true', help='Clean temporary files')
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # If no arguments are provided, show the menu
    if not any(vars(args).values()):
        args.menu = True
    
    if args.verify:
        verifier = SystemVerifier()
        verifier.run_all_verifications()
    elif args.repair:
        repair = SystemRepair()
        repair.run_all_repairs()
    elif args.maintain:
        maintainer = SystemMaintainer()
        maintainer.run_maintenance()
    elif args.test_file:
        print_file_info(args.test_file)
    elif args.clean:
        repair = SystemRepair()
        repair.fix_temporary_files()
    elif args.menu:
        main_menu()

if __name__ == "__main__":
    main()
