"""
Amulet-AI System Repair Tool
ระบบซ่อมแซมอัตโนมัติสำหรับ Amulet-AI
"""

import os
import sys
import shutil
from pathlib import Path
import logging
import json
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('repair')

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

class SystemRepair:
    """System repair class"""
    
    def __init__(self):
        self.start_time = time.time()
        self.fixed_issues = 0
        self.remaining_issues = 0
    
    def fix_temporary_files(self):
        """Remove temporary files"""
        print_header("🧹 REMOVING TEMPORARY FILES")
        
        temp_patterns = ["*.tmp", "*.bak", "*.swp", "*~", "temp_*", "*.pyc", "*.pyo"]
        temp_files = []
        
        for pattern in temp_patterns:
            temp_files.extend(list(PROJECT_ROOT.glob(f"**/{pattern}")))
        
        if not temp_files:
            print_info("No temporary files found")
            return
        
        for file_path in temp_files:
            try:
                file_path.unlink()
                print_success(f"Removed temporary file: {file_path.relative_to(PROJECT_ROOT)}")
                self.fixed_issues += 1
            except Exception as e:
                print_error(f"Failed to remove {file_path.relative_to(PROJECT_ROOT)}: {e}")
                self.remaining_issues += 1
    
    def fix_empty_directories(self):
        """Fix empty directories by adding .gitkeep files"""
        print_header("📁 FIXING EMPTY DIRECTORIES")
        
        empty_dirs = []
        for path in PROJECT_ROOT.glob("**"):
            if path.is_dir() and path.name != '__pycache__' and not any(path.iterdir()):
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
    
    def ensure_tests_directory(self):
        """Ensure the tests directory exists"""
        print_header("🧪 ENSURING TESTS DIRECTORY")
        
        tests_dir = PROJECT_ROOT / "tests"
        
        if tests_dir.exists() and tests_dir.is_dir():
            print_info("Tests directory already exists")
            return
        
        try:
            tests_dir.mkdir()
            print_success("Created tests directory")
            
            # Create a basic conftest.py file
            conftest_path = tests_dir / "conftest.py"
            with open(conftest_path, 'w') as f:
                f.write("""\"\"\"
Test configuration for Amulet-AI
การตั้งค่าสำหรับการทดสอบระบบ Amulet-AI
\"\"\"
import pytest
import sys
import os
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def test_config():
    \"\"\"Fixture for test configuration\"\"\"
    return {
        "test_mode": True,
        "use_mock_data": True
    }

@pytest.fixture
def sample_image_path():
    \"\"\"Fixture that returns a path to a sample test image\"\"\"
    images_dir = Path(__file__).parent / "test_images"
    images_dir.mkdir(exist_ok=True)
    
    # Return a path even if the file doesn't exist yet
    return images_dir / "sample_amulet.jpg"
""")
            
            # Create a basic test file
            test_api_path = tests_dir / "test_api.py"
            with open(test_api_path, 'w') as f:
                f.write("""\"\"\"
API Tests for Amulet-AI
การทดสอบ API ของระบบ Amulet-AI
\"\"\"
import pytest
import sys
import os
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_api_imports():
    \"\"\"Test that API modules can be imported\"\"\"
    # This will raise an exception if imports fail
    from backend import api
    assert api is not None

def test_config_imports():
    \"\"\"Test that configuration modules can be imported\"\"\"
    from utils.config_manager import Config, get_config
    assert Config is not None
    assert get_config is not None

def test_config_functionality(test_config):
    \"\"\"Test configuration functionality\"\"\"
    from utils.config_manager import Config
    
    # Create a test config file
    config = Config("test_config.json")
    config.set("test_key", "test_value")
    
    # Verify the value was set
    assert config.get("test_key") == "test_value"
""")
            
            # Create test_images directory
            test_images_dir = tests_dir / "test_images"
            test_images_dir.mkdir(exist_ok=True)
            
            # Add .gitkeep to test_images
            with open(test_images_dir / ".gitkeep", 'w') as f:
                f.write("# This directory will contain test images\n")
            
            print_success("Created basic test files")
            self.fixed_issues += 1
        except Exception as e:
            print_error(f"Failed to create tests directory: {e}")
            self.remaining_issues += 1
    
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
    
    def run_all_repairs(self):
        """Run all repair operations"""
        print_header("🔧 AMULET-AI SYSTEM REPAIR", Colors.MAGENTA)
        
        self.fix_temporary_files()
        self.fix_empty_directories()
        self.ensure_tests_directory()
        self.fix_requirements()
        self.verify_pillow_import()
        
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        
        # Print summary
        print_header("📊 REPAIR SUMMARY", Colors.MAGENTA)
        print_info(f"Issues fixed: {self.fixed_issues}")
        print_info(f"Issues remaining: {self.remaining_issues}")
        print_info(f"Repair completed in {elapsed:.2f} seconds")
        
        if self.remaining_issues == 0:
            print_success("System repair completed successfully! ✨")
            print_info("You can now run verify_system.py to check the system")
            return True
        else:
            print_warning(f"System repair completed with {self.remaining_issues} remaining issues. 🔔")
            print_info("Please check the output for details")
            return False

def main():
    """Main function"""
    try:
        repair = SystemRepair()
        success = repair.run_all_repairs()
        
        # Return exit code
        sys.exit(0 if success else 1)
    except Exception as e:
        print_error(f"Repair failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
