"""
Amulet-AI System Maintenance Tool

This tool helps maintain the Amulet-AI system by:
1. Cleaning up temporary and unnecessary files
2. Optimizing the directory structure
3. Verifying the integrity of important system files
4. Performing regular maintenance tasks
"""

import os
import shutil
from pathlib import Path
import json
import logging
import hashlib
import sys
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('maintenance')

# Define the project root
PROJECT_ROOT = Path(__file__).parent.absolute()

# Define important directories
DIRS = {
    'backend': PROJECT_ROOT / 'backend',
    'frontend': PROJECT_ROOT / 'frontend',
    'ai_models': PROJECT_ROOT / 'ai_models',
    'utils': PROJECT_ROOT / 'utils',
    'docs': PROJECT_ROOT / 'docs',
    'tests': PROJECT_ROOT / 'tests',
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
        print("\n⚙️ Checking critical system files...")
        missing_files = []
        
        for file_path in CRITICAL_FILES:
            full_path = PROJECT_ROOT / file_path
            if not full_path.exists():
                missing_files.append(file_path)
                logger.warning(f"Missing critical file: {file_path}")
                self.warnings += 1
                
        if missing_files:
            print(f"⚠️ Missing {len(missing_files)} critical files:")
            for file in missing_files:
                print(f"  - {file}")
            return False
        else:
            print("✅ All critical files present")
            return True

    def clean_temp_files(self):
        """Clean temporary files"""
        print("\n🧹 Cleaning temporary files...")
        
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
        
        print(f"✅ Removed {self.files_cleaned} files and {self.dirs_cleaned} directories")

    def verify_config(self):
        """Verify config.json file integrity"""
        print("\n⚙️ Verifying configuration file...")
        config_path = PROJECT_ROOT / 'config.json'
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check for essential config keys
            required_keys = ['api', 'models', 'system']
            missing_keys = [key for key in required_keys if key not in config]
            
            if missing_keys:
                print(f"⚠️ Missing configuration sections: {', '.join(missing_keys)}")
                self.warnings += 1
                return False
            else:
                print("✅ Configuration file is valid")
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
        print("\n🧹 Cleaning up redundant files...")
        
        # Files that can be safely removed if they exist
        redundant_files = [
            'README_updated.md',
            'docs/SYSTEM_GUIDE_updated.md',
            'cleanup_files_phase2.py',
            'cleanup_unused_files.py',
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
        
        print(f"✅ Removed {removed_count} redundant files")

    def create_backup(self):
        """Create a backup of critical files"""
        print("\n💾 Creating backup of critical files...")
        
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
                    logger.error(f"Failed to backup {file_path}: {str(e)}")
                    self.errors += 1
        
        print(f"✅ Backed up {backed_up} files to {backup_subdir}")

    def check_model_files(self):
        """Check for AI model files"""
        print("\n🤖 Checking AI model files...")
        
        model_files = [
            'ai_models/amulet_model.h5',
            'ai_models/amulet_model.tflite',
        ]
        
        missing_models = [file for file in model_files if not (PROJECT_ROOT / file).exists()]
        
        if missing_models:
            print(f"⚠️ Missing model files: {len(missing_models)}")
            for file in missing_models:
                print(f"  - {file}")
            print("\n💡 Run 'python setup_models.py' to download missing models")
            self.warnings += 1
            return False
        else:
            print("✅ All model files are present")
            return True

    def run_maintenance(self):
        """Run all maintenance tasks"""
        print("\n" + "="*80)
        print("🛠️  AMULET-AI SYSTEM MAINTENANCE 🛠️")
        print("="*80)
        
        try:
            self.check_critical_files()
            self.clean_temp_files()
            self.verify_config()
            self.check_model_files()
            self.cleanup_redundant_files()
            self.create_backup()
            
            # Calculate elapsed time
            elapsed = time.time() - self.start_time
            
            # Print summary
            print("\n" + "="*80)
            print(f"✅ Maintenance completed in {elapsed:.2f} seconds")
            print(f"📊 Summary:")
            print(f"  - Files cleaned: {self.files_cleaned}")
            print(f"  - Directories cleaned: {self.dirs_cleaned}")
            print(f"  - Issues fixed: {self.fixed_issues}")
            print(f"  - Warnings: {self.warnings}")
            print(f"  - Errors: {self.errors}")
            print("="*80)
            
            if self.warnings > 0 or self.errors > 0:
                print("\n⚠️ Some issues were detected. Please check the logs for details.")
            else:
                print("\n✅ System is in good condition!")
                
        except Exception as e:
            print(f"\n❌ Maintenance failed: {str(e)}")
            logger.error(f"Maintenance failed: {str(e)}")
            return False
            
        return True

if __name__ == "__main__":
    maintainer = SystemMaintainer()
    success = maintainer.run_maintenance()
    
    # Return exit code
    sys.exit(0 if success else 1)
