"""
This script identifies and deletes unnecessary files from the Amulet-AI project.
Phase 2: Removing additional files identified in screenshots.
"""

import os
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cleanup_phase2')

# Define the project root
PROJECT_ROOT = Path(__file__).parent.absolute()

# Files to be removed from root directory (based on screenshot)
ROOT_FILES_TO_REMOVE = [
    'final_test.py',
    'fix_and_run_system.py',
    'fix_streamlit.bat',
    'launch_real_ai_backend.bat',
    'launch_real_ai_complete.bat', 
    'quick_test.py',
    'run_streamlit_fast.bat',
    'run_streamlit_fast.py',
    'start_amulet_system.bat',
    'start_system.bat',
    'test_api_import.py',
    'test_real_ai_system.py',
    'test_simple.py',
    'test_system.py',
    'unified_launcher.py',
    'emergency_fix.py'
]

# Files to be removed from backend directory (based on screenshot)
BACKEND_FILES_TO_CLEAN = [
    'api_simple.py',         # Keep only main api.py
    'minimal_api.py',        # Redundant with optimized versions
    'mock_api.py',           # Keep real implementation
]

# Zero KB files in backend (likely empty or placeholder files)
BACKEND_ZERO_KB_FILES = [
    'optimized_api.py',      # 0 KB - Either empty or need proper implementation
    'recommend_optimized.py' # 0 KB - Either empty or need proper implementation
]

# Files in AI models to consider (if appropriate)
AI_MODELS_CONSIDER = [
    'improved_step5_training.py',  # Review if needed
    'steps_6_and_7.py',            # May be redundant with final_steps_6_and_7.py
    'train_advanced_amulet_ai_fixed.py',  # May be redundant
    'train_advanced_amulet_ai.py',       # May be redundant if fixed version exists
    'emergency_training.py'        # May be redundant
]

def confirm_action(message):
    """Ask for confirmation before proceeding"""
    response = input(f"{message} (y/n): ").lower().strip()
    return response == 'y' or response == 'yes'

def main():
    # Print banner
    print("="*80)
    print("Amulet-AI Project Cleanup Phase 2")
    print("="*80)
    print("This script will remove additional unnecessary files identified in screenshots.")
    print()
    
    # Check for files to remove
    files_to_remove = []
    
    # Check root files
    for file_path in ROOT_FILES_TO_REMOVE:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            files_to_remove.append(file_path)
    
    # Check backend files
    backend_dir = PROJECT_ROOT / 'backend'
    for file_path in BACKEND_FILES_TO_CLEAN:
        full_path = backend_dir / file_path
        if full_path.exists():
            files_to_remove.append(os.path.join('backend', file_path))
    
    # Check AI models directory for duplicate/redundant training files
    ai_models_dir = PROJECT_ROOT / 'ai_models'
    for file_path in AI_MODELS_CONSIDER:
        full_path = ai_models_dir / file_path
        if full_path.exists():
            files_to_remove.append(os.path.join('ai_models', file_path))
    
    # Display files to be removed
    if files_to_remove:
        print("The following files will be removed:")
        for file in sorted(files_to_remove):
            print(f"  - {file}")
    else:
        print("No files found to remove.")
        return
    
    # Confirm removal
    if not confirm_action("\nDo you want to proceed with removal?"):
        print("Operation cancelled.")
        return
    
    # Perform removal
    removed_files = []
    failed_files = []
    
    for file_path in files_to_remove:
        full_path = PROJECT_ROOT / file_path
        try:
            if full_path.exists():
                if full_path.is_file():
                    full_path.unlink()
                    logger.info(f"Removed file: {file_path}")
                    removed_files.append(file_path)
                else:
                    shutil.rmtree(full_path)
                    logger.info(f"Removed directory: {file_path}")
                    removed_files.append(file_path)
        except Exception as e:
            logger.error(f"Failed to remove {file_path}: {str(e)}")
            failed_files.append(file_path)
    
    # Summary
    print("\nCleanup Summary:")
    print(f"  - Successfully removed: {len(removed_files)} files/directories")
    if failed_files:
        print(f"  - Failed to remove: {len(failed_files)} files/directories")
        print("\nThe following files could not be removed:")
        for file in sorted(failed_files):
            print(f"  - {file}")
        print("\nYou may need to close any applications using these files and try again.")
    
    print("\nCleanup completed.")

if __name__ == "__main__":
    main()
