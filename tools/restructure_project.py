"""
Amulet-AI Project Restructure Tool
เครื่องมือปรับโครงสร้างโปรเจค Amulet-AI

สคริปต์นี้จะช่วยในการจัดระเบียบโครงสร้างไฟล์ของโปรเจค Amulet-AI:
- ย้ายไฟล์ไปยังโฟลเดอร์ที่เหมาะสม
- รวมไฟล์ที่มีฟังก์ชันการทำงานคล้ายกัน
- ปรับปรุงเส้นทางการอ้างอิง (import paths)
- สร้างเอกสารประกอบสำหรับโครงสร้างใหม่
"""

import os
import sys
import shutil
import argparse
import re
from pathlib import Path
import time
import logging
import json

# Add parent directory to path to import from tools
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

try:
    from tools.amulet_toolkit import (
        Colors, print_colored, print_success, print_warning, 
        print_error, print_info, print_header, read_file
    )
except ImportError:
    # Fallback if tools module isn't available yet
    class Colors:
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        MAGENTA = '\033[95m'
        ENDC = '\033[0m'

    def print_colored(color, message):
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
    
    def read_file(file_path, encoding='utf-8'):
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()

# =====================================================================
# Constants
# =====================================================================

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=PROJECT_ROOT / "logs" / "restructure.log",
    filemode='w'
)
logger = logging.getLogger('restructure')

# Define restructuring rules
RESTRUCTURE_RULES = [
    # กฎสำหรับย้ายไฟล์ไปยัง tests/
    {
        'source_patterns': ['test_*.py', 'verify_*.py', 'check_*.py'],
        'exclude': ['test_api.py', 'test_system.py'],
        'target_dir': 'tests',
        'description': 'Move test files to tests directory'
    },
    # กฎสำหรับย้ายไฟล์ไปยัง scripts/
    {
        'source_patterns': ['*.bat', '*.sh', 'launch_*.py', '*_launcher.py', 'start.py', 'setup_*.py'],
        'exclude': ['initialize_structure.bat'],
        'target_dir': 'scripts',
        'description': 'Move launcher scripts to scripts directory'
    },
    # กฎสำหรับย้ายไฟล์การตั้งค่าไปยัง config/
    {
        'source_patterns': ['config*.json'],
        'exclude': ['ai_models/config_advanced.json'],
        'target_dir': 'config',
        'description': 'Move configuration files to config directory'
    },
    # กฎสำหรับย้ายเอกสารไปยัง docs/
    {
        'source_patterns': ['README*.md', 'LICENSE', 'CHANGELOG.md'],
        'exclude': ['ai_models/README_ADVANCED.md', 'README.md'],
        'target_dir': 'docs',
        'description': 'Move documentation to docs directory'
    },
    # กฎสำหรับย้ายเอกสาร API ไปยัง docs/api/
    {
        'source_patterns': ['API*.md'],
        'exclude': [],
        'target_dir': 'docs/api',
        'description': 'Move API documentation to docs/api directory'
    },
    # กฎสำหรับย้ายคู่มือไปยัง docs/guides/
    {
        'source_patterns': ['GUIDE*.md', 'USAGE*.md', 'MANUAL*.md'],
        'exclude': [],
        'target_dir': 'docs/guides',
        'description': 'Move guides to docs/guides directory'
    },
    # กฎสำหรับย้ายไฟล์เก่าไปยัง archive/
    {
        'source_patterns': ['old_*.py', 'deprecated_*.py', 'archive_*.py', 'backup_*.py'],
        'exclude': [],
        'target_dir': 'archive',
        'description': 'Move deprecated files to archive directory'
    }
]

# กฎสำหรับจัดระเบียบภายในโฟลเดอร์
INTERNAL_RESTRUCTURE_RULES = [
    # จัดระเบียบภายในโฟลเดอร์ ai_models/
    {
        'folder': 'ai_models',
        'subfolders': [
            {'name': 'core', 'patterns': ['amulet_model.*', 'labels.json']},
            {'name': 'training', 'patterns': ['*training*.py', 'train_*.py']},
            {'name': 'pipelines', 'patterns': ['*pipeline*.py', '*processor*.py']},
            {'name': 'evaluation', 'patterns': ['test_*.py', '*evaluation*.py', '*metrics*.py']},
            {'name': 'configs', 'patterns': ['config_*.json', '*requirements*.txt']},
            {'name': 'docs', 'patterns': ['README*.md']}
        ],
        'description': 'Organize AI models directory'
    },
    # จัดระเบียบภายในโฟลเดอร์ backend/
    {
        'folder': 'backend',
        'subfolders': [
            {'name': 'api', 'patterns': ['api*.py', 'optimized_api.py']},
            {'name': 'models', 'patterns': ['*model_loader*.py', 'real_model_loader.py']},
            {'name': 'services', 'patterns': ['*_service.py', 'valuation.py', 'price_estimator.py', 'recommend*.py', 'similarity_search.py', 'market_scraper.py']},
            {'name': 'config', 'patterns': ['config.py']},
            {'name': 'tests', 'patterns': ['test_*.py']}
        ],
        'description': 'Organize backend directory'
    },
    # จัดระเบียบภายในโฟลเดอร์ frontend/
    {
        'folder': 'frontend',
        'subfolders': [
            {'name': 'pages', 'patterns': ['*_page.py', '*_view.py']},
            {'name': 'components', 'patterns': ['*_component.py', '*_widget.py']},
            {'name': 'utils', 'patterns': ['utils.py', 'helpers.py']}
        ],
        'description': 'Organize frontend directory'
    },
    # จัดระเบียบภายในโฟลเดอร์ docs/
    {
        'folder': 'docs',
        'subfolders': [
            {'name': 'api', 'patterns': ['API*.md']},
            {'name': 'guides', 'patterns': ['GUIDE*.md', 'USAGE*.md', 'MANUAL*.md']},
            {'name': 'system', 'patterns': ['SYSTEM*.md', 'ARCHITECTURE*.md']},
            {'name': 'development', 'patterns': ['DEVELOPMENT*.md', 'DEPLOYMENT*.md', 'CONTRIBUTING*.md']}
        ],
        'description': 'Organize documentation directory'
    },
    # จัดระเบียบภายในโฟลเดอร์ utils/
    {
        'folder': 'utils',
        'subfolders': [
            {'name': 'config', 'patterns': ['config*.py']},
            {'name': 'image', 'patterns': ['image*.py']},
            {'name': 'logging', 'patterns': ['log*.py', 'error*.py']},
            {'name': 'data', 'patterns': ['data*.py', '*_data.py']}
        ],
        'description': 'Organize utilities directory'
    },
    # จัดระเบียบภายในโฟลเดอร์ tests/
    {
        'folder': 'tests',
        'subfolders': [
            {'name': 'unit', 'patterns': ['test_*_unit.py']},
            {'name': 'integration', 'patterns': ['test_*_integration.py']},
            {'name': 'fixtures', 'patterns': ['conftest.py', '*_fixture.py']},
            {'name': 'data', 'patterns': ['test_images/*']}
        ],
        'description': 'Organize tests directory'
    }
]

# Define file consolidation rules
CONSOLIDATION_RULES = [
    {
        'name': 'utils.py',
        'target_dir': 'utils',
        'source_files': ['utils/image_utils.py', 'utils/config_manager.py', 'utils/logger.py'],
        'description': 'Consolidate utility functions'
    },
    {
        'name': 'models.py',
        'target_dir': 'backend',
        'source_files': ['backend/model_loader.py', 'backend/optimized_model_loader.py'],
        'description': 'Consolidate model loader functions'
    },
    {
        'name': 'api.py',
        'target_dir': 'backend',
        'source_files': ['backend/api.py', 'backend/api_simple.py', 'backend/minimal_api.py', 'backend/optimized_api.py'],
        'description': 'Consolidate API functions'
    }
]

# =====================================================================
# Utility Functions
# =====================================================================

def backup_project():
    """Create a backup of the project before restructuring"""
    print_header("📦 CREATING PROJECT BACKUP")
    
    backup_dir = PROJECT_ROOT / 'backup'
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = backup_dir / f"backup_{timestamp}"
    
    if not backup_dir.exists():
        backup_dir.mkdir()
    
    backup_path.mkdir()
    
    try:
        # Copy all files except very large directories
        for item in PROJECT_ROOT.glob('*'):
            if item.name in ['.git', '__pycache__', 'backup', '.venv', 'venv', 'env']:
                continue
                
            if item.is_file():
                shutil.copy2(item, backup_path / item.name)
            elif item.is_dir():
                # Use shutil.copytree for directories
                shutil.copytree(
                    item, 
                    backup_path / item.name,
                    ignore=shutil.ignore_patterns('*.pyc', '*__pycache__*', '*.h5', '*.tflite')
                )
        
        print_success(f"Project backed up to: {backup_path}")
        logger.info(f"Project backed up to: {backup_path}")
        return True
    except Exception as e:
        print_error(f"Backup failed: {e}")
        logger.error(f"Backup failed: {e}")
        return False

def find_files_by_patterns(patterns, exclude=None):
    """Find files matching the given patterns"""
    if exclude is None:
        exclude = []
        
    files = []
    for pattern in patterns:
        for file_path in PROJECT_ROOT.glob(f"**/{pattern}"):
            # Convert to relative path for easier comparison
            rel_path = file_path.relative_to(PROJECT_ROOT)
            str_path = str(rel_path)
            
            # Check if file should be excluded
            if any(ex in str_path for ex in exclude):
                continue
                
            files.append(file_path)
    
    return files

def ensure_directory(dir_path):
    """Ensure directory exists, create if it doesn't"""
    if not dir_path.exists():
        dir_path.mkdir(parents=True)
        print_info(f"Created directory: {dir_path.relative_to(PROJECT_ROOT)}")
        
    return dir_path

def move_file(source, target_dir, dry_run=False):
    """Move a file to the target directory"""
    source = Path(source)
    target_dir = Path(target_dir)
    
    # Ensure target directory exists
    if not dry_run:
        ensure_directory(target_dir)
    
    # Calculate destination path
    dest = target_dir / source.name
    
    # Log the operation
    rel_source = source.relative_to(PROJECT_ROOT)
    rel_dest = dest.relative_to(PROJECT_ROOT)
    logger.info(f"Moving file: {rel_source} -> {rel_dest}")
    
    if dry_run:
        print_info(f"Would move: {rel_source} -> {rel_dest}")
        return True
    
    try:
        # Move the file
        shutil.move(source, dest)
        print_success(f"Moved: {rel_source} -> {rel_dest}")
        return True
    except Exception as e:
        print_error(f"Failed to move {rel_source}: {e}")
        logger.error(f"Failed to move {rel_source}: {e}")
        return False

def copy_file(source, target, dry_run=False):
    """Copy a file to the target path"""
    source = Path(source)
    target = Path(target)
    
    # Ensure target directory exists
    if not dry_run:
        ensure_directory(target.parent)
    
    # Log the operation
    rel_source = source.relative_to(PROJECT_ROOT)
    rel_target = target.relative_to(PROJECT_ROOT)
    logger.info(f"Copying file: {rel_source} -> {rel_target}")
    
    if dry_run:
        print_info(f"Would copy: {rel_source} -> {rel_target}")
        return True
    
    try:
        # Copy the file
        shutil.copy2(source, target)
        print_success(f"Copied: {rel_source} -> {rel_target}")
        return True
    except Exception as e:
        print_error(f"Failed to copy {rel_source}: {e}")
        logger.error(f"Failed to copy {rel_source}: {e}")
        return False

def consolidate_files(sources, target, dry_run=False):
    """Consolidate multiple source files into a single target file"""
    sources = [Path(s) for s in sources]
    target = Path(target)
    
    # Check if sources exist
    missing_sources = [s for s in sources if not s.exists()]
    if missing_sources:
        print_warning(f"Missing source files: {missing_sources}")
        return False
    
    # Ensure target directory exists
    if not dry_run:
        ensure_directory(target.parent)
    
    # Log the operation
    rel_sources = [s.relative_to(PROJECT_ROOT) for s in sources]
    rel_target = target.relative_to(PROJECT_ROOT)
    logger.info(f"Consolidating files: {rel_sources} -> {rel_target}")
    
    if dry_run:
        print_info(f"Would consolidate: {rel_sources} -> {rel_target}")
        return True
    
    try:
        # Create the consolidated file
        with open(target, 'w', encoding='utf-8') as outfile:
            # Write header
            outfile.write(f'''"""
Consolidated file: {target.name}
Generated by restructure.py on {time.strftime("%Y-%m-%d %H:%M:%S")}

This file combines the functionality of:
{chr(10).join([f"- {s.name}" for s in sources])}
"""

''')
            
            # Process each source file
            for i, source in enumerate(sources):
                # Read the source file
                try:
                    content = read_file(source)
                    
                    # Add a separator comment
                    outfile.write(f"\n# {'='*78}\n")
                    outfile.write(f"# Source file: {source.name}\n")
                    outfile.write(f"# {'='*78}\n\n")
                    
                    # Remove module docstring if not the first file
                    if i > 0:
                        content = re.sub(r'^""".*?"""', '', content, flags=re.DOTALL)
                    
                    # Remove any existing imports of previously consolidated files
                    for prev_source in sources[:i]:
                        prev_module = prev_source.stem
                        content = re.sub(
                            rf'(from\s+.*?{prev_module}\s+import.*?\n|import\s+.*?{prev_module}.*?\n)', 
                            '', 
                            content
                        )
                    
                    # Write the content
                    outfile.write(content)
                    outfile.write('\n\n')
                    
                except Exception as e:
                    print_error(f"Error processing {source}: {e}")
                    logger.error(f"Error processing {source}: {e}")
        
        print_success(f"Consolidated: {rel_sources} -> {rel_target}")
        return True
    except Exception as e:
        print_error(f"Failed to consolidate files: {e}")
        logger.error(f"Failed to consolidate files: {e}")
        return False

def update_imports(file_path, old_imports, new_imports, dry_run=False):
    """Update import statements in a file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print_warning(f"File does not exist: {file_path}")
        return False
    
    try:
        content = read_file(file_path)
        
        # Log the operation
        rel_path = file_path.relative_to(PROJECT_ROOT)
        logger.info(f"Updating imports in: {rel_path}")
        
        # Replace imports
        new_content = content
        for old, new in zip(old_imports, new_imports):
            new_content = re.sub(
                rf'(from\s+{old}\s+import|import\s+{old})', 
                f'\\1'.replace(old, new), 
                new_content
            )
        
        # Check if content changed
        if new_content == content:
            print_info(f"No import changes needed in: {rel_path}")
            return True
        
        if dry_run:
            print_info(f"Would update imports in: {rel_path}")
            return True
        
        # Write updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print_success(f"Updated imports in: {rel_path}")
        return True
    except Exception as e:
        print_error(f"Failed to update imports in {file_path}: {e}")
        logger.error(f"Failed to update imports in {file_path}: {e}")
        return False

def generate_structure_doc():
    """Generate documentation of the new structure"""
    print_header("📝 GENERATING STRUCTURE DOCUMENTATION")
    
    doc_path = PROJECT_ROOT / "docs" / "PROJECT_STRUCTURE.md"
    
    try:
        # Build directory tree
        def get_tree(directory, prefix="", is_last=True, ignore_patterns=None, depth=0, max_depth=4):
            if ignore_patterns is None:
                ignore_patterns = ['__pycache__', '.git', '.ipynb_checkpoints', '.venv', 'venv', 'env', '*.pyc']
            
            if depth > max_depth:
                return ["    " + prefix + "..."]
                
            output = []
            dir_path = Path(directory)
            
            # Sort items: directories first, then files
            try:
                items = list(dir_path.iterdir())
            except PermissionError:
                return [prefix + "📁 [Permission denied]"]
                
            dirs = sorted([item for item in items if item.is_dir()])
            files = sorted([item for item in items if item.is_file()])
            sorted_items = dirs + files
            
            # Filter out ignored patterns
            sorted_items = [
                item for item in sorted_items 
                if not any(pattern in str(item) for pattern in ignore_patterns)
                and not any(pattern.startswith('*') and pattern[1:] in item.suffix for pattern in ignore_patterns)
            ]
            
            for i, item in enumerate(sorted_items):
                is_last_item = i == len(sorted_items) - 1
                
                # Skip ignored patterns
                if any(pattern in item.name for pattern in ignore_patterns):
                    continue
                
                # Add item to tree
                connector = "└── " if is_last_item else "├── "
                if item.is_dir():
                    # ใช้ emoji สำหรับโฟลเดอร์
                    output.append(f"{prefix}{connector}📁 {item.name}")
                else:
                    # ใช้ emoji ตามประเภทของไฟล์
                    if item.suffix in ['.py']:
                        icon = "🐍"  # Python files
                    elif item.suffix in ['.md', '.txt']:
                        icon = "📄"  # Documentation files
                    elif item.suffix in ['.json', '.yaml', '.yml']:
                        icon = "⚙️"  # Configuration files
                    elif item.suffix in ['.bat', '.sh', '.ps1']:
                        icon = "🔧"  # Script files
                    elif item.suffix in ['.jpg', '.png', '.gif', '.jpeg']:
                        icon = "🖼️"  # Image files
                    elif item.suffix in ['.h5', '.tflite', '.pb']:
                        icon = "🧠"  # AI model files
                    else:
                        icon = "📎"  # Other files
                    output.append(f"{prefix}{connector}{icon} {item.name}")
                
                # Recursively process directories
                if item.is_dir():
                    extension = "    " if is_last_item else "│   "
                    output.extend(
                        get_tree(
                            item, 
                            prefix=prefix + extension, 
                            ignore_patterns=ignore_patterns,
                            depth=depth+1,
                            max_depth=max_depth
                        )
                    )
            
            return output
        
        # Get the tree
        tree_lines = ["```", "Project Structure", ""] + get_tree(PROJECT_ROOT) + ["```"]
        
        # Create the documentation
        content = f"""# Amulet-AI Project Structure

This document describes the structure of the Amulet-AI project after reorganization.

## Directory Structure

{chr(10).join(tree_lines)}

## Main Directory Descriptions

- **ai_models/**: Contains AI model files, training scripts, and model-related utilities
  - **core/**: Core model files and labels
  - **training/**: Training scripts and modules
  - **pipelines/**: Data processing pipelines
  - **evaluation/**: Model testing and evaluation
  - **configs/**: Model configurations
  - **docs/**: Model documentation

- **backend/**: Backend API and server code
  - **api/**: API endpoints and interfaces
  - **models/**: Model loading and processing
  - **services/**: Backend services (valuation, recommendations, etc.)
  - **config/**: Backend configuration
  - **tests/**: Backend tests

- **frontend/**: Frontend UI code
  - **pages/**: Main pages and views
  - **components/**: Reusable UI components
  - **utils/**: Frontend utilities

- **docs/**: Documentation files
  - **api/**: API documentation
  - **guides/**: User and developer guides
  - **system/**: System architecture and design
  - **development/**: Development and deployment guides

- **scripts/**: Launch and utility scripts
- **tests/**: Test files
  - **unit/**: Unit tests
  - **integration/**: Integration tests
  - **fixtures/**: Test fixtures
  - **data/**: Test data

- **tools/**: Maintenance and utility tools
- **utils/**: Utility functions and modules
  - **config/**: Configuration utilities
  - **image/**: Image processing utilities
  - **logging/**: Logging and error handling
  - **data/**: Data processing utilities

## File Consolidation

Several files were consolidated to reduce redundancy:

1. **utils/utils.py**: Combined utility functions from multiple source files
2. **backend/models.py**: Consolidated model loader implementations
3. **backend/api.py**: Unified API implementations

## Migration Changes

Original files that were moved:
{chr(10).join(['- ' + rule['description'] for rule in RESTRUCTURE_RULES])}

## Internal Folder Organization

The following folders were organized with more detailed structure:
{chr(10).join(['- ' + rule['description'] for rule in INTERNAL_RESTRUCTURE_RULES])}
"""
        
        # Write the documentation
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print_success(f"Generated structure documentation: {doc_path.relative_to(PROJECT_ROOT)}")
        return True
    except Exception as e:
        print_error(f"Failed to generate structure documentation: {e}")
        logger.error(f"Failed to generate structure documentation: {e}")
        return False

# =====================================================================
# Restructuring Functions
# =====================================================================

def restructure_folder_contents(root_dir, rules, dry_run=False):
    """
    จัดโครงสร้างภายในโฟลเดอร์ตามกฎที่กำหนด
    
    Args:
        root_dir (Path): ไดเรกทอรีรูท
        rules (list): กฎสำหรับจัดโครงสร้างภายในโฟลเดอร์
        dry_run (bool): ถ้าเป็นจริง จะแสดงเฉพาะการเปลี่ยนแปลงที่จะเกิดขึ้นโดยไม่ทำจริง
        
    Returns:
        int: จำนวนไฟล์ที่ย้าย
    """
    moved_count = 0
    
    for rule in rules:
        folder_path = root_dir / rule['folder']
        if not folder_path.exists():
            print_warning(f"Folder does not exist: {folder_path.relative_to(root_dir)}")
            continue
            
        print_info(f"\n📁 Organizing {rule['folder']} folder - {rule['description']}")
        
        # สร้างโฟลเดอร์ย่อย
        for subfolder in rule['subfolders']:
            subfolder_path = folder_path / subfolder['name']
            if not subfolder_path.exists() and not dry_run:
                subfolder_path.mkdir(exist_ok=True, parents=True)
                print_info(f"  Created subfolder: {subfolder_path.relative_to(root_dir)}")
                
                # สร้างไฟล์ .gitkeep เพื่อให้ git ติดตามโฟลเดอร์เปล่า
                if list(subfolder_path.glob('*')) == []:
                    gitkeep_path = subfolder_path / '.gitkeep'
                    if not dry_run:
                        with open(gitkeep_path, 'w') as f:
                            f.write('# This file ensures the directory is not empty\n# and will be tracked by git')
                        print_info(f"  Added .gitkeep to: {subfolder_path.relative_to(root_dir)}")
            
            # ค้นหาไฟล์ที่ตรงกับรูปแบบและย้าย
            for pattern in subfolder['patterns']:
                for file_path in folder_path.glob(pattern):
                    if file_path.is_file() and file_path.name != '.gitkeep':
                        new_path = subfolder_path / file_path.name
                        
                        # ตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์ย่อยอยู่แล้วหรือไม่
                        relative_path = file_path.relative_to(folder_path)
                        parent_dir = relative_path.parts[0] if len(relative_path.parts) > 1 else None
                        
                        # ย้ายเฉพาะไฟล์ที่ไม่ได้อยู่ในโฟลเดอร์ย่อยที่ต้องการ
                        if parent_dir != subfolder['name']:
                            print_info(f"  {'[DRY RUN] Would move' if dry_run else 'Moving'}: {file_path.relative_to(root_dir)} → {new_path.relative_to(root_dir)}")
                            if not dry_run:
                                try:
                                    # สร้างโฟลเดอร์เป้าหมายถ้ายังไม่มี
                                    new_path.parent.mkdir(exist_ok=True, parents=True)
                                    # ย้ายไฟล์
                                    shutil.move(str(file_path), str(new_path))
                                    moved_count += 1
                                except Exception as e:
                                    print_error(f"    Error: {e}")
    
    return moved_count

def apply_restructure_rules(dry_run=False):
    """Apply restructuring rules to move files"""
    print_header("🔄 APPLYING RESTRUCTURING RULES")
    
    moved_files = 0
    failed_moves = 0
    
    for rule in RESTRUCTURE_RULES:
        print_info(f"\n➡️ {rule['description']}")
        
        # Find files matching patterns
        files = find_files_by_patterns(rule['source_patterns'], rule['exclude'])
        
        if not files:
            print_warning(f"No files found for rule: {rule['description']}")
            continue
        
        print_info(f"Found {len(files)} files to move")
        
        # Move each file
        target_dir = PROJECT_ROOT / rule['target_dir']
        for file_path in files:
            if move_file(file_path, target_dir, dry_run):
                moved_files += 1
            else:
                failed_moves += 1
    
    print_header("📊 RESTRUCTURING SUMMARY")
    print_info(f"Files moved: {moved_files}")
    print_info(f"Failed moves: {failed_moves}")
    
    return moved_files > 0 and failed_moves == 0

def apply_consolidation_rules(dry_run=False):
    """Apply consolidation rules to combine files"""
    print_header("🔄 APPLYING CONSOLIDATION RULES")
    
    consolidated_files = 0
    failed_consolidations = 0
    
    for rule in CONSOLIDATION_RULES:
        print_info(f"\n➡️ {rule['description']}")
        
        # Prepare paths
        target_file = PROJECT_ROOT / rule['target_dir'] / rule['name']
        source_files = [PROJECT_ROOT / src for src in rule['source_files']]
        
        # Consolidate files
        if consolidate_files(source_files, target_file, dry_run):
            consolidated_files += 1
        else:
            failed_consolidations += 1
    
    print_header("📊 CONSOLIDATION SUMMARY")
    print_info(f"File sets consolidated: {consolidated_files}")
    print_info(f"Failed consolidations: {failed_consolidations}")
    
    return consolidated_files > 0 and failed_consolidations == 0

def update_import_references(dry_run=False):
    """Update import references in Python files"""
    print_header("🔄 UPDATING IMPORT REFERENCES")
    
    # Find all Python files
    python_files = list(PROJECT_ROOT.glob("**/*.py"))
    
    updated_files = 0
    failed_updates = 0
    
    # Define import mappings based on restructuring
    import_mappings = []
    
    # Add mappings for consolidation rules
    for rule in CONSOLIDATION_RULES:
        target_module = f"{rule['target_dir']}.{Path(rule['name']).stem}"
        for source_file in rule['source_files']:
            source_module = ".".join(Path(source_file).parts[:-1] + (Path(source_file).stem,))
            import_mappings.append((source_module, target_module))
    
    # Update imports in each file
    for file_path in python_files:
        old_imports = [mapping[0] for mapping in import_mappings]
        new_imports = [mapping[1] for mapping in import_mappings]
        
        if update_imports(file_path, old_imports, new_imports, dry_run):
            updated_files += 1
        else:
            failed_updates += 1
    
    print_header("📊 IMPORT UPDATE SUMMARY")
    print_info(f"Files updated: {updated_files}")
    print_info(f"Failed updates: {failed_updates}")
    
    return failed_updates == 0

# =====================================================================
# Main Functions
# =====================================================================

def restructure_project(dry_run=False):
    """Restructure the entire project"""
    print_header("🚀 STARTING AMULET-AI PROJECT RESTRUCTURING", Colors.MAGENTA)
    
    # Record start time
    start_time = time.time()
    
    # Create necessary directories
    required_dirs = [
        'scripts', 'config', 'tools', 'docs', 'archive',
        'ai_models/core', 'ai_models/training', 'ai_models/pipelines', 'ai_models/evaluation', 'ai_models/configs', 'ai_models/docs',
        'backend/api', 'backend/models', 'backend/services', 'backend/config', 'backend/tests',
        'frontend/pages', 'frontend/components', 'frontend/utils',
        'docs/api', 'docs/guides', 'docs/system', 'docs/development',
        'utils/config', 'utils/image', 'utils/logging', 'utils/data',
        'tests/unit', 'tests/integration', 'tests/fixtures', 'tests/data'
    ]
    
    for dir_name in required_dirs:
        dir_path = PROJECT_ROOT / dir_name
        ensure_directory(dir_path)
        
        # สร้างไฟล์ .gitkeep เพื่อให้ git ติดตามโฟลเดอร์เปล่า
        if list(dir_path.glob('*')) == [] and not dry_run:
            gitkeep_path = dir_path / '.gitkeep'
            with open(gitkeep_path, 'w') as f:
                f.write('# This file ensures the directory is not empty\n# and will be tracked by git')
    
    if not dry_run:
        # Backup the project first
        if not backup_project():
            print_error("Backup failed. Aborting restructuring.")
            return False
    
    # Apply restructuring rules
    if not apply_restructure_rules(dry_run):
        print_warning("Some issues occurred during restructuring.")
    
    # Apply consolidation rules
    if not apply_consolidation_rules(dry_run):
        print_warning("Some issues occurred during file consolidation.")
    
    # Apply internal folder restructuring
    print_header("🔄 APPLYING INTERNAL FOLDER RESTRUCTURING")
    moved_internal_files = restructure_folder_contents(PROJECT_ROOT, INTERNAL_RESTRUCTURE_RULES, dry_run)
    print_info(f"Internal files moved: {moved_internal_files}")
    
    # Update import references
    if not update_import_references(dry_run):
        print_warning("Some issues occurred during import updates.")
    
    # Generate documentation
    if not dry_run:
        generate_structure_doc()
    
    # Calculate elapsed time
    elapsed = time.time() - start_time
    
    print_header("✅ PROJECT RESTRUCTURING COMPLETE", Colors.MAGENTA)
    print_info(f"Restructuring completed in {elapsed:.2f} seconds")
    
    if dry_run:
        print_warning("This was a dry run - no changes were made.")
        print_info("Run without --dry-run to apply changes.")
    else:
        print_success("The project has been restructured.")
        print_info("Please check the logs for details.")
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Amulet-AI Project Restructuring Tool')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without applying them')
    parser.add_argument('--backup-only', action='store_true', help='Only create a backup, no restructuring')
    args = parser.parse_args()
    
    # Ensure logs directory exists
    logs_dir = PROJECT_ROOT / "logs"
    if not logs_dir.exists():
        logs_dir.mkdir(parents=True)
    
    if args.backup_only:
        backup_project()
    else:
        restructure_project(dry_run=args.dry_run)

if __name__ == "__main__":
    main()
