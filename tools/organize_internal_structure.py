"""
Amulet-AI Internal Folder Organization Tool
เครื่องมือจัดระเบียบภายในโฟลเดอร์สำหรับโปรเจค Amulet-AI

สคริปต์นี้จะช่วยในการจัดระเบียบโครงสร้างภายในโฟลเดอร์ของโปรเจค Amulet-AI:
- จัดหมวดหมู่ไฟล์ภายในโฟลเดอร์หลัก
- สร้างโฟลเดอร์ย่อยตามประเภทการใช้งาน
- ย้ายไฟล์ไปยังโฟลเดอร์ย่อยที่เหมาะสม
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import from restructure_project.py
try:
    from tools.restructure_project import (
        PROJECT_ROOT, INTERNAL_RESTRUCTURE_RULES, 
        restructure_folder_contents, print_header, 
        print_success, print_info, print_warning, 
        print_error, Colors, generate_structure_doc
    )
except ImportError:
    print("Error: Could not import from restructure_project.py")
    print("Please make sure the file exists and is accessible.")
    sys.exit(1)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Amulet-AI Internal Folder Organization Tool')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without applying them')
    parser.add_argument('--folder', type=str, help='Organize only a specific folder (e.g., "ai_models")')
    args = parser.parse_args()
    
    print_header("🚀 STARTING AMULET-AI INTERNAL FOLDER ORGANIZATION", Colors.MAGENTA)
    
    # Filter rules if a specific folder is specified
    rules = INTERNAL_RESTRUCTURE_RULES
    if args.folder:
        rules = [rule for rule in INTERNAL_RESTRUCTURE_RULES if rule['folder'] == args.folder]
        if not rules:
            print_error(f"No rules found for folder: {args.folder}")
            print_info("Available folders:")
            for rule in INTERNAL_RESTRUCTURE_RULES:
                print_info(f"- {rule['folder']}")
            return 1
    
    # Create necessary directories
    for rule in rules:
        folder_path = PROJECT_ROOT / rule['folder']
        if not folder_path.exists():
            print_warning(f"Folder not found: {rule['folder']}")
            continue
            
        for subfolder in rule['subfolders']:
            subfolder_path = folder_path / subfolder['name']
            if not subfolder_path.exists() and not args.dry_run:
                subfolder_path.mkdir(exist_ok=True, parents=True)
                print_info(f"Created subfolder: {subfolder_path.relative_to(PROJECT_ROOT)}")
                
    # Apply internal folder restructuring
    moved_count = restructure_folder_contents(PROJECT_ROOT, rules, args.dry_run)
    
    # Generate updated documentation
    if not args.dry_run and moved_count > 0:
        generate_structure_doc()
    
    print_header("✅ INTERNAL FOLDER ORGANIZATION COMPLETE", Colors.MAGENTA)
    print_info(f"Files moved: {moved_count}")
    
    if args.dry_run:
        print_warning("This was a dry run - no changes were made.")
        print_info("Run without --dry-run to apply changes.")
    else:
        print_success("The internal folder structure has been organized.")
        print_info("Please check the documentation for details.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
