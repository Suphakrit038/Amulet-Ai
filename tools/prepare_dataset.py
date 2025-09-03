#!/usr/bin/env python3
"""
Amulet-AI Dataset Preparation Tool
เครื่องมือเตรียมข้อมูลฝึกสอน Amulet-AI
"""
import os
import sys
import json
import shutil
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Thai to English folder name mappings
FOLDER_MAPPINGS = {
    'พระพุทธเจ้าในวิหาร': 'buddha_in_vihara',
    'พระสมเด็จฐานสิงห์': 'somdej_thansing',
    'พระสมเด็จประทานพร พุทธกวัก': 'somdej_pudtagueg',
    'พระสมเด็จหลังรูปเหมือน': 'somdej_portrait_back',
    'พระสรรค์': 'phra_san',
    'พระสิวลี': 'phra_sivali',
    'สมเด็จพิมพ์ปรกโพธิ์ 9 ใบ': 'somdej_prok_bodhi',
    'สมเด็จแหวกม่าน': 'somdej_waek_man',
    'ออกวัดหนองอีดุก': 'wat_nong_e_duk',
    'somdej-fatherguay': 'somdej_fatherguay'
}

# Reverse mapping for display
REVERSE_MAPPINGS = {v: k for k, v in FOLDER_MAPPINGS.items()}

def check_dataset_structure(base_path):
    """Check if the dataset has the expected structure."""
    base_path = Path(base_path)
    
    # Check if dataset folders exist
    dataset_path = base_path / 'dataset'
    dataset_split_path = base_path / 'dataset_split'
    
    if not dataset_path.exists() or not dataset_split_path.exists():
        logger.error(f"Missing required dataset folders. Make sure both dataset/ and dataset_split/ folders exist.")
        return False
    
    # Check if we have any Thai folder names
    thai_folders = []
    for folder in dataset_path.iterdir():
        if folder.is_dir() and any(c in 'กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฤฦะัาำิีึืุูเแโใไ็่้๊๋์ฯๆ' for c in folder.name):
            thai_folders.append(folder.name)
    
    for folder in dataset_split_path.glob('*/*'):
        if folder.is_dir() and any(c in 'กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฤฦะัาำิีึืุูเแโใไ็่้๊๋์ฯๆ' for c in folder.name):
            thai_folders.append(folder.name)
    
    if not thai_folders:
        logger.info("Dataset appears to be already in English format. No Thai folder names found.")
        return True
    
    logger.info(f"Found {len(thai_folders)} folders with Thai names. Reorganization needed.")
    return False

def create_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")

def organize_dataset(base_path, dry_run=False):
    """
    Organize the dataset by:
    1. Converting Thai folder names to English
    2. Organizing front/back images into proper subdirectories
    """
    base_path = Path(base_path)
    dataset_path = base_path / 'dataset'
    dataset_split_path = base_path / 'dataset_split'
    
    stats = {
        'main_dataset': {'moved': 0, 'skipped': 0, 'errors': 0},
        'dataset_split': {'moved': 0, 'skipped': 0, 'errors': 0}
    }
    
    # Process main dataset
    for thai_name, english_name in FOLDER_MAPPINGS.items():
        thai_folder = dataset_path / thai_name
        if not thai_folder.exists():
            continue
        
        english_folder = dataset_path / english_name
        
        # Create front and back directories
        front_dir = english_folder / 'front_color'
        back_dir = english_folder / 'back_color'
        
        if not dry_run:
            create_directory(english_folder)
            create_directory(front_dir)
            create_directory(back_dir)
        
        # Move files to appropriate directories
        for file_path in thai_folder.glob('*.*'):
            if not file_path.is_file():
                continue
            
            filename = file_path.name.lower()
            
            try:
                if 'front' in filename:
                    dest_path = front_dir / file_path.name
                    if not dry_run:
                        if not dest_path.exists():
                            shutil.copy2(file_path, dest_path)
                            logger.info(f"Copied: {file_path} -> {dest_path}")
                            stats['main_dataset']['moved'] += 1
                        else:
                            logger.warning(f"Skipped (already exists): {dest_path}")
                            stats['main_dataset']['skipped'] += 1
                elif 'back' in filename:
                    dest_path = back_dir / file_path.name
                    if not dry_run:
                        if not dest_path.exists():
                            shutil.copy2(file_path, dest_path)
                            logger.info(f"Copied: {file_path} -> {dest_path}")
                            stats['main_dataset']['moved'] += 1
                        else:
                            logger.warning(f"Skipped (already exists): {dest_path}")
                            stats['main_dataset']['skipped'] += 1
                else:
                    # If not clearly front or back, put in front by default
                    dest_path = front_dir / file_path.name
                    if not dry_run:
                        if not dest_path.exists():
                            shutil.copy2(file_path, dest_path)
                            logger.info(f"Copied (default to front): {file_path} -> {dest_path}")
                            stats['main_dataset']['moved'] += 1
                        else:
                            logger.warning(f"Skipped (already exists): {dest_path}")
                            stats['main_dataset']['skipped'] += 1
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                stats['main_dataset']['errors'] += 1
    
    # Process dataset_split (train/test/validation)
    for split_folder in ['train', 'test', 'validation']:
        split_path = dataset_split_path / split_folder
        if not split_path.exists():
            continue
        
        for thai_name, english_name in FOLDER_MAPPINGS.items():
            thai_folder = split_path / thai_name
            if not thai_folder.exists():
                continue
            
            english_folder = split_path / english_name
            
            # Create front and back directories
            front_dir = english_folder / 'front_color'
            back_dir = english_folder / 'back_color'
            
            if not dry_run:
                create_directory(english_folder)
                create_directory(front_dir)
                create_directory(back_dir)
            
            # Move files to appropriate directories
            for file_path in thai_folder.glob('*.*'):
                if not file_path.is_file():
                    continue
                
                filename = file_path.name.lower()
                
                try:
                    if 'front' in filename:
                        dest_path = front_dir / file_path.name
                        if not dry_run:
                            if not dest_path.exists():
                                shutil.copy2(file_path, dest_path)
                                logger.info(f"Copied: {file_path} -> {dest_path}")
                                stats['dataset_split']['moved'] += 1
                            else:
                                logger.warning(f"Skipped (already exists): {dest_path}")
                                stats['dataset_split']['skipped'] += 1
                    elif 'back' in filename:
                        dest_path = back_dir / file_path.name
                        if not dry_run:
                            if not dest_path.exists():
                                shutil.copy2(file_path, dest_path)
                                logger.info(f"Copied: {file_path} -> {dest_path}")
                                stats['dataset_split']['moved'] += 1
                            else:
                                logger.warning(f"Skipped (already exists): {dest_path}")
                                stats['dataset_split']['skipped'] += 1
                    else:
                        # If not clearly front or back, put in front by default
                        dest_path = front_dir / file_path.name
                        if not dry_run:
                            if not dest_path.exists():
                                shutil.copy2(file_path, dest_path)
                                logger.info(f"Copied (default to front): {file_path} -> {dest_path}")
                                stats['dataset_split']['moved'] += 1
                            else:
                                logger.warning(f"Skipped (already exists): {dest_path}")
                                stats['dataset_split']['skipped'] += 1
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    stats['dataset_split']['errors'] += 1
    
    # Save reorganization report
    if not dry_run:
        report_path = base_path / 'dataset_organized' / 'reorganization_report.json'
        create_directory(base_path / 'dataset_organized')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'stats': stats,
                'mappings': FOLDER_MAPPINGS,
                'timestamp': import_time(),
                'total_files_moved': stats['main_dataset']['moved'] + stats['dataset_split']['moved'],
                'total_files_skipped': stats['main_dataset']['skipped'] + stats['dataset_split']['skipped'],
                'total_errors': stats['main_dataset']['errors'] + stats['dataset_split']['errors']
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote reorganization report to: {report_path}")
    
    return stats

def import_time():
    """Return current time as a string."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def remove_empty_thai_folders(base_path, dry_run=False):
    """Remove empty Thai folders after reorganization."""
    base_path = Path(base_path)
    
    # Function to check if directory contains only empty directories
    def is_effectively_empty(path):
        if not path.is_dir():
            return False
        
        for item in path.iterdir():
            if item.is_file():
                return False
            if item.is_dir() and not is_effectively_empty(item):
                return False
        
        return True
    
    # Find Thai folders in dataset and dataset_split
    thai_folders = []
    
    # Check dataset folder
    dataset_path = base_path / 'dataset'
    for folder in dataset_path.iterdir():
        if folder.is_dir() and any(c in 'กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฤฦะัาำิีึืุูเแโใไ็่้๊๋์ฯๆ' for c in folder.name):
            if is_effectively_empty(folder):
                thai_folders.append(folder)
    
    # Check dataset_split folder
    dataset_split_path = base_path / 'dataset_split'
    for split_folder in ['train', 'test', 'validation']:
        split_path = dataset_split_path / split_folder
        if not split_path.exists():
            continue
        
        for folder in split_path.iterdir():
            if folder.is_dir() and any(c in 'กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฤฦะัาำิีึืุูเแโใไ็่้๊๋์ฯๆ' for c in folder.name):
                if is_effectively_empty(folder):
                    thai_folders.append(folder)
    
    # Remove empty Thai folders
    removed_count = 0
    for folder in thai_folders:
        if not dry_run:
            try:
                shutil.rmtree(folder)
                logger.info(f"Removed empty folder: {folder}")
                removed_count += 1
            except Exception as e:
                logger.error(f"Error removing folder {folder}: {str(e)}")
        else:
            logger.info(f"Would remove folder: {folder}")
            removed_count += 1
    
    return removed_count

def print_summary(stats):
    """Print a summary of the reorganization."""
    print("\n" + "=" * 60)
    print("📊 Reorganization Summary")
    print("=" * 60 + "\n")
    
    print("Main Dataset:")
    print(f"  ✅ Files moved: {stats['main_dataset']['moved']}")
    print(f"  ⚠️ Files skipped: {stats['main_dataset']['skipped']}")
    print(f"  ❌ Files with errors: {stats['main_dataset']['errors']}")
    
    print("\nDataset Split:")
    print(f"  ✅ Files moved: {stats['dataset_split']['moved']}")
    print(f"  ⚠️ Files skipped: {stats['dataset_split']['skipped']}")
    print(f"  ❌ Files with errors: {stats['dataset_split']['errors']}")
    
    print("\nTotal:")
    print(f"  ✅ Files moved: {stats['main_dataset']['moved'] + stats['dataset_split']['moved']}")
    print(f"  ⚠️ Files skipped: {stats['main_dataset']['skipped'] + stats['dataset_split']['skipped']}")
    print(f"  ❌ Files with errors: {stats['main_dataset']['errors'] + stats['dataset_split']['errors']}")
    
    print("\nDone!")

def main():
    parser = argparse.ArgumentParser(description='Amulet-AI Dataset Preparation Tool')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without actually making changes')
    parser.add_argument('--check-only', action='store_true', help='Only check if reorganization is needed')
    parser.add_argument('--remove-empty', action='store_true', help='Remove empty Thai folders after reorganization')
    parser.add_argument('--base-path', type=str, default='.', help='Base path of the Amulet-AI project')
    
    args = parser.parse_args()
    
    # Convert relative path to absolute
    base_path = os.path.abspath(args.base_path)
    
    if not os.path.exists(base_path):
        logger.error(f"Base path does not exist: {base_path}")
        return 1
    
    # Check if we need to reorganize
    if args.check_only:
        if check_dataset_structure(base_path):
            print("✅ Dataset structure is already in English format. No reorganization needed.")
        else:
            print("⚠️ Dataset contains Thai folder names. Reorganization is recommended.")
        return 0
    
    # Perform reorganization
    if args.dry_run:
        print("🔍 DRY RUN: No actual changes will be made")
    
    stats = organize_dataset(base_path, args.dry_run)
    print_summary(stats)
    
    # Remove empty Thai folders if requested
    if args.remove_empty:
        print("\nRemoving empty Thai folders...")
        removed = remove_empty_thai_folders(base_path, args.dry_run)
        print(f"Removed {removed} empty Thai folders.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
