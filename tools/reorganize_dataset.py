#!/usr/bin/env python3
"""
Amulet-AI Dataset Reorganization Tool
เครื่องมือจัดระเบียบข้อมูลฝึกสอน Amulet-AI
"""
import os
import shutil
import json
import argparse
from pathlib import Path
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dataset_reorganize')

# Define the project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Define dataset paths
DATASET_PATH = PROJECT_ROOT / "dataset"
DATASET_ORGANIZED_PATH = PROJECT_ROOT / "dataset_organized"
DATASET_SPLIT_PATH = PROJECT_ROOT / "dataset_split"

# Define new structure with clearer English names and Thai name mappings
CATEGORY_MAPPING = {
    # Original folder name : (New English folder name, Thai folder name)
    "somdej-fatherguay": ("somdej_fatherguay", "สมเด็จหลวงพ่อกวย"),
    "พระพุทธเจ้าในวิหาร": ("buddha_in_vihara", "พระพุทธเจ้าในวิหาร"),
    "พระสมเด็จฐานสิงห์": ("somdej_lion_base", "พระสมเด็จฐานสิงห์"),
    "พระสมเด็จประทานพร พุทธกวัก": ("somdej_buddha_blessing", "พระสมเด็จประทานพร พุทธกวัก"),
    "พระสมเด็จหลังรูปเหมือน": ("somdej_portrait_back", "พระสมเด็จหลังรูปเหมือน"),
    "พระสรรค์": ("phra_san", "พระสรรค์"),
    "พระสิวลี": ("phra_sivali", "พระสิวลี"),
    "สมเด็จพิมพ์ปรกโพธิ์ 9 ใบ": ("somdej_prok_bodhi", "สมเด็จพิมพ์ปรกโพธิ์ 9 ใบ"),
    "สมเด็จแหวกม่าน": ("somdej_waek_man", "สมเด็จแหวกม่าน"),
    "ออกวัดหนองอีดุก": ("wat_nong_e_duk", "ออกวัดหนองอีดุก"),
    "วัดหนองอีดุก": ("wat_nong_e_duk_misc", "วัดหนองอีดุก")
}

# Define subfolders for organizing by image characteristics
IMAGE_CHARACTERISTIC_FOLDERS = {
    "front": "front_view",  # ด้านหน้า
    "back": "back_view",    # ด้านหลัง
    "bw": "black_white",    # ขาวดำ
    "color": "color"        # สี
}

def ensure_directory(dir_path):
    """Create directory if it doesn't exist"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory exists: {dir_path}")

def get_image_characteristics(filename):
    """Determine image characteristics from filename"""
    characteristics = []
    
    # Check if front or back view
    if "front" in filename.lower():
        characteristics.append("front")
    elif "back" in filename.lower() or "-b" in filename.lower():
        characteristics.append("back")
    else:
        characteristics.append("unknown_view")
    
    # Check if black and white or color
    if "bw" in filename.lower() or "(bw)" in filename.lower():
        characteristics.append("bw")
    else:
        characteristics.append("color")
    
    return characteristics

def organize_dataset(dry_run=False, target_folder=None):
    """Organize the dataset with a new structure"""
    # Ensure all required directories exist
    ensure_directory(DATASET_ORGANIZED_PATH)
    
    # Create the target folder list
    if target_folder:
        if target_folder in CATEGORY_MAPPING:
            target_folders = [target_folder]
        else:
            logger.error(f"Target folder '{target_folder}' not found in category mapping")
            return False
    else:
        target_folders = list(CATEGORY_MAPPING.keys())
    
    # Track statistics
    moved_files = 0
    failed_moves = 0
    skipped_files = 0
    
    # Create labels dictionary
    labels = {
        "categories": [],
        "mapping": {}
    }
    
    # Create each category folder and organize within
    for original_folder in target_folders:
        english_folder, thai_folder = CATEGORY_MAPPING[original_folder]
        
        # Add to labels
        labels["categories"].append(english_folder)
        labels["mapping"][english_folder] = {
            "thai_name": thai_folder,
            "original_folder": original_folder
        }
        
        # Create category folder
        category_dir = DATASET_ORGANIZED_PATH / english_folder
        ensure_directory(category_dir)
        
        # Create characteristic subfolders
        for characteristic_type, characteristic_folder in IMAGE_CHARACTERISTIC_FOLDERS.items():
            ensure_directory(category_dir / characteristic_folder)
        
        # Process files in the original folder
        original_dir = DATASET_PATH / original_folder
        if not original_dir.exists():
            logger.warning(f"Original directory does not exist: {original_dir}")
            continue
            
        for file_path in original_dir.glob("*"):
            if not file_path.is_file():
                continue
                
            # Get file characteristics
            characteristics = get_image_characteristics(file_path.name)
            
            # Determine subfolder based on characteristics
            subfolder = "_".join(characteristics)
            
            # Destination paths
            dest_dir = category_dir / subfolder
            ensure_directory(dest_dir)
            dest_path = dest_dir / file_path.name
            
            # Check if destination already exists
            if dest_path.exists():
                logger.warning(f"Destination file already exists: {dest_path}")
                skipped_files += 1
                continue
            
            # Move the file
            if dry_run:
                logger.info(f"Would move: {file_path} -> {dest_path}")
                moved_files += 1
            else:
                try:
                    shutil.copy2(file_path, dest_path)
                    logger.info(f"Copied: {file_path} -> {dest_path}")
                    moved_files += 1
                except Exception as e:
                    logger.error(f"Error copying file {file_path} to {dest_path}: {e}")
                    failed_moves += 1
    
    # Save labels.json
    labels_path = DATASET_ORGANIZED_PATH / "labels.json"
    if dry_run:
        logger.info(f"Would write labels to: {labels_path}")
    else:
        try:
            with open(labels_path, 'w', encoding='utf-8') as f:
                json.dump(labels, f, ensure_ascii=False, indent=2)
            logger.info(f"Wrote labels to: {labels_path}")
        except Exception as e:
            logger.error(f"Error writing labels to {labels_path}: {e}")
    
    # Generate karaoke-style labels (combining English and Thai)
    labels_karaoke = {
        "categories": [],
        "mapping": {}
    }
    
    for original_folder, (english_folder, thai_folder) in CATEGORY_MAPPING.items():
        karaoke_name = f"{english_folder} ({thai_folder})"
        labels_karaoke["categories"].append(karaoke_name)
        labels_karaoke["mapping"][karaoke_name] = {
            "english_name": english_folder,
            "thai_name": thai_folder,
            "original_folder": original_folder
        }
    
    # Save labels_karaoke.json
    labels_karaoke_path = DATASET_ORGANIZED_PATH / "labels_karaoke.json"
    if dry_run:
        logger.info(f"Would write karaoke labels to: {labels_karaoke_path}")
    else:
        try:
            with open(labels_karaoke_path, 'w', encoding='utf-8') as f:
                json.dump(labels_karaoke, f, ensure_ascii=False, indent=2)
            logger.info(f"Wrote karaoke labels to: {labels_karaoke_path}")
        except Exception as e:
            logger.error(f"Error writing karaoke labels to {labels_karaoke_path}: {e}")
    
    return moved_files, skipped_files, failed_moves

def organize_dataset_split(dry_run=False):
    """Organize the dataset_split folder structure"""
    # Ensure required directories exist
    for split in ["train", "validation", "test"]:
        ensure_directory(DATASET_SPLIT_PATH / split)
    
    # Track statistics
    moved_files = 0
    failed_moves = 0
    skipped_files = 0
    
    # Create the new structure in dataset_split
    for original_folder, (english_folder, thai_folder) in CATEGORY_MAPPING.items():
        # For each split (train, validation, test)
        for split in ["train", "validation", "test"]:
            original_split_dir = DATASET_SPLIT_PATH / split / original_folder
            
            if not original_split_dir.exists():
                continue
                
            # Create the new folder structure
            new_split_dir = DATASET_SPLIT_PATH / split / english_folder
            ensure_directory(new_split_dir)
            
            # Process each file
            for file_path in original_split_dir.glob("*"):
                if not file_path.is_file():
                    continue
                    
                # Get file characteristics
                characteristics = get_image_characteristics(file_path.name)
                
                # Determine subfolder based on characteristics
                subfolder = "_".join(characteristics)
                
                # Destination paths
                dest_dir = new_split_dir / subfolder
                ensure_directory(dest_dir)
                dest_path = dest_dir / file_path.name
                
                # Check if destination already exists
                if dest_path.exists():
                    logger.warning(f"Destination file already exists: {dest_path}")
                    skipped_files += 1
                    continue
                
                # Move the file
                if dry_run:
                    logger.info(f"Would move: {file_path} -> {dest_path}")
                    moved_files += 1
                else:
                    try:
                        shutil.copy2(file_path, dest_path)
                        logger.info(f"Copied: {file_path} -> {dest_path}")
                        moved_files += 1
                    except Exception as e:
                        logger.error(f"Error copying file {file_path} to {dest_path}: {e}")
                        failed_moves += 1
    
    return moved_files, skipped_files, failed_moves

def generate_report(moved_main, skipped_main, failed_main, moved_split, skipped_split, failed_split):
    """Generate a report of the reorganization"""
    report = {
        "dataset_organized": {
            "moved_files": moved_main,
            "skipped_files": skipped_main,
            "failed_moves": failed_main,
            "total_processed": moved_main + skipped_main + failed_main
        },
        "dataset_split": {
            "moved_files": moved_split,
            "skipped_files": skipped_split,
            "failed_moves": failed_split,
            "total_processed": moved_split + skipped_split + failed_split
        },
        "total": {
            "moved_files": moved_main + moved_split,
            "skipped_files": skipped_main + skipped_split,
            "failed_moves": failed_main + failed_split,
            "total_processed": moved_main + skipped_main + failed_main + moved_split + skipped_split + failed_split
        },
        "category_mapping": {
            original: {"english": english, "thai": thai}
            for original, (english, thai) in CATEGORY_MAPPING.items()
        }
    }
    
    # Save the report
    report_path = PROJECT_ROOT / "dataset_organized" / "reorganization_report.json"
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Wrote reorganization report to: {report_path}")
    except Exception as e:
        logger.error(f"Error writing reorganization report to {report_path}: {e}")
    
    return report

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Amulet-AI Dataset Reorganization Tool")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--folder", help="Process only a specific category folder")
    parser.add_argument("--skip-split", action="store_true", help="Skip reorganizing the dataset_split folder")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🏺 Amulet-AI Dataset Reorganization Tool")
    print("=" * 60)
    
    if args.dry_run:
        print("Running in dry-run mode (no files will be moved)")
    
    # Reorganize the main dataset
    print("\n📂 Reorganizing main dataset...")
    moved_main, skipped_main, failed_main = organize_dataset(args.dry_run, args.folder)
    
    # Reorganize the dataset_split folder
    moved_split, skipped_split, failed_split = 0, 0, 0
    if not args.skip_split:
        print("\n📂 Reorganizing dataset_split...")
        moved_split, skipped_split, failed_split = organize_dataset_split(args.dry_run)
    
    # Generate report
    if not args.dry_run:
        report = generate_report(moved_main, skipped_main, failed_main, moved_split, skipped_split, failed_split)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Reorganization Summary")
    print("=" * 60)
    
    print(f"\nMain Dataset:")
    print(f"  ✅ Files to move: {moved_main}")
    print(f"  ⚠️ Files to skip: {skipped_main}")
    print(f"  ❌ Files with errors: {failed_main}")
    
    if not args.skip_split:
        print(f"\nDataset Split:")
        print(f"  ✅ Files to move: {moved_split}")
        print(f"  ⚠️ Files to skip: {skipped_split}")
        print(f"  ❌ Files with errors: {failed_split}")
    
    print(f"\nTotal:")
    print(f"  ✅ Files to move: {moved_main + moved_split}")
    print(f"  ⚠️ Files to skip: {skipped_main + skipped_split}")
    print(f"  ❌ Files with errors: {failed_main + failed_split}")
    
    if args.dry_run:
        print("\n⚠️ This was a dry run. No files were actually moved.")
        print("   Run without --dry-run to perform the reorganization.")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
