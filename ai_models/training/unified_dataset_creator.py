"""
📦 Unified Dataset Creator for Amulet Classification
สคริปต์สำหรับรวมข้อมูลและแบ่งข้อมูลสำหรับฝึกฝน
"""
import os
import sys
import json
import shutil
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dataset_creation.log")
    ]
)
logger = logging.getLogger("dataset_creation")

class UnifiedDatasetCreator:
    """Creates a unified dataset from multiple source directories with train/val/test splits"""
    
    def __init__(self, config: Dict):
        """Initialize with configuration"""
        self.config = config
        self.source_dirs = config['source_dirs']
        self.output_dir = Path(config['output_dir'])
        self.split_ratio = config.get('split_ratio', {'train': 0.7, 'validation': 0.15, 'test': 0.15})
        self.min_samples = config.get('min_samples', 10)
        self.max_samples = config.get('max_samples', None)
        self.small_class_threshold = config.get('small_class_threshold', 30)
        self.small_class_strategy = config.get('small_class_strategy', 'all_train')
        self.organize_by_class = config.get('organize_by_class', True)
        self.class_map = {}
        self.class_stats = {}
        self.image_extensions = {'.jpg', '.jpeg', '.png'}
        self.validate_config()
        
    def validate_config(self):
        """Validate configuration parameters"""
        # Check split ratio
        split_sum = sum(self.split_ratio.values())
        if not abs(split_sum - 1.0) < 0.001:
            raise ValueError(f"Split ratio values must sum to 1.0, got {split_sum}")
        
        # Check source directories
        for source_dir in self.source_dirs:
            if not Path(source_dir).exists():
                logger.warning(f"Source directory {source_dir} does not exist, it will be skipped")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Configuration validated")
    
    def scan_source_directories(self) -> Dict:
        """
        Scan source directories to collect class and file information
        Returns a dictionary of classes and their files
        """
        classes = {}
        logger.info("Scanning source directories...")
        
        for source_dir in self.source_dirs:
            source_path = Path(source_dir)
            if not source_path.exists():
                continue
                
            # Check if this is a class directory or a directory containing class directories
            if any(f.is_dir() for f in source_path.iterdir()):
                # Directory contains subdirectories - assume each is a class
                for class_dir in source_path.iterdir():
                    if not class_dir.is_dir():
                        continue
                        
                    class_name = class_dir.name
                    class_files = self.get_image_files(class_dir)
                    
                    if class_name not in classes:
                        classes[class_name] = []
                    
                    classes[class_name].extend(class_files)
                    logger.info(f"Found {len(class_files)} images in class {class_name} from {source_dir}")
            else:
                # Assume this is a single class directory
                class_name = source_path.name
                class_files = self.get_image_files(source_path)
                
                if class_name not in classes:
                    classes[class_name] = []
                
                classes[class_name].extend(class_files)
                logger.info(f"Found {len(class_files)} images in class {class_name} from {source_dir}")
        
        # Generate class statistics
        for class_name, files in classes.items():
            self.class_stats[class_name] = {
                'total_files': len(files),
                'is_small_class': len(files) < self.small_class_threshold
            }
        
        # Create class map
        for idx, class_name in enumerate(sorted(classes.keys())):
            self.class_map[str(idx)] = class_name
        
        # Save class info
        with open(self.output_dir / 'class_info.json', 'w', encoding='utf-8') as f:
            json.dump(self.class_stats, f, indent=2, ensure_ascii=False)
        
        # Save class map
        with open(self.output_dir / 'labels.json', 'w', encoding='utf-8') as f:
            json.dump(self.class_map, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Found {len(classes)} classes with a total of {sum(len(files) for files in classes.values())} images")
        return classes
    
    def get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files in a directory (recursively)"""
        image_files = []
        
        for file_path in directory.glob('**/*'):
            if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                image_files.append(file_path)
        
        return image_files
    
    def create_splits(self, classes: Dict) -> Dict:
        """
        Create train/validation/test splits for each class
        Returns a dictionary with split information
        """
        splits = {'train': {}, 'validation': {}, 'test': {}}
        
        for class_name, files in classes.items():
            # Check if this is a small class
            is_small_class = len(files) < self.small_class_threshold
            
            # For small classes, follow the configured strategy
            if is_small_class and self.small_class_strategy == 'all_train':
                # Put all samples in training set
                splits['train'][class_name] = files
                splits['validation'][class_name] = []
                splits['test'][class_name] = []
                continue
            
            # Shuffle files to ensure random selection
            random.shuffle(files)
            
            # Limit the number of samples if configured
            if self.max_samples and len(files) > self.max_samples:
                files = files[:self.max_samples]
            
            # Check if we have enough samples
            if len(files) < self.min_samples:
                logger.warning(f"Class {class_name} has only {len(files)} samples, below minimum of {self.min_samples}")
                continue
            
            # Calculate split sizes
            train_size = int(len(files) * self.split_ratio['train'])
            val_size = int(len(files) * self.split_ratio['validation'])
            
            # Ensure we have at least some samples in each split
            train_size = max(train_size, 1)
            val_size = max(val_size, 1)
            
            # Adjust if needed to ensure we don't exceed the file count
            if train_size + val_size > len(files):
                # Reduce validation size if needed
                val_size = min(val_size, len(files) - train_size)
            
            # Create splits
            train_files = files[:train_size]
            val_files = files[train_size:train_size + val_size]
            test_files = files[train_size + val_size:]
            
            splits['train'][class_name] = train_files
            splits['validation'][class_name] = val_files
            splits['test'][class_name] = test_files
            
            logger.info(
                f"Split for {class_name}: "
                f"Train={len(train_files)}, "
                f"Validation={len(val_files)}, "
                f"Test={len(test_files)}"
            )
        
        return splits
    
    def create_unified_dataset(self, splits: Dict):
        """Create the unified dataset structure"""
        logger.info("Creating unified dataset structure...")
        
        # Create split directories
        for split_name in splits.keys():
            split_dir = self.output_dir / split_name
            split_dir.mkdir(exist_ok=True)
        
        # Copy files to appropriate locations
        for split_name, class_files in splits.items():
            split_dir = self.output_dir / split_name
            
            for class_name, files in tqdm(class_files.items(), desc=f"Processing {split_name} split"):
                class_dir = split_dir / class_name
                class_dir.mkdir(exist_ok=True)
                
                for src_file in files:
                    # Create destination path
                    dest_file = class_dir / src_file.name
                    
                    # Handle filename conflicts by adding a unique suffix
                    if dest_file.exists():
                        suffix = 1
                        while True:
                            new_name = f"{dest_file.stem}_{suffix}{dest_file.suffix}"
                            dest_file = class_dir / new_name
                            if not dest_file.exists():
                                break
                            suffix += 1
                    
                    # Copy file
                    try:
                        shutil.copy2(src_file, dest_file)
                    except Exception as e:
                        logger.error(f"Error copying {src_file} to {dest_file}: {e}")
        
        logger.info("Unified dataset created successfully")
    
    def create_dataset_summary(self):
        """Create a summary of the dataset"""
        logger.info("Creating dataset summary...")
        
        # Collect statistics
        stats = {
            'total_classes': len(self.class_map),
            'total_images': 0,
            'splits': {},
            'class_distribution': {},
            'small_classes': []
        }
        
        # Count images in each split and class
        for split_name in ['train', 'validation', 'test']:
            split_dir = self.output_dir / split_name
            if not split_dir.exists():
                continue
                
            split_count = 0
            class_counts = {}
            
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                    
                class_name = class_dir.name
                image_count = len(list(self.get_image_files(class_dir)))
                
                class_counts[class_name] = image_count
                split_count += image_count
                
                # Update class distribution
                if class_name not in stats['class_distribution']:
                    stats['class_distribution'][class_name] = {}
                
                stats['class_distribution'][class_name][split_name] = image_count
            
            stats['splits'][split_name] = {
                'total_images': split_count,
                'class_counts': class_counts
            }
            
            stats['total_images'] += split_count
        
        # Identify small classes
        for class_name, distribution in stats['class_distribution'].items():
            total_class_images = sum(distribution.values())
            if total_class_images < self.small_class_threshold:
                stats['small_classes'].append({
                    'name': class_name,
                    'total_images': total_class_images
                })
        
        # Save summary
        summary_path = self.output_dir / 'dataset_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # Create a visualization of class distribution
        self.visualize_class_distribution(stats)
        
        logger.info(f"Dataset summary created and saved to {summary_path}")
        
        return stats
    
    def visualize_class_distribution(self, stats: Dict):
        """Create a visualization of class distribution"""
        plt.figure(figsize=(12, 8))
        
        # Extract data for plotting
        class_names = list(stats['class_distribution'].keys())
        train_counts = []
        val_counts = []
        test_counts = []
        
        for class_name in class_names:
            distribution = stats['class_distribution'][class_name]
            train_counts.append(distribution.get('train', 0))
            val_counts.append(distribution.get('validation', 0))
            test_counts.append(distribution.get('test', 0))
        
        # Create bar chart
        bar_width = 0.25
        index = np.arange(len(class_names))
        
        plt.bar(index, train_counts, bar_width, label='Train')
        plt.bar(index + bar_width, val_counts, bar_width, label='Validation')
        plt.bar(index + 2 * bar_width, test_counts, bar_width, label='Test')
        
        plt.xlabel('Class')
        plt.ylabel('Number of Images')
        plt.title('Class Distribution across Splits')
        plt.xticks(index + bar_width, class_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_unified_dataset_complete(self):
        """Complete process to create unified dataset"""
        # Scan source directories
        classes = self.scan_source_directories()
        
        # Create splits
        splits = self.create_splits(classes)
        
        # Create dataset structure
        self.create_unified_dataset(splits)
        
        # Create summary
        summary = self.create_dataset_summary()
        
        logger.info("Unified dataset creation completed successfully!")
        return summary

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='Create unified dataset for Amulet Classification')
    parser.add_argument('--sources', nargs='+', required=True, help='Source directories containing amulet images')
    parser.add_argument('--output', required=True, help='Output directory for the unified dataset')
    parser.add_argument('--train', type=float, default=0.7, help='Training set ratio (default: 0.7)')
    parser.add_argument('--val', type=float, default=0.15, help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test', type=float, default=0.15, help='Test set ratio (default: 0.15)')
    parser.add_argument('--min-samples', type=int, default=10, help='Minimum samples per class (default: 10)')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum samples per class (default: None)')
    parser.add_argument('--small-class', type=int, default=30, help='Threshold for small classes (default: 30)')
    parser.add_argument('--small-class-strategy', choices=['all_train', 'proportional'], default='all_train',
                        help='Strategy for handling small classes (default: all_train)')
    
    args = parser.parse_args()
    
    # Check if split ratios sum to 1
    split_sum = args.train + args.val + args.test
    if not abs(split_sum - 1.0) < 0.001:
        parser.error(f"Split ratios must sum to 1.0, got {split_sum}")
    
    # Create configuration
    config = {
        'source_dirs': args.sources,
        'output_dir': args.output,
        'split_ratio': {
            'train': args.train,
            'validation': args.val,
            'test': args.test
        },
        'min_samples': args.min_samples,
        'max_samples': args.max_samples,
        'small_class_threshold': args.small_class,
        'small_class_strategy': args.small_class_strategy
    }
    
    # Create unified dataset
    creator = UnifiedDatasetCreator(config)
    summary = creator.create_unified_dataset_complete()
    
    print(f"Unified dataset created successfully!")
    print(f"Total classes: {summary['total_classes']}")
    print(f"Total images: {summary['total_images']}")
    print(f"Train split: {summary['splits']['train']['total_images']} images")
    print(f"Validation split: {summary['splits']['validation']['total_images']} images")
    print(f"Test split: {summary['splits']['test']['total_images']} images")
    
    if summary['small_classes']:
        print(f"\nSmall classes ({len(summary['small_classes'])}):")
        for small_class in summary['small_classes']:
            print(f"  - {small_class['name']}: {small_class['total_images']} images")

if __name__ == "__main__":
    main()
