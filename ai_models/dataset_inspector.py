#!/usr/bin/env python3
"""
🔍 Dataset Inspector for Amulet Recognition Project
Comprehensive analysis of image dataset for ML pipeline optimization

This tool provides deep insights into dataset characteristics to inform:
- Augmentation strategies (class-specific multipliers)
- Cross-validation approach (stratified k-fold recommendations)
- Data quality issues (corrupted files, duplicates)
- Preprocessing requirements (image sizes, channels)

Author: AI Assistant
Compatible: Python 3.13+
Dependencies: opencv-python, numpy, hashlib, json, argparse
"""

import os
import sys
import json
import hashlib
import argparse
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import cv2
import numpy as np


@dataclass
class ImageInfo:
    """Information about a single image file"""
    path: str
    width: int
    height: int
    channels: int
    size_bytes: int
    file_hash: str
    is_corrupted: bool = False
    error_message: str = ""


@dataclass
class ClassStats:
    """Statistics for a single class"""
    class_name: str
    total_images: int
    corrupted_images: int
    duplicate_groups: List[List[str]]
    size_distribution: Dict[str, int]  # "WxHxC": count
    sample_paths: List[str]
    recommended_augmentation: int
    health_status: str  # "healthy", "warning", "critical"


@dataclass
class DatasetSummary:
    """Overall dataset summary"""
    total_images: int
    total_classes: int
    avg_per_class: float
    min_per_class: int
    max_per_class: int
    imbalance_ratio: float
    total_corrupted: int
    total_duplicates: int
    recommended_k_fold: int
    cv_strategy_note: str


class DatasetInspector:
    """
    Comprehensive dataset inspection and analysis tool
    
    Performs deep analysis of image classification datasets to provide
    actionable insights for ML pipeline optimization.
    """
    
    def __init__(self, 
                 data_dir: str = "dataset",
                 min_target_per_class: int = 30,
                 sample_preview: int = 3):
        """
        Initialize dataset inspector
        
        Args:
            data_dir: Path to dataset directory (structure: dataset/<class>/*.jpg)
            min_target_per_class: Target minimum samples per class for augmentation
            sample_preview: Number of sample file paths to save per class
        """
        self.data_dir = Path(data_dir)
        self.min_target_per_class = min_target_per_class
        self.sample_preview = sample_preview
        
        # Valid image extensions
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Storage for analysis results
        self.class_stats: Dict[str, ClassStats] = {}
        self.summary: Optional[DatasetSummary] = None
        self.optional_data: Dict[str, Any] = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('DatasetInspector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def load_optional_files(self) -> None:
        """Load optional JSON files (labels.json, consolidation_report.json)"""
        self.logger.info("🔍 Loading optional JSON files...")
        
        # Load labels.json
        labels_file = Path("data_base/labels.json")
        if labels_file.exists():
            try:
                with open(labels_file, 'r', encoding='utf-8') as f:
                    self.optional_data['labels'] = json.load(f)
                self.logger.info(f"✅ Loaded labels.json with {len(self.optional_data['labels'])} entries")
            except Exception as e:
                self.logger.warning(f"⚠️ Error loading labels.json: {e}")
        
        # Load consolidation_report.json
        consolidation_file = Path("data_base/consolidation_report.json")
        if consolidation_file.exists():
            try:
                with open(consolidation_file, 'r', encoding='utf-8') as f:
                    self.optional_data['consolidation_report'] = json.load(f)
                self.logger.info("✅ Loaded consolidation_report.json")
            except Exception as e:
                self.logger.warning(f"⚠️ Error loading consolidation_report.json: {e}")
    
    def compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA-1 hash of file for duplicate detection
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA-1 hash string
        """
        hash_sha1 = hashlib.sha1()
        try:
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha1.update(chunk)
            return hash_sha1.hexdigest()
        except Exception as e:
            self.logger.warning(f"⚠️ Error hashing {file_path}: {e}")
            return "hash_error"
    
    def analyze_image(self, file_path: Path) -> ImageInfo:
        """
        Analyze a single image file
        
        Args:
            file_path: Path to image file
            
        Returns:
            ImageInfo object with analysis results
        """
        info = ImageInfo(
            path=str(file_path),
            width=0, height=0, channels=0,
            size_bytes=0, file_hash=""
        )
        
        try:
            # Get file size
            info.size_bytes = file_path.stat().st_size
            
            # Compute hash
            info.file_hash = self.compute_file_hash(file_path)
            
            # Try to read image with OpenCV
            image = cv2.imread(str(file_path))
            
            if image is None:
                info.is_corrupted = True
                info.error_message = "cv2.imread returned None"
                return info
            
            # Extract image properties
            info.height, info.width = image.shape[:2]
            info.channels = image.shape[2] if len(image.shape) == 3 else 1
            
        except Exception as e:
            info.is_corrupted = True
            info.error_message = str(e)
            self.logger.warning(f"⚠️ Error analyzing {file_path}: {e}")
        
        return info
    
    def scan_class_directory(self, class_dir: Path) -> ClassStats:
        """
        Scan a single class directory and analyze all images
        
        Args:
            class_dir: Path to class directory
            
        Returns:
            ClassStats object with analysis results
        """
        class_name = class_dir.name
        self.logger.info(f"📂 Scanning class: {class_name}")
        
        # Find all image files
        image_files = []
        for ext in self.valid_extensions:
            image_files.extend(class_dir.glob(f"*{ext}"))
            image_files.extend(class_dir.glob(f"*{ext.upper()}"))
        
        # Initialize class stats
        stats = ClassStats(
            class_name=class_name,
            total_images=len(image_files),
            corrupted_images=0,
            duplicate_groups=[],
            size_distribution={},
            sample_paths=[],
            recommended_augmentation=0,
            health_status="healthy"
        )
        
        if not image_files:
            stats.health_status = "critical"
            self.logger.warning(f"⚠️ No images found in {class_dir}")
            return stats
        
        # Analyze each image
        image_infos = []
        hash_to_paths = defaultdict(list)
        
        for img_file in image_files:
            info = self.analyze_image(img_file)
            image_infos.append(info)
            
            # Track corrupted images
            if info.is_corrupted:
                stats.corrupted_images += 1
            else:
                # Group by hash for duplicate detection
                hash_to_paths[info.file_hash].append(str(img_file))
                
                # Track size distribution
                size_key = f"{info.width}x{info.height}x{info.channels}"
                stats.size_distribution[size_key] = stats.size_distribution.get(size_key, 0) + 1
        
        # Find duplicate groups (hash appears multiple times)
        for file_hash, paths in hash_to_paths.items():
            if len(paths) > 1:
                stats.duplicate_groups.append(paths)
        
        # Select sample paths (non-corrupted images)
        valid_infos = [info for info in image_infos if not info.is_corrupted]
        stats.sample_paths = [info.path for info in valid_infos[:self.sample_preview]]
        
        # Calculate recommended augmentation
        valid_count = len(valid_infos)
        if valid_count < self.min_target_per_class:
            stats.recommended_augmentation = max(1, self.min_target_per_class // valid_count)
        
        # Determine health status
        if valid_count == 0:
            stats.health_status = "critical"
        elif valid_count < 5:
            stats.health_status = "critical"
        elif valid_count < 10 or stats.corrupted_images > valid_count * 0.1:
            stats.health_status = "warning"
        
        self.logger.info(f"  📊 {class_name}: {valid_count} valid, "
                        f"{stats.corrupted_images} corrupted, "
                        f"{len(stats.duplicate_groups)} duplicate groups")
        
        return stats
    
    def scan_dataset(self) -> None:
        """Scan entire dataset and analyze all classes"""
        self.logger.info(f"🚀 Starting dataset scan: {self.data_dir}")
        
        if not self.data_dir.exists():
            self.logger.error(f"❌ Dataset directory not found: {self.data_dir}")
            sys.exit(1)
        
        # Find all class directories
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        if not class_dirs:
            self.logger.error(f"❌ No class directories found in {self.data_dir}")
            sys.exit(1)
        
        self.logger.info(f"📁 Found {len(class_dirs)} class directories")
        
        # Scan each class
        for class_dir in sorted(class_dirs):
            try:
                stats = self.scan_class_directory(class_dir)
                self.class_stats[stats.class_name] = stats
            except Exception as e:
                self.logger.error(f"❌ Error scanning {class_dir}: {e}")
        
        # Generate summary statistics
        self._generate_summary()
    
    def _generate_summary(self) -> None:
        """Generate overall dataset summary statistics"""
        if not self.class_stats:
            self.logger.warning("⚠️ No class statistics available for summary")
            return
        
        # Calculate totals
        total_images = sum(stats.total_images for stats in self.class_stats.values())
        total_valid = sum(stats.total_images - stats.corrupted_images 
                         for stats in self.class_stats.values())
        total_corrupted = sum(stats.corrupted_images for stats in self.class_stats.values())
        total_duplicates = sum(len(group) - 1 for stats in self.class_stats.values() 
                              for group in stats.duplicate_groups)
        
        # Per-class statistics
        valid_counts = [stats.total_images - stats.corrupted_images 
                       for stats in self.class_stats.values()]
        
        min_per_class = min(valid_counts) if valid_counts else 0
        max_per_class = max(valid_counts) if valid_counts else 0
        avg_per_class = sum(valid_counts) / len(valid_counts) if valid_counts else 0
        imbalance_ratio = max_per_class / min_per_class if min_per_class > 0 else float('inf')
        
        # CV strategy recommendation
        recommended_k, cv_note = self.suggest_cv_strategy(min_per_class)
        
        self.summary = DatasetSummary(
            total_images=total_valid,
            total_classes=len(self.class_stats),
            avg_per_class=avg_per_class,
            min_per_class=min_per_class,
            max_per_class=max_per_class,
            imbalance_ratio=imbalance_ratio,
            total_corrupted=total_corrupted,
            total_duplicates=total_duplicates,
            recommended_k_fold=recommended_k,
            cv_strategy_note=cv_note
        )
        
        self.logger.info(f"📊 Dataset Summary: {total_valid} valid images, "
                        f"{len(self.class_stats)} classes, "
                        f"imbalance ratio: {imbalance_ratio:.1f}:1")
    
    def suggest_cv_strategy(self, min_samples_per_class: int) -> Tuple[int, str]:
        """
        Suggest cross-validation strategy based on dataset size
        
        Args:
            min_samples_per_class: Minimum number of samples in any class
            
        Returns:
            Tuple of (recommended_k, explanation_note)
        """
        if min_samples_per_class < 2:
            return 2, "❌ Critical: Some classes have <2 samples. Consider merging/excluding classes or collecting more data."
        elif min_samples_per_class < 5:
            k = min(3, min_samples_per_class)
            return k, f"⚠️ Warning: Small classes detected. Use {k}-fold CV with caution. Consider data augmentation."
        elif min_samples_per_class < 10:
            return 5, "✅ Use 5-fold stratified CV. Small dataset - augmentation recommended."
        else:
            return 5, "✅ Use 5-fold stratified CV. Dataset size adequate for robust validation."
    
    def generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if not self.summary:
            return ["❌ No analysis performed yet"]
        
        # Critical issues
        critical_classes = [name for name, stats in self.class_stats.items() 
                           if stats.health_status == "critical"]
        if critical_classes:
            recommendations.append(
                f"🚨 CRITICAL: {len(critical_classes)} classes have <5 samples: {critical_classes[:5]}..."
                f" Consider merging, excluding, or aggressive augmentation."
            )
        
        # Corruption issues
        if self.summary.total_corrupted > self.summary.total_images * 0.05:
            recommendations.append(
                f"⚠️ HIGH CORRUPTION: {self.summary.total_corrupted} corrupted files "
                f"({self.summary.total_corrupted/self.summary.total_images*100:.1f}%). "
                f"Clean dataset before training."
            )
        
        # Duplicate issues
        if self.summary.total_duplicates > self.summary.total_images * 0.05:
            recommendations.append(
                f"⚠️ MANY DUPLICATES: {self.summary.total_duplicates} duplicate images "
                f"({self.summary.total_duplicates/self.summary.total_images*100:.1f}%). "
                f"Consider deduplication."
            )
        
        # Imbalance issues
        if self.summary.imbalance_ratio > 10:
            recommendations.append(
                f"⚠️ HIGH IMBALANCE: {self.summary.imbalance_ratio:.1f}:1 ratio. "
                f"Use class weighting and stratified sampling."
            )
        
        # Augmentation recommendations
        classes_needing_aug = [name for name, stats in self.class_stats.items() 
                              if stats.recommended_augmentation > 1]
        if classes_needing_aug:
            recommendations.append(
                f"📈 AUGMENTATION NEEDED: {len(classes_needing_aug)} classes need "
                f"augmentation to reach {self.min_target_per_class} samples each."
            )
        
        # Size standardization
        all_sizes = set()
        for stats in self.class_stats.values():
            all_sizes.update(stats.size_distribution.keys())
        
        if len(all_sizes) > 5:
            recommendations.append(
                f"📐 SIZE STANDARDIZATION: {len(all_sizes)} different image sizes detected. "
                f"Standardize to 224x224 for CNN compatibility."
            )
        
        if not recommendations:
            recommendations.append("✅ Dataset looks healthy! Ready for training.")
        
        return recommendations
    
    def save_json_report(self, out_dir: Path) -> None:
        """Save detailed JSON report"""
        report_data = {
            "summary": asdict(self.summary) if self.summary else {},
            "class_statistics": {name: asdict(stats) for name, stats in self.class_stats.items()},
            "optional_data": self.optional_data,
            "recommendations": self.generate_recommendations(),
            "metadata": {
                "inspector_version": "1.0.0",
                "scan_timestamp": datetime.now().isoformat(),
                "data_directory": str(self.data_dir),
                "min_target_per_class": self.min_target_per_class
            }
        }
        
        json_file = out_dir / "dataset_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 JSON report saved: {json_file}")
    
    def save_markdown_report(self, out_dir: Path) -> None:
        """Save human-readable Markdown report"""
        if not self.summary:
            self.logger.warning("⚠️ No summary available for Markdown report")
            return
        
        md_content = []
        
        # Header
        md_content.append("# 🔍 Dataset Inspection Report")
        md_content.append(f"**Dataset Path:** `{self.data_dir}`")
        md_content.append(f"**Scan Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_content.append("")
        
        # Executive Summary
        md_content.append("## 📊 Executive Summary")
        md_content.append("")
        md_content.append(f"- **Total Images:** {self.summary.total_images:,}")
        md_content.append(f"- **Classes:** {self.summary.total_classes}")
        md_content.append(f"- **Average per Class:** {self.summary.avg_per_class:.1f}")
        md_content.append(f"- **Class Range:** {self.summary.min_per_class} - {self.summary.max_per_class}")
        md_content.append(f"- **Imbalance Ratio:** {self.summary.imbalance_ratio:.1f}:1")
        md_content.append(f"- **Corrupted Files:** {self.summary.total_corrupted}")
        md_content.append(f"- **Duplicates:** {self.summary.total_duplicates}")
        md_content.append("")
        
        # Cross-Validation Recommendation
        md_content.append("## 🎯 Cross-Validation Strategy")
        md_content.append("")
        md_content.append(f"**Recommended K-Fold:** {self.summary.recommended_k_fold}")
        md_content.append(f"**Strategy Note:** {self.summary.cv_strategy_note}")
        md_content.append("")
        
        # Per-Class Details
        md_content.append("## 📋 Per-Class Analysis")
        md_content.append("")
        md_content.append("| Class | Images | Corrupted | Duplicates | Health | Aug Factor | Sample Sizes |")
        md_content.append("|-------|---------|-----------|------------|---------|------------|--------------|")
        
        for name, stats in sorted(self.class_stats.items()):
            valid_count = stats.total_images - stats.corrupted_images
            duplicate_count = sum(len(group) - 1 for group in stats.duplicate_groups)
            
            # Health status emoji
            health_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "🚨"}.get(stats.health_status, "❓")
            
            # Top 3 most common sizes
            top_sizes = sorted(stats.size_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
            sizes_str = ", ".join([f"{size}({count})" for size, count in top_sizes])
            
            md_content.append(
                f"| {name} | {valid_count} | {stats.corrupted_images} | {duplicate_count} | "
                f"{health_emoji} {stats.health_status} | {stats.recommended_augmentation}x | {sizes_str} |"
            )
        
        md_content.append("")
        
        # Recommendations
        md_content.append("## 🎯 Recommendations")
        md_content.append("")
        for i, rec in enumerate(self.generate_recommendations(), 1):
            md_content.append(f"{i}. {rec}")
        md_content.append("")
        
        # Sample Files (for manual review)
        md_content.append("## 📸 Sample Files (First 3 per class)")
        md_content.append("")
        for name, stats in sorted(self.class_stats.items()):
            if stats.sample_paths:
                md_content.append(f"**{name}:**")
                for path in stats.sample_paths:
                    md_content.append(f"- `{path}`")
                md_content.append("")
        
        # Save to file
        md_file = out_dir / "dataset_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))
        
        self.logger.info(f"📝 Markdown report saved: {md_file}")
    
    def print_console_summary(self) -> None:
        """Print concise summary to console"""
        if not self.summary:
            print("❌ No analysis performed")
            return
        
        print("\n" + "="*60)
        print("🔍 DATASET INSPECTION SUMMARY")
        print("="*60)
        
        print(f"📊 Dataset: {self.data_dir}")
        print(f"📈 Total Images: {self.summary.total_images:,}")
        print(f"📁 Classes: {self.summary.total_classes}")
        print(f"⚖️ Balance: {self.summary.min_per_class} - {self.summary.max_per_class} ({self.summary.imbalance_ratio:.1f}:1)")
        
        if self.summary.total_corrupted > 0:
            print(f"🚨 Corrupted: {self.summary.total_corrupted}")
        
        if self.summary.total_duplicates > 0:
            print(f"🔄 Duplicates: {self.summary.total_duplicates}")
        
        print(f"🎯 Recommended CV: {self.summary.recommended_k_fold}-fold")
        
        # Health status summary
        health_counts = Counter(stats.health_status for stats in self.class_stats.values())
        print(f"💊 Health: {health_counts.get('healthy', 0)} healthy, "
              f"{health_counts.get('warning', 0)} warning, "
              f"{health_counts.get('critical', 0)} critical")
        
        print("\n🎯 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(self.generate_recommendations()[:3], 1):
            print(f"  {i}. {rec}")
        
        print("="*60)
    
    def get_exit_code(self) -> int:
        """Determine appropriate exit code based on analysis results"""
        if not self.summary:
            return 1  # No analysis performed
        
        # Critical exit conditions
        critical_classes = sum(1 for stats in self.class_stats.values() 
                              if stats.health_status == "critical")
        
        high_corruption = self.summary.total_corrupted > self.summary.total_images * 0.10
        
        if critical_classes > 0 or high_corruption:
            return 2  # Critical issues found
        
        # Warning conditions
        warning_classes = sum(1 for stats in self.class_stats.values() 
                             if stats.health_status == "warning")
        
        if warning_classes > 0 or self.summary.total_corrupted > 0:
            return 1  # Warnings found
        
        return 0  # All good
    
    def run_inspection(self, out_dir: str = "reports") -> int:
        """
        Run complete dataset inspection
        
        Args:
            out_dir: Output directory for reports
            
        Returns:
            Exit code (0=success, 1=warnings, 2=critical issues)
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load optional files
            self.load_optional_files()
            
            # Scan dataset
            self.scan_dataset()
            
            # Generate reports
            self.save_json_report(out_path)
            self.save_markdown_report(out_path)
            
            # Print summary
            self.print_console_summary()
            
            return self.get_exit_code()
            
        except Exception as e:
            self.logger.error(f"❌ Inspection failed: {e}")
            return 3  # System error


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="🔍 Dataset Inspector for Image Classification Projects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset_inspector.py
  python dataset_inspector.py --data-dir /path/to/dataset --min-target-per-class 50
  python dataset_inspector.py --out-dir analysis_results --sample-preview 5
        """
    )
    
    parser.add_argument(
        "--data-dir", 
        default="dataset",
        help="Path to dataset directory (default: dataset)"
    )
    
    parser.add_argument(
        "--out-dir",
        default="reports", 
        help="Output directory for reports (default: reports)"
    )
    
    parser.add_argument(
        "--min-target-per-class",
        type=int,
        default=30,
        help="Target minimum samples per class for augmentation recommendations (default: 30)"
    )
    
    parser.add_argument(
        "--sample-preview",
        type=int, 
        default=3,
        help="Number of sample file paths to save per class (default: 3)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger('DatasetInspector').setLevel(logging.DEBUG)
    
    # Create inspector and run
    inspector = DatasetInspector(
        data_dir=args.data_dir,
        min_target_per_class=args.min_target_per_class,
        sample_preview=args.sample_preview
    )
    
    exit_code = inspector.run_inspection(args.out_dir)
    
    print(f"\n🏁 Inspection completed with exit code: {exit_code}")
    if exit_code == 0:
        print("✅ Dataset ready for ML pipeline!")
    elif exit_code == 1:
        print("⚠️ Dataset has warnings - review recommendations")
    elif exit_code == 2:
        print("🚨 Dataset has critical issues - address before training")
    else:
        print("❌ System error during inspection")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()