#!/usr/bin/env python3
"""
🔬 FID/KID Validation Runner
===========================

Command-line tool for running FID and KID validation on datasets.
Supports both synthetic vs real image comparison and dataset quality assessment.

Features:
- Batch processing for large datasets
- Multiple comparison modes
- Comprehensive reporting
- Integration with training pipeline

Usage:
    python run_fid_kid.py --real_dir path/to/real --fake_dir path/to/synthetic
    python run_fid_kid.py --dataset_dir path/to/dataset --mode quality_check

Author: Amulet-AI Team
Date: October 2, 2025
"""

import argparse
import logging
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

from .fid_kid import FIDCalculator, KIDCalculator, compute_fid, compute_kid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """Simple image dataset for FID/KID calculation"""
    
    def __init__(self, image_dir: str, transform: transforms.Compose = None, max_images: int = None):
        self.image_dir = Path(image_dir)
        self.transform = transform or self._default_transform()
        
        # Supported image extensions
        self.extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Find all image files
        self.image_paths = []
        for ext in self.extensions:
            self.image_paths.extend(list(self.image_dir.rglob(f'*{ext}')))
            self.image_paths.extend(list(self.image_dir.rglob(f'*{ext.upper()}')))
        
        # Limit number of images if specified
        if max_images and max_images < len(self.image_paths):
            import random
            random.seed(42)
            self.image_paths = random.sample(self.image_paths, max_images)
        
        logger.info(f"Found {len(self.image_paths)} images in {image_dir}")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
    
    def _default_transform(self):
        """Default transform for FID/KID (resize to 299x299 for InceptionV3)"""
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1] range
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, 299, 299)


class FIDKIDValidator:
    """
    🎯 FID/KID Validation Runner
    
    Comprehensive validation tool for image quality assessment using
    FID and KID metrics.
    
    Features:
    - Multiple comparison modes
    - Batch processing for memory efficiency
    - Statistical analysis
    - Comprehensive reporting
    - Integration with MLOps pipelines
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        device: str = 'auto',
        max_images_per_dataset: Optional[int] = None
    ):
        """
        Initialize FID/KID Validator
        
        Args:
            batch_size: Batch size for processing
            num_workers: Number of data loading workers
            device: Device for computation
            max_images_per_dataset: Limit images per dataset (for large datasets)
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_images_per_dataset = max_images_per_dataset
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize calculators
        try:
            self.fid_calculator = FIDCalculator(device=str(self.device))
            self.kid_calculator = KIDCalculator(device=str(self.device))
            self.available = True
        except ImportError as e:
            logger.error(f"FID/KID calculators not available: {e}")
            self.available = False
        
        logger.info(f"FIDKIDValidator initialized on {self.device}")
    
    def load_dataset_images(self, dataset_dir: str, max_images: Optional[int] = None) -> torch.Tensor:
        """
        Load images from directory as tensor
        
        Args:
            dataset_dir: Directory containing images
            max_images: Maximum number of images to load
            
        Returns:
            Tensor of images (N, 3, H, W)
        """
        max_imgs = max_images or self.max_images_per_dataset
        
        dataset = ImageDataset(dataset_dir, max_images=max_imgs)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        images = []
        logger.info(f"Loading images from {dataset_dir}...")
        
        for batch in tqdm(dataloader, desc="Loading images"):
            images.append(batch)
        
        images = torch.cat(images, dim=0)
        logger.info(f"Loaded {len(images)} images from {dataset_dir}")
        
        return images
    
    def compute_fid_kid(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute FID and KID between two image sets
        
        Args:
            real_images: Real images tensor (N, 3, H, W)
            fake_images: Fake/synthetic images tensor (N, 3, H, W)
            
        Returns:
            Dictionary with FID and KID scores
        """
        if not self.available:
            raise RuntimeError("FID/KID calculators not available")
        
        logger.info(f"Computing FID/KID between {len(real_images)} real and {len(fake_images)} fake images...")
        
        # Compute FID
        fid_score = self.fid_calculator.compute(real_images, fake_images)
        
        # Compute KID
        kid_score = self.kid_calculator.compute(real_images, fake_images)
        
        results = {
            'fid': fid_score,
            'kid': kid_score,
            'num_real_images': len(real_images),
            'num_fake_images': len(fake_images)
        }
        
        logger.info(f"FID: {fid_score:.2f}, KID: {kid_score:.6f}")
        
        return results
    
    def validate_synthetic_vs_real(
        self,
        real_dir: str,
        synthetic_dir: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate synthetic images against real images
        
        Args:
            real_dir: Directory with real images
            synthetic_dir: Directory with synthetic images
            output_dir: Output directory for results
            
        Returns:
            Validation results
        """
        logger.info("🔬 Running synthetic vs real validation...")
        
        # Load images
        real_images = self.load_dataset_images(real_dir)
        synthetic_images = self.load_dataset_images(synthetic_dir)
        
        # Compute metrics
        metrics = self.compute_fid_kid(real_images, synthetic_images)
        
        # Add metadata
        results = {
            'validation_type': 'synthetic_vs_real',
            'real_dir': str(real_dir),
            'synthetic_dir': str(synthetic_dir),
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'metrics': metrics,
            'interpretation': self._interpret_scores(metrics)
        }
        
        # Save results
        if output_dir:
            self._save_results(results, output_dir, 'synthetic_vs_real_validation')
        
        return results
    
    def validate_dataset_quality(
        self,
        dataset_dir: str,
        reference_dir: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate overall dataset quality
        
        Args:
            dataset_dir: Directory with dataset to validate
            reference_dir: Optional reference dataset for comparison
            output_dir: Output directory for results
            
        Returns:
            Quality assessment results
        """
        logger.info("🔍 Running dataset quality assessment...")
        
        dataset_path = Path(dataset_dir)
        
        # If dataset has splits (train/val/test), analyze each
        if (dataset_path / 'train').exists() and (dataset_path / 'validation').exists():
            results = self._validate_split_dataset(dataset_path, reference_dir, output_dir)
        else:
            # Single dataset validation
            results = self._validate_single_dataset(dataset_path, reference_dir, output_dir)
        
        return results
    
    def _validate_split_dataset(
        self,
        dataset_path: Path,
        reference_dir: Optional[str],
        output_dir: Optional[str]
    ) -> Dict[str, Any]:
        """Validate dataset with train/val/test splits"""
        
        splits = ['train', 'validation', 'test']
        split_results = {}
        
        for split in splits:
            split_dir = dataset_path / split
            if split_dir.exists():
                logger.info(f"Analyzing {split} split...")
                
                try:
                    images = self.load_dataset_images(str(split_dir))
                    
                    split_info = {
                        'num_images': len(images),
                        'image_shape': list(images.shape),
                        'split_dir': str(split_dir)
                    }
                    
                    # If reference provided, compute FID/KID
                    if reference_dir:
                        ref_images = self.load_dataset_images(reference_dir)
                        metrics = self.compute_fid_kid(ref_images, images)
                        split_info['metrics_vs_reference'] = metrics
                    
                    split_results[split] = split_info
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze {split} split: {e}")
                    split_results[split] = {'error': str(e)}
        
        # Cross-split comparison (train vs val)
        cross_split_metrics = {}
        if 'train' in split_results and 'validation' in split_results:
            try:
                logger.info("Computing train vs validation FID/KID...")
                train_images = self.load_dataset_images(str(dataset_path / 'train'))
                val_images = self.load_dataset_images(str(dataset_path / 'validation'))
                
                cross_split_metrics['train_vs_val'] = self.compute_fid_kid(train_images, val_images)
                
            except Exception as e:
                logger.warning(f"Cross-split comparison failed: {e}")
        
        results = {
            'validation_type': 'dataset_quality_splits',
            'dataset_dir': str(dataset_path),
            'reference_dir': reference_dir,
            'timestamp': datetime.now().isoformat(),
            'split_results': split_results,
            'cross_split_metrics': cross_split_metrics,
            'summary': self._summarize_split_results(split_results, cross_split_metrics)
        }
        
        if output_dir:
            self._save_results(results, output_dir, 'dataset_quality_assessment')
        
        return results
    
    def _validate_single_dataset(
        self,
        dataset_path: Path,
        reference_dir: Optional[str],
        output_dir: Optional[str]
    ) -> Dict[str, Any]:
        """Validate single dataset directory"""
        
        images = self.load_dataset_images(str(dataset_path))
        
        results = {
            'validation_type': 'dataset_quality_single',
            'dataset_dir': str(dataset_path),
            'reference_dir': reference_dir,
            'timestamp': datetime.now().isoformat(),
            'num_images': len(images),
            'image_shape': list(images.shape)
        }
        
        # Compare with reference if provided
        if reference_dir:
            ref_images = self.load_dataset_images(reference_dir)
            metrics = self.compute_fid_kid(ref_images, images)
            results['metrics_vs_reference'] = metrics
            results['interpretation'] = self._interpret_scores(metrics)
        
        if output_dir:
            self._save_results(results, output_dir, 'single_dataset_validation')
        
        return results
    
    def _interpret_scores(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Interpret FID/KID scores"""
        fid = metrics['fid']
        kid = metrics['kid']
        
        # FID interpretation
        if fid < 10:
            fid_interpretation = "Excellent quality - very similar to real images"
        elif fid < 25:
            fid_interpretation = "Good quality - reasonably similar to real images"
        elif fid < 50:
            fid_interpretation = "Fair quality - noticeable differences from real images"
        elif fid < 100:
            fid_interpretation = "Poor quality - significant differences from real images"
        else:
            fid_interpretation = "Very poor quality - major differences from real images"
        
        # KID interpretation
        if kid < 0.001:
            kid_interpretation = "Excellent diversity and quality"
        elif kid < 0.01:
            kid_interpretation = "Good diversity and quality"
        elif kid < 0.05:
            kid_interpretation = "Fair diversity and quality"
        else:
            kid_interpretation = "Poor diversity and quality"
        
        return {
            'fid_score': fid,
            'fid_interpretation': fid_interpretation,
            'kid_score': kid,
            'kid_interpretation': kid_interpretation,
            'overall_quality': 'Good' if fid < 25 and kid < 0.01 else 'Fair' if fid < 50 and kid < 0.05 else 'Poor'
        }
    
    def _summarize_split_results(
        self,
        split_results: Dict,
        cross_split_metrics: Dict
    ) -> Dict[str, Any]:
        """Summarize results from split validation"""
        
        total_images = sum(
            r.get('num_images', 0) for r in split_results.values() 
            if isinstance(r, dict) and 'num_images' in r
        )
        
        summary = {
            'total_images': total_images,
            'splits_analyzed': list(split_results.keys()),
            'splits_with_errors': [k for k, v in split_results.items() if 'error' in v]
        }
        
        # Add cross-split analysis
        if 'train_vs_val' in cross_split_metrics:
            train_val_fid = cross_split_metrics['train_vs_val']['fid']
            summary['train_val_similarity'] = {
                'fid': train_val_fid,
                'assessment': 'Similar distributions' if train_val_fid < 20 else 'Different distributions'
            }
        
        return summary
    
    def _save_results(self, results: Dict, output_dir: str, prefix: str):
        """Save validation results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report
        json_path = output_path / f"{prefix}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save CSV summary (if metrics available)
        if 'metrics' in results:
            csv_path = output_path / f"{prefix}_summary_{timestamp}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                for key, value in results['metrics'].items():
                    writer.writerow([key, value])
        
        logger.info(f"Results saved to {output_path}")
    
    def batch_validate(
        self,
        config_file: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Run batch validation from configuration file
        
        Args:
            config_file: JSON config file with validation tasks
            output_dir: Output directory
            
        Returns:
            Batch validation results
        """
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        batch_results = {}
        
        for task_name, task_config in config.get('tasks', {}).items():
            logger.info(f"Running task: {task_name}")
            
            try:
                task_type = task_config['type']
                
                if task_type == 'synthetic_vs_real':
                    result = self.validate_synthetic_vs_real(
                        real_dir=task_config['real_dir'],
                        synthetic_dir=task_config['synthetic_dir'],
                        output_dir=output_dir
                    )
                elif task_type == 'dataset_quality':
                    result = self.validate_dataset_quality(
                        dataset_dir=task_config['dataset_dir'],
                        reference_dir=task_config.get('reference_dir'),
                        output_dir=output_dir
                    )
                else:
                    raise ValueError(f"Unknown task type: {task_type}")
                
                batch_results[task_name] = result
                
            except Exception as e:
                logger.error(f"Task {task_name} failed: {e}")
                batch_results[task_name] = {'error': str(e)}
        
        # Save batch summary
        batch_summary = {
            'batch_validation': True,
            'config_file': config_file,
            'timestamp': datetime.now().isoformat(),
            'tasks_completed': len([r for r in batch_results.values() if 'error' not in r]),
            'tasks_failed': len([r for r in batch_results.values() if 'error' in r]),
            'results': batch_results
        }
        
        self._save_results(batch_summary, output_dir, 'batch_validation')
        
        return batch_summary


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='FID/KID Validation Runner')
    
    # Operation mode
    parser.add_argument('--mode', choices=['synthetic_vs_real', 'dataset_quality', 'batch'],
                       default='synthetic_vs_real', help='Validation mode')
    
    # Paths
    parser.add_argument('--real_dir', type=str, help='Directory with real images')
    parser.add_argument('--fake_dir', '--synthetic_dir', type=str, help='Directory with synthetic images')
    parser.add_argument('--dataset_dir', type=str, help='Dataset directory to validate')
    parser.add_argument('--reference_dir', type=str, help='Reference dataset for comparison')
    parser.add_argument('--config_file', type=str, help='Config file for batch validation')
    parser.add_argument('--output_dir', type=str, default='fid_kid_results', help='Output directory')
    
    # Parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--max_images', type=int, help='Maximum images per dataset')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = FIDKIDValidator(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_images_per_dataset=args.max_images
    )
    
    if not validator.available:
        logger.error("FID/KID validation not available. Please install required dependencies.")
        return
    
    # Run validation based on mode
    try:
        if args.mode == 'synthetic_vs_real':
            if not args.real_dir or not args.fake_dir:
                parser.error("synthetic_vs_real mode requires --real_dir and --fake_dir")
            
            results = validator.validate_synthetic_vs_real(
                real_dir=args.real_dir,
                synthetic_dir=args.fake_dir,
                output_dir=args.output_dir
            )
            
            print(f"\n✅ Validation completed!")
            print(f"FID: {results['metrics']['fid']:.2f}")
            print(f"KID: {results['metrics']['kid']:.6f}")
            print(f"Quality: {results['interpretation']['overall_quality']}")
            
        elif args.mode == 'dataset_quality':
            if not args.dataset_dir:
                parser.error("dataset_quality mode requires --dataset_dir")
            
            results = validator.validate_dataset_quality(
                dataset_dir=args.dataset_dir,
                reference_dir=args.reference_dir,
                output_dir=args.output_dir
            )
            
            print(f"\n✅ Dataset quality assessment completed!")
            if 'summary' in results:
                print(f"Total images analyzed: {results['summary']['total_images']}")
            
        elif args.mode == 'batch':
            if not args.config_file:
                parser.error("batch mode requires --config_file")
            
            results = validator.batch_validate(
                config_file=args.config_file,
                output_dir=args.output_dir
            )
            
            print(f"\n✅ Batch validation completed!")
            print(f"Tasks completed: {results['tasks_completed']}")
            print(f"Tasks failed: {results['tasks_failed']}")
        
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())