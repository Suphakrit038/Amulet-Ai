"""
🔄 Advanced Data Pipeline for Maximum Quality ML
ระบบ data pipeline ขั้นสูงสำหรับ Machine Learning คุณภาพสูงสุด
"""
import numpy as np
from PIL import Image
from pathlib import Path
import json
import logging
from typing import List, Dict, Tuple, Optional, Generator
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from dataclasses import dataclass
import hashlib
import time

# Import our advanced processors
from .advanced_image_processor import advanced_processor, process_image_max_quality
from .self_supervised_learning import EmbeddingConfig

logger = logging.getLogger(__name__)

@dataclass
class DataPipelineConfig:
    """Configuration for data pipeline"""
    dataset_path: str = "dataset_organized"
    split_path: str = "dataset_split" 
    target_size: Tuple[int, int] = (512, 512)  # High resolution
    batch_size: int = 16
    num_workers: int = 4
    cache_processed: bool = True
    quality_threshold: float = 0.8
    
class HighQualityAmuletDataset(Dataset):
    """
    High-Quality Amulet Dataset for Self-Supervised Learning
    ไม่มี augmentation - เน้นคุณภาพสูงสุด
    """
    
    def __init__(self, data_dir: Path, config: DataPipelineConfig, split: str = 'train'):
        self.data_dir = data_dir
        self.config = config
        self.split = split
        self.image_paths = []
        self.labels = []
        self.metadata = []
        self.quality_cache = {}
        
        # Load data
        self._load_dataset()
        
        # Filter by quality
        self._filter_by_quality()
        
        logger.info(f"📸 Loaded {len(self.image_paths)} high-quality images for {split}")
    
    def _load_dataset(self):
        """Load dataset from organized structure"""
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            logger.error(f"❌ Split directory not found: {split_dir}")
            return
        
        # Load from each category
        for category_dir in split_dir.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                
                # Find all image files
                image_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
                for img_path in category_dir.iterdir():
                    if img_path.suffix.lower() in image_extensions:
                        self.image_paths.append(img_path)
                        self.labels.append(category_name)
                        
                        # Create metadata
                        metadata = {
                            'category': category_name,
                            'filename': img_path.name,
                            'path': str(img_path),
                            'size': img_path.stat().st_size if img_path.exists() else 0
                        }
                        self.metadata.append(metadata)
        
        logger.info(f"📂 Found {len(self.image_paths)} images in {len(set(self.labels))} categories")
    
    def _filter_by_quality(self):
        """Filter images by quality threshold"""
        high_quality_indices = []
        
        logger.info("🔍 Analyzing image quality...")
        
        for idx, img_path in enumerate(self.image_paths):
            try:
                # Load and analyze image
                image = Image.open(img_path)
                quality_metrics = advanced_processor.get_image_quality_metrics(image)
                
                # Calculate overall quality score
                quality_score = self._calculate_quality_score(quality_metrics)
                
                # Cache quality metrics
                self.quality_cache[str(img_path)] = {
                    'score': quality_score,
                    'metrics': quality_metrics
                }
                
                # Filter by threshold
                if quality_score >= self.config.quality_threshold:
                    high_quality_indices.append(idx)
                else:
                    logger.debug(f"🚫 Filtered out low quality: {img_path.name} (score: {quality_score:.3f})")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to analyze {img_path}: {e}")
                continue
        
        # Update dataset with high-quality images only
        self.image_paths = [self.image_paths[i] for i in high_quality_indices]
        self.labels = [self.labels[i] for i in high_quality_indices]
        self.metadata = [self.metadata[i] for i in high_quality_indices]
        
        logger.info(f"✅ Kept {len(self.image_paths)} high-quality images out of {len(high_quality_indices)}")
    
    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate overall quality score from metrics"""
        # Normalize and weight different metrics
        sharpness_score = min(1.0, metrics['sharpness'] / 1000.0)  # Normalize sharpness
        contrast_score = min(1.0, metrics['contrast'] / 100.0)     # Normalize contrast
        snr_score = min(1.0, metrics['snr'] / 10.0)               # Normalize SNR
        
        # Brightness penalty for very dark/bright images
        brightness_score = 1.0 - abs(metrics['brightness'] - 127.5) / 127.5
        
        # Weighted combination
        quality_score = (
            0.4 * sharpness_score +
            0.3 * contrast_score + 
            0.2 * snr_score +
            0.1 * brightness_score
        )
        
        return quality_score
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        metadata = self.metadata[idx]
        
        # Load and process image for maximum quality
        try:
            image = Image.open(img_path)
            processed_image, img_array, quality_metrics = process_image_max_quality(image)
            
            # Convert to tensor for PyTorch
            img_tensor = torch.from_numpy(img_array).float()
            
            # Remove batch dimension if present
            if img_tensor.dim() == 4:
                img_tensor = img_tensor.squeeze(0)
            
            # Ensure correct shape [C, H, W]
            if img_tensor.dim() == 3 and img_tensor.shape[0] != 3:
                img_tensor = img_tensor.permute(2, 0, 1)
            
            return {
                'image': img_tensor,
                'label': label,
                'metadata': metadata,
                'quality_metrics': quality_metrics,
                'path': str(img_path)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to load image {img_path}: {e}")
            # Return dummy data
            dummy_tensor = torch.zeros(3, 512, 512)
            return {
                'image': dummy_tensor,
                'label': label,
                'metadata': metadata,
                'quality_metrics': {},
                'path': str(img_path)
            }

class AdvancedDataPipeline:
    """
    Advanced Data Pipeline for High-Quality ML Training
    ระบบจัดการข้อมูลขั้นสูงสำหรับการเทรน ML คุณภาพสูง
    """
    
    def __init__(self, config: DataPipelineConfig):
        self.config = config
        self.datasets = {}
        self.dataloaders = {}
        self.statistics = {}
        
    def create_datasets(self) -> Dict[str, HighQualityAmuletDataset]:
        """Create datasets for train/val/test splits"""
        data_dir = Path(self.config.split_path)
        
        splits = ['train', 'validation', 'test']
        
        for split in splits:
            split_dir = data_dir / split
            if split_dir.exists():
                dataset = HighQualityAmuletDataset(data_dir, self.config, split)
                self.datasets[split] = dataset
                logger.info(f"✅ Created {split} dataset: {len(dataset)} images")
        
        return self.datasets
    
    def create_dataloaders(self) -> Dict[str, DataLoader]:
        """Create PyTorch DataLoaders"""
        for split, dataset in self.datasets.items():
            # Different settings for different splits
            if split == 'train':
                shuffle = True
                drop_last = True
            else:
                shuffle = False
                drop_last = False
            
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=self.config.num_workers,
                drop_last=drop_last,
                pin_memory=torch.cuda.is_available()
            )
            
            self.dataloaders[split] = dataloader
            logger.info(f"📦 Created {split} dataloader: {len(dataloader)} batches")
        
        return self.dataloaders
    
    def analyze_dataset_statistics(self) -> Dict:
        """Analyze comprehensive dataset statistics"""
        stats = {}
        
        for split, dataset in self.datasets.items():
            split_stats = {
                'total_images': len(dataset),
                'categories': {},
                'quality_distribution': [],
                'size_distribution': [],
                'format_distribution': {}
            }
            
            # Category distribution
            for label in dataset.labels:
                split_stats['categories'][label] = split_stats['categories'].get(label, 0) + 1
            
            # Quality and size analysis
            for metadata in dataset.metadata:
                # Quality metrics
                img_path = metadata['path']
                if img_path in dataset.quality_cache:
                    quality_score = dataset.quality_cache[img_path]['score']
                    split_stats['quality_distribution'].append(quality_score)
                
                # File size
                split_stats['size_distribution'].append(metadata['size'])
                
                # File format
                ext = Path(metadata['path']).suffix.lower()
                split_stats['format_distribution'][ext] = split_stats['format_distribution'].get(ext, 0) + 1
            
            # Calculate statistics
            if split_stats['quality_distribution']:
                split_stats['avg_quality'] = np.mean(split_stats['quality_distribution'])
                split_stats['quality_std'] = np.std(split_stats['quality_distribution'])
            
            if split_stats['size_distribution']:
                split_stats['avg_size_mb'] = np.mean(split_stats['size_distribution']) / (1024 * 1024)
                split_stats['size_std_mb'] = np.std(split_stats['size_distribution']) / (1024 * 1024)
            
            stats[split] = split_stats
        
        self.statistics = stats
        return stats
    
    def save_statistics(self, path: str):
        """Save dataset statistics"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.statistics, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📊 Saved dataset statistics to {path}")
    
    def create_embeddings_dataset(self) -> List[Dict]:
        """Create dataset for embedding extraction"""
        embeddings_data = []
        
        for split, dataset in self.datasets.items():
            for idx in range(len(dataset)):
                item = dataset[idx]
                
                embedding_item = {
                    'image_tensor': item['image'],
                    'label': item['label'],
                    'metadata': {
                        **item['metadata'],
                        'split': split,
                        'quality_metrics': item['quality_metrics']
                    }
                }
                
                embeddings_data.append(embedding_item)
        
        logger.info(f"🔗 Created embeddings dataset: {len(embeddings_data)} items")
        return embeddings_data
    
    def export_high_quality_images(self, output_dir: str, min_quality: float = 0.9):
        """Export only highest quality images"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        exported_count = 0
        
        for split, dataset in self.datasets.items():
            split_dir = output_path / split
            split_dir.mkdir(exist_ok=True)
            
            for idx in range(len(dataset)):
                item = dataset[idx]
                
                # Check quality
                if 'quality_metrics' in item and len(item['quality_metrics']) > 0:
                    quality_score = self._calculate_quality_score_from_metrics(item['quality_metrics'])
                    
                    if quality_score >= min_quality:
                        # Create category directory
                        category_dir = split_dir / item['label']
                        category_dir.mkdir(exist_ok=True)
                        
                        # Save high-quality processed image
                        img_tensor = item['image']
                        if img_tensor.dim() == 3:
                            img_array = img_tensor.permute(1, 2, 0).numpy()
                        else:
                            img_array = img_tensor.numpy()
                        
                        img_array = (img_array * 255).astype(np.uint8)
                        image = Image.fromarray(img_array)
                        
                        output_file = category_dir / f"hq_{idx:04d}.jpg"
                        image.save(output_file, 'JPEG', quality=98)
                        
                        exported_count += 1
        
        logger.info(f"📤 Exported {exported_count} highest quality images to {output_dir}")
    
    def _calculate_quality_score_from_metrics(self, metrics: Dict) -> float:
        """Calculate quality score from metrics dict"""
        if not metrics:
            return 0.0
        
        sharpness_score = min(1.0, metrics.get('sharpness', 0) / 1000.0)
        contrast_score = min(1.0, metrics.get('contrast', 0) / 100.0)
        snr_score = min(1.0, metrics.get('snr', 0) / 10.0)
        brightness_score = 1.0 - abs(metrics.get('brightness', 127.5) - 127.5) / 127.5
        
        quality_score = (
            0.4 * sharpness_score +
            0.3 * contrast_score + 
            0.2 * snr_score +
            0.1 * brightness_score
        )
        
        return quality_score

def create_advanced_pipeline(config: Optional[DataPipelineConfig] = None) -> AdvancedDataPipeline:
    """Create advanced data pipeline with default config"""
    if config is None:
        config = DataPipelineConfig()
    
    pipeline = AdvancedDataPipeline(config)
    
    # Create datasets and dataloaders
    pipeline.create_datasets()
    pipeline.create_dataloaders()
    
    # Analyze statistics
    pipeline.analyze_dataset_statistics()
    
    logger.info("🚀 Advanced data pipeline created successfully")
    return pipeline
