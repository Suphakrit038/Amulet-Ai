"""
🚀 Main Script for Advanced Amulet AI Training System (FIXED)
สคริปต์หลักสำหรับระบบเทรน AI พระเครื่องขั้นสูง

Usage:
    python train_advanced_amulet_ai_fixed.py --quick-start
    python train_advanced_amulet_ai_fixed.py --config config_advanced.json
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import torch
import numpy as np

# Add current directory to Python path for imports
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import our advanced systems
from ai_models.master_training_system import MasterTrainingSystem, MasterTrainingConfig, create_master_training_system
from ai_models.advanced_image_processor import AdvancedImageProcessor
from ai_models.self_supervised_learning import EmbeddingConfig
from ai_models.dataset_organizer import DatasetOrganizer, EmbeddingDatabase
from ai_models.advanced_data_pipeline import DataPipelineConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup training environment"""
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"🔥 CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"💾 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.info("💻 Using CPU for training")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    logger.info("🔧 Environment setup complete")

def load_config(config_path: Optional[str] = None) -> MasterTrainingConfig:
    """Load configuration from file or use defaults"""
    if config_path and Path(config_path).exists():
        logger.info(f"📂 Loading config from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Create config object (simplified mapping)
        config = MasterTrainingConfig()
        
        # Map config values
        if 'dataset_path' in config_dict:
            config.dataset_path = config_dict['dataset_path']
        if 'batch_size' in config_dict:
            config.batch_size = config_dict['batch_size']
        if 'num_epochs' in config_dict:
            config.num_epochs = config_dict['num_epochs']
        if 'learning_rate' in config_dict:
            config.learning_rate = config_dict['learning_rate']
            
        return config
    else:
        logger.info("📝 Using default configuration")
        return MasterTrainingConfig()

def validate_dataset(dataset_path: Path) -> bool:
    """Validate that the dataset exists and has proper structure"""
    logger.info(f"🔍 Validating dataset at: {dataset_path}")
    
    if not dataset_path.exists():
        logger.error(f"❌ Dataset path does not exist: {dataset_path}")
        return False
        
    if not dataset_path.is_dir():
        logger.error(f"❌ Dataset path is not a directory: {dataset_path}")
        return False
        
    # Get categories
    categories = [d for d in dataset_path.iterdir() if d.is_dir()]
    if len(categories) == 0:
        logger.error("❌ No category directories found in dataset")
        return False
    
    total_images = 0
    valid_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
    
    for category_dir in categories:
        images = []
        for ext in valid_extensions:
            images.extend(list(category_dir.glob(f"*{ext}")))
            images.extend(list(category_dir.glob(f"*{ext.upper()}")))
        
        logger.info(f"📁 {category_dir.name}: {len(images)} images")
        total_images += len(images)
        
        if len(images) == 0:
            logger.warning(f"⚠️ No images found in {category_dir.name}")
    
    logger.info(f"✅ Dataset validation passed. Total images: {total_images}")
    return True

def quick_start_config() -> MasterTrainingConfig:
    """Create quick start configuration for testing"""
    
    # Set explicit dataset path
    dataset_path = r"C:\Users\Admin\Documents\GitHub\Amulet-Ai\dataset"
    
    logger.info(f"🔧 Creating quick start config with dataset path: {dataset_path}")
    logger.info(f"📁 Path exists: {Path(dataset_path).exists()}")
    logger.info(f"📂 Is directory: {Path(dataset_path).is_dir()}")
    
    # Create config with explicit dataset path
    config = MasterTrainingConfig(
        dataset_path=dataset_path
    )
    
    # Reduce parameters for quick testing
    config.num_epochs = 10
    config.batch_size = 8
    config.patience = 5
    config.log_interval = 2
    config.warmup_epochs = 2
    
    # Verify config was created correctly
    logger.info(f"✅ Config created with dataset_path: {config.dataset_path}")
    
    return config

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Advanced Amulet AI Training System')
    parser.add_argument('--quick-start', action='store_true', 
                       help='Use quick start configuration for testing')
    parser.add_argument('--config', type=str, 
                       help='Path to configuration JSON file')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate existing model, skip training')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Load configuration
    if args.quick_start:
        logger.info("🚀 Using quick start configuration")
        config = quick_start_config()
    else:
        config = load_config(args.config)
    
    # Validate dataset
    dataset_path = Path(config.dataset_path)
    if not validate_dataset(dataset_path):
        logger.error("❌ Dataset validation failed. Exiting.")
        sys.exit(1)
    
    try:
        # Create training system
        logger.info("🏗️ Creating master training system...")
        training_system = create_master_training_system(config)
        
        if args.evaluate_only:
            # Evaluation mode
            logger.info("📊 Running evaluation only...")
            results = training_system.evaluate()
            logger.info(f"📈 Evaluation results: {results}")
        else:
            # Training mode
            logger.info("🎯 Starting training...")
            training_system.train()
            logger.info("🎉 Training completed successfully!")
            
    except Exception as e:
        logger.error(f"💥 Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
