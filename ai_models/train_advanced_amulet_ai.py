"""
🚀 Main Script for Advanced Amulet AI Training System
สคริปต์หลักสำหรับระบบเทรน AI พระเครื่องขั้นสูง

Usage:
    python train_advanced_amulet_ai.py --config config.json
    python train_advanced_amulet_ai.py --quick-start
    python train_advanced_amulet_ai.py --evaluate-only
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

# Add ai_models to path
sys.path.append(str(Path(__file__).parent))

# Import our advanced systems
from master_training_system import MasterTrainingSystem, MasterTrainingConfig, create_master_training_system
from advanced_image_processor import AdvancedImageProcessor
from self_supervised_learning import EmbeddingConfig
from dataset_organizer import DatasetOrganizer, EmbeddingDatabase
from advanced_data_pipeline import DataPipelineConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Setup training environment"""
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"🔥 CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"💾 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("🖥️ Using CPU")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    logger.info("🔧 Environment setup completed")

def load_config(config_path: Optional[str] = None) -> MasterTrainingConfig:
    """Load configuration from file or use defaults"""
    if config_path and Path(config_path).exists():
        logger.info(f"📄 Loading config from {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Create config object
        config = MasterTrainingConfig(**config_dict)
    else:
        logger.info("⚙️ Using default configuration")
        config = MasterTrainingConfig()
    
    return config

def save_config(config: MasterTrainingConfig, output_dir: str):
    """Save configuration to output directory"""
    config_path = Path(output_dir) / 'config.json'
    config_dict = {
        'dataset_path': config.dataset_path,
        'organized_path': config.organized_path,
        'split_path': config.split_path,
        'model_name': config.model_name,
        'embedding_dim': config.embedding_dim,
        'num_classes': config.num_classes,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'num_epochs': config.num_epochs,
        'warmup_epochs': config.warmup_epochs,
        'patience': config.patience,
        'temperature': config.temperature,
        'contrastive_weight': config.contrastive_weight,
        'min_quality_score': config.min_quality_score,
        'output_dir': config.output_dir,
        'save_best_only': config.save_best_only,
        'log_interval': config.log_interval,
        'use_cuda': config.use_cuda,
        'mixed_precision': config.mixed_precision,
        'num_workers': config.num_workers
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Configuration saved to {config_path}")

def validate_dataset(dataset_path: str) -> bool:
    """Validate dataset structure and content"""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        logger.error(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    # Check for category directories
    categories = [d for d in dataset_path.iterdir() if d.is_dir()]
    if len(categories) == 0:
        logger.error("❌ No category directories found in dataset")
        return False
    
    total_images = 0
    valid_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
    
    for category_dir in categories:
        images = []
        for ext in valid_extensions:
            images.extend(category_dir.glob(f'*{ext}'))
            images.extend(category_dir.glob(f'*{ext.upper()}'))
        
        category_count = len(images)
        total_images += category_count
        
        logger.info(f"📂 {category_dir.name}: {category_count} images")
        
        if category_count < 2:
            logger.warning(f"⚠️ Category {category_dir.name} has very few images ({category_count})")
    
    if total_images < 10:
        logger.error(f"❌ Too few images in dataset: {total_images}")
        return False
    
    logger.info(f"✅ Dataset validation passed. Total images: {total_images}")
    return True

def quick_start_config() -> MasterTrainingConfig:
    """Create quick start configuration for testing"""
    config = MasterTrainingConfig()
    
    # Reduce parameters for quick testing
    config.num_epochs = 10
    config.batch_size = 8
    config.patience = 5
    config.log_interval = 2
    config.warmup_epochs = 2
    
    logger.info("🚀 Quick start configuration created")
    return config

def run_training(config: MasterTrainingConfig) -> Dict:
    """Run complete training pipeline"""
    logger.info("🎯 Starting Advanced Amulet AI Training System")
    logger.info(f"📊 Configuration: {config.model_name}, {config.num_epochs} epochs, batch size {config.batch_size}")
    
    try:
        # Validate dataset
        if not validate_dataset(config.dataset_path):
            raise ValueError("Dataset validation failed")
        
        # Create training system
        training_system = create_master_training_system(config)
        
        # Save configuration
        save_config(config, config.output_dir)
        
        # Run complete pipeline
        results = training_system.run_complete_training_pipeline()
        
        logger.info("🎉 Training completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

def run_evaluation_only(config: MasterTrainingConfig) -> Dict:
    """Run evaluation only on existing model"""
    logger.info("🧪 Running evaluation only...")
    
    # Check if trained model exists
    model_path = Path(config.output_dir) / 'models' / 'best_model.pth'
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found at {model_path}")
    
    # Create system and run evaluation
    training_system = create_master_training_system(config)
    
    # Setup components
    training_system.step2_create_data_pipeline()
    training_system.step3_initialize_models()
    training_system.step4_setup_embedding_database()
    
    # Load trained model
    checkpoint = torch.load(model_path, map_location=training_system.device)
    training_system.model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    
    # Run evaluation
    results = training_system.step7_evaluate_and_visualize()
    
    logger.info("✅ Evaluation completed!")
    return results

def print_system_info():
    """Print system information"""
    print("\n" + "="*80)
    print("🎯 Advanced Amulet AI Training System")
    print("="*80)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"💻 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    print("="*80)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Advanced Amulet AI Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_advanced_amulet_ai.py --quick-start
  python train_advanced_amulet_ai.py --config my_config.json
  python train_advanced_amulet_ai.py --evaluate-only --output-dir results/
        """
    )
    
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--quick-start', action='store_true', 
                       help='Use quick start configuration for testing')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Run evaluation only on existing model')
    parser.add_argument('--dataset-path', type=str, default='dataset',
                       help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='training_output',
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print system info
    print_system_info()
    
    # Setup environment
    setup_environment()
    
    try:
        # Load or create configuration
        if args.quick_start:
            config = quick_start_config()
        else:
            config = load_config(args.config)
        
        # Override config with command line arguments
        if args.dataset_path:
            config.dataset_path = args.dataset_path
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.epochs:
            config.num_epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.learning_rate:
            config.learning_rate = args.learning_rate
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run training or evaluation
        if args.evaluate_only:
            results = run_evaluation_only(config)
        else:
            results = run_training(config)
        
        # Print summary
        print("\n" + "="*80)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📁 Results saved to: {config.output_dir}")
        print(f"📊 Model: {config.model_name}")
        print(f"🔄 Epochs: {config.num_epochs}")
        
        if 'error' not in results:
            print("✅ All steps completed successfully")
        else:
            print(f"⚠️ Completed with issues: {results['error']}")
        
        print("="*80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
