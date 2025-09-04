"""
🏃‍♀️ Run Training for Amulet Classification with Transfer Learning
สคริปต์สำหรับเริ่มต้นการฝึกฝนโมเดลด้วย Transfer Learning
"""
import os
import sys
import argparse
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("run_training.log")
    ]
)
logger = logging.getLogger("run_training")

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_data_preparation(args):
    """Run the data preparation step"""
    logger.info("Starting data preparation...")
    
    from unified_dataset_creator import UnifiedDatasetCreator
    
    # Create configuration
    config = {
        'source_dirs': args.sources,
        'output_dir': args.unified_dataset,
        'split_ratio': {
            'train': args.train_ratio,
            'validation': args.val_ratio,
            'test': args.test_ratio
        },
        'min_samples': args.min_samples,
        'max_samples': args.max_samples,
        'small_class_threshold': args.small_class_threshold,
        'small_class_strategy': args.small_class_strategy
    }
    
    # Create unified dataset
    creator = UnifiedDatasetCreator(config)
    summary = creator.create_unified_dataset_complete()
    
    logger.info(f"Data preparation completed!")
    logger.info(f"Total classes: {summary['total_classes']}")
    logger.info(f"Total images: {summary['total_images']}")
    logger.info(f"Train split: {summary['splits']['train']['total_images']} images")
    logger.info(f"Validation split: {summary['splits']['validation']['total_images']} images")
    logger.info(f"Test split: {summary['splits']['test']['total_images']} images")
    
    return summary

def run_model_training(args, data_summary=None):
    """Run the model training step"""
    logger.info("Starting model training...")
    
    from advanced_transfer_learning import TransferLearningConfig, run_transfer_learning
    
    # Create configuration
    config_overrides = {
        "model_type": args.model,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "head_only_epochs": args.head_epochs,
        "learning_rate": args.learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "use_rotate": args.use_rotate,
        "use_flip": args.use_flip,
        "data_path": args.unified_dataset,
        "output_dir": args.output_dir,
        "small_class_threshold": args.small_class_threshold,
        "small_class_strategy": args.small_class_strategy
    }
    
    # Run transfer learning
    trainer, model, history = run_transfer_learning(config_overrides)
    
    logger.info("Model training completed!")
    return trainer, model, history

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='Run Amulet Classification with Transfer Learning')
    
    # Data preparation arguments
    parser.add_argument('--skip-data-prep', action='store_true', help='Skip data preparation step')
    parser.add_argument('--sources', nargs='+', help='Source directories containing amulet images')
    parser.add_argument('--unified-dataset', default='unified_dataset', help='Output directory for the unified dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test set ratio (default: 0.15)')
    parser.add_argument('--min-samples', type=int, default=10, help='Minimum samples per class (default: 10)')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum samples per class (default: None)')
    
    # Model training arguments
    parser.add_argument('--model', choices=['efficientnet_b0', 'efficientnet_b3', 'resnet50'], 
                        default='efficientnet_b3', help='Model architecture (default: efficientnet_b3)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--head-epochs', type=int, default=10, help='Number of head-only epochs (default: 10)')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--head-learning-rate', type=float, default=1e-2, help='Head learning rate (default: 1e-2)')
    parser.add_argument('--use-rotate', action='store_true', help='Use rotation augmentation')
    parser.add_argument('--use-flip', action='store_true', help='Use flip augmentation')
    parser.add_argument('--output-dir', default='training_output', help='Output directory for training results')
    
    # Common arguments
    parser.add_argument('--small-class-threshold', type=int, default=30, 
                        help='Threshold for small classes (default: 30)')
    parser.add_argument('--small-class-strategy', choices=['all_train', 'weighted_loss', 'oversample'], 
                        default='all_train', help='Strategy for handling small classes (default: all_train)')
    
    args = parser.parse_args()
    
    # Check if split ratios sum to 1
    split_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not abs(split_sum - 1.0) < 0.001:
        parser.error(f"Split ratios must sum to 1.0, got {split_sum}")
    
    # Run data preparation if needed
    data_summary = None
    if not args.skip_data_prep:
        if not args.sources:
            parser.error("Source directories are required for data preparation")
        data_summary = run_data_preparation(args)
    
    # Run model training
    trainer, model, history = run_model_training(args, data_summary)
    
    logger.info("Complete training process finished successfully!")

if __name__ == "__main__":
    main()
