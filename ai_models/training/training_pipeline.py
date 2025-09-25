"""
Unified Training System for Amulet-AI
รวม training systems ทั้งหมดไว้ในที่เดียว
"""
import os
import sys
import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
import json

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from advanced_data_pipeline import AdvancedDataPipeline, DataPipelineConfig
from advanced_transfer_learning import TransferLearningTrainer, TransferLearningConfig

logger = logging.getLogger(__name__)

@dataclass 
class UnifiedTrainingConfig:
    """Unified configuration for all training modes"""
    # Model settings
    model_type: str = "efficientnet_b3"
    num_classes: int = 10
    
    # Data settings  
    data_path: str = "ai_models/dataset_split"
    batch_size: int = 16
    img_size: int = 224
    
    # Training settings
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Hardware settings
    use_cuda: bool = False  # Set to False for CPU-only PyTorch
    mixed_precision: bool = False
    num_workers: int = 2
    
    # Output settings
    output_dir: str = "ai_models/training_output"
    save_best_only: bool = True
    
    # Training mode
    training_mode: str = "transfer_learning"  # Options: transfer_learning, simple

class UnifiedTrainer:
    """Unified trainer that handles different training modes"""
    
    def __init__(self, config: UnifiedTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')
        
        # Setup output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized UnifiedTrainer on device: {self.device}")
        logger.info(f"Training mode: {config.training_mode}")
    
    def train(self):
        """Start training based on selected mode"""
        if self.config.training_mode == "transfer_learning":
            return self._train_transfer_learning()
        else:
            raise ValueError(f"Unknown training mode: {self.config.training_mode}")
    
    def _train_transfer_learning(self):
        """Run transfer learning training"""
        # Convert to transfer learning config
        tl_config = TransferLearningConfig(
            data_path=self.config.data_path,
            model_type=self.config.model_type,
            num_classes=self.config.num_classes,
            batch_size=self.config.batch_size,
            img_size=self.config.img_size,
            num_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            use_cuda=self.config.use_cuda,
            mixed_precision=self.config.mixed_precision,
            num_workers=self.config.num_workers,
            output_dir=self.config.output_dir,
            save_best_only=self.config.save_best_only
        )
        
        # Create and run trainer
        trainer = TransferLearningTrainer(tl_config)
        trainer.prepare_data()
        model = trainer.initialize_model()
        trainer.setup_training()
        history = trainer.train()
        
        return trainer, model, history
    
    def evaluate(self):
        """Evaluate trained model"""
        # Load best model and evaluate
        model_path = Path(self.config.output_dir) / "models" / "finetune_best_model_metric.pth"
        if model_path.exists():
            logger.info(f"Found trained model at: {model_path}")
            return True
        else:
            logger.warning("No trained model found for evaluation")
            return False
    
    def get_training_info(self):
        """Get information about training configuration"""
        info = {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "model_type": self.config.model_type,
            "training_mode": self.config.training_mode,
            "data_path": self.config.data_path,
            "output_dir": self.config.output_dir,
            "num_classes": self.config.num_classes
        }
        
        # Save info to file
        info_path = Path(self.config.output_dir) / "training_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        return info

def create_unified_trainer(config: Optional[UnifiedTrainingConfig] = None) -> UnifiedTrainer:
    """Create unified trainer with configuration"""
    if config is None:
        config = UnifiedTrainingConfig()
    
    return UnifiedTrainer(config)

def main():
    """Main training function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create trainer with default config
    trainer = create_unified_trainer()
    
    # Print training info
    info = trainer.get_training_info()
    logger.info("Training Configuration:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")
    
    # Start training
    logger.info("Starting unified training...")
    try:
        trainer_obj, model, history = trainer.train()
        logger.info("Training completed successfully!")
        
        # Evaluate
        evaluation_result = trainer.evaluate()
        logger.info(f"Evaluation completed: {evaluation_result}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()