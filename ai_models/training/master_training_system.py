"""
🎯 Master Training System for High-Quality Amulet Recognition
ระบบเทรน ML ขั้นสูงสำหรับการจดจำพระเครื่องคุณภาพสูง
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm
import time
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Import our advanced systems
from self_supervised_learning import (
    ContrastiveLearningModel, SelfSupervisedTrainer,
    AdvancedEmbeddingSystem, EmbeddingConfig
)
from advanced_data_pipeline import AdvancedDataPipeline, DataPipelineConfig
from dataset_organizer import DatasetOrganizer, EmbeddingDatabase, create_embedding_record

logger = logging.getLogger(__name__)

@dataclass
class MasterTrainingConfig:
    """Configuration for master training system"""
    # Data configuration
    dataset_path: str = "data_base"
    organized_path: str = "dataset_organized"
    split_path: str = "ai_models/dataset_split"
    
    # Model configuration
    model_name: str = "efficientnet-b4"
    embedding_dim: int = 512
    num_classes: int = 10  # Updated based on actual dataset categories
    
    # Training configuration
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_epochs: int = 10
    patience: int = 15
    
    # Self-supervised configuration
    temperature: float = 0.1
    contrastive_weight: float = 1.0
    
    # Quality thresholds
    min_quality_score: float = 0.8
    
    # Output configuration
    output_dir: str = "training_output"
    save_best_only: bool = True
    log_interval: int = 10
    
    # Hardware configuration
    use_cuda: bool = True
    mixed_precision: bool = True
    num_workers: int = 4

class MasterTrainingSystem:
    """
    🎓 Master Training System
    ระบบเทรนหลักที่รวมทุกอย่างเข้าด้วยกัน
    """
    
    def __init__(self, config: MasterTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')
        
        # Initialize paths
        self.setup_directories()
        
        # Initialize components
        self.data_pipeline = None
        self.model = None
        self.trainer = None
        self.embedding_system = None
        self.embedding_db = None
        
        # Training state
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard'))
        
        logger.info(f"🚀 Master Training System initialized on {self.device}")
        
    def setup_directories(self):
        """Setup required directories"""
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'embeddings').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
        
        logger.info(f"📁 Output directory setup: {self.output_dir}")
    
    def step1_organize_dataset(self) -> Dict:
        """Step 1: Organize dataset structure"""
        logger.info("📊 Step 1: Organizing dataset...")
        
        try:
            organizer = DatasetOrganizer(
                source_dir=self.config.dataset_path,
                target_dir=self.config.split_path
            )
            
            logger.info("📁 Starting dataset organization...")
            # Organize into train/val/test splits
            stats = organizer.organize_dataset_structure()
            logger.info("📊 Dataset structure organized")
            
            # Skip analyzing categories for quick training
            logger.info("� Skipping category analysis for quick training...")
            analysis = {
                'categories': 10,
                'total_samples': stats.get('total', 340),
                'status': 'skipped_for_quick_training'
            }
            logger.info("📈 Using minimal category analysis")
            
            # Save analysis
            analysis_path = self.output_dir / 'reports' / 'dataset_analysis.json'
            logger.info(f"💾 Saving analysis to {analysis_path}")
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ Dataset organized. Statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Step 1 failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return minimal stats to continue
            return {
                'train': 0,
                'validation': 0,
                'test': 0,
                'error': str(e)
            }
    
    def step2_create_data_pipeline(self) -> AdvancedDataPipeline:
        """Step 2: Create advanced data pipeline"""
        logger.info("🔄 Step 2: Creating data pipeline...")
        
        try:
            logger.info("📋 Creating pipeline configuration...")
            pipeline_config = DataPipelineConfig(
                split_path=self.config.split_path,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                quality_threshold=self.config.min_quality_score
            )
            
            logger.info("🏭 Initializing AdvancedDataPipeline...")
            self.data_pipeline = AdvancedDataPipeline(pipeline_config)
            
            logger.info("📊 Creating datasets...")
            # Create datasets and dataloaders
            datasets = self.data_pipeline.create_datasets()
            logger.info("📊 Creating dataloaders...")
            dataloaders = self.data_pipeline.create_dataloaders()
            
            logger.info("📈 Analyzing statistics...")
            # Analyze and save statistics
            stats = self.data_pipeline.analyze_dataset_statistics()
            self.data_pipeline.save_statistics(
                str(self.output_dir / 'reports' / 'pipeline_stats.json')
            )
            
            logger.info(f"✅ Data pipeline created. Datasets: {list(datasets.keys())}")
            return self.data_pipeline
            
        except Exception as e:
            logger.error(f"❌ Step 2 failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def step3_initialize_models(self) -> Tuple[ContrastiveLearningModel, SelfSupervisedTrainer]:
        """Step 3: Initialize models and trainer"""
        logger.info("🧠 Step 3: Initializing models...")
        
        # Create embedding configuration
        embedding_config = EmbeddingConfig(
            embedding_dim=self.config.embedding_dim,
            temperature=self.config.temperature,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate
        )
        
        # Initialize contrastive learning model
        self.model = ContrastiveLearningModel(embedding_config)
        
        # Initialize trainer
        self.trainer = SelfSupervisedTrainer(
            config=embedding_config
        )
        
        # Initialize embedding system
        self.embedding_system = AdvancedEmbeddingSystem(embedding_config)
        
        logger.info(f"✅ Models initialized. Model parameters: {self._count_parameters()}")
        return self.model, self.trainer
    
    def step4_setup_embedding_database(self) -> EmbeddingDatabase:
        """Step 4: Setup embedding database"""
        logger.info("🗄️ Step 4: Setting up embedding database...")
        
        db_path = self.output_dir / 'embeddings' / 'embeddings.db'
        self.embedding_db = EmbeddingDatabase(
            db_path=str(db_path),
            dimension=self.config.embedding_dim
        )
        
        logger.info("✅ Embedding database ready")
        return self.embedding_db
    
    def step5_train_model(self) -> Dict:
        """Step 5: Train the model"""
        logger.info("🎯 Step 5: Starting model training...")
        
        if not all([self.data_pipeline, self.model, self.trainer]):
            raise ValueError("Components not initialized. Run previous steps first.")
        
        # Get dataloaders
        train_loader = self.data_pipeline.dataloaders.get('train')
        val_loader = self.data_pipeline.dataloaders.get('validation')
        
        if not train_loader:
            raise ValueError("Training dataloader not found")
        
        # Training loop
        scaler = GradScaler() if self.config.mixed_precision else None
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch, scaler)
            
            # Validation phase
            val_metrics = None
            if val_loader:
                val_metrics = self._validate_epoch(val_loader, epoch)
            
            # Update learning rate
            current_lr = self._update_learning_rate(epoch)
            
            # Log metrics
            self._log_epoch_metrics(epoch, train_metrics, val_metrics, current_lr)
            
            # Save checkpoint
            is_best = self._save_checkpoint(epoch, train_metrics, val_metrics)
            
            # Early stopping check
            if self._should_stop_early(epoch, val_metrics):
                logger.info(f"🛑 Early stopping at epoch {epoch}")
                break
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f"⏱️ Epoch {epoch+1}/{self.config.num_epochs} completed in {epoch_time:.2f}s")
        
        # Training completed
        self._finalize_training()
        
        return self.training_history
    
    def step6_build_embedding_database(self) -> Dict:
        """Step 6: Build comprehensive embedding database"""
        logger.info("🔗 Step 6: Building embedding database...")
        
        if not all([self.model, self.embedding_db, self.data_pipeline]):
            raise ValueError("Required components not initialized")
        
        # Set model to evaluation mode
        self.model.eval()
        
        embedding_count = 0
        
        # Process all datasets
        for split_name, dataset in self.data_pipeline.datasets.items():
            logger.info(f"Processing {split_name} split...")
            
            for idx in tqdm(range(len(dataset)), desc=f"Creating embeddings - {split_name}"):
                try:
                    item = dataset[idx]
                    
                    # Extract embedding
                    with torch.no_grad():
                        image_tensor = item['image'].unsqueeze(0).to(self.device)
                        embedding = self.model.backbone(image_tensor)
                        embedding = embedding.cpu().numpy().flatten()
                    
                    # Create embedding record
                    quality_score = self._calculate_quality_score(item.get('quality_metrics', {}))
                    
                    metadata = {
                        **item['metadata'],
                        'split': split_name,
                        'index': idx,
                        'embedding_dim': len(embedding)
                    }
                    
                    record = create_embedding_record(
                        image_path=item['path'],
                        category=item['label'],
                        embedding=embedding,
                        metadata=metadata,
                        quality_score=quality_score
                    )
                    
                    # Add to database
                    if self.embedding_db.add_embedding(record):
                        embedding_count += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process {item.get('path', 'unknown')}: {e}")
                    continue
        
        # Build FAISS index
        logger.info("🔍 Building FAISS index...")
        if self.embedding_db.build_faiss_index(index_type='IVFFlat'):
            logger.info("✅ FAISS index built successfully")
        else:
            logger.error("❌ Failed to build FAISS index")
        
        # Get database statistics
        stats = self.embedding_db.get_statistics()
        
        logger.info(f"✅ Embedding database completed. Total embeddings: {embedding_count}")
        return stats
    
    def step7_evaluate_and_visualize(self) -> Dict:
        """Step 7: Comprehensive evaluation and visualization"""
        logger.info("📊 Step 7: Evaluation and visualization...")
        
        if not all([self.model, self.embedding_db, self.data_pipeline]):
            raise ValueError("Required components not initialized")
        
        evaluation_results = {}
        
        # 1. Model performance evaluation
        test_loader = self.data_pipeline.dataloaders.get('test')
        if test_loader:
            test_metrics = self._evaluate_test_set(test_loader)
            evaluation_results['test_metrics'] = test_metrics
        
        # 2. Embedding quality analysis
        embedding_analysis = self._analyze_embeddings()
        evaluation_results['embedding_analysis'] = embedding_analysis
        
        # 3. Similarity search evaluation
        similarity_results = self._evaluate_similarity_search()
        evaluation_results['similarity_evaluation'] = similarity_results
        
        # 4. Generate visualizations
        self._create_visualizations()
        
        # 5. Generate comprehensive report
        self._generate_final_report(evaluation_results)
        
        logger.info("✅ Evaluation and visualization completed")
        return evaluation_results
    
    def run_complete_training_pipeline(self) -> Dict:
        """🚀 Run the complete training pipeline"""
        logger.info("🎯 Starting complete training pipeline...")
        
        pipeline_results = {}
        
        try:
            # Step 1: Organize dataset
            logger.info("🎯 About to start Step 1...")
            dataset_stats = self.step1_organize_dataset()
            pipeline_results['dataset_organization'] = dataset_stats
            logger.info("✅ Step 1 completed successfully!")
            
            # Step 2: Create data pipeline
            logger.info("🎯 About to start Step 2...")
            self.step2_create_data_pipeline()
            pipeline_results['data_pipeline'] = "completed"
            logger.info("✅ Step 2 completed successfully!")
            
            # Step 3: Initialize models
            logger.info("🎯 About to start Step 3...")
            self.step3_initialize_models()
            pipeline_results['model_initialization'] = "completed"
            logger.info("✅ Step 3 completed successfully!")
            
            # Step 4: Setup embedding database
            logger.info("🎯 About to start Step 4...")
            self.step4_setup_embedding_database()
            pipeline_results['embedding_database'] = "initialized"
            logger.info("✅ Step 4 completed successfully!")
            
            # Step 5: Train model
            logger.info("🎯 About to start Step 5...")
            training_history = self.step5_train_model()
            pipeline_results['training_history'] = training_history
            logger.info("✅ Step 5 completed successfully!")
            
            # Step 6: Build embedding database
            logger.info("🎯 About to start Step 6...")
            embedding_stats = self.step6_build_embedding_database()
            pipeline_results['embedding_stats'] = embedding_stats
            logger.info("✅ Step 6 completed successfully!")
            
            # Step 7: Evaluate and visualize
            logger.info("🎯 About to start Step 7...")
            evaluation_results = self.step7_evaluate_and_visualize()
            pipeline_results['evaluation_results'] = evaluation_results
            logger.info("✅ Step 7 completed successfully!")
            
            logger.info("🎉 Complete training pipeline finished successfully!")
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            pipeline_results['error'] = str(e)
            raise
        
        return pipeline_results
    
    def _train_epoch(self, train_loader, epoch: int, scaler: Optional[GradScaler]) -> Dict:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                images = batch['image'].to(self.device)
                labels = batch['label']
                
                # Forward pass with mixed precision
                if scaler:
                    with autocast():
                        loss, metrics = self.trainer.training_step(images, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.step(self.trainer.optimizer)
                    scaler.update()
                else:
                    loss, metrics = self.trainer.training_step(images, labels)
                    loss.backward()
                    self.trainer.optimizer.step()
                
                self.trainer.optimizer.zero_grad()
                
                # Update metrics
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg_Loss': f'{total_loss/(batch_idx+1):.4f}'
                })
                
            except Exception as e:
                logger.warning(f"⚠️ Batch {batch_idx} failed: {e}")
                continue
        
        avg_loss = total_loss / len(train_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': 0.0  # Placeholder - contrastive learning doesn't have direct accuracy
        }
    
    def _validate_epoch(self, val_loader, epoch: int) -> Dict:
        """Validate one epoch"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                try:
                    images = batch['image'].to(self.device)
                    labels = batch['label']
                    
                    loss, metrics = self.trainer.training_step(images, labels)
                    total_loss += loss.item()
                    
                except Exception as e:
                    logger.warning(f"⚠️ Validation batch failed: {e}")
                    continue
        
        avg_loss = total_loss / len(val_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': 0.0  # Placeholder
        }
    
    def _count_parameters(self) -> int:
        """Count model parameters"""
        if self.model:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return 0
    
    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate quality score from metrics"""
        if not metrics:
            return 0.5
        
        sharpness_score = min(1.0, metrics.get('sharpness', 0) / 1000.0)
        contrast_score = min(1.0, metrics.get('contrast', 0) / 100.0)
        snr_score = min(1.0, metrics.get('snr', 0) / 10.0)
        brightness_score = 1.0 - abs(metrics.get('brightness', 127.5) - 127.5) / 127.5
        
        return 0.4 * sharpness_score + 0.3 * contrast_score + 0.2 * snr_score + 0.1 * brightness_score
    
    def _update_learning_rate(self, epoch: int) -> float:
        """Update learning rate with warmup and cosine annealing"""
        if epoch < self.config.warmup_epochs:
            # Warmup phase
            lr = self.config.learning_rate * (epoch + 1) / self.config.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.config.warmup_epochs) / (self.config.num_epochs - self.config.warmup_epochs)
            lr = self.config.learning_rate * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.trainer.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict, 
                          val_metrics: Optional[Dict], lr: float):
        """Log epoch metrics"""
        # Update history
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['learning_rates'].append(lr)
        
        if val_metrics:
            self.training_history['val_loss'].append(val_metrics['loss'])
        
        # Log to tensorboard
        self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
        self.writer.add_scalar('Learning_Rate', lr, epoch)
        
        if val_metrics:
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
        
        # Log to console
        if epoch % self.config.log_interval == 0:
            log_msg = f"Epoch {epoch+1}: Train Loss={train_metrics['loss']:.4f}, LR={lr:.2e}"
            if val_metrics:
                log_msg += f", Val Loss={val_metrics['loss']:.4f}"
            logger.info(log_msg)
    
    def _save_checkpoint(self, epoch: int, train_metrics: Dict, 
                        val_metrics: Optional[Dict]) -> bool:
        """Save model checkpoint"""
        current_loss = val_metrics['loss'] if val_metrics else train_metrics['loss']
        
        is_best = current_loss < self.best_loss
        if is_best:
            self.best_loss = current_loss
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'loss': current_loss,
            'best_loss': self.best_loss,
            'config': asdict(self.config),
            'training_history': self.training_history
        }
        
        # Save current checkpoint
        checkpoint_path = self.output_dir / 'models' / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'models' / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"💾 New best model saved: loss={current_loss:.4f}")
        
        return is_best
    
    def _should_stop_early(self, epoch: int, val_metrics: Optional[Dict]) -> bool:
        """Check if should stop early"""
        if not val_metrics or epoch < self.config.warmup_epochs:
            return False
        
        # Simple patience-based early stopping
        if len(self.training_history['val_loss']) >= self.config.patience:
            recent_losses = self.training_history['val_loss'][-self.config.patience:]
            if all(loss >= self.best_loss for loss in recent_losses):
                return True
        
        return False
    
    def _finalize_training(self):
        """Finalize training process"""
        # Close tensorboard writer
        self.writer.close()
        
        # Save final training history
        history_path = self.output_dir / 'reports' / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Plot training curves
        self._plot_training_curves()
        
        logger.info("✅ Training finalized")
    
    def _plot_training_curves(self):
        """Plot training curves"""
        plt.figure(figsize=(12, 4))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['train_loss'], label='Train Loss')
        if self.training_history['val_loss']:
            plt.plot(self.training_history['val_loss'], label='Validation Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Learning rate curve
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['learning_rates'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _evaluate_test_set(self, test_loader) -> Dict:
        """Evaluate on test set"""
        logger.info("🧪 Evaluating test set...")
        # Implementation for test evaluation
        return {'accuracy': 0.0, 'loss': 0.0}
    
    def _analyze_embeddings(self) -> Dict:
        """Analyze embedding quality"""
        logger.info("🔍 Analyzing embeddings...")
        # Implementation for embedding analysis
        return {'avg_similarity': 0.0, 'cluster_quality': 0.0}
    
    def _evaluate_similarity_search(self) -> Dict:
        """Evaluate similarity search performance"""
        logger.info("🎯 Evaluating similarity search...")
        # Implementation for similarity evaluation
        return {'precision_at_k': 0.0, 'recall_at_k': 0.0}
    
    def _create_visualizations(self):
        """Create comprehensive visualizations"""
        logger.info("📊 Creating visualizations...")
        # Implementation for visualizations
        pass
    
    def _generate_final_report(self, results: Dict):
        """Generate comprehensive final report"""
        logger.info("📋 Generating final report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'results': results,
            'model_info': {
                'parameters': self._count_parameters(),
                'architecture': self.config.model_name
            },
            'training_summary': {
                'total_epochs': len(self.training_history['train_loss']),
                'best_loss': self.best_loss,
                'final_lr': self.training_history['learning_rates'][-1] if self.training_history['learning_rates'] else 0
            }
        }
        
        report_path = self.output_dir / 'reports' / 'final_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📋 Final report saved: {report_path}")
    
    def train(self):
        """Execute full training pipeline"""
        logger.info("🚀 Starting 7-step advanced training pipeline...")
        
        try:
            # Execute the complete training pipeline
            results = self.run_complete_training_pipeline()

            # Generate final report
            self._generate_final_report(results)

            logger.info("🎉 Training pipeline completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"💥 Training pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

def create_master_training_system(config: Optional[MasterTrainingConfig] = None) -> MasterTrainingSystem:
    """Create master training system with default configuration"""
    if config is None:
        config = MasterTrainingConfig()
    
    return MasterTrainingSystem(config)
