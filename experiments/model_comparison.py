"""
🔬 Model Comparison Framework
=============================

Systematic comparison of different model architectures and training strategies

Comparison Dimensions:
1. Baseline vs Transfer Learning
2. Different backbones (ResNet, EfficientNet, MobileNet)
3. Freeze vs Finetune strategies
4. With/without augmentation
5. With/without calibration

Metrics:
- Accuracy, F1-score (macro/weighted)
- Inference latency (mean, p95)
- Model size (MB)
- Training time per epoch
- GPU memory usage

Author: Amulet-AI Team
Date: October 2, 2025
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelConfig:
    """Configuration for a model experiment"""
    name: str
    model_type: str  # 'baseline', 'transfer_learning'
    backbone: Optional[str] = None  # 'resnet50', 'efficientnet_b0', etc.
    freeze_backbone: bool = False
    use_mixup: bool = False
    use_cutmix: bool = False
    use_calibration: bool = False
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 20
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ModelResults:
    """Results from a model experiment"""
    config: ModelConfig
    
    # Performance metrics
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    f1_macro: float
    f1_weighted: float
    
    # Per-class metrics
    per_class_f1: Dict[str, float]
    
    # Inference metrics
    avg_latency_ms: float
    p95_latency_ms: float
    throughput_imgs_per_sec: float
    
    # Model characteristics
    model_size_mb: float
    num_parameters: int
    trainable_parameters: int
    
    # Training metrics
    training_time_minutes: float
    best_epoch: int
    final_loss: float
    
    # Calibration (if used)
    ece_before: Optional[float] = None
    ece_after: Optional[float] = None
    
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self):
        result = asdict(self)
        result['config'] = self.config.to_dict()
        return result


class ModelEvaluator:
    """
    🎯 Evaluate model performance comprehensively
    
    Usage:
        >>> evaluator = ModelEvaluator(model, device='cuda')
        >>> results = evaluator.evaluate(test_loader, class_names)
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def measure_latency(
        self, 
        test_loader: DataLoader, 
        num_warmup: int = 10,
        num_measure: int = 100
    ) -> Tuple[float, float, float]:
        """
        Measure inference latency
        
        Returns:
            avg_ms: Average latency in milliseconds
            p95_ms: 95th percentile latency
            throughput: Images per second
        """
        latencies = []
        
        with torch.no_grad():
            # Warmup
            for i, (images, _) in enumerate(test_loader):
                if i >= num_warmup:
                    break
                images = images.to(self.device)
                _ = self.model(images)
            
            # Measure
            for i, (images, _) in enumerate(test_loader):
                if i >= num_measure:
                    break
                
                images = images.to(self.device)
                batch_size = images.size(0)
                
                start = time.perf_counter()
                _ = self.model(images)
                torch.cuda.synchronize() if self.device == 'cuda' else None
                end = time.perf_counter()
                
                latency_ms = (end - start) * 1000 / batch_size
                latencies.append(latency_ms)
        
        avg_ms = np.mean(latencies)
        p95_ms = np.percentile(latencies, 95)
        throughput = 1000.0 / avg_ms  # images/sec
        
        return avg_ms, p95_ms, throughput
    
    def compute_accuracy(self, data_loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute accuracy and collect predictions
        
        Returns:
            accuracy: Overall accuracy
            all_preds: Predictions array
            all_labels: True labels array
        """
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100.0 * correct / total
        return accuracy, np.array(all_preds), np.array(all_labels)
    
    def get_model_size(self) -> Tuple[float, int, int]:
        """
        Get model size information
        
        Returns:
            size_mb: Model size in MB
            total_params: Total parameters
            trainable_params: Trainable parameters
        """
        # Save temporarily to measure size
        temp_path = Path("temp_model.pth")
        torch.save(self.model.state_dict(), temp_path)
        size_mb = temp_path.stat().st_size / (1024 * 1024)
        temp_path.unlink()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return size_mb, total_params, trainable_params
    
    def evaluate_full(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        class_names: List[str],
        training_time_minutes: float,
        best_epoch: int,
        final_loss: float
    ) -> Dict[str, Any]:
        """
        Complete evaluation of model
        
        Returns comprehensive metrics dictionary
        """
        print(f"🔍 Evaluating model...")
        
        # Accuracy on all splits
        train_acc, _, _ = self.compute_accuracy(train_loader)
        val_acc, _, _ = self.compute_accuracy(val_loader)
        test_acc, test_preds, test_labels = self.compute_accuracy(test_loader)
        
        print(f"  Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        print(f"  Test Acc: {test_acc:.2f}%")
        
        # F1 scores
        f1_macro = f1_score(test_labels, test_preds, average='macro')
        f1_weighted = f1_score(test_labels, test_preds, average='weighted')
        per_class_f1 = f1_score(test_labels, test_preds, average=None)
        
        per_class_f1_dict = {
            class_names[i]: float(f1_macro_i) 
            for i, f1_macro_i in enumerate(per_class_f1)
        }
        
        print(f"  F1 Macro: {f1_macro:.4f}")
        print(f"  F1 Weighted: {f1_weighted:.4f}")
        
        # Latency
        avg_lat, p95_lat, throughput = self.measure_latency(test_loader)
        print(f"  Avg Latency: {avg_lat:.2f} ms")
        print(f"  P95 Latency: {p95_lat:.2f} ms")
        print(f"  Throughput: {throughput:.1f} imgs/sec")
        
        # Model size
        size_mb, total_params, trainable_params = self.get_model_size()
        print(f"  Model Size: {size_mb:.2f} MB")
        print(f"  Total Params: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'per_class_f1': per_class_f1_dict,
            'avg_latency_ms': avg_lat,
            'p95_latency_ms': p95_lat,
            'throughput_imgs_per_sec': throughput,
            'model_size_mb': size_mb,
            'num_parameters': total_params,
            'trainable_parameters': trainable_params,
            'training_time_minutes': training_time_minutes,
            'best_epoch': best_epoch,
            'final_loss': final_loss
        }


class ModelComparison:
    """
    📊 Compare multiple models systematically
    
    Usage:
        >>> comparison = ModelComparison(save_dir='experiments/results')
        >>> comparison.add_result(config, results)
        >>> comparison.generate_report()
    """
    
    def __init__(self, save_dir: str = 'experiments/results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ModelResults] = []
    
    def add_result(self, result: ModelResults):
        """Add a model result to comparison"""
        self.results.append(result)
        
        # Save individual result
        result_file = self.save_dir / f"{result.config.name}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"✅ Saved result: {result_file}")
    
    def generate_report(self, output_file: str = 'comparison_report.md'):
        """Generate comprehensive comparison report"""
        if not self.results:
            print("⚠️  No results to compare")
            return
        
        report_path = self.save_dir / output_file
        
        with open(report_path, 'w') as f:
            f.write("# 🔬 Model Comparison Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Number of Models**: {len(self.results)}\n\n")
            
            # Performance comparison table
            f.write("## 📊 Performance Comparison\n\n")
            f.write("| Model | Test Acc | F1 Macro | F1 Weighted | Latency (ms) | Size (MB) |\n")
            f.write("|-------|----------|----------|-------------|--------------|----------|\n")
            
            for result in sorted(self.results, key=lambda x: x.test_accuracy, reverse=True):
                f.write(
                    f"| {result.config.name} | "
                    f"{result.test_accuracy:.2f}% | "
                    f"{result.f1_macro:.4f} | "
                    f"{result.f1_weighted:.4f} | "
                    f"{result.avg_latency_ms:.2f} | "
                    f"{result.model_size_mb:.2f} |\n"
                )
            
            # Best models by criterion
            f.write("\n## 🏆 Best Models\n\n")
            
            best_acc = max(self.results, key=lambda x: x.test_accuracy)
            f.write(f"- **Best Accuracy**: {best_acc.config.name} ({best_acc.test_accuracy:.2f}%)\n")
            
            best_f1 = max(self.results, key=lambda x: x.f1_macro)
            f.write(f"- **Best F1 Macro**: {best_f1.config.name} ({best_f1.f1_macro:.4f})\n")
            
            fastest = min(self.results, key=lambda x: x.avg_latency_ms)
            f.write(f"- **Fastest Inference**: {fastest.config.name} ({fastest.avg_latency_ms:.2f} ms)\n")
            
            smallest = min(self.results, key=lambda x: x.model_size_mb)
            f.write(f"- **Smallest Model**: {smallest.config.name} ({smallest.model_size_mb:.2f} MB)\n")
            
            # Per-class F1 comparison
            f.write("\n## 📈 Per-Class F1 Scores\n\n")
            class_names = list(self.results[0].per_class_f1.keys())
            
            f.write("| Model | " + " | ".join(class_names) + " |\n")
            f.write("|-------|" + "|".join(["-------"] * len(class_names)) + "|\n")
            
            for result in self.results:
                f1_values = [f"{result.per_class_f1[cls]:.3f}" for cls in class_names]
                f.write(f"| {result.config.name} | " + " | ".join(f1_values) + " |\n")
            
            # Detailed configurations
            f.write("\n## ⚙️ Model Configurations\n\n")
            for result in self.results:
                f.write(f"### {result.config.name}\n\n")
                f.write(f"- **Type**: {result.config.model_type}\n")
                if result.config.backbone:
                    f.write(f"- **Backbone**: {result.config.backbone}\n")
                f.write(f"- **Freeze Backbone**: {result.config.freeze_backbone}\n")
                f.write(f"- **MixUp**: {result.config.use_mixup}\n")
                f.write(f"- **CutMix**: {result.config.use_cutmix}\n")
                f.write(f"- **Calibration**: {result.config.use_calibration}\n")
                f.write(f"- **Learning Rate**: {result.config.learning_rate}\n")
                f.write(f"- **Batch Size**: {result.config.batch_size}\n")
                f.write(f"- **Training Time**: {result.training_time_minutes:.1f} min\n")
                f.write(f"- **Best Epoch**: {result.best_epoch}\n\n")
        
        print(f"✅ Report saved: {report_path}")
        
        # Generate plots
        self._generate_plots()
    
    def _generate_plots(self):
        """Generate comparison plots"""
        if len(self.results) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
        
        model_names = [r.config.name for r in self.results]
        
        # 1. Test Accuracy
        test_accs = [r.test_accuracy for r in self.results]
        axes[0, 0].barh(model_names, test_accs, color='skyblue')
        axes[0, 0].set_xlabel('Test Accuracy (%)')
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. F1 Scores
        f1_macros = [r.f1_macro for r in self.results]
        f1_weighteds = [r.f1_weighted for r in self.results]
        x = np.arange(len(model_names))
        width = 0.35
        axes[0, 1].bar(x - width/2, f1_macros, width, label='F1 Macro', color='coral')
        axes[0, 1].bar(x + width/2, f1_weighteds, width, label='F1 Weighted', color='lightgreen')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('F1 Score Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Latency vs Accuracy (scatter)
        latencies = [r.avg_latency_ms for r in self.results]
        axes[1, 0].scatter(latencies, test_accs, s=100, alpha=0.6, color='purple')
        for i, name in enumerate(model_names):
            axes[1, 0].annotate(name, (latencies[i], test_accs[i]), fontsize=8)
        axes[1, 0].set_xlabel('Avg Latency (ms)')
        axes[1, 0].set_ylabel('Test Accuracy (%)')
        axes[1, 0].set_title('Latency vs Accuracy Trade-off')
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Model Size
        sizes = [r.model_size_mb for r in self.results]
        axes[1, 1].barh(model_names, sizes, color='gold')
        axes[1, 1].set_xlabel('Model Size (MB)')
        axes[1, 1].set_title('Model Size Comparison')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.save_dir / 'comparison_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plots saved: {plot_path}")
        plt.close()


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("="*60)
    print("Model Comparison Framework - Demo")
    print("="*60)
    
    # Create mock results for demonstration
    comparison = ModelComparison(save_dir='experiments/demo_results')
    
    # Mock result 1: Baseline
    config1 = ModelConfig(
        name="Baseline_CNN",
        model_type="baseline",
        use_mixup=False,
        use_cutmix=False,
        learning_rate=0.001,
        batch_size=32
    )
    
    result1 = ModelResults(
        config=config1,
        train_accuracy=85.5,
        val_accuracy=78.2,
        test_accuracy=76.8,
        f1_macro=0.7543,
        f1_weighted=0.7689,
        per_class_f1={'Class1': 0.80, 'Class2': 0.75, 'Class3': 0.70, 
                      'Class4': 0.78, 'Class5': 0.72, 'Class6': 0.77},
        avg_latency_ms=8.5,
        p95_latency_ms=12.3,
        throughput_imgs_per_sec=117.6,
        model_size_mb=45.2,
        num_parameters=11234567,
        trainable_parameters=11234567,
        training_time_minutes=42.5,
        best_epoch=15,
        final_loss=0.524
    )
    
    # Mock result 2: Transfer Learning
    config2 = ModelConfig(
        name="ResNet50_Frozen",
        model_type="transfer_learning",
        backbone="resnet50",
        freeze_backbone=True,
        use_mixup=True,
        learning_rate=0.001
    )
    
    result2 = ModelResults(
        config=config2,
        train_accuracy=92.3,
        val_accuracy=84.5,
        test_accuracy=83.2,
        f1_macro=0.8245,
        f1_weighted=0.8312,
        per_class_f1={'Class1': 0.85, 'Class2': 0.82, 'Class3': 0.79,
                      'Class4': 0.84, 'Class5': 0.80, 'Class6': 0.85},
        avg_latency_ms=15.2,
        p95_latency_ms=19.8,
        throughput_imgs_per_sec=65.8,
        model_size_mb=98.5,
        num_parameters=25557032,
        trainable_parameters=4096123,
        training_time_minutes=28.3,
        best_epoch=12,
        final_loss=0.342
    )
    
    # Add results
    comparison.add_result(result1)
    comparison.add_result(result2)
    
    # Generate report
    comparison.generate_report()
    
    print("\n✅ Demo complete! Check experiments/demo_results/ for outputs")
