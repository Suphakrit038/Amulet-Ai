"""
🔍 Model Evaluator for Amulet Classification
เครื่องมือสำหรับประเมินผลโมเดลที่ฝึกฝนมาแล้ว
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_evaluation.log")
    ]
)
logger = logging.getLogger("model_evaluation")

class ModelEvaluator:
    """Evaluates a trained model on test data or custom images"""
    
    def __init__(self, model_path, config=None):
        """
        Initialize evaluator with model path
        
        Args:
            model_path: Path to the trained model checkpoint
            config: Optional configuration dictionary
        """
        self.model_path = Path(model_path)
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = []
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        logger.info(f"Loading model from {self.model_path}")
        
        # Check if the model file exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model configuration if available
        if 'config' in checkpoint:
            self.config = checkpoint['config']
            logger.info(f"Loaded model configuration from checkpoint")
        
        # Extract class names if available
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
        elif hasattr(self.config, 'num_classes'):
            # Create generic class names if not available
            self.class_names = [f"Class_{i}" for i in range(self.config['num_classes'])]
        
        # Load class mapping from labels.json if available
        labels_path = Path(self.config.get('data_path', '.')) / 'labels.json'
        if labels_path.exists():
            with open(labels_path, 'r', encoding='utf-8') as f:
                class_mapping = json.load(f)
                self.class_names = [class_mapping.get(str(i), f"Class_{i}") 
                                  for i in range(len(class_mapping))]
        
        # Load the model
        try:
            # Import model class
            from advanced_transfer_learning import TransferLearningModel, TransferLearningConfig
            
            # Create model configuration from loaded config
            model_config = TransferLearningConfig()
            for key, value in self.config.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
            
            # Create model
            self.model = TransferLearningModel(model_config)
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully: {model_config.model_type}")
            logger.info(f"Number of classes: {len(self.class_names)}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _create_data_transform(self):
        """Create data transformation for inference"""
        img_size = self.config.get('img_size', 224)
        
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def evaluate_test_set(self, test_dir):
        """
        Evaluate model on test set
        
        Args:
            test_dir: Path to test directory
        """
        logger.info(f"Evaluating model on test set: {test_dir}")
        
        # Import dataset class
        from advanced_transfer_learning import AmuletDataset
        
        test_transform = self._create_data_transform()
        
        # Create dataset and dataloader
        test_dataset = self._create_test_dataset(test_dir, test_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Evaluate
        all_preds = []
        all_labels = []
        all_paths = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                labels = batch['label']
                paths = batch['path']
                
                outputs = self.model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_paths.extend(paths)
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Create classification report
        report = classification_report(
            all_labels, all_preds,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Create results
        results = {
            'accuracy': float(accuracy),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_names': self.class_names,
            'sample_count': len(all_labels)
        }
        
        # Create detailed sample results
        sample_results = []
        for i, (path, true_label, pred_label, probs) in enumerate(
            zip(all_paths, all_labels, all_preds, all_probs)
        ):
            # Get top 3 predictions
            top_indices = np.argsort(probs)[-3:][::-1]
            top_probs = [float(probs[idx]) for idx in top_indices]
            top_classes = [self.class_names[idx] for idx in top_indices]
            
            sample_results.append({
                'path': path,
                'true_label': int(true_label),
                'true_class': self.class_names[true_label],
                'predicted_label': int(pred_label),
                'predicted_class': self.class_names[pred_label],
                'correct': true_label == pred_label,
                'top_predictions': [
                    {'class': cls, 'probability': prob}
                    for cls, prob in zip(top_classes, top_probs)
                ]
            })
        
        # Visualize results
        self._visualize_evaluation_results(results, sample_results)
        
        # Save results
        output_dir = Path('evaluation_results')
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / 'evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        with open(output_dir / 'sample_results.json', 'w', encoding='utf-8') as f:
            json.dump(sample_results, f, indent=2, ensure_ascii=False)
        
        # Log summary
        logger.info(f"Evaluation completed with accuracy: {accuracy:.4f}")
        logger.info(f"Results saved to {output_dir}")
        
        # Print summary
        print(f"\nEvaluation Results:")
        print(f"------------------")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Number of samples: {len(all_labels)}")
        print(f"\nPer-class metrics:")
        
        # Print per-class metrics
        for class_name in self.class_names:
            if class_name in report:
                metrics = report[class_name]
                print(f"  {class_name}: "
                     f"Precision={metrics['precision']:.4f}, "
                     f"Recall={metrics['recall']:.4f}, "
                     f"F1={metrics['f1-score']:.4f}, "
                     f"Samples={metrics['support']}")
        
        return results
    
    def _create_test_dataset(self, test_dir, transform):
        """Create a dataset from the test directory"""
        from advanced_transfer_learning import AmuletDataset
        
        test_dir = Path(test_dir)
        image_paths = []
        labels = []
        metadata = []
        
        # Load class mapping if available
        class_mapping = {}
        labels_path = Path(self.config.get('data_path', '.')) / 'labels.json'
        if labels_path.exists():
            with open(labels_path, 'r', encoding='utf-8') as f:
                class_mapping = json.load(f)
        
        # Iterate through class directories
        for class_dir in test_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            
            # Get class index
            class_idx = None
            for idx, name in enumerate(self.class_names):
                if name == class_name:
                    class_idx = idx
                    break
            
            if class_idx is None:
                # Try to find class index from class mapping
                for idx, name in class_mapping.items():
                    if name == class_name:
                        class_idx = int(idx)
                        break
            
            if class_idx is None:
                logger.warning(f"Class {class_name} not found in model classes, skipping")
                continue
            
            # Get all image files
            for img_path in class_dir.glob('**/*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_paths.append(str(img_path))
                    labels.append(class_idx)
                    metadata.append({
                        'class_name': class_name,
                        'view_type': self._determine_view_type(img_path)
                    })
        
        logger.info(f"Created test dataset with {len(image_paths)} images")
        
        return AmuletDataset(image_paths, labels, transform, self.class_names, metadata)
    
    def _determine_view_type(self, img_path):
        """Determine view type from filename"""
        filename = img_path.name.lower()
        
        if 'front' in filename or '-f' in filename:
            return 'front'
        elif 'back' in filename or '-b' in filename:
            return 'back'
        else:
            return 'unknown'
    
    def evaluate_single_image(self, image_path):
        """
        Evaluate model on a single image
        
        Args:
            image_path: Path to image file
        """
        logger.info(f"Evaluating single image: {image_path}")
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load and transform image
        try:
            image = Image.open(image_path).convert('RGB')
            transform = self._create_data_transform()
            tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                _, predicted = torch.max(outputs, 1)
                
                predicted_class = predicted.item()
                predicted_name = self.class_names[predicted_class]
                confidence = probs[predicted_class].item()
                
                # Get top 3 predictions
                top_indices = torch.argsort(probs, descending=True)[:3].cpu().numpy()
                top_probs = [probs[idx].item() for idx in top_indices]
                top_classes = [self.class_names[idx] for idx in top_indices]
                
                # Create result
                result = {
                    'path': str(image_path),
                    'predicted_class': predicted_name,
                    'confidence': confidence,
                    'top_predictions': [
                        {'class': cls, 'probability': prob}
                        for cls, prob in zip(top_classes, top_probs)
                    ]
                }
                
                # Create visualization
                self._visualize_single_prediction(image, result)
                
                # Print results
                print(f"\nPrediction for {image_path}:")
                print(f"Predicted class: {predicted_name}")
                print(f"Confidence: {confidence:.4f}")
                print(f"\nTop 3 predictions:")
                for cls, prob in zip(top_classes, top_probs):
                    print(f"  {cls}: {prob:.4f}")
                
                return result
                
        except Exception as e:
            logger.error(f"Error evaluating image: {e}")
            raise
    
    def _visualize_evaluation_results(self, results, sample_results):
        """Visualize evaluation results"""
        output_dir = Path('evaluation_results/visualizations')
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create confusion matrix visualization
        self._create_confusion_matrix_plot(
            np.array(results['confusion_matrix']),
            results['class_names'],
            output_dir / 'confusion_matrix.png'
        )
        
        # Create per-class metrics visualization
        self._create_class_metrics_plot(
            results['classification_report'],
            output_dir / 'class_metrics.png'
        )
        
        # Create error analysis visualization
        self._create_error_examples(
            sample_results,
            output_dir / 'error_examples.png',
            max_errors=10
        )
    
    def _create_confusion_matrix_plot(self, cm, class_names, output_path):
        """Create confusion matrix visualization"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_class_metrics_plot(self, report, output_path):
        """Create per-class metrics visualization"""
        # Extract class metrics
        classes = []
        precisions = []
        recalls = []
        f1_scores = []
        supports = []
        
        for class_name, metrics in report.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            
            classes.append(class_name)
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1-score'])
            supports.append(metrics['support'])
        
        # Create figure
        plt.figure(figsize=(14, 7))
        
        # Create bar chart
        x = np.arange(len(classes))
        width = 0.25
        
        plt.bar(x - width, precisions, width, label='Precision')
        plt.bar(x, recalls, width, label='Recall')
        plt.bar(x + width, f1_scores, width, label='F1 Score')
        
        # Add support information as text
        for i, support in enumerate(supports):
            plt.text(i, max(precisions[i], recalls[i], f1_scores[i]) + 0.05, 
                    f"n={support}", ha='center')
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Per-Class Metrics')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_error_examples(self, sample_results, output_path, max_errors=10):
        """Create error examples visualization"""
        # Get misclassified samples
        errors = [sample for sample in sample_results if not sample['correct']]
        
        # Limit number of errors to display
        if len(errors) > max_errors:
            errors = errors[:max_errors]
        
        if not errors:
            logger.info("No classification errors found")
            return
        
        # Create grid of error examples
        n_cols = min(5, len(errors))
        n_rows = (len(errors) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(n_cols * 4, n_rows * 5))
        
        for i, error in enumerate(errors):
            # Load image
            img_path = error['path']
            img = Image.open(img_path).convert('RGB')
            
            # Create subplot
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(img)
            plt.title(f"True: {error['true_class']}\nPred: {error['predicted_class']}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_single_prediction(self, image, result):
        """Visualize prediction for a single image"""
        output_dir = Path('evaluation_results/single')
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Display image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Input Image")
        plt.axis('off')
        
        # Display prediction probabilities
        plt.subplot(1, 2, 2)
        classes = [pred['class'] for pred in result['top_predictions']]
        probs = [pred['probability'] for pred in result['top_predictions']]
        
        y_pos = np.arange(len(classes))
        
        plt.barh(y_pos, probs, align='center')
        plt.yticks(y_pos, classes)
        plt.xlabel('Probability')
        plt.title('Top Predictions')
        
        # Highlight predicted class
        plt.gca().get_yticklabels()[0].set_color('red')
        
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"prediction_{Path(result['path']).stem}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a more detailed visualization with image annotation
        self._create_annotated_prediction(image, result, output_dir)
    
    def _create_annotated_prediction(self, image, result, output_dir):
        """Create annotated prediction image"""
        # Make a copy of the image
        img_width, img_height = image.size
        font_size = max(12, min(img_width, img_height) // 20)
        
        # Create a larger canvas to fit image and text
        canvas_width = img_width
        canvas_height = int(img_height * 1.3)  # Extra space for text
        canvas = Image.new('RGB', (canvas_width, canvas_height), color=(255, 255, 255))
        
        # Paste the original image
        canvas.paste(image, (0, 0))
        
        # Add text annotations
        draw = ImageDraw.Draw(canvas)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Add prediction information
        y_offset = img_height + 10
        draw.text((10, y_offset), f"Prediction: {result['top_predictions'][0]['class']}", 
                  fill=(0, 0, 0), font=font)
        
        y_offset += font_size + 5
        draw.text((10, y_offset), f"Confidence: {result['top_predictions'][0]['probability']:.4f}", 
                  fill=(0, 0, 0), font=font)
        
        # Add top 3 predictions
        y_offset += font_size + 10
        draw.text((10, y_offset), "Top Predictions:", fill=(0, 0, 0), font=font)
        
        for i, pred in enumerate(result['top_predictions']):
            y_offset += font_size + 5
            draw.text((10, y_offset), f"{i+1}. {pred['class']}: {pred['probability']:.4f}", 
                     fill=(0, 0, 0), font=font)
        
        # Save the annotated image
        output_path = output_dir / f"annotated_{Path(result['path']).stem}.png"
        canvas.save(output_path)
        
        logger.info(f"Saved annotated prediction to {output_path}")

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='Evaluate Amulet Classification Model')
    parser.add_argument('--model', required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--test-dir', help='Path to test directory for batch evaluation')
    parser.add_argument('--image', help='Path to a single image for evaluation')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model)
    
    # Evaluate model
    if args.test_dir:
        evaluator.evaluate_test_set(args.test_dir)
    elif args.image:
        evaluator.evaluate_single_image(args.image)
    else:
        parser.error("Either --test-dir or --image must be specified")

if __name__ == "__main__":
    main()
