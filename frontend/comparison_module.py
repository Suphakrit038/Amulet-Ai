"""
🔍 Image Comparison Module
โมดูลสำหรับเปรียบเทียบรูปภาพของผู้ใช้กับรูปภาพในฐานข้อมูล
"""
import os
import json
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("image_comparison.log")
    ]
)
logger = logging.getLogger("image_comparison")

class FeatureExtractor:
    """Feature extractor using trained model"""
    
    def __init__(self, model_path, device=None):
        """Initialize with model path"""
        self.model_path = Path(model_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        
        # Load model
        self._load_model()
        
        logger.info(f"Feature extractor loaded successfully on {self.device}")
    
    def _load_model(self):
        """Load the trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Determine model architecture
        if 'config' in checkpoint and 'model_type' in checkpoint['config']:
            model_type = checkpoint['config']['model_type']
        else:
            model_type = 'efficientnet_b0'  # Default
        
        # Create model based on architecture
        if model_type.startswith('efficientnet'):
            model = models.efficientnet_b0(pretrained=False)
            # Remove the classification layer to get features
            self.feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()  # Replace with Identity to get features
        else:
            model = models.resnet50(pretrained=False)
            self.feature_dim = model.fc.in_features
            model.fc = nn.Identity()  # Replace with Identity to get features
        
        # Load weights (except the classification layer)
        model_state_dict = checkpoint['model_state_dict']
        
        # Filter out classifier weights
        if model_type.startswith('efficientnet'):
            filtered_state_dict = {k: v for k, v in model_state_dict.items() 
                                 if not k.startswith('classifier')}
        else:
            filtered_state_dict = {k: v for k, v in model_state_dict.items() 
                                 if not k.startswith('fc')}
        
        # Load filtered weights
        model.load_state_dict(filtered_state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        
        self.model = model
        
        # Create transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image_path):
        """Extract features from an image"""
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Transform image
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)
            
        return features.cpu().numpy().flatten()

class ImageComparer:
    """Image comparison using feature similarity"""
    
    def __init__(self, model_path, database_dir, result_dir="comparison_results"):
        """Initialize with model path and database directory"""
        self.model_path = Path(model_path)
        self.database_dir = Path(database_dir)
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature extractor
        self.extractor = FeatureExtractor(model_path)
        
        # Load class names
        labels_path = self.database_dir / "labels.json"
        if labels_path.exists():
            with open(labels_path, 'r', encoding='utf-8') as f:
                self.class_map = json.load(f)
                self.class_names = [self.class_map[str(i)] for i in range(len(self.class_map))]
        else:
            # Use directory names as class names
            self.class_names = [d.name for d in self.database_dir.iterdir() if d.is_dir()]
        
        logger.info(f"Initialized image comparer with {len(self.class_names)} classes")
        
        # Cache for database features
        self.database_features = {}
        self.database_paths = {}
    
    def build_database_cache(self):
        """Build cache of features for all database images"""
        logger.info("Building database feature cache...")
        start_time = time.time()
        
        for class_name in self.class_names:
            class_dir = self.database_dir / class_name
            if not class_dir.exists() or not class_dir.is_dir():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Collect image paths
            image_paths = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_paths.extend(list(class_dir.glob(f"*{ext}")))
                image_paths.extend(list(class_dir.glob(f"*{ext.upper()}")))
            
            if not image_paths:
                logger.warning(f"No images found for class: {class_name}")
                continue
            
            logger.info(f"Extracting features for {len(image_paths)} images in class {class_name}")
            
            # Extract features for each image
            self.database_features[class_name] = []
            self.database_paths[class_name] = []
            
            for img_path in image_paths:
                try:
                    features = self.extractor.extract_features(img_path)
                    self.database_features[class_name].append(features)
                    self.database_paths[class_name].append(str(img_path))
                except Exception as e:
                    logger.error(f"Error extracting features from {img_path}: {e}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Database cache built in {elapsed_time:.2f} seconds")
        
        # Save cache summary
        cache_summary = {
            "classes": {},
            "total_images": 0
        }
        
        for class_name, features in self.database_features.items():
            cache_summary["classes"][class_name] = len(features)
            cache_summary["total_images"] += len(features)
        
        with open(self.result_dir / "cache_summary.json", 'w', encoding='utf-8') as f:
            json.dump(cache_summary, f, indent=2, ensure_ascii=False)
    
    def compare_image(self, image_path, top_k=5):
        """Compare an image with the database and return the most similar ones"""
        # Extract features from query image
        query_features = self.extractor.extract_features(image_path)
        
        # If database cache is empty, build it
        if not self.database_features:
            self.build_database_cache()
        
        # Compare with all images in database
        all_similarities = []
        all_paths = []
        all_classes = []
        
        for class_name, features_list in self.database_features.items():
            for i, features in enumerate(features_list):
                similarity = cosine_similarity([query_features], [features])[0][0]
                all_similarities.append(similarity)
                all_paths.append(self.database_paths[class_name][i])
                all_classes.append(class_name)
        
        # Sort by similarity (descending)
        indices = np.argsort(all_similarities)[::-1]
        top_indices = indices[:top_k]
        
        top_similarities = [all_similarities[i] for i in top_indices]
        top_paths = [all_paths[i] for i in top_indices]
        top_classes = [all_classes[i] for i in top_indices]
        
        # Create result
        result = {
            "query_image": str(image_path),
            "top_matches": [
                {
                    "path": path,
                    "class": class_name,
                    "similarity": float(similarity)
                }
                for path, class_name, similarity in zip(top_paths, top_classes, top_similarities)
            ]
        }
        
        return result
    
    def visualize_comparison(self, image_path, output_path=None, top_k=5):
        """Visualize comparison between query image and top matches"""
        # Get comparison result
        result = self.compare_image(image_path, top_k)
        
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Display query image
        plt.subplot(1, top_k + 1, 1)
        query_img = Image.open(image_path).convert('RGB')
        plt.imshow(query_img)
        plt.title("Query Image")
        plt.axis('off')
        
        # Display top matches
        for i, match in enumerate(result["top_matches"]):
            plt.subplot(1, top_k + 1, i + 2)
            match_img = Image.open(match["path"]).convert('RGB')
            plt.imshow(match_img)
            plt.title(f"{match['class']}\nSimilarity: {match['similarity']:.2f}")
            plt.axis('off')
        
        plt.tight_layout()
        
        # Save or show figure
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            output_path = self.result_dir / f"comparison_{Path(image_path).stem}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
    
    def batch_compare(self, query_dir, output_dir=None, top_k=5):
        """Compare multiple images with the database"""
        query_dir = Path(query_dir)
        if not query_dir.exists():
            raise FileNotFoundError(f"Query directory not found: {query_dir}")
        
        output_dir = Path(output_dir) if output_dir else self.result_dir / "batch_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect image paths
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_paths.extend(list(query_dir.glob(f"*{ext}")))
            image_paths.extend(list(query_dir.glob(f"*{ext.upper()}")))
        
        if not image_paths:
            logger.warning(f"No images found in query directory: {query_dir}")
            return []
        
        logger.info(f"Comparing {len(image_paths)} images")
        
        # If database cache is empty, build it
        if not self.database_features:
            self.build_database_cache()
        
        # Process each image
        results = []
        for img_path in image_paths:
            try:
                # Compare image
                result = self.compare_image(img_path, top_k)
                results.append(result)
                
                # Visualize comparison
                vis_path = output_dir / f"comparison_{img_path.stem}.png"
                self.visualize_comparison(img_path, vis_path, top_k)
                
                logger.info(f"Processed {img_path.name}")
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
        
        # Save overall results
        with open(output_dir / "batch_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
