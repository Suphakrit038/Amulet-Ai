"""
🚀 Self-Supervised Learning System with Advanced Embeddings
ระบบ Self-Supervised Learning ขั้นสูงสำหรับพระเครื่องไทย
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
import pickle
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding system"""
    embedding_dim: int = 512
    projection_dim: int = 128
    temperature: float = 0.1
    batch_size: int = 16
    epochs: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    
class ContrastiveLearningModel(nn.Module):
    """
    Self-Supervised Contrastive Learning Model
    ใช้ SimCLR approach สำหรับเรียนรู้ representation ของพระเครื่อง
    """
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config
        
        # Backbone: EfficientNet-B4 (ความละเอียดสูง)
        self.backbone = self._create_backbone()
        
        # Projection head สำหรับ contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.backbone.classifier.in_features, config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.projection_dim)
        )
        
        # Remove classifier from backbone
        self.backbone.classifier = nn.Identity()
        
    def _create_backbone(self):
        """สร้าง backbone network"""
        import torchvision.models as models
        
        # ใช้ EfficientNet-B4 สำหรับความละเอียดสูง
        backbone = models.efficientnet_b4(pretrained=True)
        
        # Freeze early layers (optional)
        for param in list(backbone.parameters())[:50]:
            param.requires_grad = False
            
        return backbone
    
    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)
        embeddings = self.projection_head(features)
        return F.normalize(embeddings, dim=1)

class SelfSupervisedTrainer:
    """
    Self-Supervised Learning Trainer
    เทรนโมเดลแบบ self-supervised ด้วย contrastive learning
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = ContrastiveLearningModel(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )
        
        logger.info(f"🚀 Self-supervised trainer initialized on {self.device}")
    
    def create_positive_pairs(self, images):
        """
        สร้าง positive pairs โดยไม่ใช้ traditional augmentation
        ใช้การปรับคุณภาพภาพแทน
        """
        positive_pairs = []
        
        for img in images:
            # View 1: Original high-quality processed image
            view1 = img
            
            # View 2: Alternative high-quality processing
            # - Different noise reduction
            # - Different sharpening level  
            # - Different contrast optimization
            view2 = self._create_alternative_view(img)
            
            positive_pairs.append((view1, view2))
            
        return positive_pairs
    
    def _create_alternative_view(self, image):
        """สร้าง alternative view ด้วยการปรับคุณภาพแบบอื่น"""
        # ใช้การปรับพารามิเตอร์การประมวลผลแทน augmentation
        # เช่น ระดับ sharpening ที่ต่างกัน, noise reduction ที่ต่างกัน
        return image  # Placeholder - จะใช้ advanced_image_processor
    
    def contrastive_loss(self, embeddings1, embeddings2, temperature=0.1):
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
        """
        batch_size = embeddings1.shape[0]
        
        # Combine embeddings
        embeddings = torch.cat([embeddings1, embeddings2], dim=0)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
        
        # Create labels (positive pairs)
        labels = torch.cat([torch.arange(batch_size) + batch_size, 
                           torch.arange(batch_size)], dim=0)
        labels = labels.to(self.device)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
        similarity_matrix.masked_fill_(mask, -float('inf'))
        
        # Calculate loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def train_epoch(self, dataloader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, images in enumerate(dataloader):
            images = images.to(self.device)
            
            # Create positive pairs
            positive_pairs = self.create_positive_pairs(images)
            
            # Forward pass for both views
            view1_embeddings = []
            view2_embeddings = []
            
            for view1, view2 in positive_pairs:
                emb1 = self.model(view1.unsqueeze(0))
                emb2 = self.model(view2.unsqueeze(0))
                view1_embeddings.append(emb1)
                view2_embeddings.append(emb2)
            
            embeddings1 = torch.cat(view1_embeddings, dim=0)
            embeddings2 = torch.cat(view2_embeddings, dim=0)
            
            # Calculate loss
            loss = self.contrastive_loss(embeddings1, embeddings2, self.config.temperature)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def train(self, dataloader, num_epochs=None):
        """Full training loop"""
        if num_epochs is None:
            num_epochs = self.config.epochs
        
        logger.info(f"🏋️ Starting self-supervised training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(dataloader)
            self.scheduler.step()
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(f'ai_models/checkpoint_epoch_{epoch+1}.pth')
        
        logger.info("✅ Self-supervised training completed")
    
    def save_checkpoint(self, path):
        """Save training checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)

class AdvancedEmbeddingSystem:
    """
    Advanced Embedding System for Amulet Recognition
    ระบบ embedding ขั้นสูงสำหรับการจดจำพระเครื่อง
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self.embeddings_db = {}
        self.faiss_index = None
        self.metadata = []
        
        # Load trained model if exists
        self._load_model()
        
    def _load_model(self):
        """Load trained self-supervised model"""
        model_path = Path('ai_models/self_supervised_model.pth')
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                self.model = ContrastiveLearningModel(self.config)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                logger.info("✅ Loaded trained self-supervised model")
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                self.model = None
        else:
            logger.warning("⚠️ No trained model found")
    
    def extract_embeddings(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings from images using trained model
        """
        if self.model is None:
            logger.error("❌ No trained model available")
            return np.random.random((len(images), self.config.embedding_dim))
        
        embeddings = []
        self.model.eval()
        
        with torch.no_grad():
            for img in images:
                # Convert to tensor
                img_tensor = torch.from_numpy(img).float()
                if len(img_tensor.shape) == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                
                # Extract embedding
                embedding = self.model.backbone(img_tensor)
                embeddings.append(embedding.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def build_faiss_index(self, embeddings: np.ndarray, use_gpu=False):
        """
        Build FAISS index for fast similarity search
        """
        dimension = embeddings.shape[1]
        
        # Create index
        if use_gpu and faiss.get_num_gpus() > 0:
            # GPU index
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, dimension)
        else:
            # CPU index with inner product (cosine similarity)
            index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        index.add(embeddings.astype(np.float32))
        
        self.faiss_index = index
        logger.info(f"🔍 Built FAISS index with {embeddings.shape[0]} embeddings")
        
        return index
    
    def find_similar_images(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Find k most similar images using FAISS
        """
        if self.faiss_index is None:
            logger.error("❌ FAISS index not built")
            return []
        
        # Normalize query
        query_normalized = query_embedding.copy()
        faiss.normalize_L2(query_normalized)
        
        # Search
        similarities, indices = self.faiss_index.search(
            query_normalized.reshape(1, -1).astype(np.float32), k
        )
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.metadata):
                result = {
                    'similarity': float(similarity),
                    'metadata': self.metadata[idx],
                    'rank': i + 1
                }
                results.append(result)
        
        return results
    
    def cluster_embeddings(self, embeddings: np.ndarray, method='kmeans', n_clusters=None):
        """
        Cluster embeddings to discover patterns
        """
        if method == 'kmeans':
            if n_clusters is None:
                n_clusters = 6  # Default to 6 amulet types
            
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(embeddings)
            
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
            labels = clusterer.fit_predict(embeddings)
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        return labels, clusterer
    
    def reduce_dimensions(self, embeddings: np.ndarray, method='tsne', n_components=2):
        """
        Reduce embedding dimensions for visualization
        """
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        elif method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        reduced = reducer.fit_transform(embeddings)
        return reduced, reducer
    
    def analyze_embedding_space(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Comprehensive analysis of embedding space
        """
        analysis = {}
        
        # 1. Clustering analysis
        kmeans_labels, kmeans_model = self.cluster_embeddings(embeddings, 'kmeans', 6)
        dbscan_labels, dbscan_model = self.cluster_embeddings(embeddings, 'dbscan')
        
        analysis['clustering'] = {
            'kmeans_labels': kmeans_labels.tolist(),
            'dbscan_labels': dbscan_labels.tolist(),
            'n_kmeans_clusters': len(set(kmeans_labels)),
            'n_dbscan_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        }
        
        # 2. Dimensionality reduction
        tsne_2d, _ = self.reduce_dimensions(embeddings, 'tsne', 2)
        pca_2d, _ = self.reduce_dimensions(embeddings, 'pca', 2)
        
        analysis['dimensionality_reduction'] = {
            'tsne_2d': tsne_2d.tolist(),
            'pca_2d': pca_2d.tolist()
        }
        
        # 3. Inter-cluster distances
        centroids = kmeans_model.cluster_centers_
        inter_distances = cosine_similarity(centroids)
        
        analysis['inter_cluster_distances'] = inter_distances.tolist()
        
        # 4. Intra-cluster analysis
        intra_cluster_stats = []
        for cluster_id in range(len(centroids)):
            cluster_embeddings = embeddings[kmeans_labels == cluster_id]
            if len(cluster_embeddings) > 1:
                similarities = cosine_similarity(cluster_embeddings)
                stats = {
                    'cluster_id': int(cluster_id),
                    'size': len(cluster_embeddings),
                    'avg_similarity': float(np.mean(similarities)),
                    'std_similarity': float(np.std(similarities))
                }
                intra_cluster_stats.append(stats)
        
        analysis['intra_cluster_stats'] = intra_cluster_stats
        
        return analysis
    
    def save_embeddings_database(self, embeddings: np.ndarray, metadata: List[Dict], path: str):
        """Save embeddings database"""
        database = {
            'embeddings': embeddings,
            'metadata': metadata,
            'config': self.config,
            'timestamp': str(np.datetime64('now'))
        }
        
        with open(path, 'wb') as f:
            pickle.dump(database, f)
        
        logger.info(f"💾 Saved embeddings database to {path}")
    
    def load_embeddings_database(self, path: str):
        """Load embeddings database"""
        with open(path, 'rb') as f:
            database = pickle.load(f)
        
        self.embeddings_db = database['embeddings']
        self.metadata = database['metadata']
        
        # Build FAISS index
        self.build_faiss_index(self.embeddings_db)
        
        logger.info(f"📂 Loaded embeddings database from {path}")
        return database

# Example usage and integration
def create_self_supervised_system():
    """Create and initialize self-supervised learning system"""
    config = EmbeddingConfig(
        embedding_dim=512,
        projection_dim=128,
        temperature=0.1,
        batch_size=16,
        epochs=200,
        learning_rate=3e-4
    )
    
    # Create trainer
    trainer = SelfSupervisedTrainer(config)
    
    # Create embedding system  
    embedding_system = AdvancedEmbeddingSystem(config)
    
    return trainer, embedding_system
