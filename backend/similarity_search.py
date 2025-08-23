"""
FAISS-based similarity search for amulet images
Handles vector embeddings and nearest neighbor search
"""
import os
import numpy as np
import faiss
import pickle
from typing import List, Tuple, Optional
import tensorflow as tf
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class SimilaritySearchEngine:
    def __init__(self, 
                 index_path: Optional[str] = None,
                 embeddings_path: Optional[str] = None,
                 metadata_path: Optional[str] = None):
        """
        Initialize FAISS similarity search engine
        
        Args:
            index_path: Path to FAISS index file
            embeddings_path: Path to embeddings file
            metadata_path: Path to metadata file (image paths, labels, etc.)
        """
        self.index_path = index_path or "models/faiss_index.bin"
        self.embeddings_path = embeddings_path or "models/embeddings.pkl"
        self.metadata_path = metadata_path or "models/metadata.pkl"
        
        self.index = None
        self.metadata = None
        self.feature_extractor = None
        self.embedding_dim = 1280  # Default for EfficientNetV2
        
        # Load existing index and metadata if available
        self._load_index()
        self._load_metadata()
    
    def _load_feature_extractor(self):
        """Load pre-trained model for feature extraction"""
        try:
            # Use EfficientNetV2 as feature extractor
            base_model = tf.keras.applications.EfficientNetV2B0(
                weights='imagenet',
                include_top=False,
                pooling='avg'
            )
            self.feature_extractor = base_model
            self.embedding_dim = base_model.output_shape[-1]
            logger.info(f"✅ Loaded feature extractor with embedding dim: {self.embedding_dim}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load feature extractor: {e}")
            self.feature_extractor = None
    
    def _load_index(self):
        """Load existing FAISS index"""
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                logger.info(f"✅ Loaded FAISS index from {self.index_path}")
                logger.info(f"📊 Index contains {self.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"⚠️ Could not load FAISS index: {e}")
                self.index = None
        else:
            logger.info("🔄 No existing FAISS index found")
    
    def _load_metadata(self):
        """Load metadata associated with embeddings"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"✅ Loaded metadata for {len(self.metadata)} items")
            except Exception as e:
                logger.warning(f"⚠️ Could not load metadata: {e}")
                self.metadata = []
        else:
            logger.info("🔄 No existing metadata found")
            self.metadata = []
    
    def _save_index(self):
        """Save FAISS index to disk"""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            logger.info(f"💾 Saved FAISS index to {self.index_path}")
        except Exception as e:
            logger.error(f"❌ Could not save FAISS index: {e}")
    
    def _save_metadata(self):
        """Save metadata to disk"""
        try:
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info(f"💾 Saved metadata to {self.metadata_path}")
        except Exception as e:
            logger.error(f"❌ Could not save metadata: {e}")
    
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract features from image using pre-trained model
        
        Args:
            image_path: Path to image file
            
        Returns:
            Feature vector or None if extraction failed
        """
        if self.feature_extractor is None:
            self._load_feature_extractor()
            
        if self.feature_extractor is None:
            logger.error("❌ Feature extractor not available")
            return None
        
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))  # EfficientNet input size
            img_array = np.array(img)
            img_array = img_array.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Extract features
            features = self.feature_extractor.predict(img_array, verbose=0)
            return features.flatten()
            
        except Exception as e:
            logger.error(f"❌ Could not extract features from {image_path}: {e}")
            return None
    
    def build_index(self, dataset_path: str):
        """
        Build FAISS index from dataset
        
        Args:
            dataset_path: Path to dataset directory
        """
        logger.info(f"🏗️ Building FAISS index from {dataset_path}")
        
        if self.feature_extractor is None:
            self._load_feature_extractor()
        
        # Collect all image paths
        image_paths = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, file))
        
        logger.info(f"📊 Found {len(image_paths)} images")
        
        if not image_paths:
            logger.warning("⚠️ No images found in dataset")
            return
        
        # Extract features for all images
        embeddings = []
        valid_metadata = []
        
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 10 == 0:
                logger.info(f"🔄 Processing image {i+1}/{len(image_paths)}")
            
            features = self.extract_features(img_path)
            if features is not None:
                embeddings.append(features)
                
                # Extract metadata from path
                rel_path = os.path.relpath(img_path, dataset_path)
                class_name = os.path.dirname(rel_path)
                
                valid_metadata.append({
                    'path': img_path,
                    'relative_path': rel_path,
                    'class_name': class_name,
                    'filename': os.path.basename(img_path)
                })
        
        if not embeddings:
            logger.error("❌ No valid embeddings extracted")
            return
        
        # Create FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        
        # Use IndexFlatIP for cosine similarity (after L2 normalization)
        faiss.normalize_L2(embeddings_array)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_array)
        
        # Save index and metadata
        self.index = index
        self.metadata = valid_metadata
        self._save_index()
        self._save_metadata()
        
        logger.info(f"✅ Built FAISS index with {len(embeddings)} vectors")
    
    def search_similar(self, query_image_path: str, k: int = 5) -> List[dict]:
        """
        Search for similar images
        
        Args:
            query_image_path: Path to query image
            k: Number of similar images to return
            
        Returns:
            List of similar images with metadata and scores
        """
        if self.index is None:
            logger.error("❌ FAISS index not loaded")
            return []
        
        # Extract features from query image
        query_features = self.extract_features(query_image_path)
        if query_features is None:
            logger.error("❌ Could not extract features from query image")
            return []
        
        # Normalize and search
        query_features = query_features.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_features)
        
        scores, indices = self.index.search(query_features, k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def get_index_stats(self) -> dict:
        """Get statistics about the current index"""
        if self.index is None:
            return {"status": "No index loaded", "total_vectors": 0}
        
        return {
            "status": "Index loaded",
            "total_vectors": self.index.ntotal,
            "dimension": self.index.d,
            "metadata_count": len(self.metadata) if self.metadata else 0
        }

# Example usage functions
def initialize_similarity_search():
    """Initialize similarity search engine for the application"""
    search_engine = SimilaritySearchEngine()
    
    # Build index if it doesn't exist
    if search_engine.index is None:
        dataset_path = "dataset"
        if os.path.exists(dataset_path):
            search_engine.build_index(dataset_path)
        else:
            logger.warning("⚠️ Dataset directory not found")
    
    return search_engine

def find_similar_amulets(image_path: str, top_k: int = 3) -> List[dict]:
    """
    Find similar amulets for a given image
    
    Args:
        image_path: Path to query image
        top_k: Number of similar results to return
        
    Returns:
        List of similar amulets with metadata
    """
    search_engine = initialize_similarity_search()
    return search_engine.search_similar(image_path, k=top_k)
