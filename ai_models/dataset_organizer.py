"""
📊 Dataset Organizer & Embedding Database Manager
จัดการ dataset และสร้าง embedding database ขั้นสูง
"""
import numpy as np
from PIL import Image
from pathlib import Path
import json
import sqlite3
import logging
from typing import List, Dict, Tuple, Optional, Union
import torch
import faiss
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import shutil
from collections import defaultdict
import concurrent.futures
from tqdm import tqdm

# Import our advanced systems
from advanced_image_processor import advanced_processor, process_image_max_quality
from self_supervised_learning import ContrastiveLearningModel, AdvancedEmbeddingSystem

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingRecord:
    """Record for embedding database"""
    id: str
    image_path: str
    category: str
    embedding: np.ndarray
    metadata: Dict
    quality_score: float
    created_at: str
    
class DatasetOrganizer:
    """
    Advanced Dataset Organizer
    จัดการ dataset อย่างเป็นระบบ
    """
    
    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.stats = defaultdict(int)
        self.category_mapping = {}
        
    def organize_dataset_structure(self, split_ratios: Dict[str, float] = None):
        """Organize dataset into train/validation/test structure"""
        if split_ratios is None:
            split_ratios = {'train': 0.8, 'validation': 0.1, 'test': 0.1}
        
        logger.info("📁 Organizing dataset structure...")
        
        # Create target directory structure
        for split in split_ratios.keys():
            (self.target_dir / split).mkdir(parents=True, exist_ok=True)
        
        # Process each category
        categories = [d for d in self.source_dir.iterdir() if d.is_dir()]
        
        for category_dir in tqdm(categories, desc="Processing categories"):
            category_name = category_dir.name
            self.category_mapping[category_name] = category_name
            
            # Get all images in category
            images = []
            extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
            
            for ext in extensions:
                images.extend(category_dir.glob(f'*{ext}'))
                images.extend(category_dir.glob(f'*{ext.upper()}'))
            
            if not images:
                logger.warning(f"⚠️ No images found in {category_name}")
                continue
            
            # Shuffle images for random split
            np.random.shuffle(images)
            
            # Calculate split indices
            n_images = len(images)
            n_train = int(n_images * split_ratios['train'])
            n_val = int(n_images * split_ratios['validation'])
            n_test = n_images - n_train - n_val
            
            # Split images
            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]
            
            # Copy images to appropriate splits
            splits_data = {
                'train': train_images,
                'validation': val_images,
                'test': test_images
            }
            
            for split, split_images in splits_data.items():
                if split_images:
                    split_category_dir = self.target_dir / split / category_name
                    split_category_dir.mkdir(parents=True, exist_ok=True)
                    
                    for img_path in split_images:
                        target_path = split_category_dir / img_path.name
                        try:
                            if not target_path.exists():
                                shutil.copy2(img_path, target_path)
                                self.stats[f'{split}_copied'] += 1
                        except Exception as e:
                            logger.error(f"❌ Failed to copy {img_path}: {e}")
            
            self.stats['categories_processed'] += 1
            self.stats[f'{category_name}_total'] = n_images
            self.stats[f'{category_name}_train'] = n_train
            self.stats[f'{category_name}_val'] = n_val
            self.stats[f'{category_name}_test'] = n_test
        
        # Save organization report
        self._save_organization_report()
        
        logger.info("✅ Dataset organization completed")
        return dict(self.stats)
    
    def _save_organization_report(self):
        """Save organization report"""
        report_path = self.target_dir / "organization_report.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'source_dir': str(self.source_dir),
            'target_dir': str(self.target_dir),
            'statistics': dict(self.stats),
            'category_mapping': self.category_mapping
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Organization report saved to {report_path}")
    
    def analyze_categories(self) -> Dict:
        """Analyze category distribution and quality"""
        analysis = {}
        
        for category_dir in self.source_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            category_analysis = {
                'name': category_name,
                'total_images': 0,
                'formats': defaultdict(int),
                'sizes': [],
                'quality_scores': [],
                'sample_paths': []
            }
            
            # Analyze images in category
            extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
            
            for img_path in category_dir.iterdir():
                if img_path.suffix.lower() in extensions:
                    category_analysis['total_images'] += 1
                    category_analysis['formats'][img_path.suffix.lower()] += 1
                    
                    try:
                        # Get file size
                        size_mb = img_path.stat().st_size / (1024 * 1024)
                        category_analysis['sizes'].append(size_mb)
                        
                        # Sample for quality analysis (first 5 images)
                        if len(category_analysis['sample_paths']) < 5:
                            category_analysis['sample_paths'].append(str(img_path))
                            
                            # Analyze quality
                            image = Image.open(img_path)
                            quality_metrics = advanced_processor.get_image_quality_metrics(image)
                            quality_score = self._calculate_quality_score(quality_metrics)
                            category_analysis['quality_scores'].append(quality_score)
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to analyze {img_path}: {e}")
            
            # Calculate statistics
            if category_analysis['sizes']:
                category_analysis['avg_size_mb'] = np.mean(category_analysis['sizes'])
                category_analysis['size_std_mb'] = np.std(category_analysis['sizes'])
            
            if category_analysis['quality_scores']:
                category_analysis['avg_quality'] = np.mean(category_analysis['quality_scores'])
                category_analysis['quality_std'] = np.std(category_analysis['quality_scores'])
            
            analysis[category_name] = category_analysis
        
        return analysis
    
    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate quality score from metrics"""
        sharpness_score = min(1.0, metrics['sharpness'] / 1000.0)
        contrast_score = min(1.0, metrics['contrast'] / 100.0)
        snr_score = min(1.0, metrics['snr'] / 10.0)
        brightness_score = 1.0 - abs(metrics['brightness'] - 127.5) / 127.5
        
        return 0.4 * sharpness_score + 0.3 * contrast_score + 0.2 * snr_score + 0.1 * brightness_score

class EmbeddingDatabase:
    """
    Advanced Embedding Database Manager
    จัดการฐานข้อมูล embedding ขั้นสูง
    """
    
    def __init__(self, db_path: str, dimension: int = 512):
        self.db_path = Path(db_path)
        self.dimension = dimension
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database
        self._init_database()
        
        # Initialize FAISS index
        self.faiss_index = None
        self.embedding_ids = []
        
    def _init_database(self):
        """Initialize SQLite database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create embeddings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    image_path TEXT NOT NULL,
                    category TEXT NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    metadata_json TEXT,
                    quality_score REAL,
                    created_at TEXT,
                    UNIQUE(image_path)
                )
            ''')
            
            # Create categories table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS categories (
                    name TEXT PRIMARY KEY,
                    count INTEGER DEFAULT 0,
                    avg_quality REAL DEFAULT 0.0,
                    created_at TEXT
                )
            ''')
            
            # Create indices for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON embeddings(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_quality ON embeddings(quality_score)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created ON embeddings(created_at)')
            
            conn.commit()
            logger.info("🗄️ Database schema initialized")
    
    def add_embedding(self, record: EmbeddingRecord) -> bool:
        """Add embedding record to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Serialize embedding
                embedding_blob = pickle.dumps(record.embedding)
                metadata_json = json.dumps(record.metadata, ensure_ascii=False)
                
                # Insert record
                cursor.execute('''
                    INSERT OR REPLACE INTO embeddings
                    (id, image_path, category, embedding_blob, metadata_json, quality_score, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.id,
                    record.image_path,
                    record.category,
                    embedding_blob,
                    metadata_json,
                    record.quality_score,
                    record.created_at
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to add embedding {record.id}: {e}")
            return False
    
    def build_faiss_index(self, index_type: str = 'IVFFlat') -> bool:
        """Build FAISS index from database embeddings"""
        try:
            # Load all embeddings
            embeddings, ids = self._load_all_embeddings()
            
            if len(embeddings) == 0:
                logger.warning("⚠️ No embeddings found to build index")
                return False
            
            embeddings_array = np.vstack(embeddings)
            
            # Create FAISS index based on type
            if index_type == 'IVFFlat':
                quantizer = faiss.IndexFlatIP(self.dimension)
                nlist = min(100, len(embeddings) // 10)  # Adaptive nlist
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                
                # Train index if needed
                if len(embeddings) >= nlist:
                    index.train(embeddings_array)
                else:
                    # Fallback to flat index for small datasets
                    index = faiss.IndexFlatIP(self.dimension)
                    
            elif index_type == 'HNSW':
                index = faiss.IndexHNSWFlat(self.dimension, 32)
                index.hnsw.efConstruction = 200
                
            else:  # Default flat index
                index = faiss.IndexFlatIP(self.dimension)
            
            # Add embeddings to index
            index.add(embeddings_array)
            
            self.faiss_index = index
            self.embedding_ids = ids
            
            # Save index to disk
            index_path = self.db_path.parent / f"faiss_index_{index_type.lower()}.bin"
            faiss.write_index(index, str(index_path))
            
            # Save ID mapping
            ids_path = self.db_path.parent / "embedding_ids.pkl"
            with open(ids_path, 'wb') as f:
                pickle.dump(ids, f)
            
            logger.info(f"🔍 FAISS index built successfully: {len(embeddings)} embeddings, {index_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to build FAISS index: {e}")
            return False
    
    def _load_all_embeddings(self) -> Tuple[List[np.ndarray], List[str]]:
        """Load all embeddings from database"""
        embeddings = []
        ids = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, embedding_blob FROM embeddings ORDER BY created_at')
            
            for row in cursor.fetchall():
                embedding_id, embedding_blob = row
                embedding = pickle.loads(embedding_blob)
                
                embeddings.append(embedding)
                ids.append(embedding_id)
        
        return embeddings, ids
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar embeddings"""
        if self.faiss_index is None:
            logger.error("❌ FAISS index not built. Call build_faiss_index() first")
            return []
        
        try:
            # Search in FAISS index
            query_embedding = query_embedding.reshape(1, -1)
            distances, indices = self.faiss_index.search(query_embedding, k)
            
            # Get detailed results from database
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.embedding_ids):
                    embedding_id = self.embedding_ids[idx]
                    record = self.get_embedding_record(embedding_id)
                    
                    if record:
                        result = {
                            'id': embedding_id,
                            'similarity': float(distance),
                            'rank': i + 1,
                            'category': record['category'],
                            'image_path': record['image_path'],
                            'quality_score': record['quality_score'],
                            'metadata': record['metadata']
                        }
                        results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to search similar embeddings: {e}")
            return []
    
    def get_embedding_record(self, embedding_id: str) -> Optional[Dict]:
        """Get embedding record by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, image_path, category, embedding_blob, metadata_json, quality_score, created_at
                    FROM embeddings WHERE id = ?
                ''', (embedding_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'image_path': row[1],
                        'category': row[2],
                        'embedding': pickle.loads(row[3]),
                        'metadata': json.loads(row[4]) if row[4] else {},
                        'quality_score': row[5],
                        'created_at': row[6]
                    }
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get embedding record {embedding_id}: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        stats = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total embeddings
                cursor.execute('SELECT COUNT(*) FROM embeddings')
                stats['total_embeddings'] = cursor.fetchone()[0]
                
                # Category distribution
                cursor.execute('''
                    SELECT category, COUNT(*), AVG(quality_score)
                    FROM embeddings GROUP BY category ORDER BY COUNT(*) DESC
                ''')
                stats['categories'] = {}
                for row in cursor.fetchall():
                    category, count, avg_quality = row
                    stats['categories'][category] = {
                        'count': count,
                        'avg_quality': avg_quality or 0.0
                    }
                
                # Quality distribution
                cursor.execute('''
                    SELECT MIN(quality_score), MAX(quality_score), AVG(quality_score)
                    FROM embeddings WHERE quality_score IS NOT NULL
                ''')
                row = cursor.fetchone()
                if row and row[0] is not None:
                    stats['quality'] = {
                        'min': row[0],
                        'max': row[1],
                        'avg': row[2]
                    }
                
                # Recent additions
                cursor.execute('''
                    SELECT COUNT(*) FROM embeddings
                    WHERE created_at > datetime('now', '-1 day')
                ''')
                stats['added_last_24h'] = cursor.fetchone()[0]
        
        except Exception as e:
            logger.error(f"❌ Failed to get statistics: {e}")
        
        return stats
    
    def export_embeddings(self, output_path: str) -> bool:
        """Export embeddings to file"""
        try:
            embeddings, ids = self._load_all_embeddings()
            
            export_data = {
                'embeddings': [emb.tolist() for emb in embeddings],
                'ids': ids,
                'dimension': self.dimension,
                'exported_at': datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"📤 Exported {len(embeddings)} embeddings to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to export embeddings: {e}")
            return False

def create_embedding_record(image_path: str, category: str, 
                          embedding: np.ndarray, metadata: Dict,
                          quality_score: float) -> EmbeddingRecord:
    """Create embedding record with auto-generated ID"""
    # Generate unique ID
    content = f"{image_path}_{category}_{datetime.now().isoformat()}"
    record_id = hashlib.md5(content.encode()).hexdigest()
    
    return EmbeddingRecord(
        id=record_id,
        image_path=image_path,
        category=category,
        embedding=embedding,
        metadata=metadata,
        quality_score=quality_score,
        created_at=datetime.now().isoformat()
    )
