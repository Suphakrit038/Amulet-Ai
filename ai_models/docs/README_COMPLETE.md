# 🔮 Complete Hybrid ML System Documentation

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Components](#components)
5. [Usage Guide](#usage-guide)
6. [Production Deployment](#production-deployment)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)
9. [Performance Optimization](#performance-optimization)
10. [Development Guide](#development-guide)

---

## 🎯 System Overview

This is a **complete production-ready hybrid machine learning system** for amulet classification that combines:

- **🧠 Deep Learning**: PyTorch CNN features for advanced pattern recognition
- **📊 Classical ML**: scikit-learn ensemble methods for robust classification
- **🔍 Computer Vision**: OpenCV traditional features (SIFT, HOG, LBP, etc.)
- **⚡ High Performance**: Optimized feature caching and batch processing
- **🚀 Production Ready**: FastAPI deployment with monitoring and health checks

### Key Features
- ✅ **Modular Architecture**: Easy to extend and maintain
- ✅ **Hybrid Feature Extraction**: CNN + Classical CV features
- ✅ **Class-aware Augmentation**: Handles imbalanced datasets
- ✅ **Comprehensive Evaluation**: Detailed performance analysis
- ✅ **Production Deployment**: Docker, systemd, API serving
- ✅ **Python 3.13 Compatible**: Latest Python version support

---

## 🏗️ Architecture

```
Hybrid ML System
├── 📊 Dataset Analysis (dataset_inspector.py)
│   ├── Corruption detection
│   ├── Duplicate identification
│   ├── Class distribution analysis
│   └── CV strategy recommendations
│
├── 🔄 Data Augmentation (augmentation_pipeline.py)
│   ├── Class-aware augmentation
│   ├── Quality validation
│   ├── Batch processing
│   └── Multiple techniques
│
├── 🧠 Feature Extraction (feature_extractors.py)
│   ├── CNN Features (PyTorch)
│   ├── Classical Features (OpenCV)
│   ├── Feature caching
│   └── Modular design
│
├── 🎯 Model Training (hybrid_trainer.py)
│   ├── Ensemble methods
│   ├── Hyperparameter tuning
│   ├── Cross-validation
│   └── Model persistence
│
├── 📊 Evaluation Suite (evaluation_suite.py)
│   ├── Performance metrics
│   ├── Feature importance
│   ├── Error analysis
│   └── Visualizations
│
└── 🚀 Production Deployment (production_deployment.py)
    ├── FastAPI service
    ├── Docker containers
    ├── Health monitoring
    └── API documentation
```

---

## 💻 Installation & Setup

### Prerequisites
- **Python 3.11+** (tested with 3.13.5)
- **4GB RAM minimum** (8GB recommended)
- **2GB disk space** for models and cache

### Step 1: Environment Setup

```bash
# Clone repository
cd c:\Users\Admin\Documents\GitHub\Amulet-Ai

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements_compatible.txt
```

### Step 2: Quick Installation Check

```bash
# Test Python version compatibility
python -c "import sys; print(f'Python {sys.version}')"

# Test core dependencies
python -c "import sklearn, cv2, numpy; print('Core dependencies OK')"

# Test PyTorch (optional but recommended)
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### Step 3: Dataset Preparation

```bash
# Organize your dataset like this:
dataset/
├── class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_2/
│   ├── image1.jpg
│   └── ...
└── class_n/
    └── ...
```

---

## 🔧 Components

### 1. 📊 Dataset Inspector (`dataset_inspector.py`)

**Purpose**: Comprehensive dataset analysis and quality assurance

**Features**:
- Image corruption detection
- Duplicate image identification
- Class distribution analysis
- Cross-validation strategy recommendations
- Dataset quality metrics

**Usage**:
```bash
# Analyze dataset
python ai_models/dataset_inspector.py --data-dir dataset --output-dir reports

# Quick inspection
python ai_models/dataset_inspector.py --data-dir dataset --quick-mode
```

**CLI Options**:
- `--data-dir`: Path to dataset directory
- `--output-dir`: Output directory for reports
- `--quick-mode`: Fast analysis (fewer checks)
- `--check-duplicates`: Enable duplicate detection
- `--corruption-threshold`: Corruption detection sensitivity

### 2. 🔄 Augmentation Pipeline (`augmentation_pipeline.py`)

**Purpose**: Class-aware data augmentation for imbalanced datasets

**Features**:
- Multiple augmentation techniques (rotation, brightness, noise, blur)
- Quality validation for augmented images
- Class-aware balancing strategies
- Batch processing support

**Usage**:
```bash
# Balance dataset with augmentation
python ai_models/augmentation_pipeline.py --input-dir dataset --output-dir dataset_augmented

# Custom augmentation settings
python ai_models/augmentation_pipeline.py \
    --input-dir dataset \
    --output-dir dataset_augmented \
    --target-samples 1000 \
    --augmentation-factor 2.0
```

**CLI Options**:
- `--input-dir`: Source dataset directory
- `--output-dir`: Augmented dataset output
- `--target-samples`: Target samples per class
- `--augmentation-factor`: Augmentation multiplier
- `--quality-threshold`: Quality validation threshold

### 3. 🧠 Feature Extractors (`feature_extractors.py`)

**Purpose**: Hybrid feature extraction combining CNN and classical methods

**Features**:
- CNN features using PyTorch (optional)
- Classical CV features (SIFT, HOG, LBP, etc.)
- Feature caching for performance
- Modular extractor design

**Usage**:
```python
from ai_models.feature_extractors import HybridFeatureExtractor, FeatureConfig

# Configure feature extraction
config = FeatureConfig(
    use_cnn_features=True,
    use_sift_features=True,
    use_hog_features=True,
    cache_features=True
)

# Initialize extractor
extractor = HybridFeatureExtractor(config)

# Extract features from images
features = extractor.extract_batch(images)
```

### 4. 🎯 Hybrid Trainer (`hybrid_trainer.py`)

**Purpose**: Complete training pipeline with ensemble methods

**Features**:
- Multiple ML algorithms (Random Forest, SVM, etc.)
- Ensemble model creation
- Hyperparameter optimization
- Cross-validation and evaluation

**Usage**:
```bash
# Train complete hybrid model
python ai_models/hybrid_trainer.py \
    --data-dir dataset \
    --output-dir ai_models/saved_models/hybrid_amulet_classifier

# Quick training (faster, fewer iterations)
python ai_models/hybrid_trainer.py \
    --data-dir dataset \
    --output-dir saved_models \
    --quick-mode
```

**CLI Options**:
- `--data-dir`: Training dataset directory
- `--output-dir`: Model output directory
- `--quick-mode`: Fast training mode
- `--cv-folds`: Cross-validation folds
- `--test-size`: Test set proportion

### 5. 📊 Evaluation Suite (`evaluation_suite.py`)

**Purpose**: Comprehensive model evaluation and analysis

**Features**:
- Detailed performance metrics
- Feature importance analysis
- Error case visualization
- Speed profiling

**Usage**:
```bash
# Comprehensive evaluation
python ai_models/evaluation_suite.py \
    --model-dir ai_models/saved_models/hybrid_amulet_classifier \
    --output-dir evaluation_reports

# Quick evaluation
python ai_models/evaluation_suite.py \
    --model-dir saved_models/model \
    --quick
```

### 6. 🚀 Production Deployment (`production_deployment.py`)

**Purpose**: Production-ready API service deployment

**Features**:
- FastAPI REST API
- Docker containerization
- Health monitoring
- Load balancing support

**Usage**:
```bash
# Start development server
python ai_models/production_deployment.py serve

# Create deployment files
python ai_models/production_deployment.py deploy --output-dir deployment

# Production server
python ai_models/production_deployment.py serve --host 0.0.0.0 --port 8000
```

---

## 🚀 Usage Guide

### Quick Start (Complete Pipeline)

```bash
# 1. Analyze your dataset
python ai_models/dataset_inspector.py --data-dir dataset

# 2. Augment if needed (for imbalanced classes)
python ai_models/augmentation_pipeline.py --input-dir dataset --output-dir dataset_balanced

# 3. Train the hybrid model
python ai_models/hybrid_trainer.py --data-dir dataset_balanced --output-dir saved_models

# 4. Evaluate the model
python ai_models/evaluation_suite.py --model-dir saved_models

# 5. Deploy to production
python ai_models/production_deployment.py serve
```

### Typical Workflow

#### Step 1: Dataset Analysis
```bash
python ai_models/dataset_inspector.py \
    --data-dir dataset \
    --output-dir analysis_reports \
    --check-duplicates \
    --verbose
```

**Expected output**:
- Dataset quality report
- Class distribution analysis
- Recommendations for improvement

#### Step 2: Data Preparation (if needed)
```bash
# If classes are imbalanced
python ai_models/augmentation_pipeline.py \
    --input-dir dataset \
    --output-dir dataset_augmented \
    --target-samples 500 \
    --verbose
```

#### Step 3: Model Training
```bash
python ai_models/hybrid_trainer.py \
    --data-dir dataset_augmented \
    --output-dir ai_models/saved_models/my_model \
    --cv-folds 5 \
    --verbose
```

**Expected output**:
- Trained ensemble model
- Feature extractors
- Training configuration
- Evaluation results

#### Step 4: Model Evaluation
```bash
python ai_models/evaluation_suite.py \
    --model-dir ai_models/saved_models/my_model \
    --output-dir evaluation_reports \
    --test-data-dir test_dataset
```

**Expected output**:
- Performance metrics
- Confusion matrices
- Feature importance plots
- Error analysis

#### Step 5: Production Deployment
```bash
# Create deployment files
python ai_models/production_deployment.py deploy

# Start service
docker-compose up -d

# Test API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"image_data": "base64_encoded_image"}'
```

---

## 🐳 Production Deployment

### Docker Deployment (Recommended)

1. **Create deployment files**:
```bash
python ai_models/production_deployment.py deploy --type docker
```

2. **Build and start**:
```bash
docker-compose up -d
```

3. **Verify deployment**:
```bash
curl http://localhost:8000/health
```

### Manual Deployment

1. **Install dependencies**:
```bash
pip install -r requirements_production.txt
```

2. **Start service**:
```bash
python -m uvicorn ai_models.production_deployment:app --host 0.0.0.0 --port 8000
```

### Systemd Service (Linux)

1. **Create service file**:
```bash
python ai_models/production_deployment.py deploy --type systemd
```

2. **Install and start**:
```bash
sudo cp hybrid-ml-service.service /etc/systemd/system/
sudo systemctl enable hybrid-ml-service
sudo systemctl start hybrid-ml-service
```

---

## 📡 API Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

#### `POST /predict`
Make predictions on base64 encoded images.

**Request**:
```json
{
    "image_data": "base64_encoded_image_data",
    "confidence_threshold": 0.5,
    "return_probabilities": true
}
```

**Response**:
```json
{
    "success": true,
    "predicted_class": "somdej_fatherguay",
    "confidence": 0.87,
    "probabilities": {
        "somdej_fatherguay": 0.87,
        "phra_san": 0.13
    },
    "processing_time_ms": 145.2,
    "model_version": "1.0.0"
}
```

#### `POST /predict/file`
Upload image file for prediction.

**Request**: Multipart form with image file

**Response**: Same as `/predict`

#### `GET /health`
Service health check.

**Response**:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "version": "1.0.0",
    "uptime_seconds": 3600.5,
    "memory_usage_mb": 512.3,
    "cpu_usage_percent": 15.2,
    "predictions_count": 1247,
    "average_response_time_ms": 142.8
}
```

#### `GET /metrics`
Detailed service metrics and statistics.

#### `GET /docs`
Interactive API documentation (Swagger UI).

### Python Client Example

```python
import requests
import base64

# Load and encode image
with open('amulet_image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Make prediction
response = requests.post(
    'http://localhost:8000/predict',
    json={
        'image_data': image_data,
        'confidence_threshold': 0.5,
        'return_probabilities': True
    }
)

result = response.json()
if result['success']:
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.3f}")
else:
    print(f"Error: {result['error_message']}")
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. PyTorch Installation Issues

**Problem**: PyTorch fails to install or import
```bash
ImportError: No module named 'torch'
```

**Solution**:
```bash
# Install CPU version (recommended for production)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or install compatible version
pip install torch==2.1.1+cpu torchvision==0.16.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

#### 2. OpenCV Issues

**Problem**: OpenCV fails to load or crashes
```bash
ImportError: libGL.so.1: cannot open shared object file
```

**Solution** (Linux):
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

**Solution** (Windows):
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

#### 3. Memory Issues

**Problem**: Out of memory during training
```bash
MemoryError: Unable to allocate array
```

**Solutions**:
- Use `--quick-mode` for training
- Reduce batch size in feature extraction
- Enable feature caching: `use_feature_cache=True`
- Reduce image resolution in preprocessing

#### 4. Model Loading Issues

**Problem**: Trained model fails to load
```bash
FileNotFoundError: Model file not found
```

**Solutions**:
- Check model directory path
- Ensure all model files are present:
  - `ensemble_model.joblib`
  - `scaler.joblib`
  - `label_encoder.joblib`
  - `training_config.json`

#### 5. API Service Issues

**Problem**: API service fails to start
```bash
Port 8000 is already in use
```

**Solutions**:
```bash
# Use different port
python ai_models/production_deployment.py serve --port 8001

# Kill existing process
lsof -ti:8000 | xargs kill -9  # Linux/Mac
# netstat -ano | findstr :8000  # Windows
```

### Performance Issues

#### Slow Prediction Speed

1. **Enable feature caching**:
```python
config = FeatureConfig(cache_features=True)
```

2. **Use CPU-only PyTorch**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

3. **Optimize image size**:
- Resize images to 224x224 or smaller
- Use JPEG compression for storage

4. **Reduce feature complexity**:
```python
config = FeatureConfig(
    use_cnn_features=True,  # Keep main CNN features
    use_sift_features=False,  # Disable expensive features
    use_surf_features=False
)
```

#### High Memory Usage

1. **Batch processing**:
```python
# Process images in smaller batches
batch_size = 32  # Reduce if needed
```

2. **Clear cache periodically**:
```python
feature_extractor.clear_cache()
```

3. **Use dimensionality reduction**:
```python
config = TrainingConfig(use_dimensionality_reduction=True)
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
export LOG_LEVEL=DEBUG  # Linux/Mac
set LOG_LEVEL=DEBUG     # Windows

python ai_models/hybrid_trainer.py --data-dir dataset --verbose
```

---

## ⚡ Performance Optimization

### Training Optimization

#### 1. Quick Mode
For faster development iterations:
```bash
python ai_models/hybrid_trainer.py --data-dir dataset --quick-mode
```

Quick mode reduces:
- Cross-validation folds (3 instead of 5)
- Hyperparameter search iterations
- Feature extraction complexity

#### 2. Feature Selection
Optimize feature extraction for your use case:

```python
# Balanced performance/speed
config = FeatureConfig(
    use_cnn_features=True,      # Good accuracy
    use_hog_features=True,      # Fast classical features
    use_lbp_features=True,      # Good for textures
    use_sift_features=False,    # Slow but accurate
    use_surf_features=False,    # Slow
    use_orb_features=False      # Fast but less accurate
)
```

#### 3. Caching Strategy
Enable intelligent caching:

```python
config = FeatureConfig(
    cache_features=True,
    cache_dir="feature_cache",
    max_cache_size_gb=2.0
)
```

### Inference Optimization

#### 1. Model Optimization
After training, optimize model for inference:

```python
# In hybrid_trainer.py, ensemble models are automatically optimized
# Dimensionality reduction and feature selection are applied
```

#### 2. Batch Processing
Process multiple images together:

```python
# Instead of one-by-one
for image in images:
    prediction = model.predict([image])

# Use batch processing
predictions = model.predict(images)
```

#### 3. Feature Caching
Cache extracted features for repeated predictions:

```python
extractor = HybridFeatureExtractor(config)
extractor.enable_cache(max_size=1000)  # Cache last 1000 features
```

### Production Optimization

#### 1. Docker Optimization

**Multi-stage build** (create optimized Dockerfile):
```dockerfile
# Build stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production stage
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "-m", "uvicorn", "ai_models.production_deployment:app", "--host", "0.0.0.0"]
```

#### 2. Load Balancing

**Nginx configuration** for multiple instances:
```nginx
upstream backend {
    server app1:8000;
    server app2:8000;
    server app3:8000;
}

server {
    location / {
        proxy_pass http://backend;
    }
}
```

#### 3. Monitoring

Enable monitoring for performance tracking:
```python
# In production_deployment.py
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter('requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('request_duration_seconds', 'Request latency')
```

### Memory Optimization

#### 1. Garbage Collection
```python
import gc

# After batch processing
del large_arrays
gc.collect()
```

#### 2. Memory Mapping
For large datasets:
```python
import numpy as np

# Use memory-mapped arrays
features = np.memmap('features.dat', dtype='float32', mode='r')
```

#### 3. Streaming Processing
For very large datasets:
```python
def process_dataset_streaming(data_dir, batch_size=32):
    for batch in get_batch_generator(data_dir, batch_size):
        yield process_batch(batch)
```

---

## 👨‍💻 Development Guide

### Project Structure

```
ai_models/
├── __init__.py                    # Package initialization
├── dataset_inspector.py          # Dataset analysis tool
├── augmentation_pipeline.py      # Data augmentation system
├── feature_extractors.py         # Hybrid feature extraction
├── hybrid_trainer.py            # Main training pipeline
├── evaluation_suite.py          # Model evaluation system
├── production_deployment.py     # API deployment service
└── docs/
    └── README_COMPLETE.md        # This documentation
```

### Adding New Features

#### 1. New Feature Extractor

Create new extractor by extending base class:

```python
from feature_extractors import BaseFeatureExtractor

class CustomFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your extractor
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        # Implement feature extraction
        features = your_feature_extraction_logic(image)
        return features
    
    def get_feature_names(self) -> List[str]:
        return ["custom_feature_1", "custom_feature_2", ...]
```

Register in `HybridFeatureExtractor`:
```python
# In feature_extractors.py
if self.config.use_custom_features:
    custom_extractor = CustomFeatureExtractor(self.config)
    custom_features = custom_extractor.extract_features(image)
    all_features.append(custom_features)
```

#### 2. New ML Algorithm

Add to ensemble in `hybrid_trainer.py`:

```python
from sklearn.ensemble import YourNewClassifier

# In create_ensemble_model method
models = [
    ('rf', RandomForestClassifier(**rf_params)),
    ('svm', SVC(**svm_params)),
    ('custom', YourNewClassifier(**custom_params))  # Add here
]
```

#### 3. New Evaluation Metric

Extend `PerformanceAnalyzer`:

```python
def calculate_custom_metric(self, y_true, y_pred, y_pred_proba):
    # Calculate your metric
    custom_score = your_metric_calculation(y_true, y_pred)
    return custom_score
```

### Testing

#### Unit Tests
```bash
# Install testing dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov=ai_models
```

#### Integration Tests
```bash
# Test complete pipeline
python test_pipeline.py

# Test API endpoints
python test_api.py
```

### Code Style

Follow PEP 8 and use type hints:

```python
from typing import List, Dict, Optional, Union
import numpy as np

def process_images(images: List[np.ndarray], 
                  config: Dict[str, Any]) -> List[str]:
    """Process images and return predictions.
    
    Args:
        images: List of input images
        config: Configuration dictionary
        
    Returns:
        List of predicted class names
    """
    # Implementation
    return predictions
```

### Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Add tests** for new functionality
4. **Update documentation**
5. **Submit pull request**

### Version Management

Update version numbers in:
- `production_deployment.py` (API version)
- `setup.py` or `pyproject.toml` (package version)
- `README.md` (documentation version)

```python
# Version format: MAJOR.MINOR.PATCH
VERSION = "1.0.0"
```

---

## 📞 Support & Resources

### Documentation
- **API Docs**: `http://localhost:8000/docs` (when service is running)
- **Code Documentation**: Inline docstrings and type hints
- **Examples**: Check `examples/` directory

### Community
- **Issues**: Use GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Wiki**: Additional documentation and tutorials

### Performance Benchmarks

Typical performance on standard hardware:

| Component | Time (seconds) | Notes |
|-----------|---------------|--------|
| Dataset Analysis | 30-120 | Depends on dataset size |
| Feature Extraction | 60-300 | CNN features take longer |
| Model Training | 120-600 | Ensemble with CV |
| Model Evaluation | 30-120 | With visualizations |
| Inference (single) | 0.1-0.5 | Per image prediction |
| Inference (batch) | 0.05-0.2 | Per image in batch |

### Hardware Recommendations

**Minimum**:
- CPU: 4 cores
- RAM: 4GB
- Storage: 2GB free space

**Recommended**:
- CPU: 8+ cores
- RAM: 8GB+
- Storage: 10GB+ SSD
- GPU: Optional (CUDA compatible)

**Production**:
- CPU: 16+ cores
- RAM: 16GB+
- Storage: 50GB+ SSD
- Network: High bandwidth for API serving

---

## 📄 License & Citation

### License
This project is released under the MIT License. See LICENSE file for details.

### Citation
If you use this system in research or production, please cite:

```bibtex
@software{hybrid_ml_system,
  title={Hybrid Machine Learning System for Image Classification},
  author={AI Assistant},
  year={2024},
  url={https://github.com/your-repo/hybrid-ml-system}
}
```

---

## 🎉 Conclusion

This complete hybrid ML system provides:

✅ **End-to-End Pipeline**: From data analysis to production deployment  
✅ **State-of-the-Art Methods**: CNN + Classical ML ensemble  
✅ **Production Ready**: Docker, API, monitoring, health checks  
✅ **Highly Configurable**: Modular design for easy customization  
✅ **Comprehensive Documentation**: Detailed guides and examples  
✅ **Performance Optimized**: Caching, batch processing, efficient algorithms  

**Quick Start Command**:
```bash
# Complete pipeline in one command
python ai_models/hybrid_trainer.py --data-dir dataset --output-dir saved_models && \
python ai_models/production_deployment.py serve
```

**🌐 Your ML service will be available at**: `http://localhost:8000`

**📚 API documentation at**: `http://localhost:8000/docs`

---

*Happy Machine Learning! 🚀*