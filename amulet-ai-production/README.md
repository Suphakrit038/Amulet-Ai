# 🔮 Amulet-AI Production

**ระบบจำแนกพระเครื่องอัจฉริยะ - Production Ready**

Thai Amulet Classification System - Optimized for Production Deployment

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start API Server
```bash
cd api
python -m uvicorn main_api:app --host 0.0.0.0 --port 8000
```

### 3. Start Frontend
```bash
cd frontend
streamlit run main_app.py --server.port 8501
```

### 4. Access Application
- **Web App**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📁 Project Structure

```
amulet-ai-production/
├── ai_models/              # AI Models & Classification Logic
│   ├── __init__.py
│   ├── enhanced_production_system.py
│   ├── updated_classifier.py
│   ├── compatibility_loader.py
│   ├── labels.json
│   └── twobranch/         # Two-Branch CNN System
├── api/                   # FastAPI Backend
│   ├── main_api.py        # Main API
│   ├── main_api_fast.py   # Fast API
│   └── main_api_secure.py # Secure API
├── frontend/              # Streamlit Frontend
│   ├── main_app.py        # Main Web App
│   ├── style.css          # Custom Styles
│   ├── components/        # UI Components
│   └── utils/             # Utilities
├── core/                  # Core Utilities
│   ├── config.py          # Configuration
│   ├── error_handling.py  # Error Management
│   ├── performance.py     # Performance Tools
│   └── security.py        # Security Features
├── organized_dataset/     # Training Dataset
│   ├── splits/            # Train/Val/Test Splits
│   ├── metadata/          # Dataset Metadata
│   └── processed/         # Processed Images
├── config/                # Configuration Files
├── deployment/            # Docker & Deployment
└── requirements.txt       # Python Dependencies
```

## 🎯 Features

- **AI Classification**: Advanced machine learning models for amulet recognition
- **Two-Branch CNN**: Deep learning architecture for enhanced accuracy
- **Web Interface**: User-friendly Streamlit dashboard
- **REST API**: FastAPI backend with comprehensive documentation
- **Security**: Built-in authentication and rate limiting
- **Performance**: Optimized for production workloads
- **Docker Ready**: Container support for easy deployment

## 🔧 Configuration

### Environment Variables
Copy `config/config_template.env` to `.env` and configure:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Frontend Configuration
FRONTEND_PORT=8501

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8501

# Model Configuration
MODEL_PATH=ai_models/
DATASET_PATH=organized_dataset/
```

### Docker Deployment

**Development:**
```bash
docker-compose -f deployment/docker-compose.dev.yml up
```

**Production:**
```bash
docker-compose -f deployment/docker-compose.prod.yml up
```

## 📊 Dataset

The system includes a comprehensive dataset with:
- **6 Amulet Classes**: phra_sivali, portrait_back, prok_bodhi_9_leaves, etc.
- **Train/Validation/Test Splits**: Organized for machine learning
- **Metadata**: Detailed information about each image
- **Augmented Data**: Enhanced training samples

## 🧠 AI Models

### Available Models:
1. **Enhanced Production System**: scikit-learn based classifier
2. **Two-Branch CNN**: PyTorch deep learning model
3. **Updated Classifier**: Latest optimized model

### Model Performance:
- High accuracy on test dataset
- Real-time inference capability
- Production-ready optimization

## 🛡️ Security Features

- Authentication & Authorization
- Rate Limiting
- Input Validation
- Error Handling
- Secure Headers

## 📈 Performance

- Optimized inference speed
- Memory-efficient processing
- Caching mechanisms
- Async operations
- Load balancing ready

## 🔍 API Endpoints

### Classification
- `POST /predict` - Classify amulet images
- `POST /batch-predict` - Batch classification
- `GET /models` - Available models info

### System
- `GET /health` - Health check
- `GET /metrics` - System metrics
- `GET /status` - Service status

## 🎨 Frontend Features

- Drag & drop image upload
- Real-time predictions
- Interactive charts
- Model comparison
- Results export

## 🚀 Deployment Options

### Local Development
```bash
# Terminal 1: API
cd api && python -m uvicorn main_api:app --reload

# Terminal 2: Frontend  
cd frontend && streamlit run main_app.py
```

### Production Deployment
- Docker containers
- Kubernetes support
- Cloud deployment ready
- CI/CD pipeline compatible

## 📞 Support

For issues and questions:
- Check API documentation at `/docs`
- Review error logs in console
- Ensure all dependencies are installed
- Verify dataset paths are correct

## 📝 License

Copyright © 2025 Amulet-AI Team. All rights reserved.

---

**Built with ❤️ for Thai Cultural Heritage Preservation**