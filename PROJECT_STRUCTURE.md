# Amulet-AI Project Structure
```
Amulet-Ai/
├── 📁 backend/                 # Backend API Services
│   ├── 🐍 api.py              # Main FastAPI application
│   ├── 🐍 model_loader.py     # AI Model management
│   ├── 🐍 valuation.py        # Price estimation system
│   ├── 🐍 recommend.py        # Market recommendation
│   ├── 🐍 config.py           # Configuration settings
│   └── 🐍 __init__.py         # Package initialization
│
├── 📁 frontend/               # Frontend Applications
│   ├── 🐍 app_streamlit.py    # Streamlit web interface
│   └── 🐍 components/         # UI Components (planned)
│
├── 📁 ai_models/             # AI Model Components
│   ├── 🐍 similarity_search.py    # FAISS similarity search
│   ├── 🐍 price_estimator.py      # Scikit-learn price models
│   ├── 🐍 market_scraper.py       # Scrapy data collection
│   ├── 🐍 train_simple.py         # TensorFlow training
│   └── 📁 saved_models/           # Trained model storage
│
├── 📁 dataset/               # Training Data
│   ├── 📁 โพธิ์ฐานบัว/          # Buddha images - Bodhi lotus base
│   ├── 📁 สีวลี/               # Sivali amulet images
│   ├── 📁 หลวงพ่อกวยแหวกม่าน/    # LP Kuay curtain-parting amulet
│   └── 📁 ฐานสิงห์/            # Lion-base Buddha images
│
├── 📁 utils/                 # Utility Functions
│   ├── 🐍 image_processor.py     # Image preprocessing
│   ├── 🐍 data_validator.py      # Data validation
│   └── 🐍 logger.py              # Logging configuration
│
├── 📁 tests/                 # Testing Suite
│   ├── 🐍 test_api.py            # API endpoint tests
│   ├── 🐍 test_models.py         # AI model tests
│   └── 🐍 test_integration.py    # Integration tests
│
├── 📁 scripts/               # Automation Scripts
│   ├── 🦇 start_backend.bat      # Backend startup
│   ├── 🦇 start_frontend.bat     # Frontend startup
│   ├── 🦇 start_system.bat       # Full system startup
│   └── 🐍 setup.py               # Project setup
│
├── 📁 docs/                  # Documentation
│   ├── 📄 API.md                 # API documentation
│   ├── 📄 DEPLOYMENT.md          # Deployment guide
│   └── 📄 DEVELOPMENT.md         # Development guide
│
├── 📄 requirements.txt        # Python dependencies
├── 📄 labels.json            # Class labels mapping
├── 📄 README.md              # Project overview
├── 📄 USAGE.md               # Usage instructions
└── 📄 .gitignore             # Git ignore rules
```

## 🎯 Current Status
- ✅ **Backend API**: Running on http://localhost:8000
- ✅ **Frontend UI**: Running on http://localhost:8501  
- 🔄 **AI Models**: Advanced simulation mode
- 📊 **Dataset**: Raw images available, needs cleaning
