# 📚 Amulet-AI API Documentation

## Overview
Advanced AI-powered Thai Buddhist Amulet Recognition and Valuation System with real image analysis and price estimation capabilities.

## 🚀 Quick Start

### 1. Start the System
```bash
# Option 1: Full system
python scripts/start_system.bat

# Option 2: Individual components
python scripts/start_backend.bat    # API Server
python scripts/start_frontend.bat   # Streamlit UI

# Option 3: Direct command
python backend/optimized_api.py     # Optimized API
```

### 2. Access the Interface
- **Streamlit UI**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## 🔗 API Endpoints

### Main Endpoints

#### `POST /predict`
Upload images for amulet recognition and valuation.

**Parameters:**
- `front` (file): Front image of amulet (required)
- `back` (file): Back image of amulet (optional)

**Response:**
```json
{
  "top1": {
    "class_id": 0,
    "class_name": "หลวงพ่อกวยแหวกม่าน",
    "confidence": 0.92
  },
  "topk": [...],
  "valuation": {
    "p05": 15000,
    "p50": 45000,
    "p95": 120000,
    "confidence": "high"
  },
  "recommendations": [...],
  "ai_mode": "advanced_simulation",
  "processing_time": 0.234
}
```

#### `GET /system-status`
Get comprehensive system status and performance metrics.

#### `GET /health`
Simple health check endpoint.

### Utility Endpoints

#### `GET /supported-formats`
Get list of supported image formats.

#### `GET /stats`
Get detailed performance statistics.

#### `POST /clear-cache`
Clear model prediction cache.

## 📊 Supported Classes

1. **หลวงพ่อกวยแหวกม่าน** - LP Kuay curtain-parting amulet
2. **โพธิ์ฐานบัว** - Buddha with lotus base
3. **ฐานสิงห์** - Lion-base Buddha
4. **สีวลี** - Sivali amulet

## 🖼️ Image Requirements

**Supported Formats:**
- JPEG, PNG (recommended)
- HEIC, HEIF, WebP, BMP, TIFF

**File Size:** Maximum 10MB per image

**Quality Guidelines:**
- Clear, well-lit images
- Front view of amulet
- Minimal background distractions
- High resolution preferred

## 🤖 AI Modes

### Advanced Simulation Mode
- **Status**: Production-ready
- **Features**: Real image analysis, color detection, texture analysis
- **Accuracy**: High simulation of trained model
- **Performance**: Fast (~0.2-0.5 seconds)

### Simple Mock Mode
- **Status**: Testing only
- **Features**: Random predictions
- **Use**: Development and testing

## 💰 Valuation System

The system provides price estimates in Thai Baht (THB) with three percentiles:

- **P05**: 5th percentile (low estimate)
- **P50**: Median price (most likely)
- **P95**: 95th percentile (high estimate)

**Factors Considered:**
- Amulet class and rarity
- Prediction confidence
- Market conditions
- Historical pricing data

## 🏪 Market Recommendations

Intelligent market recommendations based on:
- Amulet type and specialization
- Price range compatibility
- Distance and accessibility
- Market ratings and reputation

## 🔧 Configuration

### Environment Variables
```bash
AMULET_ENV=development|testing|production
TF_ENABLE_ONEDNN_OPTS=0  # Disable TensorFlow warnings
```

### Config Files
- `backend/config.py`: Main configuration
- `labels.json`: Class labels mapping
- `requirements.txt`: Python dependencies

## 🛠️ Development

### Project Structure
```
Amulet-Ai/
├── backend/              # Backend API services
├── frontend/             # Frontend applications
├── ai_models/           # AI model components
├── dataset/             # Training data
├── utils/              # Utility functions
├── tests/              # Testing suite
├── scripts/            # Automation scripts
└── docs/               # Documentation
```

### Adding New Features

1. **New Amulet Class:**
   - Update `labels.json`
   - Add training images to `dataset/`
   - Update valuation config in `backend/config.py`

2. **New AI Model:**
   - Add to `ai_models/` directory
   - Update `backend/optimized_model_loader.py`
   - Test with existing API endpoints

## 📈 Performance Monitoring

### Metrics Available
- Total requests processed
- Success/failure rates
- Average response times
- Cache hit rates
- Uptime statistics

### Monitoring Endpoints
- `/stats` - Performance metrics
- `/system-status` - System health
- `/health` - Basic health check

## 🔍 Troubleshooting

### Common Issues

**Server Won't Start:**
```bash
# Check Python environment
python --version

# Check dependencies
pip install -r requirements.txt

# Use optimized API
python backend/optimized_api.py
```

**Image Upload Errors:**
- Verify file format is supported
- Check file size (max 10MB)
- Ensure proper image encoding

**Low Prediction Confidence:**
- Use clear, well-lit images
- Ensure amulet is properly framed
- Try different angles if available

### Performance Tips

1. **Faster Predictions:**
   - Use JPEG format for images
   - Enable caching (default)
   - Use optimized API endpoint

2. **Better Accuracy:**
   - Upload high-quality images
   - Include both front and back views
   - Ensure good lighting conditions

## 🚨 Error Handling

The API provides detailed error messages:
- `400`: Bad Request (invalid file format, etc.)
- `413`: File Too Large (>10MB)
- `500`: Internal Server Error

## 🔐 Security Considerations

- File size limitations prevent DoS attacks
- Supported format validation prevents malicious uploads
- No persistent file storage (memory processing only)
- Rate limiting through server configuration

## 🎯 Future Enhancements

### Planned Features
- [ ] Real TensorFlow model integration
- [ ] Batch image processing
- [ ] Historical price tracking
- [ ] User authentication
- [ ] Advanced market analytics
- [ ] Mobile app support

### AI Improvements
- [ ] FAISS similarity search
- [ ] Scikit-learn price models
- [ ] Scrapy market data collection
- [ ] Model retraining pipeline

---

**Version**: 1.0.0  
**Last Updated**: August 2025  
**Maintained By**: Amulet-AI Development Team
