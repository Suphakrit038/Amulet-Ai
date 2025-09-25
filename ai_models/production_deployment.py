#!/usr/bin/env python3
"""
🔧 Production Model Deployment & Integration System
Complete system for deploying hybrid ML models to production environment

This system provides:
- Model loading and serving infrastructure
- API endpoint creation with FastAPI
- Input validation and preprocessing
- Output formatting and confidence scoring
- Error handling and logging
- Model versioning and health checks
- Performance monitoring and metrics

Author: AI Assistant
Compatible: Python 3.13+
Dependencies: fastapi, uvicorn, pydantic, pillow, requests
"""

import os
import sys
import json
import asyncio
import uvicorn
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging
import warnings
import traceback
from contextlib import asynccontextmanager
warnings.filterwarnings('ignore')

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import httpx

# Image processing
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# Our models
from hybrid_trainer import HybridTrainer, TrainingConfig
from feature_extractors import HybridFeatureExtractor, FeatureConfig
from evaluation_suite import ModelLoader, EvaluationConfig

# For monitoring
import time
import psutil
import threading
from collections import defaultdict, deque


class PredictionRequest(BaseModel):
    """Pydantic model for prediction requests"""
    
    image_data: str = Field(..., description="Base64 encoded image data")
    confidence_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")
    return_probabilities: bool = Field(False, description="Whether to return class probabilities")
    image_format: str = Field("auto", description="Image format (auto, jpg, png)")
    
    @validator('image_data')
    def validate_image_data(cls, v):
        if not v or len(v) < 10:
            raise ValueError("Image data cannot be empty")
        return v


class PredictionResponse(BaseModel):
    """Pydantic model for prediction responses"""
    
    success: bool = Field(..., description="Whether prediction was successful")
    predicted_class: Optional[str] = Field(None, description="Predicted class name")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    model_version: Optional[str] = Field(None, description="Model version used")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class HealthResponse(BaseModel):
    """Pydantic model for health check responses"""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    predictions_count: int = Field(..., description="Total predictions made")
    average_response_time_ms: float = Field(..., description="Average response time")
    last_prediction_time: Optional[datetime] = Field(None, description="Last prediction timestamp")


class ModelManager:
    """Manages model loading, caching, and serving"""
    
    def __init__(self, model_dir: str, config_path: Optional[str] = None):
        self.model_dir = Path(model_dir)
        self.config_path = config_path
        self.logger = self._setup_logging()
        
        # Model components
        self.model_loader = None
        self.is_loaded = False
        self.model_version = "1.0.0"
        self.load_timestamp = None
        
        # Performance tracking
        self.prediction_count = 0
        self.response_times = deque(maxlen=1000)  # Last 1000 response times
        self.error_count = 0
        
        # Configuration
        self.max_image_size = (1024, 1024)  # Max image dimensions
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('ModelManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def load_model(self) -> bool:
        """Load model asynchronously"""
        try:
            self.logger.info(f"🔄 Loading model from {self.model_dir}")
            
            # Load model in thread to avoid blocking
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, self._load_model_sync)
            
            if success:
                self.is_loaded = True
                self.load_timestamp = datetime.now()
                self.logger.info("✅ Model loaded successfully")
            else:
                self.logger.error("❌ Failed to load model")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Model loading error: {e}")
            return False
    
    def _load_model_sync(self) -> bool:
        """Synchronous model loading"""
        try:
            self.model_loader = ModelLoader(str(self.model_dir))
            return self.model_loader.load_model()
        except Exception as e:
            self.logger.error(f"Sync model loading failed: {e}")
            return False
    
    def preprocess_image(self, image_data: str, image_format: str = "auto") -> np.ndarray:
        """Preprocess image data for prediction"""
        try:
            # Decode base64 image
            if ',' in image_data:
                image_data = image_data.split(',')[1]  # Remove data:image/... prefix
            
            image_bytes = base64.b64decode(image_data)
            
            # Convert to PIL Image
            pil_image = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if pil_image.mode not in ['RGB', 'L']:
                pil_image = pil_image.convert('RGB')
            
            # Resize if too large
            if pil_image.size[0] > self.max_image_size[0] or pil_image.size[1] > self.max_image_size[1]:
                pil_image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            
            # Ensure 3 channels (RGB)
            if len(image_array.shape) == 2:  # Grayscale
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:  # RGBA
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            
            return image_array
            
        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make prediction on image"""
        start_time = time.time()
        
        try:
            if not self.is_loaded:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            # Preprocess image
            image = self.preprocess_image(request.image_data, request.image_format)
            
            # Make prediction in thread to avoid blocking
            loop = asyncio.get_event_loop()
            predicted_labels, probabilities = await loop.run_in_executor(
                None, self.model_loader.predict, [image]
            )
            
            # Process results
            predicted_class = predicted_labels[0]
            class_probabilities = probabilities[0]
            confidence = float(np.max(class_probabilities))
            
            # Check confidence threshold
            if confidence < request.confidence_threshold:
                predicted_class = "uncertain"
                confidence = 0.0
            
            # Prepare response
            response_data = {
                'success': True,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'processing_time_ms': (time.time() - start_time) * 1000,
                'model_version': self.model_version
            }
            
            # Add probabilities if requested
            if request.return_probabilities and self.model_loader.label_encoder:
                class_names = self.model_loader.label_encoder.classes_
                prob_dict = {
                    class_name: float(prob) 
                    for class_name, prob in zip(class_names, class_probabilities)
                }
                response_data['probabilities'] = prob_dict
            
            # Add metadata
            response_data['metadata'] = {
                'image_shape': image.shape,
                'prediction_id': f"pred_{self.prediction_count + 1}_{int(time.time())}"
            }
            
            # Update metrics
            self.prediction_count += 1
            self.response_times.append((time.time() - start_time) * 1000)
            
            return PredictionResponse(**response_data)
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Prediction error: {str(e)}")
            
            return PredictionResponse(
                success=False,
                error_message=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
                model_version=self.model_version
            )
    
    def get_health_status(self, start_time: datetime) -> HealthResponse:
        """Get current health status"""
        uptime = (datetime.now() - start_time).total_seconds()
        
        # System metrics
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        cpu_usage = psutil.cpu_percent(interval=None)
        
        # Average response time
        avg_response_time = np.mean(self.response_times) if self.response_times else 0.0
        
        # Last prediction time
        last_prediction = self.load_timestamp if self.prediction_count > 0 else None
        
        return HealthResponse(
            status="healthy" if self.is_loaded else "unhealthy",
            model_loaded=self.is_loaded,
            version=self.model_version,
            uptime_seconds=uptime,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            predictions_count=self.prediction_count,
            average_response_time_ms=avg_response_time,
            last_prediction_time=last_prediction
        )


# Global model manager
model_manager = None
service_start_time = datetime.now()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global model_manager
    
    # Startup
    print("🚀 Starting Hybrid ML Model Service...")
    
    # Load model
    model_dir = os.getenv("MODEL_DIR", "ai_models/saved_models/hybrid_amulet_classifier")
    model_manager = ModelManager(model_dir)
    
    success = await model_manager.load_model()
    if not success:
        print("❌ Failed to load model during startup")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Hybrid ML Model Service...")


# Create FastAPI app
app = FastAPI(
    title="🔮 Hybrid ML Model Service",
    description="Production-ready hybrid machine learning model serving API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with service information"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🔮 Hybrid ML Model Service</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .header { text-align: center; color: #333; }
            .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
            .endpoint { background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 4px; }
            .method { font-weight: bold; color: #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔮 Hybrid ML Model Service</h1>
                <p>Production-ready amulet classification API</p>
            </div>
            
            <div class="section">
                <h2>📡 Available Endpoints</h2>
                
                <div class="endpoint">
                    <span class="method">POST</span> <code>/predict</code><br>
                    Make predictions on images
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <code>/health</code><br>
                    Service health check
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <code>/docs</code><br>
                    Interactive API documentation
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <code>/metrics</code><br>
                    Service metrics and statistics
                </div>
            </div>
            
            <div class="section">
                <h2>🔧 Usage Example</h2>
                <pre><code>
import requests
import base64

# Encode image to base64
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
print(f"Predicted class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.3f}")
                </code></pre>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction on uploaded image"""
    global model_manager
    
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    return await model_manager.predict(request)


@app.post("/predict/file")
async def predict_file(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.3,
    return_probabilities: bool = False
):
    """Make prediction on uploaded file"""
    try:
        # Read file contents
        contents = await file.read()
        
        # Encode to base64
        image_data = base64.b64encode(contents).decode('utf-8')
        
        # Create request
        request = PredictionRequest(
            image_data=image_data,
            confidence_threshold=confidence_threshold,
            return_probabilities=return_probabilities
        )
        
        return await predict(request)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health check"""
    global model_manager, service_start_time
    
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    return model_manager.get_health_status(service_start_time)


@app.get("/metrics")
async def get_metrics():
    """Get detailed service metrics"""
    global model_manager, service_start_time
    
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    health = model_manager.get_health_status(service_start_time)
    
    # Additional metrics
    metrics = {
        'service_info': {
            'version': '1.0.0',
            'model_version': model_manager.model_version,
            'start_time': service_start_time.isoformat(),
            'uptime_seconds': health.uptime_seconds
        },
        'performance': {
            'total_predictions': model_manager.prediction_count,
            'total_errors': model_manager.error_count,
            'error_rate': model_manager.error_count / max(model_manager.prediction_count, 1),
            'average_response_time_ms': health.average_response_time_ms,
            'recent_response_times': list(model_manager.response_times)[-10:]  # Last 10
        },
        'system': {
            'memory_usage_mb': health.memory_usage_mb,
            'cpu_usage_percent': health.cpu_usage_percent,
            'disk_usage': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else None
        },
        'model_info': {
            'model_loaded': health.model_loaded,
            'model_path': str(model_manager.model_dir),
            'load_timestamp': model_manager.load_timestamp.isoformat() if model_manager.load_timestamp else None
        }
    }
    
    return metrics


@app.post("/reload")
async def reload_model():
    """Reload the model (admin endpoint)"""
    global model_manager
    
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    success = await model_manager.load_model()
    
    return {
        'success': success,
        'message': 'Model reloaded successfully' if success else 'Model reload failed',
        'timestamp': datetime.now().isoformat()
    }


class ProductionDeployment:
    """Handles production deployment configurations"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('ProductionDeployment')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_dockerfile(self, output_dir: str = ".") -> str:
        """Create Dockerfile for containerized deployment"""
        dockerfile_content = '''# Hybrid ML Model Service Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs evaluation_reports evaluation_plots

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "ai_models.production_deployment:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        
        dockerfile_path = Path(output_dir) / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        self.logger.info(f"📦 Dockerfile created: {dockerfile_path}")
        return str(dockerfile_path)
    
    def create_docker_compose(self, output_dir: str = ".") -> str:
        """Create docker-compose.yml for easy deployment"""
        compose_content = '''version: '3.8'

services:
  hybrid-ml-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_DIR=/app/ai_models/saved_models/hybrid_amulet_classifier
      - LOG_LEVEL=INFO
    volumes:
      - ./ai_models/saved_models:/app/ai_models/saved_models:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Add nginx reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - hybrid-ml-service
    restart: unless-stopped

networks:
  default:
    name: hybrid-ml-network
'''
        
        compose_path = Path(output_dir) / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        
        self.logger.info(f"🐳 Docker Compose file created: {compose_path}")
        return str(compose_path)
    
    def create_nginx_config(self, output_dir: str = ".") -> str:
        """Create nginx configuration for load balancing"""
        nginx_content = '''events {
    worker_connections 1024;
}

http {
    upstream hybrid_ml_backend {
        server hybrid-ml-service:8000;
        # Add more servers for load balancing
        # server hybrid-ml-service-2:8000;
    }

    server {
        listen 80;
        
        location / {
            proxy_pass http://hybrid_ml_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            
            # Increase timeout for model inference
            proxy_read_timeout 60s;
            proxy_connect_timeout 10s;
            
            # Handle large image uploads
            client_max_body_size 10M;
        }
        
        # Health check endpoint
        location /health {
            proxy_pass http://hybrid_ml_backend/health;
            access_log off;
        }
    }
}
'''
        
        nginx_path = Path(output_dir) / "nginx.conf"
        with open(nginx_path, 'w') as f:
            f.write(nginx_content)
        
        self.logger.info(f"🌐 Nginx config created: {nginx_path}")
        return str(nginx_path)
    
    def create_systemd_service(self, output_dir: str = ".") -> str:
        """Create systemd service file for Linux deployment"""
        service_content = '''[Unit]
Description=Hybrid ML Model Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/hybrid-ml-service
ExecStart=/opt/hybrid-ml-service/venv/bin/python -m uvicorn ai_models.production_deployment:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

# Environment variables
Environment=MODEL_DIR=/opt/hybrid-ml-service/ai_models/saved_models/hybrid_amulet_classifier
Environment=LOG_LEVEL=INFO

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/hybrid-ml-service/logs

[Install]
WantedBy=multi-user.target
'''
        
        service_path = Path(output_dir) / "hybrid-ml-service.service"
        with open(service_path, 'w') as f:
            f.write(service_content)
        
        self.logger.info(f"🔧 Systemd service created: {service_path}")
        return str(service_path)
    
    def create_deployment_script(self, output_dir: str = ".") -> str:
        """Create deployment script"""
        script_content = '''#!/bin/bash
# Hybrid ML Model Service Deployment Script

set -e

echo "🚀 Deploying Hybrid ML Model Service..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Build and start services
echo "📦 Building Docker image..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

echo "⏳ Waiting for service to be ready..."
sleep 10

# Health check
echo "🔍 Checking service health..."
if curl -f http://localhost:8000/health; then
    echo "✅ Service is healthy and running!"
    echo "🌐 Service available at: http://localhost:8000"
    echo "📚 API docs available at: http://localhost:8000/docs"
else
    echo "❌ Service health check failed"
    echo "📋 Service logs:"
    docker-compose logs hybrid-ml-service
    exit 1
fi

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "Useful commands:"
echo "  View logs: docker-compose logs -f"
echo "  Stop service: docker-compose down"
echo "  Restart service: docker-compose restart"
echo "  Update service: docker-compose pull && docker-compose up -d"
'''
        
        script_path = Path(output_dir) / "deploy.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        self.logger.info(f"📋 Deployment script created: {script_path}")
        return str(script_path)
    
    def create_production_requirements(self, output_dir: str = ".") -> str:
        """Create production requirements.txt"""
        requirements = '''# Production Requirements for Hybrid ML Model Service
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
httpx==0.25.2

# ML and Data Processing
scikit-learn==1.3.2
numpy==1.24.3
pandas==2.1.4
joblib==1.3.2

# Image Processing
opencv-python==4.8.1.78
Pillow==10.1.0
scikit-image==0.22.0

# PyTorch (CPU only for production stability)
torch==2.1.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
torchvision==0.16.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Monitoring and System
psutil==5.9.6
prometheus-client==0.19.0

# Optional: Advanced visualizations
plotly==5.17.0

# Development and Testing (optional)
pytest==7.4.3
requests==2.31.0
'''
        
        req_path = Path(output_dir) / "requirements_production.txt"
        with open(req_path, 'w') as f:
            f.write(requirements)
        
        self.logger.info(f"📋 Production requirements created: {req_path}")
        return str(req_path)


def run_development_server():
    """Run development server"""
    print("🚀 Starting Hybrid ML Model Service (Development Mode)")
    print("🌐 Service will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "ai_models.production_deployment:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🔮 Hybrid ML Model Production Deployment"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start the API server')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    serve_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Create deployment files')
    deploy_parser.add_argument('--output-dir', default='.', help='Output directory')
    deploy_parser.add_argument('--type', choices=['docker', 'systemd', 'all'], 
                              default='all', help='Deployment type')
    
    args = parser.parse_args()
    
    if args.command == 'serve':
        uvicorn.run(
            "ai_models.production_deployment:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
    
    elif args.command == 'deploy':
        deployment = ProductionDeployment()
        
        if args.type in ['docker', 'all']:
            deployment.create_dockerfile(args.output_dir)
            deployment.create_docker_compose(args.output_dir)
            deployment.create_nginx_config(args.output_dir)
            deployment.create_deployment_script(args.output_dir)
            deployment.create_production_requirements(args.output_dir)
        
        if args.type in ['systemd', 'all']:
            deployment.create_systemd_service(args.output_dir)
        
        print(f"✅ Deployment files created in: {args.output_dir}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()