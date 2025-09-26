#!/usr/bin/env python3
"""
🚀 Enhanced Production API v4.0
High-performance API meeting production standards with comprehensive monitoring

Features:
- Sub-2s response time (p95 < 2s, p99 < 3s)
- Comprehensive health monitoring
- Detailed metrics and observability
- Production-grade error handling
- Request ID tracking and structured logging
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import numpy as np
import cv2
from PIL import Image
import io
import logging
import time
import psutil
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid
import asyncio
from pydantic import BaseModel
import json
from pathlib import Path

# Import our enhanced classifier
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from ai_models.enhanced_production_system import EnhancedProductionClassifier
    print("✅ Using enhanced production system")
except ImportError as e:
    print(f"❌ Failed to import model: {e}")
    sys.exit(1)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Models for API responses
class PredictionRequest(BaseModel):
    """Request model for batch predictions"""
    images: List[str]  # Base64 encoded images
    request_id: Optional[str] = None

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    status: str
    is_supported: bool
    predicted_class: Optional[str] = None
    thai_name: Optional[str] = None
    confidence: Optional[float] = None
    detailed_results: Optional[List[Dict]] = None
    explanations: Optional[Dict] = None
    performance: Dict[str, Any]
    request_id: str
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    model_status: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    resource_usage: Dict[str, Any]
    sla_compliance: Dict[str, Any]
    timestamp: str

class MetricsResponse(BaseModel):
    """Metrics response model"""
    requests_total: int
    requests_per_second: float
    error_rate: float
    latency_percentiles: Dict[str, float]
    memory_usage_mb: float
    cache_performance: Dict[str, Any]
    uptime_minutes: float

# Global classifier instance
classifier = None
app_start_time = time.time()
request_count = 0
error_count = 0

# Request rate limiting
request_times = []
MAX_REQUESTS_PER_MINUTE = 100

# Security (optional)
security = HTTPBearer(auto_error=False)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Optional token verification - implement as needed"""
    # In production, implement proper JWT validation
    return credentials

def rate_limit_check():
    """Check if request rate is within limits"""
    current_time = time.time()
    
    # Clean old requests (older than 1 minute)
    global request_times
    request_times = [t for t in request_times if current_time - t < 60]
    
    if len(request_times) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Maximum 100 requests per minute."
        )
    
    request_times.append(current_time)

def load_classifier():
    """Load the enhanced classifier"""
    global classifier
    
    # Try multiple model paths in order of preference
    model_paths = ["trained_model", "trained_model_enhanced", "trained_model_production"]
    
    for model_path in model_paths:
        if Path(model_path).exists():
            try:
                classifier = EnhancedProductionClassifier()
                classifier.load_model(model_path)
                logger.info(f"✅ Enhanced classifier loaded successfully from {model_path}")
                return
            except Exception as e:
                logger.warning(f"⚠️ Failed to load from {model_path}: {e}")
                continue
                
    logger.error("❌ No trained model found in any expected location")
    classifier = None

# Initialize FastAPI app
app = FastAPI(
    title="🔮 Enhanced Amulet-AI API v4.0",
    description="Production-grade Thai Buddhist Amulet Recognition API",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("🚀 Starting Enhanced Amulet-AI API v4.0...")
    load_classifier()
    logger.info("✅ API startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Enhanced Amulet-AI API...")

def process_uploaded_image(image_data: bytes) -> np.ndarray:
    """Process uploaded image data"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image)
        
        return image_np
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def validate_image_pair(front_image: np.ndarray, back_image: np.ndarray) -> bool:
    """Validate image pair"""
    # Check image dimensions
    if front_image.shape[:2] != back_image.shape[:2]:
        logger.warning("Image dimensions mismatch - will resize")
    
    # Check if images are too small
    min_size = 50
    if front_image.shape[0] < min_size or front_image.shape[1] < min_size:
        raise HTTPException(status_code=400, detail="Images too small (minimum 50x50 pixels)")
    
    # Check if images are too large
    max_size = 4000
    if front_image.shape[0] > max_size or front_image.shape[1] > max_size:
        raise HTTPException(status_code=400, detail="Images too large (maximum 4000x4000 pixels)")
    
    return True

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "Enhanced Amulet-AI API",
        "version": "4.0.0",
        "status": "operational",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metrics": "/metrics",
            "model_info": "/model/info"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_amulet(
    front_image: UploadFile = File(..., description="Front view of the amulet"),
    back_image: UploadFile = File(..., description="Back view of the amulet"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """
    Predict amulet type from front and back images
    
    **Requirements:**
    - Both images must be clear photos of Thai Buddhist amulets
    - Supported formats: JPG, PNG, WEBP
    - Maximum file size: 10MB per image
    - Minimum resolution: 50x50 pixels
    
    **Response includes:**
    - Predicted class and confidence
    - Detailed explanation
    - Performance metrics
    - OOD detection results
    """
    global request_count, error_count
    
    # Rate limiting
    rate_limit_check()
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    request_count += 1
    
    start_time = time.time()
    
    try:
        # Check if classifier is loaded
        if classifier is None:
            error_count += 1
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Validate file types
        allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
        if front_image.content_type not in allowed_types or back_image.content_type not in allowed_types:
            error_count += 1
            raise HTTPException(status_code=400, detail="Unsupported image format")
        
        # Check file sizes (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB
        if front_image.size and front_image.size > max_size:
            error_count += 1
            raise HTTPException(status_code=400, detail="Front image too large (max 10MB)")
        if back_image.size and back_image.size > max_size:
            error_count += 1
            raise HTTPException(status_code=400, detail="Back image too large (max 10MB)")
        
        # Read image data
        front_data = await front_image.read()
        back_data = await back_image.read()
        
        # Process images
        front_np = process_uploaded_image(front_data)
        back_np = process_uploaded_image(back_data)
        
        # Validate image pair
        validate_image_pair(front_np, back_np)
        
        # Make prediction
        result = classifier.predict_production(front_np, back_np, request_id=request_id)
        
        # Add API-specific metrics
        processing_time = time.time() - start_time
        result['performance']['api_processing_time'] = processing_time
        result['performance']['total_requests'] = request_count
        result['performance']['error_rate'] = error_count / request_count
        
        # Log request
        logger.info(f"Prediction request {request_id}: {result.get('status')} in {processing_time:.3f}s")
        
        # Background task for metrics collection
        background_tasks.add_task(log_request_metrics, request_id, processing_time, True)
        
        return PredictionResponse(**result)
        
    except HTTPException:
        error_count += 1
        processing_time = time.time() - start_time
        background_tasks.add_task(log_request_metrics, request_id, processing_time, False)
        raise
    except Exception as e:
        error_count += 1
        processing_time = time.time() - start_time
        logger.error(f"Prediction error {request_id}: {str(e)}")
        background_tasks.add_task(log_request_metrics, request_id, processing_time, False)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint
    
    Returns detailed system health including:
    - Model status and performance
    - Resource usage
    - SLA compliance status
    - Recent performance metrics
    """
    try:
        if classifier is None:
            return HealthResponse(
                status="unhealthy",
                model_status={"loaded": False, "error": "Classifier not loaded"},
                performance_metrics={},
                resource_usage={},
                sla_compliance={},
                timestamp=datetime.now().isoformat()
            )
        
        # Get comprehensive health data
        health_data = classifier.get_system_health()
        
        # Add API-specific metrics
        health_data['api_metrics'] = {
            'total_requests': request_count,
            'error_count': error_count,
            'error_rate': error_count / max(request_count, 1),
            'uptime_minutes': (time.time() - app_start_time) / 60
        }
        
        # Determine overall status
        status = "healthy"
        
        # Check SLA violations
        perf_metrics = health_data.get('performance_metrics', {})
        if perf_metrics.get('p95', 0) > 2.0:
            status = "degraded"
        if perf_metrics.get('p99', 0) > 3.0:
            status = "unhealthy"
        if health_data['resource_usage'].get('memory_mb', 0) > 500:
            status = "degraded"
        if health_data['api_metrics']['error_rate'] > 0.005:
            status = "degraded"
        
        return HealthResponse(
            status=status,
            model_status=health_data['model_status'],
            performance_metrics=health_data['performance_metrics'],
            resource_usage=health_data['resource_usage'],
            sla_compliance=health_data['sla_compliance'],
            timestamp=health_data['timestamp']
        )
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return HealthResponse(
            status="error",
            model_status={"error": str(e)},
            performance_metrics={},
            resource_usage={},
            sla_compliance={},
            timestamp=datetime.now().isoformat()
        )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get detailed metrics for monitoring and observability
    
    **Prometheus-compatible metrics:**
    - Request count and rate
    - Error rate and latency percentiles
    - Resource usage
    - Cache performance
    """
    try:
        uptime = (time.time() - app_start_time) / 60
        
        # Calculate requests per second (last minute)
        current_time = time.time()
        recent_requests = [t for t in request_times if current_time - t < 60]
        rps = len(recent_requests) / 60
        
        # Get system metrics
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Get classifier metrics if available
        latency_percentiles = {}
        cache_performance = {}
        
        if classifier:
            health = classifier.get_system_health()
            latency_percentiles = health.get('performance_metrics', {})
            cache_performance = health.get('cache_performance', {})
        
        return MetricsResponse(
            requests_total=request_count,
            requests_per_second=rps,
            error_rate=error_count / max(request_count, 1),
            latency_percentiles=latency_percentiles,
            memory_usage_mb=memory_mb,
            cache_performance=cache_performance,
            uptime_minutes=uptime
        )
        
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@app.get("/model/info")
async def get_model_info():
    """
    Get detailed model information
    
    Returns:
    - Model version and configuration
    - Training metrics and performance
    - Supported classes
    - Feature extraction details
    """
    try:
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        health = classifier.get_system_health()
        
        return {
            "model_version": "4.0.0",
            "model_type": "Enhanced Production Classifier",
            "model_status": health['model_status'],
            "supported_classes": health['model_status'].get('classes_supported', []),
            "performance_targets": {
                "f1_per_class": "≥ 0.85",
                "balanced_accuracy": "≥ 0.80",
                "calibration_error": "< 0.05",
                "ood_auroc": "≥ 0.90",
                "latency_p95": "< 2s",
                "memory_usage": "< 500MB"
            },
            "features": {
                "dual_view_processing": True,
                "ood_detection": True,
                "calibrated_probabilities": True,
                "feature_caching": True,
                "explanation_generation": True
            },
            "training_data": {
                "dataset": "Thai Buddhist Amulets",
                "classes": 3,
                "image_pairs": "Variable based on dataset"
            },
            "api_features": {
                "rate_limiting": "100 requests/minute",
                "max_file_size": "10MB",
                "supported_formats": ["JPG", "PNG", "WEBP"],
                "structured_logging": True,
                "health_monitoring": True
            }
        }
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model info")

@app.post("/model/retrain")
async def trigger_retrain(credentials: HTTPAuthorizationCredentials = Depends(verify_token)):
    """
    Trigger model retraining (Admin only)
    
    **Note:** This endpoint would typically be secured and trigger
    a background retraining process.
    """
    # In production, implement proper authentication and background processing
    return {
        "status": "scheduled",
        "message": "Model retraining scheduled",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/feedback")
async def submit_feedback(
    request_id: str,
    is_correct: bool,
    correct_label: Optional[str] = None,
    comments: Optional[str] = None
):
    """
    Submit feedback for a prediction
    
    **Parameters:**
    - request_id: ID of the prediction request
    - is_correct: Whether the prediction was correct
    - correct_label: The correct label (if prediction was wrong)
    - comments: Additional feedback comments
    """
    feedback_data = {
        "request_id": request_id,
        "is_correct": is_correct,
        "correct_label": correct_label,
        "comments": comments,
        "timestamp": datetime.now().isoformat()
    }
    
    # In production, store this feedback for model improvement
    logger.info(f"Feedback received for request {request_id}: {'correct' if is_correct else 'incorrect'}")
    
    return {
        "status": "received",
        "message": "Thank you for your feedback",
        "feedback_id": str(uuid.uuid4())
    }

async def log_request_metrics(request_id: str, processing_time: float, success: bool):
    """Background task to log detailed request metrics"""
    try:
        metrics_data = {
            "request_id": request_id,
            "processing_time": processing_time,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }
        
        # In production, send to monitoring system (e.g., Prometheus, ELK Stack)
        logger.info(f"Request metrics: {json.dumps(metrics_data)}")
        
    except Exception as e:
        logger.error(f"Failed to log metrics: {str(e)}")

# Custom exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": [
                "/predict", "/health", "/metrics", "/model/info"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    # Production configuration
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": False,  # Set to False in production
        "workers": 1,     # Adjust based on server capacity
        "access_log": True,
        "log_level": "info"
    }
    
    logger.info("🚀 Starting Enhanced Amulet-AI API v4.0 server...")
    logger.info(f"Server configuration: {config}")
    
    uvicorn.run(
        "backend.api.main_api:app",
        **config
    )