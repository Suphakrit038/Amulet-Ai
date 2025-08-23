"""
Configuration Management for Amulet-AI System
จัดการการตั้งค่าทั้งหมดในที่เดียว
"""
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset"
AI_MODELS_PATH = PROJECT_ROOT / "ai_models" / "saved_models"
LOGS_PATH = PROJECT_ROOT / "logs"

# Ensure directories exist
AI_MODELS_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

@dataclass
class APIConfig:
    """API Server Configuration"""
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    log_level: str = "INFO"
    cors_origins: List[str] = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]

@dataclass
class ModelConfig:
    """AI Model Configuration"""
    # Model paths
    tensorflow_model_path: Optional[str] = None
    labels_file: str = str(PROJECT_ROOT / "labels.json")
    
    # Model settings
    input_size: tuple = (224, 224)
    batch_size: int = 32
    confidence_threshold: float = 0.5
    max_predictions: int = 3
    
    # Advanced simulation settings
    use_advanced_simulation: bool = True
    simulation_confidence_range: tuple = (0.7, 0.95)
    
    # Supported image formats
    supported_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [
                'image/jpeg', 'image/jpg', 'image/png', 
                'image/heic', 'image/heif', 'image/webp', 
                'image/bmp', 'image/tiff'
            ]

@dataclass
class DataConfig:
    """Data Processing Configuration"""
    dataset_path: str = str(DATASET_PATH)
    image_extensions: List[str] = None
    max_image_size: int = 1024  # pixels
    quality_threshold: float = 0.8
    
    def __post_init__(self):
        if self.image_extensions is None:
            self.image_extensions = ['.jpg', '.jpeg', '.png', '.heic', '.webp']

@dataclass
class PriceConfig:
    """Price Estimation Configuration"""
    # Base price ranges (THB)
    base_prices: Dict[str, Dict[str, int]] = None
    
    # Price variation factors
    condition_multiplier: Dict[str, float] = None
    rarity_multiplier: Dict[str, float] = None
    market_trend_factor: float = 1.0
    
    def __post_init__(self):
        if self.base_prices is None:
            self.base_prices = {
                "หลวงพ่อกวยแหวกม่าน": {"low": 15000, "mid": 45000, "high": 120000},
                "โพธิ์ฐานบัว": {"low": 8000, "mid": 25000, "high": 75000},
                "ฐานสิงห์": {"low": 12000, "mid": 35000, "high": 85000},
                "สีวลี": {"low": 5000, "mid": 18000, "high": 50000}
            }
        
        if self.condition_multiplier is None:
            self.condition_multiplier = {
                "excellent": 1.3, "good": 1.0, "fair": 0.7, "poor": 0.4
            }
        
        if self.rarity_multiplier is None:
            self.rarity_multiplier = {
                "very_rare": 2.5, "rare": 1.8, "uncommon": 1.2, "common": 1.0
            }

@dataclass
class SystemConfig:
    """Main System Configuration"""
    # Environment
    environment: str = "development"  # development, testing, production
    debug: bool = True
    
    # Component configs
    api: APIConfig = APIConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    price: PriceConfig = PriceConfig()
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance
    max_concurrent_requests: int = 10
    request_timeout: int = 30  # seconds
    cache_ttl: int = 300  # seconds
    
    def is_production(self) -> bool:
        return self.environment == "production"
    
    def is_development(self) -> bool:
        return self.environment == "development"

# Global configuration instance
config = SystemConfig()

# Environment-based overrides
if os.getenv("AMULET_ENV") == "production":
    config.environment = "production"
    config.debug = False
    config.api.reload = False
    config.log_level = "WARNING"
elif os.getenv("AMULET_ENV") == "testing":
    config.environment = "testing"
    config.model.use_advanced_simulation = True

# Function to get configuration
def get_config() -> SystemConfig:
    """Get the current system configuration"""
    return config

def update_config(**kwargs) -> None:
    """Update configuration values"""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

# Export commonly used configs
api_config = config.api
model_config = config.model
data_config = config.data
price_config = config.price
