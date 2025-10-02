#!/usr/bin/env python3
"""
Configuration management for Amulet-AI
Centralized configuration with environment variable support and validation
"""

import os
from typing import Optional, List
from pathlib import Path

class Config:
    """Centralized configuration management"""
    
    # API Configuration
    API_HOST: str = os.getenv("AMULET_API_HOST", "localhost")
    API_PORT: int = int(os.getenv("AMULET_API_PORT", "8000"))
    API_PROTOCOL: str = os.getenv("AMULET_API_PROTOCOL", "http")
    
    @property
    def API_BASE_URL(self) -> str:
        return f"{self.API_PROTOCOL}://{self.API_HOST}:{self.API_PORT}"
    
    # Security Configuration
    SECRET_KEY: str = os.getenv("AMULET_SECRET_KEY", "amulet-ai-development-key-2025")
    ALLOWED_ORIGINS: List[str] = os.getenv("AMULET_ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:8511").split(",")
    TOKEN_ALGORITHM: str = os.getenv("AMULET_TOKEN_ALGORITHM", "HS256")
    TOKEN_EXPIRE_MINUTES: int = int(os.getenv("AMULET_TOKEN_EXPIRE_MINUTES", "30"))
    REQUIRE_AUTH: bool = os.getenv("AMULET_REQUIRE_AUTH", "false").lower() == "true"
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = int(os.getenv("AMULET_MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # 10MB
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    UPLOAD_TIMEOUT: int = int(os.getenv("AMULET_UPLOAD_TIMEOUT", "60"))
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE: int = int(os.getenv("AMULET_MAX_REQUESTS_PER_MINUTE", "100"))
    RATE_LIMIT_ENABLED: bool = os.getenv("AMULET_RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS: int = int(os.getenv("AMULET_RATE_LIMIT_REQUESTS", "60"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("AMULET_RATE_LIMIT_WINDOW", "60"))
    
    # Model Configuration
    # Model paths and configurations
    MODEL_PATH = "trained_model"  # Path to the trained model directory
    TWOBRANCH_PATHS: List[str] = ["trained_twobranch", "ai_models/twobranch"]
    MODEL_CACHE_SIZE: int = int(os.getenv("AMULET_MODEL_CACHE_SIZE", "10"))
    USE_MOCK_MODEL: bool = os.getenv("AMULET_USE_MOCK_MODEL", "false").lower() == "true"
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("AMULET_LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("AMULET_LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    LOG_FILE: Optional[str] = os.getenv("AMULET_LOG_FILE")
    
    # Performance Configuration
    ENABLE_ASYNC_PROCESSING: bool = os.getenv("AMULET_ENABLE_ASYNC_PROCESSING", "true").lower() == "true"
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("AMULET_MAX_CONCURRENT_REQUESTS", "10"))
    CACHE_TTL_SECONDS: int = int(os.getenv("AMULET_CACHE_TTL_SECONDS", "300"))
    
    # Cache Configuration
    CACHE_MAX_SIZE: int = int(os.getenv("AMULET_CACHE_MAX_SIZE", str(100 * 1024 * 1024)))  # 100MB
    CACHE_DEFAULT_TTL: int = int(os.getenv("AMULET_CACHE_DEFAULT_TTL", "3600"))  # 1 hour
    IMAGE_CACHE_SIZE: int = int(os.getenv("AMULET_IMAGE_CACHE_SIZE", "500"))
    IMAGE_CACHE_TTL: int = int(os.getenv("AMULET_IMAGE_CACHE_TTL", "1800"))  # 30 minutes
    
    # Memory Management
    MEMORY_WARNING_THRESHOLD: float = float(os.getenv("AMULET_MEMORY_WARNING_THRESHOLD", "0.8"))  # 80%
    MEMORY_CRITICAL_THRESHOLD: float = float(os.getenv("AMULET_MEMORY_CRITICAL_THRESHOLD", "0.9"))  # 90%
    MMAP_THRESHOLD: int = int(os.getenv("AMULET_MMAP_THRESHOLD", str(50 * 1024 * 1024)))  # 50MB
    LARGE_IMAGE_THRESHOLD: int = int(os.getenv("AMULET_LARGE_IMAGE_THRESHOLD", str(10 * 1024 * 1024)))  # 10MB
    
    # Performance Settings
    CONNECTION_POOL_SIZE: int = int(os.getenv("AMULET_CONNECTION_POOL_SIZE", "100"))
    CHUNK_SIZE: int = int(os.getenv("AMULET_CHUNK_SIZE", "8192"))
    THREAD_POOL_SIZE: int = int(os.getenv("AMULET_THREAD_POOL_SIZE", "4"))
    
    # Frontend Configuration
    FRONTEND_CACHE_IMAGES: bool = os.getenv("AMULET_FRONTEND_CACHE_IMAGES", "true").lower() == "true"
    FRONTEND_CSS_CACHE: bool = os.getenv("AMULET_FRONTEND_CSS_CACHE", "true").lower() == "true"
    
    # Development Configuration
    DEBUG: bool = os.getenv("AMULET_DEBUG", "false").lower() == "true"
    RELOAD: bool = os.getenv("AMULET_RELOAD", "false").lower() == "true"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Check required paths exist
        if not Path(cls.MODEL_PATH).exists() and not cls.DEBUG:
            errors.append(f"Model path does not exist: {cls.MODEL_PATH}")
        
        # Validate file size limits
        if cls.MAX_FILE_SIZE > 50 * 1024 * 1024:  # 50MB max
            errors.append("MAX_FILE_SIZE too large (max 50MB)")
        
        # Validate security settings
        if cls.SECRET_KEY == "your-secret-key-change-in-production" and not cls.DEBUG:
            errors.append("SECRET_KEY must be changed in production")
        
        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False
        
        return True
    
    @classmethod
    def load_env_file(cls, env_file: str = ".env") -> None:
        """Load environment variables from file"""
        env_path = Path(env_file)
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()

# Global configuration instance
config = Config()

# Validate configuration on import
if not config.validate():
    print("Warning: Configuration validation failed!")