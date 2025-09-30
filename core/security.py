#!/usr/bin/env python3
"""
Security utilities for Amulet-AI
JWT token management, input sanitization, and security helpers
"""

try:
    import jwt
except ImportError:
    print("Warning: PyJWT not installed. JWT features will not work.")
    jwt = None
import html
import re
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path
from .config import config

class SecurityManager:
    """Centralized security management"""
    
    def __init__(self):
        self.secret_key = config.SECRET_KEY
        self.algorithm = config.TOKEN_ALGORITHM
        self.expire_minutes = config.TOKEN_EXPIRE_MINUTES
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.expire_minutes)
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    @staticmethod
    def sanitize_html(text: str) -> str:
        """Sanitize HTML input to prevent XSS"""
        if not isinstance(text, str):
            return str(text)
        return html.escape(text, quote=True)
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        if not isinstance(filename, str):
            filename = str(filename)
        
        # Remove path separators and dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'\.\.+', '.', filename)
        filename = filename.strip('. ')
        
        # Limit length
        if len(filename) > 255:
            name, ext = Path(filename).stem, Path(filename).suffix
            filename = name[:250-len(ext)] + ext
        
        return filename or "unnamed_file"
    
    @staticmethod
    def validate_image_type(content_type: str, file_data: bytes) -> bool:
        """Validate image type by checking both MIME type and file signature"""
        if content_type not in config.ALLOWED_IMAGE_TYPES:
            return False
        
        # Check file signatures (magic numbers)
        signatures = {
            b'\xFF\xD8\xFF': ['image/jpeg', 'image/jpg'],  # JPEG
            b'\x89PNG\r\n\x1a\n': ['image/png'],           # PNG
            b'GIF87a': ['image/gif'],                       # GIF87a
            b'GIF89a': ['image/gif'],                       # GIF89a
            b'RIFF': ['image/webp'],                        # WebP (partial)
        }
        
        for signature, mime_types in signatures.items():
            if file_data.startswith(signature):
                return content_type in mime_types
        
        return False
    
    @staticmethod
    def generate_secure_filename(original_filename: str) -> str:
        """Generate secure filename with random component"""
        secure_name = SecurityManager.sanitize_filename(original_filename)
        name, ext = Path(secure_name).stem, Path(secure_name).suffix
        random_component = secrets.token_hex(8)
        return f"{name}_{random_component}{ext}"
    
    @staticmethod
    def hash_file_content(file_data: bytes) -> str:
        """Generate SHA256 hash of file content"""
        return hashlib.sha256(file_data).hexdigest()
    
    @staticmethod
    def validate_request_size(content_length: Optional[int]) -> bool:
        """Validate request content length"""
        if content_length is None:
            return True  # Let the server handle it
        return content_length <= config.MAX_FILE_SIZE
    
    @staticmethod
    def is_safe_origin(origin: str) -> bool:
        """Check if origin is in allowed list"""
        return origin in config.ALLOWED_ORIGINS
    
    @staticmethod
    def rate_limit_key(identifier: str) -> str:
        """Generate rate limiting key"""
        return f"rate_limit:{hashlib.md5(identifier.encode()).hexdigest()}"

class InputValidator:
    """Input validation utilities"""
    
    @staticmethod
    def validate_image_dimensions(width: int, height: int) -> bool:
        """Validate image dimensions"""
        min_size = 50
        max_size = 4000
        
        return (min_size <= width <= max_size and 
                min_size <= height <= max_size)
    
    @staticmethod
    def validate_file_size(size: int) -> bool:
        """Validate file size"""
        return 0 < size <= config.MAX_FILE_SIZE
    
    @staticmethod
    def validate_api_params(params: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate API parameters and return errors"""
        errors = {}
        
        # Validate model parameter
        if 'model' in params:
            valid_models = ['enhanced', 'twobranch', 'auto']
            if params['model'] not in valid_models:
                errors.setdefault('model', []).append(f"Must be one of: {valid_models}")
        
        # Validate preprocess parameter
        if 'preprocess' in params:
            valid_preprocess = ['standard', 'advanced']
            if params['preprocess'] not in valid_preprocess:
                errors.setdefault('preprocess', []).append(f"Must be one of: {valid_preprocess}")
        
        return errors

# Global security manager instance
security = SecurityManager()
validator = InputValidator()