#!/usr/bin/env python3
"""
Enhanced Error Handling for Amulet-AI
Comprehensive exception handling with proper logging and retry mechanisms
"""

import logging
import functools
import time
import asyncio
from typing import Any, Callable, Dict, Optional, Type, Union
from datetime import datetime
import traceback
from contextlib import contextmanager
import sys
from .config import config

class AmuletError(Exception):
    """Base exception for Amulet-AI"""
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "AMULET_ERROR"
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()

class ValidationError(AmuletError):
    """Validation related errors"""
    def __init__(self, message: str, field: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field

class ModelError(AmuletError):
    """Model loading and prediction errors"""
    def __init__(self, message: str, model_name: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "MODEL_ERROR", details)
        self.model_name = model_name

class ProcessingError(AmuletError):
    """Image processing errors"""
    def __init__(self, message: str, stage: str = None, details: Dict[str, Any] = None):
        super().__init__(message, "PROCESSING_ERROR", details)
        self.stage = stage

class NetworkError(AmuletError):
    """Network and API related errors"""
    def __init__(self, message: str, status_code: int = None, details: Dict[str, Any] = None):
        super().__init__(message, "NETWORK_ERROR", details)
        self.status_code = status_code

class SecurityError(AmuletError):
    """Security related errors"""
    def __init__(self, message: str, security_level: str = "HIGH", details: Dict[str, Any] = None):
        super().__init__(message, "SECURITY_ERROR", details)
        self.security_level = security_level

class ErrorLogger:
    """Enhanced error logging with different levels and formatting"""
    
    def __init__(self, name: str = "amulet_ai"):
        self.logger = logging.getLogger(name)
        self.setup_logger()
    
    def setup_logger(self):
        """Setup logger with proper formatting"""
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None, level: str = "ERROR"):
        """Log error with context and proper formatting"""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        
        if isinstance(error, AmuletError):
            error_data.update({
                "error_code": error.error_code,
                "details": error.details
            })
        
        # Add traceback for debugging
        if level in ["ERROR", "CRITICAL"]:
            error_data["traceback"] = traceback.format_exc()
        
        log_method = getattr(self.logger, level.lower())
        log_method(f"Error occurred: {error_data}")
    
    def log_security_event(self, event: str, details: Dict[str, Any] = None):
        """Log security-related events"""
        security_data = {
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.logger.warning(f"SECURITY EVENT: {security_data}")
    
    def log_performance(self, operation: str, duration: float, details: Dict[str, Any] = None):
        """Log performance metrics"""
        perf_data = {
            "operation": operation,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.logger.info(f"PERFORMANCE: {perf_data}")

# Global error logger instance
error_logger = ErrorLogger()

def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retrying failed operations with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        error_logger.log_error(
                            e, 
                            context={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "max_retries": max_retries
                            }
                        )
                        raise
                    
                    wait_time = delay * (backoff_factor ** attempt)
                    error_logger.logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {wait_time:.2f}s: {str(e)}"
                    )
                    time.sleep(wait_time)
            
            raise last_exception
        
        return wrapper
    return decorator

def async_retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Async version of retry decorator"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        error_logger.log_error(
                            e,
                            context={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "max_retries": max_retries
                            }
                        )
                        raise
                    
                    wait_time = delay * (backoff_factor ** attempt)
                    error_logger.logger.warning(
                        f"Async retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {wait_time:.2f}s: {str(e)}"
                    )
                    await asyncio.sleep(wait_time)
            
            raise last_exception
        
        return wrapper
    return decorator

@contextmanager
def error_context(operation: str, **context_data):
    """Context manager for error handling with automatic logging"""
    start_time = time.time()
    try:
        error_logger.logger.info(f"Starting operation: {operation}")
        yield
        
        duration = time.time() - start_time
        error_logger.log_performance(operation, duration, context_data)
        
    except Exception as e:
        duration = time.time() - start_time
        context_data.update({
            "operation": operation,
            "duration": duration
        })
        
        error_logger.log_error(e, context_data)
        raise

def safe_execute(func: Callable, default_return: Any = None, log_errors: bool = True) -> Any:
    """Safely execute a function and return default value on error"""
    try:
        return func()
    except Exception as e:
        if log_errors:
            error_logger.log_error(e, context={"function": func.__name__})
        return default_return

def validate_and_sanitize(
    value: Any,
    expected_type: Type,
    max_length: int = None,
    allowed_values: list = None,
    sanitizer: Callable = None
) -> Any:
    """Validate and sanitize input values"""
    try:
        # Type validation
        if not isinstance(value, expected_type):
            if expected_type == str and value is not None:
                value = str(value)
            else:
                raise ValidationError(
                    f"Expected {expected_type.__name__}, got {type(value).__name__}",
                    details={"value": str(value)[:100]}
                )
        
        # String length validation
        if expected_type == str and max_length and len(value) > max_length:
            raise ValidationError(
                f"Value too long: {len(value)} > {max_length}",
                details={"value_length": len(value), "max_length": max_length}
            )
        
        # Allowed values validation
        if allowed_values and value not in allowed_values:
            raise ValidationError(
                f"Value not in allowed list: {value}",
                details={"value": str(value), "allowed": allowed_values}
            )
        
        # Apply sanitizer if provided
        if sanitizer:
            value = sanitizer(value)
        
        return value
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Validation failed: {str(e)}")

class CircuitBreaker:
    """Circuit breaker pattern for failing services"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if time.time() - self.last_failure_time < self.timeout:
                    raise NetworkError("Circuit breaker is OPEN")
                else:
                    self.state = "HALF_OPEN"
            
            try:
                result = func(*args, **kwargs)
                self.on_success()
                return result
            except Exception as e:
                self.on_failure()
                raise
        
        return wrapper
    
    def on_success(self):
        """Called on successful execution"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def on_failure(self):
        """Called on failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

# Export commonly used decorators and utilities
__all__ = [
    'AmuletError', 'ValidationError', 'ModelError', 'ProcessingError', 
    'NetworkError', 'SecurityError', 'ErrorLogger', 'error_logger',
    'retry_on_failure', 'async_retry_on_failure', 'error_context',
    'safe_execute', 'validate_and_sanitize', 'CircuitBreaker'
]