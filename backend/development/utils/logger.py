"""
Advanced Logging System for Amulet-AI
ระบบบันทึกและตรวจสอบการทำงานแบบครบวงจร
"""
import os
import sys
import time
import logging
import functools
import inspect
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Union

# Try to import main config, but provide fallbacks
try:
    from backend.config.app_config import get_config
    config = get_config()
    LOG_LEVEL = config.log_level
    LOG_FORMAT = config.log_format
except ImportError:
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Define base log path
BASE_DIR = Path(__file__).parent.parent.parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True, parents=True)

# Configure root logger
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "amulet_ai.log", encoding="utf-8")
    ]
)

# Performance tracking
performance_data = {
    "total_calls": {},
    "total_time": {},
    "avg_time": {},
    "min_time": {},
    "max_time": {},
    "last_call": {}
}

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger with the specified name
    
    Args:
        name: Name for the logger, typically the module name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Add file handler for this specific module
    module_log_file = LOGS_DIR / f"{name}.log"
    file_handler = logging.FileHandler(module_log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)
    
    return logger

def performance_monitor(operation_name: str = None) -> Callable:
    """
    Decorator to monitor function performance
    
    Args:
        operation_name: Name of the operation being monitored
        
    Returns:
        Decorated function with performance monitoring
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use function name if operation_name not provided
            op_name = operation_name or func.__name__
            
            # Initialize performance data if needed
            if op_name not in performance_data["total_calls"]:
                performance_data["total_calls"][op_name] = 0
                performance_data["total_time"][op_name] = 0
                performance_data["min_time"][op_name] = float('inf')
                performance_data["max_time"][op_name] = 0
            
            # Record start time
            start_time = time.time()
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Update performance data
            performance_data["total_calls"][op_name] += 1
            performance_data["total_time"][op_name] += elapsed_time
            performance_data["avg_time"][op_name] = (
                performance_data["total_time"][op_name] / 
                performance_data["total_calls"][op_name]
            )
            performance_data["min_time"][op_name] = min(
                performance_data["min_time"][op_name], 
                elapsed_time
            )
            performance_data["max_time"][op_name] = max(
                performance_data["max_time"][op_name], 
                elapsed_time
            )
            performance_data["last_call"][op_name] = elapsed_time
            
            # Log performance information for slow operations
            if elapsed_time > 1.0:  # Log operations taking more than 1 second
                module_logger = logging.getLogger(func.__module__)
                module_logger.warning(
                    f"⏱️ Slow operation - {op_name} took {elapsed_time:.2f}s to complete"
                )
            
            return result
        
        return wrapper
    
    # Handle case where decorator is used without parentheses
    if callable(operation_name):
        func = operation_name
        operation_name = func.__name__
        return decorator(func)
    
    return decorator

class PerformanceContext:
    """Context manager for tracking performance of code blocks"""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        """Initialize the context manager"""
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        
    def __enter__(self):
        """Start timing when entering the context"""
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log timing when exiting the context"""
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 1.0:  # Log operations taking more than 1 second
                self.logger.warning(
                    f"⏱️ Slow operation - {self.operation_name} took {elapsed_time:.2f}s to complete"
                )
            else:
                self.logger.debug(
                    f"⏱️ Operation {self.operation_name} took {elapsed_time:.4f}s to complete"
                )

def track_performance(operation_name: str) -> PerformanceContext:
    """
    Get a context manager for tracking performance
    
    Args:
        operation_name: Name of the operation to track
        
    Returns:
        PerformanceContext instance
    """
    # Get the caller's module name to use the right logger
    frame = inspect.currentframe().f_back
    module_name = frame.f_globals.get('__name__', __name__)
    logger = logging.getLogger(module_name)
    
    return PerformanceContext(operation_name, logger)

def get_performance_stats() -> Dict[str, Dict[str, float]]:
    """
    Get collected performance statistics
    
    Returns:
        Dictionary with performance statistics
    """
    return performance_data

def log_exception(logger: logging.Logger, exc: Exception, context: str = ""):
    """
    Log an exception with detailed information
    
    Args:
        logger: Logger to use
        exc: Exception to log
        context: Optional context information
    """
    if context:
        logger.error(f"Exception in {context}: {exc}", exc_info=True)
    else:
        logger.error(f"Exception: {exc}", exc_info=True)

def setup_module_logger(module_name: str) -> logging.Logger:
    """
    Set up a logger for a specific module with both console and file output
    
    Args:
        module_name: Name of the module
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Create console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console)
    
    # Create file handler
    log_file = LOGS_DIR / f"{module_name.replace('.', '_')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)
    
    return logger
