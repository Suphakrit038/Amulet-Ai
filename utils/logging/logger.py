"""
🏺 Amulet-AI Logging Utilities
Advanced Logging and Performance Tracking System
ระบบจัดการ log และ error tracking ขั้นสูง
"""
import logging
import json
import os
import time
import functools
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from pathlib import Path

class AmuletLogger:
    """Advanced logging system for Amulet-AI"""
    
    def __init__(self, name: str = "amulet_ai", log_dir: Optional[str] = None):
        self.name = name
        
        # Use dev-tools/logs if available, otherwise create logs
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            # Try to find dev-tools/logs, fallback to logs
            project_root = Path(__file__).parent.parent.parent
            dev_logs = project_root / "dev-tools" / "logs"
            if dev_logs.exists():
                self.log_dir = dev_logs
            else:
                self.log_dir = project_root / "logs"
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup advanced logger with multiple handlers"""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler with rotation
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Error log file
        error_file = self.log_dir / f"{self.name}_errors.log"
        error_handler = logging.FileHandler(error_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Enhanced formatter with Thai support
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str, extra: Optional[Dict] = None):
        """Log info message with optional extra data"""
        self.logger.info(self._format_message(message, extra))
    
    def warning(self, message: str, extra: Optional[Dict] = None):
        """Log warning message with optional extra data"""
        self.logger.warning(self._format_message(message, extra))
    
    def error(self, message: str, extra: Optional[Dict] = None, exc_info: bool = False):
        """Log error message with optional extra data and exception info"""
        self.logger.error(self._format_message(message, extra), exc_info=exc_info)
    
    def debug(self, message: str, extra: Optional[Dict] = None):
        """Log debug message with optional extra data"""
        self.logger.debug(self._format_message(message, extra))
    
    def critical(self, message: str, extra: Optional[Dict] = None, exc_info: bool = True):
        """Log critical message with exception info"""
        self.logger.critical(self._format_message(message, extra), exc_info=exc_info)
    
    def log_api_request(self, endpoint: str, method: str, status_code: int, 
                       response_time: float, client_ip: str = "unknown"):
        """Log API request details"""
        extra_data = {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time_ms": round(response_time * 1000, 2),
            "client_ip": client_ip
        }
        
        level = "info" if 200 <= status_code < 400 else "warning" if status_code < 500 else "error"
        message = f"API {method} {endpoint} - {status_code} - {response_time*1000:.2f}ms"
        
        getattr(self, level)(message, extra_data)
    
    def _format_message(self, message: str, extra: Optional[Dict] = None) -> str:
        """Format message with extra data in JSON format"""
        if extra:
            extra_str = json.dumps(extra, ensure_ascii=False, indent=2)
            return f"{message} | Extra: {extra_str}"
        return message

class AmuletPerformanceTracker:
    """Advanced performance tracking for Amulet-AI operations"""
    
    def __init__(self, logger: AmuletLogger):
        self.logger = logger
        self.stats = {}
        self.active_operations = {}
    
    def start_timer(self, operation: str, metadata: Optional[Dict] = None):
        """Start timing an operation with optional metadata"""
        start_time = time.time()
        operation_id = f"{operation}_{int(start_time)}"
        
        self.active_operations[operation] = {
            'id': operation_id,
            'start_time': start_time,
            'metadata': metadata or {}
        }
        
        self.logger.debug(f"Started operation: {operation}", {'operation_id': operation_id})
        return operation_id
    
    def end_timer(self, operation: str, success: bool = True, result_data: Optional[Dict] = None):
        """End timing operation and log performance metrics"""
        if operation not in self.active_operations:
            self.logger.warning(f"Operation {operation} not found in active operations")
            return
        
        end_time = time.time()
        start_data = self.active_operations.pop(operation)
        duration = end_time - start_data['start_time']
        
        performance_data = {
            'operation': operation,
            'operation_id': start_data['id'],
            'duration_seconds': round(duration, 4),
            'duration_ms': round(duration * 1000, 2),
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'metadata': start_data['metadata']
        }
        
        if result_data:
            performance_data['result'] = result_data
        
        # Store in stats
        if operation not in self.stats:
            self.stats[operation] = []
        self.stats[operation].append(performance_data)
        
        # Log performance
        status = "✅ SUCCESS" if success else "❌ FAILED"
        message = f"Operation {operation} {status} | Duration: {duration*1000:.2f}ms"
        
        if success and duration < 1.0:  # Fast operations - info level
            self.logger.info(message, performance_data)
        elif success:  # Slow operations - warning level
            self.logger.warning(f"SLOW {message}", performance_data)
        else:  # Failed operations - error level
            self.logger.error(message, performance_data)
    
    def get_performance_summary(self, operation: str = None) -> Dict[str, Any]:
        """Get performance statistics summary"""
        if operation:
            ops = self.stats.get(operation, [])
            if not ops:
                return {}
                
            durations = [op['duration_seconds'] for op in ops]
            successes = [op['success'] for op in ops]
            
            return {
                'operation': operation,
                'total_calls': len(ops),
                'success_rate': sum(successes) / len(successes) * 100,
                'avg_duration_ms': round(sum(durations) / len(durations) * 1000, 2),
                'min_duration_ms': round(min(durations) * 1000, 2),
                'max_duration_ms': round(max(durations) * 1000, 2),
                'last_call': ops[-1]['timestamp']
            }
        
        # Summary for all operations
        summary = {}
        for op_name, op_data in self.stats.items():
            summary[op_name] = self.get_performance_summary(op_name)
        
        return summary

def performance_monitor(operation_name: str = None):
    """Decorator for automatic performance monitoring"""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracker = default_performance_tracker
            tracker.start_timer(op_name, {
                'function': func.__name__,
                'module': func.__module__,
                'args_count': len(args),
                'kwargs_count': len(kwargs)
            })
            
            try:
                result = func(*args, **kwargs)
                tracker.end_timer(op_name, success=True, result_data={'has_result': result is not None})
                return result
            except Exception as e:
                tracker.end_timer(op_name, success=False, result_data={'error': str(e), 'error_type': type(e).__name__})
                raise
        
        return wrapper
    return decorator

# Global instances - Updated to use new class names
default_logger = AmuletLogger()
default_performance_tracker = AmuletPerformanceTracker(default_logger)

# Convenience functions with enhanced features
def log_info(message: str, extra: Optional[Dict] = None):
    """Enhanced info logging function"""
    default_logger.info(message, extra)

def log_error(message: str, extra: Optional[Dict] = None, exc_info: bool = False):
    """Enhanced error logging function with exception info"""
    default_logger.error(message, extra, exc_info)

def log_warning(message: str, extra: Optional[Dict] = None):
    """Enhanced warning logging function"""
    default_logger.warning(message, extra)

def log_debug(message: str, extra: Optional[Dict] = None):
    """Enhanced debug logging function"""
    default_logger.debug(message, extra)

def log_api_call(endpoint: str, method: str, status_code: int, response_time: float, client_ip: str = "unknown"):
    """Log API call with structured data"""
    default_logger.log_api_request(endpoint, method, status_code, response_time, client_ip)

def get_logger(name: str = "amulet_ai") -> AmuletLogger:
    """Get a named logger instance"""
    return AmuletLogger(name)

def track_performance(operation: str):
    """Context manager for performance tracking"""
    class PerformanceContext:
        def __init__(self, op_name: str):
            self.operation = op_name
            self.tracker = default_performance_tracker
        
        def __enter__(self):
            self.tracker.start_timer(self.operation)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            success = exc_type is None
            error_data = {'error': str(exc_val), 'error_type': exc_type.__name__} if exc_type else None
            self.tracker.end_timer(self.operation, success, error_data)
    
    return PerformanceContext(operation)
