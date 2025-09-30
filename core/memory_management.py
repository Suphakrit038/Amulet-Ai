#!/usr/bin/env python3
"""
Memory Management Module for Amulet-AI
Streaming file handling, garbage collection, and memory usage monitoring
"""

import gc
import psutil
import threading
import time
import weakref
import mmap
import os
from typing import Generator, Optional, Dict, Any, BinaryIO
from dataclasses import dataclass
from datetime import datetime
import io
from contextlib import contextmanager
from .config import config
from .error_handling import ProcessingError, error_logger

@dataclass
class MemoryUsage:
    """Memory usage statistics"""
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    available_mb: float
    timestamp: datetime

class MemoryMonitor:
    """Monitor and manage memory usage"""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._process = psutil.Process()
        self._lock = threading.Lock()
        self._memory_log = []
        self._max_log_entries = 1000
        
    def get_current_usage(self) -> MemoryUsage:
        """Get current memory usage"""
        try:
            memory_info = self._process.memory_info()
            system_memory = psutil.virtual_memory()
            
            return MemoryUsage(
                rss_mb=memory_info.rss / 1024 / 1024,
                vms_mb=memory_info.vms / 1024 / 1024,
                percent=system_memory.percent / 100,
                available_mb=system_memory.available / 1024 / 1024,
                timestamp=datetime.now()
            )
        except Exception as e:
            error_logger.log_error(e, context={"operation": "memory_monitoring"})
            return MemoryUsage(0, 0, 0, 0, datetime.now())
    
    def log_usage(self, operation: str = None):
        """Log current memory usage"""
        usage = self.get_current_usage()
        
        with self._lock:
            self._memory_log.append({
                'usage': usage,
                'operation': operation,
                'timestamp': usage.timestamp.isoformat()
            })
            
            # Trim log if too large
            if len(self._memory_log) > self._max_log_entries:
                self._memory_log = self._memory_log[-self._max_log_entries:]
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """Check if system is under memory pressure"""
        usage = self.get_current_usage()
        
        pressure_level = "normal"
        if usage.percent > self.critical_threshold:
            pressure_level = "critical"
        elif usage.percent > self.warning_threshold:
            pressure_level = "warning"
        
        return {
            "level": pressure_level,
            "usage": usage,
            "should_gc": pressure_level in ["warning", "critical"],
            "should_clear_cache": pressure_level == "critical"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        with self._lock:
            if not self._memory_log:
                return {"no_data": True}
            
            recent_entries = self._memory_log[-100:]  # Last 100 entries
            rss_values = [entry['usage'].rss_mb for entry in recent_entries]
            
            return {
                "current": self.get_current_usage().__dict__,
                "recent_avg_rss_mb": sum(rss_values) / len(rss_values),
                "recent_max_rss_mb": max(rss_values),
                "recent_min_rss_mb": min(rss_values),
                "log_entries": len(self._memory_log),
                "pressure_check": self.check_memory_pressure()
            }

class StreamingFileHandler:
    """Handle large files with streaming and memory-efficient processing"""
    
    def __init__(self, chunk_size: int = 8192):
        self.chunk_size = chunk_size
        self.memory_monitor = MemoryMonitor()
    
    @contextmanager
    def open_large_file(self, file_path: str, mode: str = 'rb'):
        """Context manager for large file handling with memory monitoring"""
        file_handle = None
        try:
            self.memory_monitor.log_usage(f"opening_file_{os.path.basename(file_path)}")
            
            file_size = os.path.getsize(file_path)
            
            # Use memory mapping for very large files
            if file_size > config.MMAP_THRESHOLD:
                file_handle = open(file_path, mode)
                with mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    yield mmapped_file
            else:
                file_handle = open(file_path, mode)
                yield file_handle
                
        except Exception as e:
            raise ProcessingError(f"Failed to open file {file_path}: {e}")
        finally:
            if file_handle:
                file_handle.close()
            self.memory_monitor.log_usage(f"closed_file_{os.path.basename(file_path)}")
    
    def stream_file_chunks(self, file_handle: BinaryIO) -> Generator[bytes, None, None]:
        """Stream file in chunks"""
        try:
            while True:
                # Check memory pressure before reading next chunk
                pressure = self.memory_monitor.check_memory_pressure()
                if pressure["should_gc"]:
                    gc.collect()
                
                chunk = file_handle.read(self.chunk_size)
                if not chunk:
                    break
                yield chunk
                
        except Exception as e:
            raise ProcessingError(f"Failed to stream file chunks: {e}")
    
    def process_large_image(self, image_data: bytes, max_size: int = None) -> bytes:
        """Process large image with memory management"""
        max_size = max_size or config.MAX_FILE_SIZE
        
        if len(image_data) > max_size:
            raise ProcessingError(f"Image too large: {len(image_data)} > {max_size}")
        
        try:
            self.memory_monitor.log_usage("image_processing_start")
            
            # Check memory pressure
            pressure = self.memory_monitor.check_memory_pressure()
            if pressure["level"] == "critical":
                gc.collect()
                
                # Re-check after GC
                pressure = self.memory_monitor.check_memory_pressure()
                if pressure["level"] == "critical":
                    raise ProcessingError("Insufficient memory for image processing")
            
            # Process image in memory-efficient way
            image_io = io.BytesIO(image_data)
            
            # For very large images, process in chunks
            if len(image_data) > config.LARGE_IMAGE_THRESHOLD:
                return self._process_large_image_chunked(image_io)
            else:
                return image_data
                
        finally:
            self.memory_monitor.log_usage("image_processing_end")
    
    def _process_large_image_chunked(self, image_io: io.BytesIO) -> bytes:
        """Process large image in chunks to manage memory"""
        try:
            from PIL import Image
            
            # Load image with PIL for processing
            image = Image.open(image_io)
            
            # Resize if too large
            max_dimension = 2048
            if max(image.size) > max_dimension:
                ratio = max_dimension / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save back to bytes
            output_io = io.BytesIO()
            image.save(output_io, format='JPEG', quality=85, optimize=True)
            return output_io.getvalue()
            
        except Exception as e:
            raise ProcessingError(f"Failed to process large image: {e}")

class GarbageCollectionManager:
    """Manage garbage collection and memory cleanup"""
    
    def __init__(self):
        self.memory_monitor = MemoryMonitor()
        self._cleanup_lock = threading.Lock()
        self._last_cleanup = datetime.now()
        self._cleanup_interval = 300  # 5 minutes
        
    def should_cleanup(self) -> bool:
        """Check if cleanup should be performed"""
        time_since_cleanup = (datetime.now() - self._last_cleanup).total_seconds()
        pressure = self.memory_monitor.check_memory_pressure()
        
        return (time_since_cleanup > self._cleanup_interval or 
                pressure["should_gc"])
    
    def perform_cleanup(self, force: bool = False) -> Dict[str, Any]:
        """Perform garbage collection and cleanup"""
        if not force and not self.should_cleanup():
            return {"skipped": True, "reason": "not_needed"}
        
        with self._cleanup_lock:
            try:
                self.memory_monitor.log_usage("cleanup_start")
                
                # Record memory before cleanup
                before_memory = self.memory_monitor.get_current_usage()
                
                # Perform garbage collection
                collected = []
                for generation in range(3):
                    collected.append(gc.collect(generation))
                
                # Clear weak references
                weakref.getweakrefs
                
                # Record memory after cleanup
                after_memory = self.memory_monitor.get_current_usage()
                
                self._last_cleanup = datetime.now()
                self.memory_monitor.log_usage("cleanup_end")
                
                memory_freed = before_memory.rss_mb - after_memory.rss_mb
                
                return {
                    "performed": True,
                    "collections": collected,
                    "memory_before_mb": before_memory.rss_mb,
                    "memory_after_mb": after_memory.rss_mb,
                    "memory_freed_mb": memory_freed,
                    "timestamp": self._last_cleanup.isoformat()
                }
                
            except Exception as e:
                error_logger.log_error(e, context={"operation": "garbage_collection"})
                return {"error": str(e)}

class WeakReferenceManager:
    """Manage weak references to prevent memory leaks"""
    
    def __init__(self):
        self._refs = {}
        self._callbacks = {}
    
    def register(self, obj: Any, name: str, callback: callable = None):
        """Register an object with weak reference"""
        def cleanup_callback(ref):
            if name in self._refs:
                del self._refs[name]
            if callback:
                callback()
        
        self._refs[name] = weakref.ref(obj, cleanup_callback)
        if callback:
            self._callbacks[name] = callback
    
    def get(self, name: str) -> Optional[Any]:
        """Get object by name if still alive"""
        ref = self._refs.get(name)
        return ref() if ref else None
    
    def cleanup_dead_refs(self):
        """Remove dead references"""
        dead_refs = [name for name, ref in self._refs.items() if ref() is None]
        for name in dead_refs:
            del self._refs[name]
            if name in self._callbacks:
                del self._callbacks[name]

# Global instances
memory_monitor = MemoryMonitor()
streaming_handler = StreamingFileHandler()
gc_manager = GarbageCollectionManager()
weak_ref_manager = WeakReferenceManager()

def memory_efficient_operation(operation_name: str):
    """Decorator for memory-efficient operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                memory_monitor.log_usage(f"{operation_name}_start")
                
                # Check memory pressure before operation
                pressure = memory_monitor.check_memory_pressure()
                if pressure["should_gc"]:
                    gc_manager.perform_cleanup()
                
                result = func(*args, **kwargs)
                
                # Check memory after operation
                memory_monitor.log_usage(f"{operation_name}_end")
                
                return result
                
            except Exception as e:
                memory_monitor.log_usage(f"{operation_name}_error")
                raise
        
        return wrapper
    return decorator

@contextmanager
def memory_limit_context(max_memory_mb: float):
    """Context manager to enforce memory limits"""
    initial_memory = memory_monitor.get_current_usage()
    
    try:
        yield
    finally:
        current_memory = memory_monitor.get_current_usage()
        if current_memory.rss_mb > initial_memory.rss_mb + max_memory_mb:
            error_logger.logger.warning(
                f"Memory limit exceeded: {current_memory.rss_mb:.1f}MB > "
                f"{initial_memory.rss_mb + max_memory_mb:.1f}MB"
            )
            gc_manager.perform_cleanup(force=True)

def schedule_memory_cleanup():
    """Schedule periodic memory cleanup"""
    def cleanup_task():
        try:
            # Perform cleanup if needed
            gc_manager.perform_cleanup()
            
            # Cleanup weak references
            weak_ref_manager.cleanup_dead_refs()
            
            # Schedule next cleanup
            threading.Timer(300, cleanup_task).start()  # Every 5 minutes
            
        except Exception as e:
            error_logger.log_error(e, context={"operation": "scheduled_cleanup"})
    
    # Start cleanup timer
    threading.Timer(300, cleanup_task).start()

# Initialize cleanup on module import
schedule_memory_cleanup()

# Export public interface
__all__ = [
    'MemoryUsage', 'MemoryMonitor', 'StreamingFileHandler', 
    'GarbageCollectionManager', 'WeakReferenceManager',
    'memory_monitor', 'streaming_handler', 'gc_manager', 'weak_ref_manager',
    'memory_efficient_operation', 'memory_limit_context', 'schedule_memory_cleanup'
]