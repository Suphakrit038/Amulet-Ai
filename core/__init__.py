#!/usr/bin/env python3
"""
Core modules for Amulet-AI
Enhanced features including error handling, memory management, 
performance optimization, and thread safety
"""

from .config import config
from .error_handling import (
    AmuletError, ValidationError, ModelError, ProcessingError,
    NetworkError, SecurityError, ErrorLogger, error_logger,
    retry_on_failure, async_retry_on_failure, error_context,
    safe_execute, validate_and_sanitize, CircuitBreaker
)
from .memory_management import (
    MemoryUsage, MemoryMonitor, StreamingFileHandler,
    GarbageCollectionManager, WeakReferenceManager,
    memory_monitor, streaming_handler, gc_manager, weak_ref_manager,
    memory_efficient_operation, memory_limit_context, schedule_memory_cleanup
)
from .performance import (
    TTLCache, ImageCache, ConnectionPool, PerformanceMonitor,
    image_cache, connection_pool, performance_monitor,
    timed_operation, get_file_hash, stream_large_file,
    memory_efficient_b64encode, schedule_cleanup
)
from .thread_safety import (
    LockType, ThreadSafeStats, ThreadSafeDict, ThreadSafeQueue,
    AtomicCounter, LockManager, ThreadPoolManager, AsyncSafeDataStructures,
    global_stats, lock_manager, thread_pool_manager,
    thread_safe_operation, atomic_operation, cleanup_thread_resources
)
from .security import security, validator
from .rate_limiter import rate_limiter, get_client_id, apply_rate_limit

__all__ = [
    # Config
    'config',
    
    # Error Handling
    'AmuletError', 'ValidationError', 'ModelError', 'ProcessingError',
    'NetworkError', 'SecurityError', 'ErrorLogger', 'error_logger',
    'retry_on_failure', 'async_retry_on_failure', 'error_context',
    'safe_execute', 'validate_and_sanitize', 'CircuitBreaker',
    
    # Memory Management
    'MemoryUsage', 'MemoryMonitor', 'StreamingFileHandler',
    'GarbageCollectionManager', 'WeakReferenceManager',
    'memory_monitor', 'streaming_handler', 'gc_manager', 'weak_ref_manager',
    'memory_efficient_operation', 'memory_limit_context', 'schedule_memory_cleanup',
    
    # Performance
    'TTLCache', 'ImageCache', 'ConnectionPool', 'PerformanceMonitor',
    'image_cache', 'connection_pool', 'performance_monitor',
    'timed_operation', 'get_file_hash', 'stream_large_file',
    'memory_efficient_b64encode', 'schedule_cleanup',
    
    # Thread Safety
    'LockType', 'ThreadSafeStats', 'ThreadSafeDict', 'ThreadSafeQueue',
    'AtomicCounter', 'LockManager', 'ThreadPoolManager', 'AsyncSafeDataStructures',
    'global_stats', 'lock_manager', 'thread_pool_manager',
    'thread_safe_operation', 'atomic_operation', 'cleanup_thread_resources',
    
    # Security
    'security', 'validator',
    
    # Rate Limiting
    'rate_limiter', 'get_client_id', 'apply_rate_limit'
]