#!/usr/bin/env python3
"""
Performance Optimization Module for Amulet-AI
Caching, async I/O, connection pooling, and performance monitoring
"""

import asyncio
import aiohttp
import aiofiles
import time
import base64
import hashlib
import json
import threading
from typing import Any, Dict, Optional, Union, List
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from dataclasses import dataclass, field
from collections import defaultdict
import weakref
import gc
from .config import config

@dataclass
class CacheEntry:
    """Cache entry with expiration and metadata"""
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0

class TTLCache:
    """Thread-safe TTL (Time To Live) cache with size limits"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._total_size = 0
        self._stats = defaultdict(int)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats['misses'] += 1
                return None
            
            # Check expiration
            if entry.expires_at and datetime.now() > entry.expires_at:
                del self._cache[key]
                self._total_size -= entry.size_bytes
                self._stats['expired'] += 1
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            self._stats['hits'] += 1
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache"""
        with self._lock:
            # Calculate size
            size_bytes = self._estimate_size(value)
            
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._total_size -= old_entry.size_bytes
            
            # Check if we need to evict entries
            while (len(self._cache) >= self.max_size or 
                   self._total_size + size_bytes > config.CACHE_MAX_SIZE):
                self._evict_lru()
            
            # Create new entry
            ttl = ttl or self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None
            
            entry = CacheEntry(
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at,
                size_bytes=size_bytes
            )
            
            self._cache[key] = entry
            self._total_size += size_bytes
            self._stats['puts'] += 1
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self._cache:
            return
        
        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed
        )
        
        entry = self._cache.pop(lru_key)
        self._total_size -= entry.size_bytes
        self._stats['evictions'] += 1
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value"""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, bytes):
                return len(value)
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(
                    self._estimate_size(k) + self._estimate_size(v)
                    for k, v in value.items()
                )
            else:
                return len(str(value).encode('utf-8'))
        except Exception:
            return 1024  # Default estimate
    
    def cleanup_expired(self):
        """Remove expired entries"""
        with self._lock:
            now = datetime.now()
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.expires_at and now > entry.expires_at
            ]
            
            for key in expired_keys:
                entry = self._cache.pop(key)
                self._total_size -= entry.size_bytes
                self._stats['expired'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / max(total_requests, 1)
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'total_size_bytes': self._total_size,
                'hit_rate': hit_rate,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'puts': self._stats['puts'],
                'evictions': self._stats['evictions'],
                'expired': self._stats['expired']
            }
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._total_size = 0
            self._stats.clear()

class ImageCache(TTLCache):
    """Specialized cache for base64 encoded images"""
    
    def __init__(self, max_size: int = 500, default_ttl: int = 1800):
        super().__init__(max_size, default_ttl)
    
    def cache_image(self, image_data: bytes, image_name: str = None) -> str:
        """Cache image and return cache key"""
        # Generate cache key from image hash
        image_hash = hashlib.sha256(image_data).hexdigest()
        cache_key = f"image:{image_hash}"
        
        # Check if already cached
        cached = self.get(cache_key)
        if cached:
            return cache_key
        
        # Encode and cache
        try:
            base64_data = base64.b64encode(image_data).decode('utf-8')
            self.put(cache_key, {
                'data': base64_data,
                'name': image_name,
                'size': len(image_data),
                'hash': image_hash
            })
            return cache_key
        except Exception as e:
            raise ProcessingError(f"Failed to cache image: {e}")
    
    def get_image(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached image data"""
        return self.get(cache_key)

class ConnectionPool:
    """Async HTTP connection pool"""
    
    def __init__(self, max_connections: int = 100, timeout: int = 30):
        self.max_connections = max_connections
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            async with self._lock:
                if self._session is None or self._session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=self.max_connections,
                        limit_per_host=20,
                        keepalive_timeout=60,
                        enable_cleanup_closed=True
                    )
                    
                    timeout = aiohttp.ClientTimeout(total=self.timeout)
                    
                    self._session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        headers={
                            'User-Agent': 'Amulet-AI/4.0.0',
                            'Accept': 'application/json'
                        }
                    )
        
        return self._session
    
    async def close(self):
        """Close connection pool"""
        if self._session and not self._session.closed:
            await self._session.close()

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self._metrics = defaultdict(list)
        self._lock = threading.Lock()
    
    def record_timing(self, operation: str, duration: float, metadata: Dict[str, Any] = None):
        """Record timing for an operation"""
        with self._lock:
            self._metrics[f"{operation}_timing"].append({
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            })
            
            # Keep only recent entries
            max_entries = 1000
            if len(self._metrics[f"{operation}_timing"]) > max_entries:
                self._metrics[f"{operation}_timing"] = self._metrics[f"{operation}_timing"][-max_entries:]
    
    def record_counter(self, metric: str, value: int = 1):
        """Record counter metric"""
        with self._lock:
            self._metrics[f"{metric}_count"].append({
                'value': value,
                'timestamp': datetime.now().isoformat()
            })
    
    def get_stats(self, operation: str = None) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._lock:
            if operation:
                timings = self._metrics.get(f"{operation}_timing", [])
                if timings:
                    durations = [t['duration'] for t in timings[-100:]]  # Last 100
                    return {
                        'count': len(timings),
                        'avg_duration': sum(durations) / len(durations),
                        'min_duration': min(durations),
                        'max_duration': max(durations),
                        'recent_count': len([t for t in timings if 
                                           datetime.fromisoformat(t['timestamp']) > 
                                           datetime.now() - timedelta(minutes=5)])
                    }
                return {'count': 0}
            
            return {key: len(values) for key, values in self._metrics.items()}

# Global instances
image_cache = ImageCache()
connection_pool = ConnectionPool()
performance_monitor = PerformanceMonitor()

def timed_operation(operation_name: str):
    """Decorator to time operations and record metrics"""
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                performance_monitor.record_timing(operation_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                performance_monitor.record_timing(
                    f"{operation_name}_error", 
                    duration, 
                    {'error': str(e)}
                )
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                performance_monitor.record_timing(operation_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                performance_monitor.record_timing(
                    f"{operation_name}_error", 
                    duration, 
                    {'error': str(e)}
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

@lru_cache(maxsize=128)
def get_file_hash(file_path: str) -> str:
    """Get cached file hash"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return ""

async def stream_large_file(file_path: str, chunk_size: int = 8192):
    """Stream large files asynchronously"""
    try:
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(chunk_size):
                yield chunk
    except Exception as e:
        raise ProcessingError(f"Failed to stream file {file_path}: {e}")

def memory_efficient_b64encode(data: bytes, chunk_size: int = 57) -> str:
    """Memory-efficient base64 encoding for large data"""
    import base64
    
    if len(data) <= chunk_size * 100:  # Small data, encode normally
        return base64.b64encode(data).decode('utf-8')
    
    # Large data, encode in chunks
    result = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        result.append(base64.b64encode(chunk).decode('utf-8'))
    
    return ''.join(result)

def schedule_cleanup():
    """Schedule periodic cleanup operations"""
    def cleanup_task():
        try:
            # Cleanup expired cache entries
            image_cache.cleanup_expired()
            
            # Force garbage collection
            gc.collect()
            
            # Schedule next cleanup
            threading.Timer(300, cleanup_task).start()  # Every 5 minutes
        except Exception as e:
            print(f"Cleanup task failed: {e}")
    
    # Start cleanup timer
    threading.Timer(300, cleanup_task).start()

# Initialize cleanup on module import
schedule_cleanup()

# Export public interface
__all__ = [
    'TTLCache', 'ImageCache', 'ConnectionPool', 'PerformanceMonitor',
    'image_cache', 'connection_pool', 'performance_monitor',
    'timed_operation', 'get_file_hash', 'stream_large_file',
    'memory_efficient_b64encode', 'schedule_cleanup'
]