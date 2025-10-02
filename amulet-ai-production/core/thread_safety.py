#!/usr/bin/env python3
"""
Thread Safety Module for Amulet-AI
Thread-safe data structures, locks, and atomic operations
"""

import threading
import queue
import time
import weakref
from typing import Any, Dict, Optional, Callable, TypeVar, Generic, List
from datetime import datetime
from collections import defaultdict, deque
from contextlib import contextmanager
import concurrent.futures
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from .error_handling import AmuletError, error_logger

T = TypeVar('T')

class LockType(Enum):
    """Types of locks available"""
    LOCK = "lock"
    RLOCK = "rlock"
    SEMAPHORE = "semaphore"
    CONDITION = "condition"

@dataclass
class ThreadSafeStats:
    """Thread-safe statistics container"""
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _counters: Dict[str, int] = field(default_factory=dict)
    _timings: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    def increment(self, key: str, value: int = 1):
        """Atomically increment a counter"""
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value
    
    def record_timing(self, operation: str, duration: float):
        """Record timing for an operation"""
        with self._lock:
            self._timings[operation].append(duration)
            # Keep only recent timings
            if len(self._timings[operation]) > 1000:
                self._timings[operation] = self._timings[operation][-500:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self._lock:
            stats = {"counters": self._counters.copy()}
            
            for operation, timings in self._timings.items():
                if timings:
                    stats[f"{operation}_avg"] = sum(timings) / len(timings)
                    stats[f"{operation}_count"] = len(timings)
            
            return stats

class ThreadSafeDict(Generic[T]):
    """Thread-safe dictionary with additional features"""
    
    def __init__(self):
        self._data: Dict[str, T] = {}
        self._lock = threading.RLock()
        self._access_times: Dict[str, datetime] = {}
    
    def get(self, key: str, default: T = None) -> T:
        """Get value with thread safety"""
        with self._lock:
            self._access_times[key] = datetime.now()
            return self._data.get(key, default)
    
    def set(self, key: str, value: T) -> None:
        """Set value with thread safety"""
        with self._lock:
            self._data[key] = value
            self._access_times[key] = datetime.now()
    
    def update(self, other: Dict[str, T]) -> None:
        """Update with multiple values atomically"""
        with self._lock:
            now = datetime.now()
            self._data.update(other)
            for key in other.keys():
                self._access_times[key] = now
    
    def pop(self, key: str, default: T = None) -> T:
        """Remove and return value"""
        with self._lock:
            self._access_times.pop(key, None)
            return self._data.pop(key, default)
    
    def keys(self) -> List[str]:
        """Get all keys"""
        with self._lock:
            return list(self._data.keys())
    
    def values(self) -> List[T]:
        """Get all values"""
        with self._lock:
            return list(self._data.values())
    
    def items(self) -> List[tuple]:
        """Get all items"""
        with self._lock:
            return list(self._data.items())
    
    def size(self) -> int:
        """Get size of dictionary"""
        with self._lock:
            return len(self._data)
    
    def clear(self) -> None:
        """Clear all data"""
        with self._lock:
            self._data.clear()
            self._access_times.clear()
    
    def cleanup_old(self, max_age_seconds: int = 3600) -> int:
        """Remove entries older than max_age_seconds"""
        with self._lock:
            now = datetime.now()
            old_keys = []
            
            for key, access_time in self._access_times.items():
                if (now - access_time).total_seconds() > max_age_seconds:
                    old_keys.append(key)
            
            for key in old_keys:
                self._data.pop(key, None)
                self._access_times.pop(key, None)
            
            return len(old_keys)

class ThreadSafeQueue(Generic[T]):
    """Thread-safe queue with additional features"""
    
    def __init__(self, maxsize: int = 0):
        self._queue = queue.Queue(maxsize=maxsize)
        self._stats = ThreadSafeStats()
    
    def put(self, item: T, timeout: Optional[float] = None) -> bool:
        """Put item in queue"""
        try:
            self._queue.put(item, timeout=timeout)
            self._stats.increment("items_added")
            return True
        except queue.Full:
            self._stats.increment("put_timeouts")
            return False
    
    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """Get item from queue"""
        try:
            item = self._queue.get(timeout=timeout)
            self._stats.increment("items_retrieved")
            return item
        except queue.Empty:
            self._stats.increment("get_timeouts")
            return None
    
    def size(self) -> int:
        """Get queue size"""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self._queue.empty()
    
    def full(self) -> bool:
        """Check if queue is full"""
        return self._queue.full()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        stats = self._stats.get_stats()
        stats.update({
            "current_size": self.size(),
            "is_empty": self.empty(),
            "is_full": self.full()
        })
        return stats

class AtomicCounter:
    """Thread-safe atomic counter"""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self, step: int = 1) -> int:
        """Atomically increment and return new value"""
        with self._lock:
            self._value += step
            return self._value
    
    def decrement(self, step: int = 1) -> int:
        """Atomically decrement and return new value"""
        with self._lock:
            self._value -= step
            return self._value
    
    def get(self) -> int:
        """Get current value"""
        with self._lock:
            return self._value
    
    def set(self, value: int) -> int:
        """Set value and return previous value"""
        with self._lock:
            old_value = self._value
            self._value = value
            return old_value
    
    def compare_and_swap(self, expected: int, new_value: int) -> bool:
        """Compare and swap operation"""
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False

class LockManager:
    """Manage different types of locks"""
    
    def __init__(self):
        self._locks: Dict[str, Any] = {}
        self._lock_types: Dict[str, LockType] = {}
        self._manager_lock = threading.RLock()
    
    def get_lock(self, name: str, lock_type: LockType = LockType.LOCK, **kwargs) -> Any:
        """Get or create a named lock"""
        with self._manager_lock:
            if name not in self._locks:
                if lock_type == LockType.LOCK:
                    self._locks[name] = threading.Lock()
                elif lock_type == LockType.RLOCK:
                    self._locks[name] = threading.RLock()
                elif lock_type == LockType.SEMAPHORE:
                    value = kwargs.get('value', 1)
                    self._locks[name] = threading.Semaphore(value)
                elif lock_type == LockType.CONDITION:
                    lock = kwargs.get('lock', threading.RLock())
                    self._locks[name] = threading.Condition(lock)
                
                self._lock_types[name] = lock_type
            
            return self._locks[name]
    
    @contextmanager
    def acquire_lock(self, name: str, timeout: Optional[float] = None):
        """Context manager for acquiring locks"""
        lock = self.get_lock(name)
        acquired = False
        
        try:
            if hasattr(lock, 'acquire'):
                acquired = lock.acquire(timeout=timeout) if timeout else lock.acquire()
            else:
                # For semaphores and other lock types
                acquired = lock.acquire(timeout=timeout) if timeout else lock.acquire()
            
            if not acquired:
                raise AmuletError(f"Failed to acquire lock '{name}' within timeout")
            
            yield lock
            
        finally:
            if acquired and hasattr(lock, 'release'):
                lock.release()
    
    def cleanup_unused_locks(self, max_age_seconds: int = 3600):
        """Remove unused locks (simple cleanup)"""
        # This is a simplified cleanup - in production, you'd track usage times
        pass

class ThreadPoolManager:
    """Manage thread pools for different operations"""
    
    def __init__(self):
        self._pools: Dict[str, concurrent.futures.ThreadPoolExecutor] = {}
        self._pool_stats = ThreadSafeStats()
        self._manager_lock = threading.RLock()
    
    def get_pool(self, name: str, max_workers: int = None) -> concurrent.futures.ThreadPoolExecutor:
        """Get or create a named thread pool"""
        with self._manager_lock:
            if name not in self._pools:
                max_workers = max_workers or 4
                self._pools[name] = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix=f"amulet_{name}"
                )
            
            return self._pools[name]
    
    def submit_task(self, pool_name: str, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit task to named pool"""
        pool = self.get_pool(pool_name)
        self._pool_stats.increment(f"{pool_name}_tasks_submitted")
        
        def wrapped_func(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                self._pool_stats.record_timing(f"{pool_name}_task", duration)
                self._pool_stats.increment(f"{pool_name}_tasks_completed")
                return result
            except Exception as e:
                self._pool_stats.increment(f"{pool_name}_tasks_failed")
                error_logger.log_error(e, context={
                    "pool": pool_name,
                    "function": func.__name__
                })
                raise
        
        return pool.submit(wrapped_func, *args, **kwargs)
    
    def shutdown_pool(self, name: str, wait: bool = True):
        """Shutdown a named pool"""
        with self._manager_lock:
            if name in self._pools:
                pool = self._pools.pop(name)
                pool.shutdown(wait=wait)
    
    def shutdown_all(self, wait: bool = True):
        """Shutdown all pools"""
        with self._manager_lock:
            for name in list(self._pools.keys()):
                self.shutdown_pool(name, wait)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread pool statistics"""
        return self._pool_stats.get_stats()

class AsyncSafeDataStructures:
    """Async-safe data structures"""
    
    @staticmethod
    def create_async_dict() -> Dict[str, Any]:
        """Create async-safe dictionary"""
        return {}  # In Python, dict operations are atomic for single items
    
    @staticmethod
    def create_async_queue(maxsize: int = 0) -> asyncio.Queue:
        """Create async-safe queue"""
        return asyncio.Queue(maxsize=maxsize)
    
    @staticmethod
    def create_async_lock() -> asyncio.Lock:
        """Create async lock"""
        return asyncio.Lock()
    
    @staticmethod
    def create_async_semaphore(value: int = 1) -> asyncio.Semaphore:
        """Create async semaphore"""
        return asyncio.Semaphore(value)

# Global instances
global_stats = ThreadSafeStats()
lock_manager = LockManager()
thread_pool_manager = ThreadPoolManager()

def thread_safe_operation(lock_name: str = None, timeout: float = None):
    """Decorator for thread-safe operations"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            actual_lock_name = lock_name or f"func_{func.__name__}"
            
            try:
                with lock_manager.acquire_lock(actual_lock_name, timeout=timeout):
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    global_stats.record_timing(func.__name__, duration)
                    global_stats.increment(f"{func.__name__}_calls")
                    
                    return result
            except Exception as e:
                global_stats.increment(f"{func.__name__}_errors")
                raise
        
        return wrapper
    return decorator

def atomic_operation(counter_name: str = None):
    """Decorator for atomic counter operations"""
    def decorator(func: Callable) -> Callable:
        counter = AtomicCounter()
        
        def wrapper(*args, **kwargs):
            call_id = counter.increment()
            try:
                result = func(*args, **kwargs)
                global_stats.increment(f"{func.__name__}_completed")
                return result
            except Exception as e:
                global_stats.increment(f"{func.__name__}_failed")
                raise
        
        return wrapper
    return decorator

# Cleanup function
def cleanup_thread_resources():
    """Clean up thread-related resources"""
    try:
        thread_pool_manager.shutdown_all(wait=True)
        lock_manager.cleanup_unused_locks()
    except Exception as e:
        error_logger.log_error(e, context={"operation": "thread_cleanup"})

# Export public interface
__all__ = [
    'LockType', 'ThreadSafeStats', 'ThreadSafeDict', 'ThreadSafeQueue',
    'AtomicCounter', 'LockManager', 'ThreadPoolManager', 'AsyncSafeDataStructures',
    'global_stats', 'lock_manager', 'thread_pool_manager',
    'thread_safe_operation', 'atomic_operation', 'cleanup_thread_resources'
]