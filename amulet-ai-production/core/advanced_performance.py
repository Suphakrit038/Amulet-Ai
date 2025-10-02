#!/usr/bin/env python3
"""
⚡ Advanced Performance Optimization Module
เพิ่มประสิทธิภาพระบบด้วย caching, compression, และ monitoring
"""

import os
import time
import asyncio
import hashlib
import pickle
import gzip
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json

import psutil
import numpy as np
from PIL import Image
import io

class AdvancedCacheManager:
    """Advanced caching system with TTL, compression, and analytics"""
    
    def __init__(self, max_memory_mb: int = 100, default_ttl: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.compression_enabled = True
        
    def _get_cache_key(self, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, str):
            return hashlib.md5(data.encode()).hexdigest()
        elif isinstance(data, (bytes, bytearray)):
            return hashlib.md5(data).hexdigest()
        else:
            return hashlib.md5(str(data).encode()).hexdigest()
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data using gzip"""
        if self.compression_enabled:
            serialized = pickle.dumps(data)
            return gzip.compress(serialized)
        else:
            return pickle.dumps(data)
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data"""
        if self.compression_enabled:
            try:
                decompressed = gzip.decompress(compressed_data)
                return pickle.loads(decompressed)
            except:
                # Fallback for non-compressed data
                return pickle.loads(compressed_data)
        else:
            return pickle.loads(compressed_data)
    
    def _cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, (data, expiry_time) in self.cache.items():
            if expiry_time < current_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
    
    def _check_memory_limit(self):
        """Check and enforce memory limits"""
        # Estimate memory usage
        total_size = sum(len(data) for data, _ in self.cache.values())
        
        if total_size > self.max_memory_bytes:
            # Remove least recently used items
            sorted_keys = sorted(
                self.access_times.items(),
                key=lambda x: x[1]
            )
            
            # Remove oldest 25% of items
            remove_count = max(1, len(sorted_keys) // 4)
            for key, _ in sorted_keys[:remove_count]:
                if key in self.cache:
                    del self.cache[key]
                del self.access_times[key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        self._cleanup_expired()
        
        if key in self.cache:
            data, expiry_time = self.cache[key]
            if expiry_time > time.time():
                self.access_times[key] = time.time()
                self.hit_count += 1
                return self._decompress_data(data)
            else:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
        
        self.miss_count += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache"""
        try:
            compressed_data = self._compress_data(value)
            expiry_time = time.time() + (ttl or self.default_ttl)
            
            self.cache[key] = (compressed_data, expiry_time)
            self.access_times[key] = time.time()
            
            self._check_memory_limit()
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    def invalidate(self, pattern: str = None):
        """Invalidate cache entries"""
        if pattern:
            # Remove keys matching pattern
            keys_to_remove = [key for key in self.cache.keys() if pattern in key]
            for key in keys_to_remove:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
        else:
            # Clear all cache
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        cache_size = len(self.cache)
        memory_usage = sum(len(data) for data, _ in self.cache.values())
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': cache_size,
            'memory_usage_bytes': memory_usage,
            'memory_usage_mb': memory_usage / (1024 * 1024),
            'compression_enabled': self.compression_enabled
        }

class ImageOptimizer:
    """Advanced image optimization for performance"""
    
    def __init__(self):
        self.webp_quality = 85
        self.max_dimension = 1024
        self.cache = AdvancedCacheManager(max_memory_mb=50)
    
    def optimize_image(self, image_data: bytes, format: str = 'WEBP') -> bytes:
        """Optimize image for web delivery"""
        cache_key = hashlib.md5(image_data).hexdigest()
        cached_result = self.cache.get(f"optimized_{cache_key}_{format}")
        
        if cached_result:
            return cached_result
        
        try:
            # Open image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            # Resize if too large
            if max(image.size) > self.max_dimension:
                ratio = self.max_dimension / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save optimized
            output = io.BytesIO()
            if format.upper() == 'WEBP':
                image.save(output, format='WEBP', quality=self.webp_quality, optimize=True)
            elif format.upper() == 'JPEG':
                image.save(output, format='JPEG', quality=self.webp_quality, optimize=True)
            else:
                image.save(output, format=format, optimize=True)
            
            optimized_data = output.getvalue()
            
            # Cache result
            self.cache.set(f"optimized_{cache_key}_{format}", optimized_data, ttl=3600)
            
            return optimized_data
            
        except Exception as e:
            print(f"Image optimization error: {e}")
            return image_data
    
    def get_image_info(self, image_data: bytes) -> Dict[str, Any]:
        """Get image information"""
        try:
            image = Image.open(io.BytesIO(image_data))
            return {
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'width': image.size[0],
                'height': image.size[1],
                'file_size_bytes': len(image_data),
                'file_size_mb': len(image_data) / (1024 * 1024)
            }
        except Exception as e:
            return {'error': str(e)}

class SystemMonitor:
    """Advanced system monitoring and analytics"""
    
    def __init__(self):
        self.metrics_history = []
        self.alerts = []
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'response_time': 2.0
        }
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': psutil.virtual_memory()._asdict(),
            'disk': psutil.disk_usage('.')._asdict(),
            'network': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {},
            'process_count': len(psutil.pids()),
            'boot_time': psutil.boot_time()
        }
        
        # Calculate derived metrics
        metrics['memory_percent'] = metrics['memory']['percent']
        metrics['disk_percent'] = (metrics['disk']['used'] / metrics['disk']['total']) * 100
        metrics['memory_available_gb'] = metrics['memory']['available'] / (1024**3)
        metrics['disk_free_gb'] = metrics['disk']['free'] / (1024**3)
        
        # Store in history
        self.metrics_history.append(metrics)
        
        # Keep only last 100 measurements
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        return metrics
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for alert conditions"""
        alerts = []
        
        if metrics['cpu_percent'] > self.thresholds['cpu_percent']:
            alerts.append({
                'type': 'high_cpu',
                'message': f"High CPU usage: {metrics['cpu_percent']:.1f}%",
                'severity': 'warning',
                'timestamp': metrics['timestamp']
            })
        
        if metrics['memory_percent'] > self.thresholds['memory_percent']:
            alerts.append({
                'type': 'high_memory',
                'message': f"High memory usage: {metrics['memory_percent']:.1f}%",
                'severity': 'warning',
                'timestamp': metrics['timestamp']
            })
        
        if metrics['disk_percent'] > self.thresholds['disk_percent']:
            alerts.append({
                'type': 'high_disk',
                'message': f"High disk usage: {metrics['disk_percent']:.1f}%",
                'severity': 'critical',
                'timestamp': metrics['timestamp']
            })
        
        # Add new alerts
        self.alerts.extend(alerts)
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics_history:
            return {'error': 'No metrics available'}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        cpu_values = [m['cpu_percent'] for m in recent_metrics]
        memory_values = [m['memory_percent'] for m in recent_metrics]
        
        return {
            'current_metrics': self.metrics_history[-1] if self.metrics_history else {},
            'averages': {
                'cpu_percent': sum(cpu_values) / len(cpu_values),
                'memory_percent': sum(memory_values) / len(memory_values)
            },
            'peaks': {
                'cpu_percent': max(cpu_values),
                'memory_percent': max(memory_values)
            },
            'alerts_count': len(self.alerts),
            'active_alerts': [alert for alert in self.alerts if alert['severity'] == 'critical'],
            'health_score': self._calculate_health_score()
        }
    
    def _calculate_health_score(self) -> float:
        """Calculate system health score (0-100)"""
        if not self.metrics_history:
            return 50.0
        
        latest = self.metrics_history[-1]
        
        # Score components (0-100 each)
        cpu_score = max(0, 100 - latest['cpu_percent'])
        memory_score = max(0, 100 - latest['memory_percent'])
        disk_score = max(0, 100 - latest['disk_percent'])
        
        # Weighted average
        health_score = (cpu_score * 0.3 + memory_score * 0.4 + disk_score * 0.3)
        
        # Penalty for active critical alerts
        critical_alerts = len([a for a in self.alerts if a['severity'] == 'critical'])
        health_score -= (critical_alerts * 10)
        
        return max(0, min(100, health_score))

class PerformanceOptimizer:
    """Main performance optimization coordinator"""
    
    def __init__(self):
        self.cache_manager = AdvancedCacheManager()
        self.image_optimizer = ImageOptimizer()
        self.system_monitor = SystemMonitor()
        self.optimization_enabled = True
        
    async def optimize_api_response(self, data: Any, cache_key: str = None) -> Any:
        """Optimize API response with caching"""
        if not self.optimization_enabled:
            return data
        
        if cache_key:
            # Try to get from cache
            cached_data = self.cache_manager.get(cache_key)
            if cached_data:
                return cached_data
            
            # Cache the response
            self.cache_manager.set(cache_key, data)
        
        return data
    
    async def optimize_image_upload(self, image_data: bytes) -> bytes:
        """Optimize uploaded image"""
        if not self.optimization_enabled:
            return image_data
        
        return self.image_optimizer.optimize_image(image_data)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        return {
            'cache_stats': self.cache_manager.get_stats(),
            'image_optimization': {
                'cache_stats': self.image_optimizer.cache.get_stats(),
                'webp_quality': self.image_optimizer.webp_quality,
                'max_dimension': self.image_optimizer.max_dimension
            },
            'system_performance': self.system_monitor.get_performance_summary(),
            'optimization_enabled': self.optimization_enabled
        }
    
    def start_monitoring(self, interval: int = 60):
        """Start background system monitoring"""
        async def monitor_loop():
            while True:
                try:
                    self.system_monitor.collect_metrics()
                    await asyncio.sleep(interval)
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    await asyncio.sleep(interval)
        
        asyncio.create_task(monitor_loop())

# Global optimizer instance
global_optimizer = PerformanceOptimizer()

# Utility functions for easy use
def get_cache_manager() -> AdvancedCacheManager:
    """Get global cache manager"""
    return global_optimizer.cache_manager

def get_image_optimizer() -> ImageOptimizer:
    """Get global image optimizer"""
    return global_optimizer.image_optimizer

def get_system_monitor() -> SystemMonitor:
    """Get global system monitor"""
    return global_optimizer.system_monitor

def get_performance_stats() -> Dict[str, Any]:
    """Get all performance statistics"""
    return global_optimizer.get_optimization_stats()

# Example usage and testing
async def test_performance_optimization():
    """Test performance optimization features"""
    print("🧪 Testing Performance Optimization...")
    
    # Test caching
    cache = get_cache_manager()
    cache.set("test_key", {"message": "Hello World"}, ttl=60)
    cached_result = cache.get("test_key")
    print(f"✅ Cache test: {cached_result}")
    
    # Test image optimization
    img_opt = get_image_optimizer()
    # Create a simple test image
    test_image = Image.new('RGB', (800, 600), color='red')
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format='PNG')
    img_data = img_bytes.getvalue()
    
    optimized = img_opt.optimize_image(img_data, 'WEBP')
    print(f"✅ Image optimization: {len(img_data)} -> {len(optimized)} bytes")
    
    # Test system monitoring
    monitor = get_system_monitor()
    metrics = monitor.collect_metrics()
    print(f"✅ System monitoring: CPU {metrics['cpu_percent']:.1f}%")
    
    # Get performance stats
    stats = get_performance_stats()
    print(f"✅ Performance stats collected: {len(stats)} categories")

if __name__ == "__main__":
    asyncio.run(test_performance_optimization())