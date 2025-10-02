#!/usr/bin/env python3
"""
Rate limiting utilities for Amulet-AI
In-memory rate limiting with configurable limits
"""

import time
import threading
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from .config import config

@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests: int
    window: int  # seconds
    
class InMemoryRateLimiter:
    """Thread-safe in-memory rate limiter"""
    
    def __init__(self):
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = threading.RLock()
        
        # Rate limits for different endpoints
        self.limits = {
            'classify': RateLimit(requests=config.RATE_LIMIT_REQUESTS, 
                                window=config.RATE_LIMIT_WINDOW),
            'health': RateLimit(requests=100, window=60),
            'default': RateLimit(requests=30, window=60)
        }
    
    def is_allowed(self, key: str, endpoint: str = 'default') -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is allowed under rate limit
        Returns (is_allowed, rate_limit_info)
        """
        with self._lock:
            now = time.time()
            limit = self.limits.get(endpoint, self.limits['default'])
            
            # Clean old requests
            self._requests[key] = [
                req_time for req_time in self._requests[key]
                if now - req_time < limit.window
            ]
            
            current_requests = len(self._requests[key])
            remaining = max(0, limit.requests - current_requests)
            
            if current_requests >= limit.requests:
                # Rate limit exceeded
                oldest_request = min(self._requests[key]) if self._requests[key] else now
                reset_time = int(oldest_request + limit.window)
                
                return False, {
                    'limit': limit.requests,
                    'remaining': 0,
                    'reset': reset_time,
                    'retry_after': int(reset_time - now)
                }
            
            # Allow request
            self._requests[key].append(now)
            reset_time = int(now + limit.window)
            
            return True, {
                'limit': limit.requests,
                'remaining': remaining - 1,
                'reset': reset_time,
                'retry_after': 0
            }
    
    def clear_expired(self, max_age: int = 3600):
        """Clear expired rate limit entries"""
        with self._lock:
            now = time.time()
            expired_keys = []
            
            for key, requests in self._requests.items():
                # Remove requests older than max_age
                recent_requests = [
                    req_time for req_time in requests
                    if now - req_time < max_age
                ]
                
                if not recent_requests:
                    expired_keys.append(key)
                else:
                    self._requests[key] = recent_requests
            
            # Remove empty entries
            for key in expired_keys:
                del self._requests[key]
    
    def get_stats(self) -> Dict[str, int]:
        """Get rate limiter statistics"""
        with self._lock:
            return {
                'active_keys': len(self._requests),
                'total_requests': sum(len(requests) for requests in self._requests.values())
            }

# Global rate limiter instance
rate_limiter = InMemoryRateLimiter()

def get_client_id(request) -> str:
    """Extract client identifier from request"""
    # Try to get real IP from headers (for reverse proxy setups)
    forwarded_for = getattr(request, 'headers', {}).get('x-forwarded-for')
    if forwarded_for:
        # Take the first IP from the list
        client_ip = forwarded_for.split(',')[0].strip()
    else:
        # Fallback to direct client IP
        client_ip = getattr(request, 'client', {}).get('host', 'unknown')
    
    return f"ip:{client_ip}"

def apply_rate_limit(client_id: str, endpoint: str = 'default') -> Tuple[bool, Dict[str, int]]:
    """Apply rate limiting to a request"""
    return rate_limiter.is_allowed(client_id, endpoint)