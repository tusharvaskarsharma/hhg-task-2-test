import threading
import time
import hashlib
import json
from collections import OrderedDict
from typing import Any, Optional, Dict
from backend.config import settings

class QueryCache:
    def __init__(self, max_size: int = settings.CACHE_MAX_SIZE, ttl_seconds: int = settings.CACHE_TTL_SECONDS, version: str = settings.CACHE_VERSION):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.version = version
        
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.lock = threading.RLock()
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0
        self.expirations = 0

    def _generate_key(self, normalized_query: str, language: str, top_k: int) -> str:
        key_data = {
            "q": normalized_query,
            "l": language,
            "k": top_k,
            "v": self.version
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()

    def get(self, normalized_query: str, language: str, top_k: int) -> Optional[Any]:
        if not settings.CACHE_ENABLED:
            return None
            
        key = self._generate_key(normalized_query, language, top_k)
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check TTL
                if self.ttl_seconds > 0:
                    if time.time() - entry["timestamp"] > self.ttl_seconds:
                        # Expired
                        self.expirations += 1
                        self.misses += 1
                        del self.cache[key]
                        return None
                        
                # Hit
                self.hits += 1
                self.cache.move_to_end(key) # LRU update
                return entry["value"]
                
            # Miss
            self.misses += 1
            return None

    def set(self, normalized_query: str, language: str, top_k: int, value: Any):
        if not settings.CACHE_ENABLED:
            return
            
        key = self._generate_key(normalized_query, language, top_k)
        
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            elif len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
                self.evictions += 1
                
            self.cache[key] = {
                "value": value,
                "timestamp": time.time()
            }
            self.sets += 1

    def delete(self, key: str):
        with self.lock:
            if key in self.cache:
                del self.cache[key]

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self.sets = 0
            self.evictions = 0
            self.expirations = 0

    def contains(self, key: str) -> bool:
        with self.lock:
            return key in self.cache

    def stats(self) -> Dict[str, Any]:
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "enabled": settings.CACHE_ENABLED,
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "sets": self.sets,
                "evictions": self.evictions,
                "expirations": self.expirations
            }

# Global singleton instance
cache_instance = QueryCache()
