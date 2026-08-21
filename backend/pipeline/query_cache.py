import threading
import time
import hashlib
import json
from collections import OrderedDict
from typing import Any, Optional, Dict, Tuple
from backend.config import settings

class LRUCache:
    def __init__(self, max_size: int, ttl_seconds: int):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.lock = threading.RLock()
        
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0
        self.expirations = 0
        
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if self.ttl_seconds > 0 and time.time() - entry["timestamp"] > self.ttl_seconds:
                    self.expirations += 1
                    self.misses += 1
                    del self.cache[key]
                    return None
                
                self.hits += 1
                self.cache.move_to_end(key)
                return entry["value"]
                
            self.misses += 1
            return None

    def set(self, key: str, value: Any):
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

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self.sets = 0
            self.evictions = 0
            self.expirations = 0
            
    def delete(self, key: str):
        with self.lock:
            if key in self.cache:
                del self.cache[key]

    def stats(self) -> Dict[str, Any]:
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "sets": self.sets,
                "evictions": self.evictions,
                "expirations": self.expirations
            }


class TwoLevelCache:
    def __init__(self):
        self.retrieval_enabled = settings.RETRIEVAL_CACHE_ENABLED
        self.response_enabled = settings.RESPONSE_CACHE_ENABLED
        self.schema_version = settings.CACHE_SCHEMA_VERSION
        
        self.retrieval_cache = LRUCache(settings.RETRIEVAL_CACHE_MAX_SIZE, settings.RETRIEVAL_CACHE_TTL_SECONDS)
        self.response_cache = LRUCache(settings.RESPONSE_CACHE_MAX_SIZE, settings.RESPONSE_CACHE_TTL_SECONDS)
        
        # For request coalescing (single-flight)
        self.in_progress_events: Dict[str, threading.Event] = {}
        self.coalesce_lock = threading.Lock()
        self.coalesce_timeout = settings.SLM_COALESCE_TIMEOUT_SECONDS
        
        # SLM call counter for instrumentation
        self._slm_call_count = 0
        self._slm_call_lock = threading.Lock()

    def increment_slm_calls(self):
        with self._slm_call_lock:
            self._slm_call_count += 1
    
    @property
    def slm_call_count(self) -> int:
        with self._slm_call_lock:
            return self._slm_call_count

    def reset_slm_call_count(self):
        with self._slm_call_lock:
            self._slm_call_count = 0

    @staticmethod
    def _normalize_for_key(query: str) -> str:
        """Lowercase normalization for cache key consistency.
        NFC normalization is done in preprocess_query; this adds
        case-insensitive matching without affecting retrieval."""
        return query.lower()

    # --- RETRIEVAL CACHE ---
    def _generate_retrieval_key(self, normalized_query: str, language: str, top_k: int) -> str:
        key_data = {
            "q": self._normalize_for_key(normalized_query),
            "l": language,
            "k": top_k,
            "v": self.schema_version
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()

    def get_retrieval(self, normalized_query: str, language: str, top_k: int) -> Optional[Any]:
        if not self.retrieval_enabled:
            return None
        key = self._generate_retrieval_key(normalized_query, language, top_k)
        return self.retrieval_cache.get(key)
        
    def set_retrieval(self, normalized_query: str, language: str, top_k: int, value: Any):
        if not self.retrieval_enabled:
            return
        key = self._generate_retrieval_key(normalized_query, language, top_k)
        self.retrieval_cache.set(key, value)
        
    # --- Backward compatibility aliases for retrieval cache ---
    def get(self, normalized_query: str, language: str, top_k: int) -> Optional[Any]:
        return self.get_retrieval(normalized_query, language, top_k)
        
    def set(self, normalized_query: str, language: str, top_k: int, value: Any):
        return self.set_retrieval(normalized_query, language, top_k, value)


    # --- RESPONSE CACHE ---
    def generate_response_key(self, normalized_query: str, language: str, top_k: int, 
                              generate: bool, slm_provider: str, slm_model: str,
                              prompt_version: str, grounding_version: str, artifact_version: str) -> str:
        key_data = {
            "q": self._normalize_for_key(normalized_query),
            "l": language,
            "k": top_k,
            "g": generate,
            "p": slm_provider,
            "m": slm_model,
            "pv": prompt_version,
            "gv": grounding_version,
            "a": artifact_version,
            "v": self.schema_version
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()

    def get_or_wait_response(self, key: str) -> Tuple[Optional[Any], bool]:
        """
        Returns (cached_value, should_compute)
        - If value is in cache, returns (value, False)
        - If value is not in cache and no one is computing, returns (None, True).
          Caller MUST call set_response or release_response_lock later.
        - If someone else is computing, waits for them (with timeout), then returns
          (value, False) or (None, True) if the computation failed/timed out.
        """
        if not self.response_enabled:
            return None, True
            
        with self.coalesce_lock:
            # 1. Check cache first
            val = self.response_cache.get(key)
            if val is not None:
                return val, False
                
            # 2. Check if another thread is computing it
            if key in self.in_progress_events:
                event = self.in_progress_events[key]
                should_wait = True
            else:
                # 3. We are the first, mark as computing
                self.in_progress_events[key] = threading.Event()
                return None, True
                
        # Wait for the other thread to finish (with timeout to prevent deadlock)
        if should_wait:
            event.wait(timeout=self.coalesce_timeout)
            # Try getting from cache again
            val = self.response_cache.get(key)
            if val is not None:
                return val, False
            # Timeout or computation failed — caller should compute independently
            return None, True

    def set_response(self, key: str, value: Any):
        if not self.response_enabled:
            self.release_response_lock(key)
            return
            
        self.response_cache.set(key, value)
        self.release_response_lock(key)
        
    def release_response_lock(self, key: str):
        """Release the lock/event for a key if computation failed or finished."""
        with self.coalesce_lock:
            if key in self.in_progress_events:
                event = self.in_progress_events.pop(key)
                event.set()

    # --- GENERAL ---
    def clear(self):
        self.retrieval_cache.clear()
        self.response_cache.clear()
        self.reset_slm_call_count()

    def stats(self) -> Dict[str, Any]:
        ret_stats = self.retrieval_cache.stats()
        ret_stats["enabled"] = self.retrieval_enabled
        resp_stats = self.response_cache.stats()
        resp_stats["enabled"] = self.response_enabled
        
        return {
            "enabled": self.retrieval_enabled, # For backward compatibility
            "retrieval": ret_stats,
            "response": resp_stats,
            "slm_call_count": self.slm_call_count,
            # Flattened retrieval stats for backward compatibility with benchmark_cache.py
            "size": ret_stats["size"],
            "max_size": ret_stats["max_size"],
            "hits": ret_stats["hits"],
            "misses": ret_stats["misses"],
            "hit_rate": ret_stats["hit_rate"],
            "sets": ret_stats["sets"],
            "evictions": ret_stats["evictions"],
            "expirations": ret_stats["expirations"]
        }

# Global singleton instance
cache_instance = TwoLevelCache()
