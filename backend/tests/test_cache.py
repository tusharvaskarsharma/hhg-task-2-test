import pytest
import time
from backend.pipeline.query_cache import QueryCache
from backend.config import settings

def test_cache_set_get():
    cache = QueryCache(max_size=10, ttl_seconds=3600)
    cache.set("q1", "hi", 10, [{"id": "doc1"}])
    
    val = cache.get("q1", "hi", 10)
    assert val == [{"id": "doc1"}]
    
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0

def test_cache_miss():
    cache = QueryCache(max_size=10, ttl_seconds=3600)
    val = cache.get("missing", "hi", 10)
    assert val is None
    
    stats = cache.stats()
    assert stats["misses"] == 1

def test_cache_ttl():
    cache = QueryCache(max_size=10, ttl_seconds=1)
    cache.set("q1", "hi", 10, "val")
    
    # modify internal timestamp to simulate expiration
    # Need to generate key to modify it
    key = cache._generate_key("q1", "hi", 10)
    cache.cache[key]["timestamp"] = time.time() - 2.0
    
    val = cache.get("q1", "hi", 10)
    assert val is None
    
    stats = cache.stats()
    assert stats["expirations"] == 1
    assert stats["misses"] == 1

def test_cache_ttl_disabled():
    cache = QueryCache(max_size=10, ttl_seconds=0)
    cache.set("q1", "hi", 10, "val")
    
    key = cache._generate_key("q1", "hi", 10)
    cache.cache[key]["timestamp"] = time.time() - 10000.0
    
    val = cache.get("q1", "hi", 10)
    assert val == "val"

def test_cache_lru_eviction():
    cache = QueryCache(max_size=2, ttl_seconds=3600)
    cache.set("q1", "hi", 10, "v1")
    cache.set("q2", "hi", 10, "v2")
    cache.set("q3", "hi", 10, "v3") # should evict q1
    
    assert cache.get("q1", "hi", 10) is None
    assert cache.get("q2", "hi", 10) == "v2"
    assert cache.get("q3", "hi", 10) == "v3"
    
    stats = cache.stats()
    assert stats["evictions"] == 1

def test_cache_clear_delete():
    cache = QueryCache(max_size=10, ttl_seconds=3600)
    cache.set("q1", "hi", 10, "v1")
    key = cache._generate_key("q1", "hi", 10)
    
    cache.delete(key)
    assert cache.get("q1", "hi", 10) is None
    
    cache.set("q1", "hi", 10, "v1")
    cache.clear()
    assert cache.get("q1", "hi", 10) is None
    assert cache.stats()["size"] == 0

def test_cache_disabled(monkeypatch):
    monkeypatch.setattr(settings, "CACHE_ENABLED", False)
    
    cache = QueryCache(max_size=10, ttl_seconds=3600)
    cache.set("q1", "hi", 10, "v1")
    
    val = cache.get("q1", "hi", 10)
    assert val is None
    
    stats = cache.stats()
    assert stats["enabled"] is False

def test_deterministic_keys():
    cache = QueryCache(max_size=10)
    k1 = cache._generate_key("hello", "hi", 10)
    k2 = cache._generate_key("hello", "en", 10)
    k3 = cache._generate_key("hello", "hi", 5)
    
    assert k1 != k2
    assert k1 != k3
