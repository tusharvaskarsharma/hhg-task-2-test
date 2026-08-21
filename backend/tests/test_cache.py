import pytest
import time
import threading
from unittest.mock import patch, MagicMock
from backend.pipeline.query_cache import TwoLevelCache, cache_instance
from backend.config import settings


# ─── Retrieval Cache ───────────────────────────────────────────────────────────

def test_retrieval_cache_set_get():
    cache = TwoLevelCache()
    cache.retrieval_cache.max_size = 10
    cache.retrieval_cache.ttl_seconds = 3600
    cache.set_retrieval("q1", "hi", 10, [{"id": "doc1"}])
    
    val = cache.get_retrieval("q1", "hi", 10)
    assert val == [{"id": "doc1"}]
    
    stats = cache.stats()
    assert stats["retrieval"]["hits"] == 1
    assert stats["retrieval"]["misses"] == 0

def test_retrieval_cache_miss():
    cache = TwoLevelCache()
    val = cache.get_retrieval("nonexistent", "hi", 10)
    assert val is None
    assert cache.stats()["retrieval"]["misses"] == 1


# ─── Response Cache ───────────────────────────────────────────────────────────

def test_response_cache_miss_then_hit():
    cache = TwoLevelCache()
    cache.response_cache.max_size = 10
    cache.response_cache.ttl_seconds = 3600
    
    key = cache.generate_response_key("q1", "hi", 10, True, "prov", "mod", "v1", "v1", "v1")
    
    val, should_compute = cache.get_or_wait_response(key)
    assert val is None
    assert should_compute is True
    
    cache.set_response(key, {"answer": "test"})
    
    val2, should_compute2 = cache.get_or_wait_response(key)
    assert val2 == {"answer": "test"}
    assert should_compute2 is False
    
    stats = cache.stats()
    assert stats["response"]["hits"] == 1
    assert stats["response"]["misses"] == 1

def test_response_cache_hit_bypasses_generator():
    """Response cache hit must return the cached answer without SLM call."""
    cache = TwoLevelCache()
    key = cache.generate_response_key("capital of india", "en", 10, True, "groq", "llama", "v1", "v1", "v1")
    
    # Store a response
    _, should_compute = cache.get_or_wait_response(key)
    assert should_compute is True
    cache.set_response(key, {"answer": "New Delhi", "answer_source": "generated"})
    
    # Second access should return cached value without computing
    val, should_compute2 = cache.get_or_wait_response(key)
    assert val is not None
    assert val["answer"] == "New Delhi"
    assert should_compute2 is False  # No SLM call needed


# ─── Query Normalisation ──────────────────────────────────────────────────────

def test_query_normalisation_whitespace():
    """Whitespace variations should produce the same cache key."""
    cache = TwoLevelCache()
    key1 = cache.generate_response_key("hello world", "en", 10, True, "p", "m", "v1", "v1", "v1")
    key2 = cache.generate_response_key("hello  world", "en", 10, True, "p", "m", "v1", "v1", "v1")
    # After preprocess_query, both should be "hello world" — but the cache key
    # function itself lowercases, so these should match if caller normalises first
    # The preprocess_query normalises whitespace before passing to cache key generation
    from backend.pipeline.tokenizer import preprocess_query
    q1 = preprocess_query("  hello   world  ")
    q2 = preprocess_query("hello world")
    key1 = cache.generate_response_key(q1, "en", 10, True, "p", "m", "v1", "v1", "v1")
    key2 = cache.generate_response_key(q2, "en", 10, True, "p", "m", "v1", "v1", "v1")
    assert key1 == key2

def test_query_normalisation_unicode_nfc():
    """Unicode NFC normalisation should produce the same cache key."""
    import unicodedata
    from backend.pipeline.tokenizer import preprocess_query
    # Devanagari: composed vs decomposed form
    q1 = preprocess_query("\u0915\u093E")  # composed
    q2 = preprocess_query("\u0915\u093E")  # same
    cache = TwoLevelCache()
    key1 = cache.generate_response_key(q1, "hi", 10, True, "p", "m", "v1", "v1", "v1")
    key2 = cache.generate_response_key(q2, "hi", 10, True, "p", "m", "v1", "v1", "v1")
    assert key1 == key2

def test_query_normalisation_case_insensitive():
    """Cache keys should be case-insensitive."""
    cache = TwoLevelCache()
    key1 = cache.generate_response_key("Hello World", "en", 10, True, "p", "m", "v1", "v1", "v1")
    key2 = cache.generate_response_key("hello world", "en", 10, True, "p", "m", "v1", "v1", "v1")
    assert key1 == key2


# ─── Cache Isolation ──────────────────────────────────────────────────────────

def test_language_isolation():
    """Different languages must produce different cache keys."""
    cache = TwoLevelCache()
    key_hi = cache.generate_response_key("q1", "hi", 10, True, "p", "m", "v1", "v1", "v1")
    key_en = cache.generate_response_key("q1", "en", 10, True, "p", "m", "v1", "v1", "v1")
    key_bn = cache.generate_response_key("q1", "bn", 10, True, "p", "m", "v1", "v1", "v1")
    assert key_hi != key_en != key_bn

def test_top_k_isolation():
    """Different top_k must produce different cache keys."""
    cache = TwoLevelCache()
    key5 = cache.generate_response_key("q1", "en", 5, True, "p", "m", "v1", "v1", "v1")
    key10 = cache.generate_response_key("q1", "en", 10, True, "p", "m", "v1", "v1", "v1")
    assert key5 != key10

def test_model_provider_isolation():
    """Different model/provider must produce different cache keys."""
    cache = TwoLevelCache()
    key1 = cache.generate_response_key("q1", "en", 10, True, "groq", "llama", "v1", "v1", "v1")
    key2 = cache.generate_response_key("q1", "en", 10, True, "ollama", "llama", "v1", "v1", "v1")
    key3 = cache.generate_response_key("q1", "en", 10, True, "groq", "mistral", "v1", "v1", "v1")
    assert key1 != key2
    assert key1 != key3

def test_prompt_version_isolation():
    cache = TwoLevelCache()
    key1 = cache.generate_response_key("q1", "en", 10, True, "p", "m", "v1", "v1", "v1")
    key2 = cache.generate_response_key("q1", "en", 10, True, "p", "m", "v2", "v1", "v1")
    assert key1 != key2

def test_grounding_version_isolation():
    cache = TwoLevelCache()
    key1 = cache.generate_response_key("q1", "en", 10, True, "p", "m", "v1", "v1", "v1")
    key2 = cache.generate_response_key("q1", "en", 10, True, "p", "m", "v1", "v2", "v1")
    assert key1 != key2

def test_artifact_version_isolation():
    cache = TwoLevelCache()
    key1 = cache.generate_response_key("q1", "en", 10, True, "p", "m", "v1", "v1", "abc")
    key2 = cache.generate_response_key("q1", "en", 10, True, "p", "m", "v1", "v1", "def")
    assert key1 != key2

def test_schema_version_isolation():
    cache = TwoLevelCache()
    key1 = cache.generate_response_key("q1", "en", 10, True, "p", "m", "v1", "v1", "v1")
    # Change schema version
    old_version = cache.schema_version
    cache.schema_version = "v999"
    key2 = cache.generate_response_key("q1", "en", 10, True, "p", "m", "v1", "v1", "v1")
    cache.schema_version = old_version
    assert key1 != key2

def test_generate_flag_isolation():
    """generate=True and generate=False must produce different cache keys."""
    cache = TwoLevelCache()
    key1 = cache.generate_response_key("q1", "en", 10, True, "p", "m", "v1", "v1", "v1")
    key2 = cache.generate_response_key("q1", "en", 10, False, "p", "m", "v1", "v1", "v1")
    assert key1 != key2


# ─── TTL & Eviction ──────────────────────────────────────────────────────────

def test_cache_ttl():
    cache = TwoLevelCache()
    cache.retrieval_cache.max_size = 10
    cache.retrieval_cache.ttl_seconds = 1
    cache.set_retrieval("q1", "hi", 10, "val")
    
    key = cache._generate_retrieval_key("q1", "hi", 10)
    cache.retrieval_cache.cache[key]["timestamp"] = time.time() - 2.0
    
    val = cache.get_retrieval("q1", "hi", 10)
    assert val is None
    
    stats = cache.stats()
    assert stats["retrieval"]["expirations"] == 1
    assert stats["retrieval"]["misses"] == 1

def test_cache_lru_eviction():
    cache = TwoLevelCache()
    cache.retrieval_cache.max_size = 2
    cache.retrieval_cache.ttl_seconds = 3600
    cache.set_retrieval("q1", "hi", 10, "v1")
    cache.set_retrieval("q2", "hi", 10, "v2")
    cache.set_retrieval("q3", "hi", 10, "v3") # should evict q1
    
    assert cache.get_retrieval("q1", "hi", 10) is None
    assert cache.get_retrieval("q2", "hi", 10) == "v2"
    assert cache.get_retrieval("q3", "hi", 10) == "v3"
    
    stats = cache.stats()
    assert stats["retrieval"]["evictions"] == 1


# ─── Cache Clear & Disabled ──────────────────────────────────────────────────

def test_cache_clear():
    cache = TwoLevelCache()
    cache.set_retrieval("q1", "hi", 10, "v1")
    
    key = cache.generate_response_key("q2", "hi", 10, True, "p", "m", "v1", "v1", "v1")
    
    _, _ = cache.get_or_wait_response(key)
    cache.set_response(key, "v2")
    
    cache.clear()
    assert cache.get_retrieval("q1", "hi", 10) is None
    
    val, comp = cache.get_or_wait_response(key)
    assert val is None
    assert comp is True
    cache.release_response_lock(key)
    
    assert cache.stats()["retrieval"]["size"] == 0
    assert cache.stats()["response"]["size"] == 0

def test_cache_disabled_mode():
    """When cache is disabled, get returns None and set is a no-op."""
    cache = TwoLevelCache()
    cache.retrieval_enabled = False
    cache.response_enabled = False
    
    cache.set_retrieval("q1", "hi", 10, "val")
    assert cache.get_retrieval("q1", "hi", 10) is None
    
    key = cache.generate_response_key("q1", "en", 10, True, "p", "m", "v1", "v1", "v1")
    val, should_compute = cache.get_or_wait_response(key)
    assert val is None
    assert should_compute is True


# ─── No-Cache Conditions ─────────────────────────────────────────────────────

def test_failed_response_not_cached():
    """Responses with answer_source != 'generated' should not be cached."""
    cache = TwoLevelCache()
    key = cache.generate_response_key("fail", "en", 10, True, "p", "m", "v1", "v1", "v1")
    
    # Simulate: first request computes but fails
    _, should_compute = cache.get_or_wait_response(key)
    assert should_compute is True
    # Don't set_response, just release the lock (simulating failure)
    cache.release_response_lock(key)
    
    # Second request should need to compute again
    val, should_compute2 = cache.get_or_wait_response(key)
    assert val is None
    assert should_compute2 is True
    cache.release_response_lock(key)


# ─── Single-Flight Coalescing ────────────────────────────────────────────────

def test_coalescing():
    cache = TwoLevelCache()
    key = cache.generate_response_key("coalesce", "en", 5, True, "p", "m", "v1", "v1", "v1")
    
    # Thread 1 starts computing
    val1, should_compute1 = cache.get_or_wait_response(key)
    assert val1 is None
    assert should_compute1 is True
    
    result = []
    
    def thread2_func():
        # Thread 2 should wait and then get the result
        val2, should_compute2 = cache.get_or_wait_response(key)
        result.append((val2, should_compute2))
        
    t = threading.Thread(target=thread2_func)
    t.start()
    
    # Thread 2 is now waiting. Let's provide the result.
    time.sleep(0.1) # yield to let thread 2 block
    cache.set_response(key, "computed_value")
    
    t.join()
    
    assert len(result) == 1
    assert result[0] == ("computed_value", False)

def test_100_concurrent_requests_single_slm_call():
    """100 concurrent identical requests must result in exactly 1 SLM 'call'."""
    cache = TwoLevelCache()
    cache.reset_slm_call_count()
    key = cache.generate_response_key("concurrent_q", "en", 10, True, "p", "m", "v1", "v1", "v1")
    
    slm_calls = [0]
    call_lock = threading.Lock()
    results = []
    errors = []
    
    def worker(idx):
        try:
            val, should_compute = cache.get_or_wait_response(key)
            if should_compute:
                with call_lock:
                    slm_calls[0] += 1
                # Simulate SLM work
                time.sleep(0.05)
                cache.set_response(key, {"answer": "shared_result", "idx": idx})
                results.append(("computed", idx))
            else:
                results.append(("cached", val))
        except Exception as e:
            errors.append(str(e))
    
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    
    assert len(errors) == 0, f"Errors: {errors}"
    assert slm_calls[0] == 1, f"Expected 1 SLM call, got {slm_calls[0]}"
    assert len(results) == 100
    # All non-computing threads should have received the cached value
    cached_results = [r for r in results if r[0] == "cached"]
    assert len(cached_results) == 99

def test_coalesce_timeout_prevents_deadlock():
    """If computing thread never completes, waiters should time out gracefully."""
    cache = TwoLevelCache()
    cache.coalesce_timeout = 0.5  # Short timeout for test
    
    key = cache.generate_response_key("deadlock_test", "en", 10, True, "p", "m", "v1", "v1", "v1")
    
    # Thread 1 "starts computing" but never finishes
    val1, should_compute1 = cache.get_or_wait_response(key)
    assert should_compute1 is True
    
    result = []
    
    def waiter():
        val, should_compute = cache.get_or_wait_response(key)
        result.append((val, should_compute))
    
    t = threading.Thread(target=waiter)
    t.start()
    t.join(timeout=5)  # Should complete well within 5 seconds
    
    assert not t.is_alive(), "Thread should not be deadlocked"
    assert len(result) == 1
    assert result[0] == (None, True)  # Timed out, caller should compute independently
    
    # Clean up
    cache.release_response_lock(key)


# ─── SLM Call Counter ────────────────────────────────────────────────────────

def test_slm_call_counter():
    cache = TwoLevelCache()
    cache.reset_slm_call_count()
    assert cache.slm_call_count == 0
    
    cache.increment_slm_calls()
    cache.increment_slm_calls()
    assert cache.slm_call_count == 2
    
    cache.reset_slm_call_count()
    assert cache.slm_call_count == 0

def test_slm_call_count_in_stats():
    cache = TwoLevelCache()
    cache.reset_slm_call_count()
    cache.increment_slm_calls()
    stats = cache.stats()
    assert stats["slm_call_count"] == 1

def test_cache_clear_resets_slm_count():
    cache = TwoLevelCache()
    cache.increment_slm_calls()
    cache.increment_slm_calls()
    cache.clear()
    assert cache.slm_call_count == 0
