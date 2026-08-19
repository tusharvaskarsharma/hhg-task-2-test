import pytest
import math
from backend.evaluation.latency_benchmark import calculate_latency_stats, Timer
import time

def test_latency_stats_empty():
    stats = calculate_latency_stats([])
    assert stats["count"] == 0
    assert stats["p50_ms"] == 0.0

def test_latency_stats_single():
    stats = calculate_latency_stats([10.0])
    assert stats["count"] == 1
    assert stats["p50_ms"] == 10.0
    assert stats["p95_ms"] == 10.0
    assert stats["min_ms"] == 10.0
    assert stats["max_ms"] == 10.0
    assert stats["mean_ms"] == 10.0

def test_latency_stats_repeated():
    stats = calculate_latency_stats([5.0, 5.0, 5.0])
    assert stats["p50_ms"] == 5.0
    assert stats["p95_ms"] == 5.0

def test_latency_stats_percentiles():
    # 1 to 100
    data = list(range(1, 101))
    stats = calculate_latency_stats(data)
    
    assert stats["count"] == 100
    assert stats["min_ms"] == 1.0
    assert stats["max_ms"] == 100.0
    assert stats["mean_ms"] == 50.5
    
    # p50 of 1..100 should be 50.5
    assert math.isclose(stats["p50_ms"], 50.5)
    
    # p95 should be 95.05
    assert math.isclose(stats["p95_ms"], 95.05)

def test_timer_context():
    with Timer() as t:
        time.sleep(0.01)
        
    assert t.duration_ms > 0
    assert t.duration_ms >= 10.0 # roughly 10ms
