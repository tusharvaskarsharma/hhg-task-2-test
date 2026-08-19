import time
import math
from typing import List, Dict, Any

def calculate_latency_stats(latencies: List[float]) -> Dict[str, Any]:
    if not latencies:
        return {
            "count": 0,
            "min_ms": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0
        }
        
    sorted_lats = sorted(latencies)
    n = len(sorted_lats)
    
    # Calculate min, max, mean
    min_ms = sorted_lats[0]
    max_ms = sorted_lats[-1]
    mean_ms = sum(sorted_lats) / n
    
    # Calculate p50 and p95 correctly
    def percentile(p):
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_lats[int(k)]
        d0 = sorted_lats[int(f)] * (c - k)
        d1 = sorted_lats[int(c)] * (k - f)
        return d0 + d1
        
    p50_ms = percentile(0.50)
    p95_ms = percentile(0.95)
    
    return {
        "count": n,
        "min_ms": float(min_ms),
        "mean_ms": float(mean_ms),
        "p50_ms": float(p50_ms),
        "p95_ms": float(p95_ms),
        "max_ms": float(max_ms)
    }

class Timer:
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        
    @property
    def duration_ms(self):
        return (self.end_time - self.start_time) * 1000.0
