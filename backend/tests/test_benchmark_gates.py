import pytest
import subprocess
import sys

def test_benchmark_gates_enforced():
    """
    Synthetic test to ensure that the run_final_benchmark script has the correct 
    strict latency gates implemented.
    """
    with open("run_final_benchmark.py", "r") as f:
        content = f.read()
        
    assert "p50_val > 50 or p95_val > 50 or p99_val > 50" in content, "Strict latency gate (50ms) missing"
    assert "slm_calls > 0" in content, "SLM zero-call invariant missing"
    assert "status_str = \"FAIL" in content, "Failure assignment missing"
    assert "sys.exit(1)" in content, "Failure exit missing"
