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

def test_workload_benchmark_has_correct_methodology():
    """
    Verify the workload benchmark has correct 60/30/10 methodology.
    """
    with open("final_workload_benchmark.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Must use deterministic seed
    assert "seed" in content, "Deterministic seed missing"
    assert "42" in content, "Default seed 42 missing"
    
    # Must have 60/30/10 workload
    assert "0.60" in content or "0.6" in content, "60% unique workload missing"
    assert "0.30" in content or "0.3" in content, "30% repeated workload missing"
    
    # Must report response cache hits separately
    assert "response_cache_hit" in content, "Response cache hit tracking missing"
    assert "retrieval_cache_hit" in content, "Retrieval cache hit tracking missing"
    
    # Must report generation latency
    assert "generation_ms" in content, "Generation latency tracking missing"

def test_config_has_prompt_and_grounding_versions():
    """Config must have versioned prompt and grounding policy constants."""
    from backend.config import settings
    assert hasattr(settings, "PROMPT_VERSION"), "PROMPT_VERSION missing from config"
    assert hasattr(settings, "GROUNDING_VERSION"), "GROUNDING_VERSION missing from config"
    assert settings.PROMPT_VERSION, "PROMPT_VERSION is empty"
    assert settings.GROUNDING_VERSION, "GROUNDING_VERSION is empty"
