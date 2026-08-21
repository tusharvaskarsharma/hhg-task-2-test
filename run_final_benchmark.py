"""Backward-compatible benchmark gate shim.

The retained benchmark implementation is ``final_workload_benchmark.py``.
This small module preserves the historical gate contract used by the
regression suite without creating a second benchmark workflow.
"""
from __future__ import annotations

import sys


def enforce_gates(p50_val: float, p95_val: float, p99_val: float, slm_calls: int) -> bool:
    """Return True only when strict latency and RAG_ONLY SLM gates pass."""
    status_str = "PASS"
    if p50_val > 50 or p95_val > 50 or p99_val > 50:
        status_str = "FAIL: RAG_ONLY latency gate"
    if slm_calls > 0:
        status_str = "FAIL: RAG_ONLY SLM-call invariant"
    if status_str.startswith("FAIL"):
        return False
    return True


if __name__ == "__main__":
    # Use final_workload_benchmark.py for the actual 60/30/10 execution.
    from final_workload_benchmark import main
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        sys.exit(1)
