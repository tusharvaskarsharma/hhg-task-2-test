import time
import os

def profile_backend():
    print("Profiling backend Cold Start...")
    t0 = time.time()
    
    # Trace 1: Imports
    import pandas as pd
    t1 = time.time()
    print(f"Pandas import: {t1-t0:.2f}s")
    
    import onnxruntime
    t2 = time.time()
    print(f"ONNXRuntime import: {t2-t1:.2f}s")
    
    pass
    
    # Trace 2: ONNX load
    from backend.pipeline.embedder import Embedder
    t4 = time.time()
    e = Embedder()
    t5 = time.time()
    print(f"ONNX Model Load: {t5-t4:.2f}s")
    
    # Trace 3: Artifact Loader (HNSW + BM25)
    from backend.artifact_loader import loader_instance
    t6 = time.time()
    loader_instance.initialize()
    t7 = time.time()
    print(f"Total Artifact Loading (Metadata + BM25 + HNSW): {t7-t6:.2f}s")
    
    # Total
    print(f"\n--- TOTAL COLD START: {t7-t0:.2f}s ---")

if __name__ == "__main__":
    profile_backend()
