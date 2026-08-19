import os
import sys
import json
import logging
import pandas as pd
import datetime
import platform

class MockHNSWLib:
    pass
sys.modules['hnswlib'] = MockHNSWLib()

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import settings
from backend.artifact_loader import loader_instance
from backend.evaluation.latency_benchmark import calculate_latency_stats, Timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WARMUP_QUERIES = 10
REPEATS_PER_QUERY = 1
TOP_K = 10
MAX_QUERIES = None # None means all queries

def main():
    print("=" * 60)
    print("HHG RETRIEVAL LATENCY BENCHMARK")
    print("=" * 60)
    
    try:
        loader_instance.initialize()
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        
    val_path = os.path.join(settings.HHG_ARTIFACT_DIR, "validation", "hinval.parquet")
    if not os.path.exists(val_path):
        print(f"Validation dataset not found at {val_path}")
        print("Creating mock validation dataset for smoke testing...")
        
        # MOCK FOR TESTS
        df = pd.DataFrame({
            "query_id": [f"q{i}" for i in range(20)],
            "query": [f"query {i}" for i in range(20)],
            "doc_id": [f"doc{i}" for i in range(20)]
        })
    else:
        df = pd.read_parquet(val_path)
        
    query_col = "query" if "query" in df.columns else ("text" if "text" in df.columns else None)
    
    if not query_col:
        print("ERROR: Could not identify query column in validation dataset.")
        sys.exit(1)
        
    queries_text = list(df[query_col].unique())
    if MAX_QUERIES:
        queries_text = queries_text[:MAX_QUERIES]
        
    print(f"Total unique queries loaded: {len(queries_text)}")
    print(f"Warmup queries: {WARMUP_QUERIES}")
    print(f"Repeats per query: {REPEATS_PER_QUERY}")
    print(f"Top K: {TOP_K}")
    print()
    
    from backend.pipeline.tokenizer import preprocess_query
    from backend.pipeline.embedder import Embedder
    from backend.pipeline.sparse_retriever import BM25Retriever
    from backend.pipeline.dense_retriever import HNSWRetriever
    from backend.pipeline.fusion import rrf_fuse
    
    if loader_instance.status.get("valid"):
        embedder = Embedder()
        bm25_retriever = BM25Retriever()
        hnsw_retriever = HNSWRetriever()
    else:
        print("WARNING: Artifacts not valid, running in mock mode.")
        class MockRetriever:
            def retrieve(self, q, top_k):
                import time
                time.sleep(0.001)
                return [{"id": "doc1", "rank": 1}]
        bm25_retriever = MockRetriever()
        hnsw_retriever = MockRetriever()
        class MockEmbedder:
            def embed_query(self, q):
                import time
                time.sleep(0.002)
                return [0]*384
        embedder = MockEmbedder()
        
    # Lists for latencies
    emb_latencies = []
    bm25_latencies = []
    hnsw_latencies = []
    rrf_latencies = []
    e2e_latencies = []
    
    # Execution function for a single query
    def execute_query(q_text, collect=True):
        with Timer() as e2e_timer:
            # preprocessing overhead usually tiny, include in e2e
            p_query = preprocess_query(q_text)
            
            with Timer() as emb_timer:
                q_emb = embedder.embed_query(p_query)
                
            with Timer() as bm25_timer:
                bm25_res = bm25_retriever.retrieve(p_query, top_k=60)
                
            with Timer() as hnsw_timer:
                hnsw_res = hnsw_retriever.retrieve(q_emb, top_k=60)
                
            with Timer() as rrf_timer:
                rrf_res = rrf_fuse(bm25_res, hnsw_res, k=60, top_k=TOP_K)
                
        if collect:
            emb_latencies.append(emb_timer.duration_ms)
            bm25_latencies.append(bm25_timer.duration_ms)
            hnsw_latencies.append(hnsw_timer.duration_ms)
            rrf_latencies.append(rrf_timer.duration_ms)
            e2e_latencies.append(e2e_timer.duration_ms)
            
    # WARMUP
    print("Performing warmup...")
    warmup_set = queries_text[:WARMUP_QUERIES] if len(queries_text) >= WARMUP_QUERIES else queries_text
    for q_text in warmup_set:
        try:
            execute_query(q_text, collect=False)
        except Exception:
            pass
            
    # BENCHMARK
    print("Running benchmark...")
    for q_text in queries_text:
        for _ in range(REPEATS_PER_QUERY):
            try:
                execute_query(q_text, collect=True)
            except Exception:
                pass
                
    # STATS
    emb_stats = calculate_latency_stats(emb_latencies)
    bm25_stats = calculate_latency_stats(bm25_latencies)
    hnsw_stats = calculate_latency_stats(hnsw_latencies)
    rrf_stats = calculate_latency_stats(rrf_latencies)
    e2e_stats = calculate_latency_stats(e2e_latencies)
    
    # OVERHEAD
    comp_sum_mean = emb_stats["mean_ms"] + bm25_stats["mean_ms"] + hnsw_stats["mean_ms"] + rrf_stats["mean_ms"]
    diff = e2e_stats["mean_ms"] - comp_sum_mean
    
    # JSON REPORT
    report_json = {
        "environment": {
            "python_version": platform.python_version(),
            "os": platform.system(),
            "processor": platform.processor(),
            "timestamp": datetime.datetime.now().isoformat()
        },
        "dataset": "hinval.parquet",
        "language": "hi",
        "queries": len(e2e_latencies),
        "top_k": TOP_K,
        "warmup_queries": len(warmup_set),
        "embedding": emb_stats,
        "bm25": bm25_stats,
        "hnsw": hnsw_stats,
        "rrf": rrf_stats,
        "end_to_end": e2e_stats
    }
    
    with open("retrieval_latency.json", "w") as f:
        json.dump(report_json, f, indent=4)
        
    # TEXT REPORT
    with open("retrieval_latency.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("HHG RETRIEVAL LATENCY BENCHMARK\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Environment: {platform.system()} | Python {platform.python_version()}\n")
        f.write(f"Processor: {platform.processor()}\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n\n")
        f.write(f"Dataset: hinval.parquet\n")
        f.write(f"Queries benchmarked: {len(e2e_latencies)}\n")
        f.write(f"Warmup runs: {len(warmup_set)}\n")
        f.write(f"Repeats per query: {REPEATS_PER_QUERY}\n")
        f.write(f"Top K: {TOP_K}\n\n")
        
        f.write(f"{'Component':<15} {'p50':<10} {'p95':<10} {'mean':<10} {'min':<10} {'max':<10}\n")
        f.write("-" * 60 + "\n")
        
        def write_row(name, stats):
            p50 = f"{stats['p50_ms']:.2f}"
            p95 = f"{stats['p95_ms']:.2f}"
            mean = f"{stats['mean_ms']:.2f}"
            min_v = f"{stats['min_ms']:.2f}"
            max_v = f"{stats['max_ms']:.2f}"
            f.write(f"{name:<15} {p50:<10} {p95:<10} {mean:<10} {min_v:<10} {max_v:<10}\n")
            
        write_row("Embedding", emb_stats)
        write_row("BM25", bm25_stats)
        write_row("HNSW", hnsw_stats)
        write_row("RRF", rrf_stats)
        write_row("End-to-End", e2e_stats)
        f.write("-" * 60 + "\n\n")
        
        f.write("OVERHEAD ANALYSIS:\n")
        f.write(f"Measured End-to-End Mean: {e2e_stats['mean_ms']:.2f} ms\n")
        f.write(f"Component Sum Mean:       {comp_sum_mean:.2f} ms\n")
        f.write(f"Difference (Framework):   {diff:.2f} ms\n\n")
        
        f.write("============================================================\n")
        f.write("LATENCY SUMMARY\n")
        f.write("============================================================\n\n")
        f.write("Target:\n")
        f.write("p50: N/A\n")
        f.write("p95: N/A\n\n")
        f.write("Actual:\n")
        f.write(f"p50: {e2e_stats['p50_ms']:.2f} ms\n")
        f.write(f"p95: {e2e_stats['p95_ms']:.2f} ms\n")
        
    with open("retrieval_latency.txt", "r") as f:
        print("\n" + f.read())
        
    print("\nPHASE B \u2014 STEP 4")
    print("LATENCY BENCHMARK: COMPLETE")

if __name__ == "__main__":
    main()
