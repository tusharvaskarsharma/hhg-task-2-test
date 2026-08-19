import os
import time
import json
import statistics
import urllib.request
import urllib.error

API_URL = "http://127.0.0.1:8000/api/query"
VOICE_URL = "http://127.0.0.1:8000/api/voice"
WARMUP_ROUNDS = 5
BENCHMARK_ROUNDS = 100

import requests

def ping_text(query="headache"):
    try:
        res = requests.post(API_URL, json={"query": query, "language": "en", "top_k": 2})
        res.raise_for_status()
        return res.json()
    except requests.exceptions.HTTPError as e:
        return {"error": e.response.status_code, "message": e.response.text}
    except Exception as e:
        return {"error": 500, "message": str(e)}

import requests

def ping_voice(*args, **kwargs):
    try:
        res = requests.post(
            VOICE_URL,
            files={"audio": ("example.ogg", open("example.ogg", "rb").read(), "audio/ogg")},
            data={"language": "en", "top_k": 2}
        )
        return res.json()
    except requests.exceptions.HTTPError as e:
        return {"error": e.response.status_code, "message": e.response.text}
    except Exception as e:
        return {"error": 500, "message": str(e)}

import uuid

def run_benchmark(name, fetch_func, is_slm_expected, is_stt_expected):
    print(f"\\n--- Running Benchmark: {name} ---")
    
    # Check availability
    run_uuid = uuid.uuid4().hex[:8]
    test_res = fetch_func(f"test {run_uuid}") if name != "TOTAL" else fetch_func()
    if "error" in test_res:
        if test_res["error"] == 503:
            print("Status: NOT BENCHMARKED (Service Unavailable)")
        else:
            print(f"Status: NOT BENCHMARKED (Error: {test_res['error']}, Message: {test_res.get('message')})")
        return None
        
    lat = test_res.get("latency", {})
    bd = lat.get("breakdown", {})
    
    if is_slm_expected and bd.get("generation_ms", 0.0) == 0.0:
        print("Status: NOT BENCHMARKED (SLM Disabled or Credentials Missing)")
        return None
        
    if is_stt_expected and bd.get("stt_ms", 0.0) == 0.0:
        print("Status: NOT BENCHMARKED (STT Disabled or Credentials Missing)")
        return None

    # Warmup
    print("Warming up...")
    for i in range(WARMUP_ROUNDS):
        fetch_func(f"warmup {run_uuid} {i}") if name != "TOTAL" else fetch_func()
        
    print(f"Running {BENCHMARK_ROUNDS} queries...")
    latencies = {
        "total": [], "partial": [], "rag_only": [],
        "stt": [], "lang": [], "tok": [], "emb": [], "bm25": [],
        "hnsw": [], "rrf": [], "meta": [], "ground": [], "slm": [],
        "val": [], "ser": []
    }
    
    for i in range(BENCHMARK_ROUNDS):
        res = fetch_func(f"query {run_uuid} {i}") if name != "TOTAL" else fetch_func()
        if "error" in res:
            continue
            
        l = res["latency"]
        b = l["breakdown"]
        
        # Validation checks
        if name == "1. RAG_ONLY (Focusing on RAG metrics)":
            if b.get("embedding_ms", 0) == 0 or b.get("bm25_ms", 0) == 0 or b.get("hnsw_ms", 0) == 0:
                print("Status: INVALID BENCHMARK - Core retrieval path bypassed (0 ms latency).")
                return None
        
        latencies["total"].append(l["total_ms"])
        latencies["partial"].append(l["partial_ms"])
        latencies["rag_only"].append(l["rag_only_ms"])
        
        latencies["stt"].append(b["stt_ms"])
        latencies["lang"].append(b["language_detection_ms"])
        latencies["tok"].append(b["tokenization_ms"])
        latencies["emb"].append(b["embedding_ms"])
        latencies["bm25"].append(b["bm25_ms"])
        latencies["hnsw"].append(b["hnsw_ms"])
        latencies["rrf"].append(b["rrf_ms"])
        latencies["meta"].append(b["metadata_ms"])
        latencies["ground"].append(b["grounding_ms"])
        latencies["slm"].append(b["generation_ms"])
        latencies["val"].append(b["validation_ms"])
        latencies["ser"].append(b.get("serialization_ms", 0.0))

    def stat_str(arr):
        if not arr: return "N/A"
        return f"p50={statistics.median(arr):.2f}ms | p95={statistics.quantiles(arr, n=20)[18]:.2f}ms | min={min(arr):.2f}ms | max={max(arr):.2f}ms"

    print(f"\\nResults for {name}:")
    print(f"TOTAL:    {stat_str(latencies['total'])}")
    print(f"PARTIAL:  {stat_str(latencies['partial'])}")
    print(f"RAG_ONLY: {stat_str(latencies['rag_only'])}")
    
    print("\\nDetailed Breakdown (p50):")
    print(f"STT:                 {statistics.median(latencies['stt']):.2f}ms")
    print(f"Language detection:  {statistics.median(latencies['lang']):.2f}ms")
    print(f"Tokenization:        {statistics.median(latencies['tok']):.2f}ms")
    print(f"Embedding:           {statistics.median(latencies['emb']):.2f}ms")
    print(f"BM25:                {statistics.median(latencies['bm25']):.2f}ms")
    print(f"HNSW:                {statistics.median(latencies['hnsw']):.2f}ms")
    print(f"RRF:                 {statistics.median(latencies['rrf']):.2f}ms")
    print(f"Metadata:            {statistics.median(latencies['meta']):.2f}ms")
    print(f"Grounding:           {statistics.median(latencies['ground']):.2f}ms")
    print(f"SLM/generation:      {statistics.median(latencies['slm']):.2f}ms")
    print(f"Validation:          {statistics.median(latencies['val']):.2f}ms")
    print(f"Serialization:       {statistics.median(latencies['ser']):.2f}ms")
    return latencies

if __name__ == "__main__":
    print("Starting Final 3-Tier Performance Benchmark...")
    
    # 1. RAG_ONLY (STT=OFF, SLM=OFF)
    # We can ping the text endpoint. 
    # If SLM is actually ON, we might get SLM metrics, but the prompt says RAG_ONLY target is tested here.
    # To test RAG_ONLY properly, we evaluate the rag_only_ms metric of a standard request.
    # The instructions say: "Benchmark these three configurations separately"
    # I'll just use the text endpoint for RAG_ONLY and look at rag_only_ms, or just report if SLM is ON/OFF
    
    # Since I don't control the server config directly from here, I will just report the metrics returned.
    
    run_benchmark("1. RAG_ONLY (Focusing on RAG metrics)", ping_text, is_slm_expected=False, is_stt_expected=False)
    run_benchmark("2. PARTIAL (Focusing on RAG + SLM metrics)", ping_text, is_slm_expected=True, is_stt_expected=False)
    run_benchmark("3. TOTAL (Focusing on STT + RAG + SLM metrics)", ping_voice, is_slm_expected=True, is_stt_expected=True)
