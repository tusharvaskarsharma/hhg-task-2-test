import os
import time
import json
import statistics
import requests
import subprocess
from pathlib import Path

# Paths
ENV_FILE = Path("backend/.env")

def set_env(slm_enabled="true"):
    content = ENV_FILE.read_text()
    import re
    content = re.sub(r'HHG_SLM_ENABLED=.*', f'HHG_SLM_ENABLED={slm_enabled}', content)
    ENV_FILE.write_text(content)

def start_backend():
    proc = subprocess.Popen(["uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000", "--env-file", "backend/.env"], cwd=str(Path.cwd()))
    # Wait for readiness
    for _ in range(60):
        try:
            r = requests.get("http://127.0.0.1:8000/api/ready")
            if r.status_code == 200:
                print("Backend is ready.")
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    else:
        print("Backend failed to start in time!")
    return proc

def run_test(fetch_func, name):
    print(f"Running {name}...")
    # Warmup
    for i in range(5):
        fetch_func(f"warmup {i}")
    
    results = []
    latencies = {
        "total": [], "partial": [], "rag_only": [],
        "stt": [], "lang": [], "tok": [], "emb": [], "bm25": [],
        "hnsw": [], "rrf": [], "meta": [], "ground": [], "slm": [],
        "val": [], "ser": []
    }
    
    for i in range(100):
        res = fetch_func(f"query {i}")
        if "error" in res: continue
        
        l = res["latency"]
        b = l["breakdown"]
        
        latencies["total"].append(l.get("total_ms", 0))
        latencies["partial"].append(l.get("partial_ms", 0))
        latencies["rag_only"].append(l.get("rag_only_ms", 0))
        
        latencies["stt"].append(b.get("stt_ms", 0))
        latencies["lang"].append(b.get("language_detection_ms", 0))
        latencies["tok"].append(b.get("tokenization_ms", 0))
        latencies["emb"].append(b.get("embedding_ms", 0))
        latencies["bm25"].append(b.get("bm25_ms", 0))
        latencies["hnsw"].append(b.get("hnsw_ms", 0))
        latencies["rrf"].append(b.get("rrf_ms", 0))
        latencies["meta"].append(b.get("metadata_ms", 0))
        latencies["ground"].append(b.get("grounding_ms", 0))
        latencies["slm"].append(b.get("generation_ms", 0))
        latencies["val"].append(b.get("validation_ms", 0))
        latencies["ser"].append(b.get("serialization_ms", 0))
        
    return latencies

def ping_text(query):
    return requests.post("http://127.0.0.1:8000/api/query", json={"query": query, "language": "en", "top_k": 2}).json()

audio_bytes = open("example.ogg", "rb").read()

def ping_voice(query):
    return requests.post(
        "http://127.0.0.1:8000/api/voice",
        files={"audio": ("example.ogg", audio_bytes, "audio/ogg")},
        data={"language": "en", "top_k": 2}
    ).json()

def fmt(arr):
    if not arr: return "N/A"
    return f"{statistics.median(arr):.2f}"

def main():
    import psutil
    # Kill existing servers
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if 'uvicorn' in str(proc.info['cmdline']):
            proc.kill()
    
    # 1. RAG_ONLY (STT=OFF, SLM=OFF)
    set_env("false")
    backend = start_backend()
    rag_stats = run_test(ping_text, "RAG_ONLY")
    backend.kill()
    backend.wait()
    
    # 2. PARTIAL & TOTAL (SLM=ON)
    set_env("true")
    backend = start_backend()
    partial_stats = run_test(ping_text, "PARTIAL")
    total_stats = run_test(ping_voice, "TOTAL")
    backend.kill()
    backend.wait()
    
    # Generate Markdown Table
    print("Generating Benchmark Results...")
    md = "# Benchmark Results\n\n"
    
    def gen_table(stats, name, key):
        nonlocal md
        md += f"## {name}\n\n"
        md += f"- **p50**: {statistics.median(stats[key]):.2f} ms\n"
        md += f"- **p95**: {statistics.quantiles(stats[key], n=20)[18]:.2f} ms\n"
        md += f"- **min**: {min(stats[key]):.2f} ms\n"
        md += f"- **max**: {max(stats[key]):.2f} ms\n\n"
        md += "| Metric | p50 (ms) |\n|---|---|\n"
        md += f"| STT | {fmt(stats['stt'])} |\n"
        md += f"| Language Detection | {fmt(stats['lang'])} |\n"
        md += f"| Tokenization | {fmt(stats['tok'])} |\n"
        md += f"| Embedding | {fmt(stats['emb'])} |\n"
        md += f"| BM25 | {fmt(stats['bm25'])} |\n"
        md += f"| HNSW | {fmt(stats['hnsw'])} |\n"
        md += f"| RRF | {fmt(stats['rrf'])} |\n"
        md += f"| Metadata | {fmt(stats['meta'])} |\n"
        md += f"| Grounding | {fmt(stats['ground'])} |\n"
        md += f"| SLM / Generation | {fmt(stats['slm'])} |\n"
        md += f"| Validation | {fmt(stats['val'])} |\n"
        md += f"| Serialization | {fmt(stats['ser'])} |\n\n"
    
    gen_table(rag_stats, "1. RAG_ONLY (STT=OFF, SLM=OFF)", "rag_only")
    gen_table(partial_stats, "2. PARTIAL (STT=OFF, SLM=ON)", "partial")
    gen_table(total_stats, "3. TOTAL (STT=ON, SLM=ON)", "total")
    
    with open("benchmark_table.md", "w") as f:
        f.write(md)
    print(md)

if __name__ == "__main__":
    main()
