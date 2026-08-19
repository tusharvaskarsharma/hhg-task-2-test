import os
import sys
import time
import json
import uuid
import subprocess
import requests
import numpy as np
from datetime import datetime

def update_env(slm_enabled: bool):
    env_path = 'backend/.env'
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    with open(env_path, 'w') as f:
        for line in lines:
            if line.startswith('HHG_SLM_ENABLED='):
                f.write(f'HHG_SLM_ENABLED={str(slm_enabled).lower()}\n')
            elif line.startswith('CACHE_ENABLED='):
                continue
            else:
                f.write(line)
        f.write('CACHE_ENABLED=false\n')

def wait_for_server(timeout=120):
    for _ in range(timeout):
        try:
            r = requests.get('http://127.0.0.1:8000/api/ready', timeout=1)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def print_table(components_data):
    print(f"{'stage':<25} {'avg':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'min':>8} {'max':>8}")
    print("-" * 79)
    for stage, latencies in components_data.items():
        if not latencies:
            print(f"{stage:<25} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
            continue
        avg = np.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        _min = np.min(latencies)
        _max = np.max(latencies)
        print(f"{stage:<25} {avg:>8.2f} {p50:>8.2f} {p95:>8.2f} {p99:>8.2f} {_min:>8.2f} {_max:>8.2f}")

def format_table_md(components_data):
    lines = []
    lines.append("| stage | avg | p50 | p95 | p99 | min | max |")
    lines.append("|---|---|---|---|---|---|---|")
    for stage, latencies in components_data.items():
        if not latencies:
            lines.append(f"| {stage} | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        avg = np.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        _min = np.min(latencies)
        _max = np.max(latencies)
        lines.append(f"| {stage} | {avg:.2f} | {p50:.2f} | {p95:.2f} | {p99:.2f} | {_min:.2f} | {_max:.2f} |")
    return "\n".join(lines)

def run_benchmark(name, endpoint, is_voice=False):
    print(f"\n--- Running Benchmark: {name} ---")
    url = f'http://127.0.0.1:8000/api/{endpoint}'
    
    print("Sending 10 warmup requests...")
    for i in range(10):
        q = f'what is the capital of india {uuid.uuid4()}'
        try:
            if is_voice:
                with open('example.ogg', 'rb') as f:
                    requests.post(url, files={'audio': ('example.ogg', f, 'audio/ogg')}, data={'language': 'en'})
            else:
                requests.post(url, json={'query': q, 'language': 'en', 'top_k': 2})
        except:
            pass
            
    print("Running 100 unique uncached requests...")
    
    components = {}
    expected_keys = []
    if name == "RAG_ONLY":
        expected_keys = ["language detection", "tokenization", "embedding", "BM25", "HNSW", "RRF", "metadata lookup", "grounding", "serialization", "total RAG_ONLY"]
    elif name == "PARTIAL":
        expected_keys = ["language detection", "tokenization", "embedding", "BM25", "HNSW", "RRF", "metadata lookup", "grounding", "serialization", "SLM/generation", "validation", "total PARTIAL"]
    elif name == "TOTAL":
        expected_keys = ["STT", "language detection", "tokenization", "embedding", "BM25", "HNSW", "RRF", "metadata lookup", "grounding", "serialization", "SLM/generation", "validation", "total TOTAL"]

    for k in expected_keys:
        components[k] = []
        
    failures = 0
    not_benchmarked = False

    for i in range(100):
        q = f'what is the capital of india {uuid.uuid4()}'
        try:
            if is_voice:
                with open('example.ogg', 'rb') as f:
                    r = requests.post(url, files={'audio': ('example.ogg', f, 'audio/ogg')}, data={'language': 'en'})
            else:
                r = requests.post(url, json={'query': q, 'language': 'en', 'top_k': 2})
                
            if name != "RAG_ONLY":
                if name == "TOTAL":
                    time.sleep(1.5)
                else:
                    time.sleep(0.5)
                
            if r.status_code != 200:
                failures += 1
                if failures >= 10:
                    not_benchmarked = True
                    break
                continue
                
            data = r.json()
            l = data['latency']
            bd = l.get('breakdown', {})
            
            if bd.get('embedding_ms', 0) == 0 or bd.get('bm25_ms', 0) == 0 or bd.get('hnsw_ms', 0) == 0:
                print("INVALID: embedding_ms, bm25_ms, or hnsw_ms is 0.")
                sys.exit(1)
                
            if name == "RAG_ONLY":
                components["language detection"].append(bd.get("language_detection_ms", 0))
                components["tokenization"].append(bd.get("tokenization_ms", 0))
                components["embedding"].append(bd.get("embedding_ms", 0))
                components["BM25"].append(bd.get("bm25_ms", 0))
                components["HNSW"].append(bd.get("hnsw_ms", 0))
                components["RRF"].append(bd.get("rrf_ms", 0))
                components["metadata lookup"].append(bd.get("metadata_ms", 0))
                components["grounding"].append(bd.get("grounding_ms", 0))
                components["serialization"].append(bd.get("serialization_ms", 0))
                components["total RAG_ONLY"].append(l.get("rag_only_ms", 0))
                
            elif name == "PARTIAL":
                components["language detection"].append(bd.get("language_detection_ms", 0))
                components["tokenization"].append(bd.get("tokenization_ms", 0))
                components["embedding"].append(bd.get("embedding_ms", 0))
                components["BM25"].append(bd.get("bm25_ms", 0))
                components["HNSW"].append(bd.get("hnsw_ms", 0))
                components["RRF"].append(bd.get("rrf_ms", 0))
                components["metadata lookup"].append(bd.get("metadata_ms", 0))
                components["grounding"].append(bd.get("grounding_ms", 0))
                components["serialization"].append(bd.get("serialization_ms", 0))
                components["SLM/generation"].append(bd.get("generation_ms", 0))
                components["validation"].append(bd.get("validation_ms", 0))
                components["total PARTIAL"].append(l.get("partial_ms", 0))

            elif name == "TOTAL":
                components["STT"].append(bd.get("stt_ms", 0))
                components["language detection"].append(bd.get("language_detection_ms", 0))
                components["tokenization"].append(bd.get("tokenization_ms", 0))
                components["embedding"].append(bd.get("embedding_ms", 0))
                components["BM25"].append(bd.get("bm25_ms", 0))
                components["HNSW"].append(bd.get("hnsw_ms", 0))
                components["RRF"].append(bd.get("rrf_ms", 0))
                components["metadata lookup"].append(bd.get("metadata_ms", 0))
                components["grounding"].append(bd.get("grounding_ms", 0))
                components["serialization"].append(bd.get("serialization_ms", 0))
                components["SLM/generation"].append(bd.get("generation_ms", 0))
                components["validation"].append(bd.get("validation_ms", 0))
                components["total TOTAL"].append(l.get("total_ms", 0))
        except Exception as e:
            print(f"Exception during request: {e}")
            failures += 1
            if failures >= 10:
                not_benchmarked = True
                break

    if not_benchmarked:
        print(f"Results for {name}: NOT BENCHMARKED")
        return None
        
    print_table(components)
    return components

def main():
    subprocess.run("taskkill /F /IM uvicorn.exe /T", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    update_env(False)
    t0 = time.time()
    p1 = subprocess.Popen(["uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000", "--env-file", "backend/.env"])
    if not wait_for_server():
        print("Server failed to start")
        sys.exit(1)
    startup_time_rag = time.time() - t0
    
    rag_data = run_benchmark("RAG_ONLY", "query")
    
    p1.terminate()
    p1.wait()
    subprocess.run("taskkill /F /IM uvicorn.exe /T", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    update_env(True)
    t0 = time.time()
    p2 = subprocess.Popen(["uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000", "--env-file", "backend/.env"])
    if not wait_for_server():
        print("Server failed to start")
        sys.exit(1)
    startup_time_slm = time.time() - t0
    
    part_data = run_benchmark("PARTIAL", "query")
    tot_data = run_benchmark("TOTAL", "voice", is_voice=True)
    
    p2.terminate()
    p2.wait()
    subprocess.run("taskkill /F /IM uvicorn.exe /T", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not rag_data:
        print("RAG_ONLY benchmarking failed. Exiting.")
        sys.exit(1)
        
    rag_p50 = np.percentile(rag_data["total RAG_ONLY"], 50)
    rag_p95 = np.percentile(rag_data["total RAG_ONLY"], 95)
    
    part_p50 = np.percentile(part_data["total PARTIAL"], 50) if part_data else 0
    tot_p50 = np.percentile(tot_data["total TOTAL"], 50) if tot_data else 0
    
    if rag_p50 <= 50 and rag_p95 <= 50:
        verdict = "PASS"
    else:
        verdict = "FAIL"
        
    validation_logic = "TOTAL >= PARTIAL >= RAG_ONLY: "
    if part_data and tot_data:
        if tot_p50 >= part_p50 >= rag_p50:
            validation_logic += "PASS"
        else:
            validation_logic += "FAIL"
    else:
        validation_logic += "N/A (NOT BENCHMARKED)"

    print("\n=================================")
    print("FINAL REPORT")
    print("=================================")
    print(f"COLD START")
    print(f"startup/warmup time (SLM OFF): {startup_time_rag:.2f} s")
    print(f"startup/warmup time (SLM ON): {startup_time_slm:.2f} s")
    print("\nWARM BENCHMARK")
    print("100 unique requests")
    print(f"\nRAG_ONLY: p50={rag_p50:.2f}ms, p95={rag_p95:.2f}ms")
    
    md_content = f"# Benchmark Results\n\n"
    md_content += f"- **Timestamp:** {datetime.now().isoformat()}\n"
    md_content += f"- **Number of Requests:** 100 per tier\n"
    md_content += f"- **Warmup Status:** Completed\n"
    md_content += f"- **Cache-Bypass Status:** UUIDs used\n"
    md_content += f"- **Cold-start Latency (SLM OFF):** {startup_time_rag:.2f} s\n"
    md_content += f"- **Cold-start Latency (SLM ON):** {startup_time_slm:.2f} s\n\n"
    
    md_content += f"## RAG_ONLY Statistics\n"
    md_content += format_table_md(rag_data) + "\n\n"
    
    md_content += f"## PARTIAL Statistics\n"
    if part_data:
        md_content += format_table_md(part_data) + "\n\n"
    else:
        md_content += "NOT BENCHMARKED\n\n"
        
    md_content += f"## TOTAL Statistics\n"
    if tot_data:
        md_content += format_table_md(tot_data) + "\n\n"
    else:
        md_content += "NOT BENCHMARKED\n\n"
        
    md_content += f"## Final Verdict\n"
    md_content += f"**RAG_ONLY Target (<=50ms):** {verdict}\n"
    md_content += f"**Logic Check:** {validation_logic}\n"
    
    with open("benchmark_results.md", "w") as f:
        f.write(md_content)
        
    print("\nResults saved to benchmark_results.md")
    print("\n" + verdict)
    
    if verdict == "FAIL":
        sys.exit(1)

if __name__ == '__main__':
    main()
