import os
import sys
import time
import json
import uuid
import subprocess
import requests
import numpy as np

def update_env(slm_enabled: bool):
    env_path = 'backend/.env'
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    with open(env_path, 'w') as f:
        for line in lines:
            if line.startswith('HHG_SLM_ENABLED='):
                f.write(f'HHG_SLM_ENABLED={str(slm_enabled).lower()}\n')
            else:
                f.write(line)

def wait_for_server():
    for _ in range(60):
        try:
            r = requests.get('http://127.0.0.1:8000/api/ready', timeout=1)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def run_benchmark(name, endpoint, is_voice=False):
    print(f"\n--- Running Benchmark: {name} ---")
    url = f'http://127.0.0.1:8000/api/{endpoint}'
    
    # Warmup
    print("Sending 10 warmup requests...")
    for i in range(10):
        q = f'what is the capital of india {uuid.uuid4()}'
        if is_voice:
            with open('example.ogg', 'rb') as f:
                requests.post(url, files={'audio': ('example.ogg', f, 'audio/ogg')}, data={'language': 'en'})
        else:
            requests.post(url, json={'query': q, 'language': 'en', 'top_k': 2})
            
    print("Running 100 unique uncached requests...")
    latencies = []
    rag_latencies = []
    partial_latencies = []
    total_latencies = []
    
    all_breakdowns = []
    
    for i in range(100):
        q = f'what is the capital of india {uuid.uuid4()}'
        if is_voice:
            with open('example.ogg', 'rb') as f:
                r = requests.post(url, files={'audio': ('example.ogg', f, 'audio/ogg')}, data={'language': 'en'})
        else:
            r = requests.post(url, json={'query': q, 'language': 'en', 'top_k': 2})
            
        if name != "RAG_ONLY":
            time.sleep(0.5)
            
        if r.status_code != 200:
            print(f"Error: {r.status_code} {r.text}")
            continue
            
        data = r.json()
        l = data['latency']
        
        # Based on endpoint and config, determine the reported metric
        if name == "RAG_ONLY":
            metric = l['rag_only_ms']
        elif name == "PARTIAL":
            metric = l['partial_ms'] if is_voice else l['total_ms']
        else:
            metric = l['total_ms']
            
        latencies.append(metric)
        rag_latencies.append(l.get('rag_only_ms', 0))
        partial_latencies.append(l.get('partial_ms', 0))
        total_latencies.append(l.get('total_ms', 0))
        all_breakdowns.append(l.get('breakdown', {}))
        
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    _min = np.min(latencies)
    _max = np.max(latencies)
    
    print(f"Results for {name}:")
    print(f"p50: {p50:.2f} ms")
    print(f"p95: {p95:.2f} ms")
    print(f"Min: {_min:.2f} ms")
    print(f"Max: {_max:.2f} ms")
    
    print("\nComponent Breakdown (p50):")
    for key in all_breakdowns[0].keys():
        vals = [b.get(key, 0) for b in all_breakdowns]
        print(f"{key}: {np.percentile(vals, 50):.2f} ms")
        
    return p50, p95, _min, _max

def main():
    # Kill existing server if any
    subprocess.run("taskkill /F /IM uvicorn.exe /T", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # RAG_ONLY
    update_env(False)
    t0 = time.time()
    p = subprocess.Popen(["uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000", "--env-file", "backend/.env"])
    if not wait_for_server():
        print("Server failed to start")
        sys.exit(1)
    t1 = time.time()
    startup_time = t1 - t0
    print(f"Startup/Warmup time (SLM OFF): {startup_time:.2f} s")
    
    rag_p50, rag_p95, rag_min, rag_max = run_benchmark("RAG_ONLY", "query")
    
    p.terminate()
    p.wait()
    subprocess.run("taskkill /F /IM uvicorn.exe /T", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # PARTIAL and TOTAL
    update_env(True)
    t0 = time.time()
    p = subprocess.Popen(["uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000", "--env-file", "backend/.env"])
    if not wait_for_server():
        print("Server failed to start")
        sys.exit(1)
    t1 = time.time()
    startup_time_slm = t1 - t0
    print(f"\nStartup/Warmup time (SLM ON): {startup_time_slm:.2f} s")
    
    part_p50, part_p95, part_min, part_max = run_benchmark("PARTIAL", "query")
    tot_p50, tot_p95, tot_min, tot_max = run_benchmark("TOTAL", "voice", is_voice=True)
    
    p.terminate()
    p.wait()
    subprocess.run("taskkill /F /IM uvicorn.exe /T", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    verdict = "PASS" if (rag_p50 <= 50 and rag_p95 <= 50) else "FAIL"
    
    print("\n=================================")
    print("FINAL REPORT")
    print("=================================")
    print(f"COLD START (SLM ON): {startup_time_slm:.2f} s")
    print("\nWARM 100-REQUEST BENCHMARK:")
    print(f"RAG_ONLY: p50={rag_p50:.2f} / p95={rag_p95:.2f} / min={rag_min:.2f} / max={rag_max:.2f}")
    print(f"PARTIAL:  p50={part_p50:.2f} / p95={part_p95:.2f} / min={part_min:.2f} / max={part_max:.2f}")
    print(f"TOTAL:    p50={tot_p50:.2f} / p95={tot_p95:.2f} / min={tot_min:.2f} / max={tot_max:.2f}")
    
    print("\nFINAL VERDICT:")
    print(f"{verdict} (RAG_ONLY <= 50ms requirement)")
    
if __name__ == '__main__':
    main()
