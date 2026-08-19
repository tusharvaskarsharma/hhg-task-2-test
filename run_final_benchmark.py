import os
import sys
import time
import json
import uuid
import subprocess
import requests
import argparse
import numpy as np
import platform
import hashlib
from datetime import datetime, timezone

def get_commit_sha():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        return "unknown"

def get_artifact_sha(base_dir):
    manifest = os.path.join(base_dir, 'hhg_rag_artifacts', 'build_manifest.json')
    if os.path.exists(manifest):
        with open(manifest, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    return "missing"

def update_env(slm_enabled: bool, saaras_enabled: bool = False):
    env_path = 'backend/.env'
    if not os.path.exists(env_path):
        return
        
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    with open(env_path, 'w') as f:
        for line in lines:
            if line.startswith('HHG_SLM_ENABLED='):
                f.write(f'HHG_SLM_ENABLED={str(slm_enabled).lower()}\n')
            elif line.startswith('HHG_SAARAS_ENABLED='):
                f.write(f'HHG_SAARAS_ENABLED={str(saaras_enabled).lower()}\n')
            elif line.startswith('HHG_CACHE_ENABLED='):
                f.write('HHG_CACHE_ENABLED=false\n')
            else:
                f.write(line)

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

def run_benchmark(mode, num_requests, delay, is_voice=False, generate=False):
    print(f"\n--- Running Benchmark: {mode} ---")
    endpoint = "voice" if is_voice else "query"
    url = f'http://127.0.0.1:8000/api/{endpoint}'
    
    # Warmup
    warmups = 10 if not is_voice else min(10, num_requests)
    print(f"Sending {warmups} warmup requests...")
    for i in range(warmups):
        q = f'what is the capital of india {uuid.uuid4()}'
        try:
            if is_voice:
                if os.path.exists('example.ogg'):
                    with open('example.ogg', 'rb') as f:
                        requests.post(url, files={'audio': ('example.ogg', f, 'audio/ogg')}, data={'language': 'en', 'generate': str(generate).lower()})
            else:
                requests.post(url, json={'query': q, 'language': 'en', 'top_k': 2, 'generate': generate})
        except:
            pass
        time.sleep(delay)
            
    print(f"Running {num_requests} measured requests...")
    
    components = {
        "language detection": [], "tokenization": [], "embedding": [], 
        "BM25": [], "HNSW": [], "RRF": [], "metadata lookup": [], 
        "extractive": [], "serialization": []
    }
    
    if mode in ["partial", "total"]:
        components.update({"grounding": [], "generation": [], "validation": []})
        
    if mode == "total":
        components["STT"] = []
        
    components[f"total_{mode}"] = []
        
    failures = 0
    not_benchmarked = False

    for i in range(num_requests):
        q = f'what is the capital of india {uuid.uuid4()}'
        try:
            if is_voice:
                if os.path.exists('example.ogg'):
                    with open('example.ogg', 'rb') as f:
                        r = requests.post(url, files={'audio': ('example.ogg', f, 'audio/ogg')}, data={'language': 'en', 'generate': str(generate).lower()})
                else:
                    failures += 1
                    continue
            else:
                r = requests.post(url, json={'query': q, 'language': 'en', 'top_k': 2, 'generate': generate})
                
            time.sleep(delay)
                
            if r.status_code == 429:
                print("Received 429 Too Many Requests. Stopping.")
                not_benchmarked = True
                break
                
            if r.status_code != 200:
                print(f"Request failed: {r.status_code} {r.text}")
                failures += 1
                if failures >= min(5, num_requests):
                    not_benchmarked = True
                    break
                continue
                
            data = r.json()
            l = data.get('latency', {})
            bd = l.get('breakdown', {})
            
            components["language detection"].append(bd.get("language_detection_ms", 0))
            components["tokenization"].append(bd.get("tokenization_ms", 0))
            components["embedding"].append(bd.get("embedding_ms", 0))
            components["BM25"].append(bd.get("bm25_ms", 0))
            components["HNSW"].append(bd.get("hnsw_ms", 0))
            components["RRF"].append(bd.get("rrf_ms", 0))
            components["metadata lookup"].append(bd.get("metadata_ms", 0))
            components["extractive"].append(bd.get("extractive_ms", 0))
            components["serialization"].append(bd.get("serialization_ms", 0))
            
            if mode in ["partial", "total"]:
                components["grounding"].append(bd.get("grounding_ms", 0))
                components["generation"].append(bd.get("generation_ms", 0))
                components["validation"].append(bd.get("validation_ms", 0))
                
            if mode == "total":
                components["STT"].append(bd.get("stt_ms", 0))
                components[f"total_{mode}"].append(l.get("total_ms", 0))
            elif mode == "partial":
                components[f"total_{mode}"].append(l.get("partial_ms", 0))
            else:
                components[f"total_{mode}"].append(l.get("rag_only_ms", 0))
                
        except Exception as e:
            print(f"Exception during request: {e}")
            failures += 1
            if failures >= min(5, num_requests):
                not_benchmarked = True
                break

    if not_benchmarked or not components[f"total_{mode}"]:
        print(f"Results for {mode}: NOT BENCHMARKED")
        return None
        
    return components

def format_table_md(components_data):
    lines = ["| stage | avg | p50 | p95 | p99 | min | max |", "|---|---|---|---|---|---|---|"]
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rag-only", "partial", "total"], required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    # Ensure previous uvicorn instances are killed
    subprocess.run("taskkill /F /IM uvicorn.exe /T", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    slm_enabled = args.mode in ["partial", "total"]
    saaras_enabled = args.mode == "total"
    
    update_env(slm_enabled, saaras_enabled)
    
    print("Starting uvicorn...")
    p = subprocess.Popen(["uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000", "--env-file", "backend/.env"])
    if not wait_for_server():
        print("Server failed to start")
        p.terminate()
        sys.exit(1)
        
    data = None
    if args.mode == "rag-only":
        data = run_benchmark("rag-only", args.requests, args.delay, is_voice=False, generate=False)
    elif args.mode == "partial":
        data = run_benchmark("partial", args.requests, args.delay, is_voice=False, generate=True)
    elif args.mode == "total":
        data = run_benchmark("total", args.requests, args.delay, is_voice=True, generate=True)
        
    p.terminate()
    p.wait()
    subprocess.run("taskkill /F /IM uvicorn.exe /T", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    report_path = "reports/latency_report.md"
    
    # Create or update report
    content = ""
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            content = f.read()
    else:
        content = f"# HHG Latency Report\n\n## Environment\n- **Platform:** {platform.platform()}\n- **Python:** {sys.version.split()[0]}\n- **Commit:** {get_commit_sha()}\n- **Artifact manifest SHA256:** {get_artifact_sha(os.getcwd())}\n- **Benchmark date:** {datetime.now(timezone.utc).isoformat()}\n\n"

    # Clean existing section if any
    mode_titles = {"rag-only": "RAG_ONLY Tier", "partial": "PARTIAL Tier", "total": "TOTAL Tier"}
    title = f"## {mode_titles[args.mode]}"
    
    # Minimal append for simplicity, but best to write a fresh block
    new_block = f"\n{title}\nConfiguration: `slm_enabled={slm_enabled}`, `saaras_enabled={saaras_enabled}`\n\n"
    if data:
        new_block += format_table_md(data) + "\n\n"
    else:
        new_block += "**NOT MEASURED** (failed or rate-limited)\n\n"

    with open(report_path, "a") as f:
        f.write(new_block)
        
    print(f"Appended results to {report_path}")

if __name__ == '__main__':
    main()
