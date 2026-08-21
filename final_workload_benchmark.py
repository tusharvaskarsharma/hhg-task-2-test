import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
BASE_URL = "http://127.0.0.1:8000"
LANGUAGES = ["hi", "en", "bn"]


def pct(values, p):
    if not values:
        return None
    values = sorted(float(v) for v in values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def stats(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None, "mean": None}
    return {
        "count": len(vals),
        "p50": pct(vals, 50),
        "p95": pct(vals, 95),
        "p99": pct(vals, 99),
        "max": max(vals),
        "mean": statistics.mean(vals),
    }


def load_ground_truth(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    by_lang = {lang: [] for lang in LANGUAGES}
    for item in data["queries"]:
        if item.get("language") in by_lang and item.get("query"):
            by_lang[item["language"]].append(item)
    return by_lang


def make_workload(by_lang, per_language, seed):
    rng = random.Random(seed)
    workload = []
    for lang in LANGUAGES:
        source = list(by_lang[lang])
        if len(source) < per_language:
            raise RuntimeError(f"not enough ground-truth queries for {lang}: {len(source)}")
        rng.shuffle(source)
        unique_n = int(per_language * 0.60)
        repeat_n = int(per_language * 0.30)
        ood_n = per_language - unique_n - repeat_n
        unique = source[:unique_n]
        repeats = [dict(source[i % unique_n]) for i in range(repeat_n)]
        for idx, item in enumerate(repeats):
            item["workload_kind"] = "repeated"
            item["repeat_of"] = unique[idx % unique_n]["query_id"]
        ood_templates = {
            "hi": ["यह प्रश्न उपलब्ध संग्रह के बाहर की एक काल्पनिक जानकारी के बारे में है {}"],
            "en": ["This is an intentionally out-of-dataset question about an unavailable topic {}"],
            "bn": ["এটি সংগ্রহের বাইরের একটি কাল্পনিক বিষয় সম্পর্কে প্রশ্ন {}"],
        }
        entries = []
        for item in unique:
            item = dict(item)
            item["workload_kind"] = "unique"
            item["repeat_of"] = None
            entries.append(item)
        for item in repeats:
            entries.append(item)
        for idx in range(ood_n):
            entries.append({
                "query_id": f"ood:{lang}:{idx}",
                "language": lang,
                "query": ood_templates[lang][0].format(seed + idx),
                "gold_answer": "",
                "relevant_passage_ids": [],
                "workload_kind": "out_of_dataset",
                "repeat_of": None,
            })
        rng.shuffle(entries)
        workload.extend(entries)
    return workload


def wait_ready(proc, timeout=300):
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"backend exited with code {proc.returncode}: {last}")
        try:
            response = requests.get(f"{BASE_URL}/api/ready", timeout=2)
            last = f"status={response.status_code} body={response.text[:200]}"
            if response.status_code == 200:
                return
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(1)
    raise TimeoutError(f"backend did not become ready: {last}")


def clear_backend_cache():
    """Clear backend cache before benchmark run for cold-start measurement."""
    try:
        # POST to health endpoint to get cache stats, then restart to clear
        # Since there's no explicit cache-clear endpoint, we rely on backend restart
        pass
    except Exception:
        pass


def call(item, generate):
    started = time.perf_counter()
    response = requests.post(
        f"{BASE_URL}/api/query",
        json={"query": item["query"], "language": item["language"], "top_k": 10, "generate": generate},
        timeout=30,
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    body = response.json()
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {body}")
    latency = body.get("latency", {})
    breakdown = latency.get("breakdown", {})
    cache = body.get("cache", {})
    return {
        "query_id": item["query_id"],
        "language": item["language"],
        "workload_kind": item["workload_kind"],
        "query": item["query"],
        "wall_ms": wall_ms,
        "total_ms": latency.get("total_ms"),
        "partial_ms": latency.get("partial_ms"),
        "rag_only_ms": latency.get("rag_only_ms"),
        "generation_ms": breakdown.get("generation_ms", 0.0),
        "cache_layer": cache.get("cache_layer", "none"),
        "retrieval_cache_hit": bool(cache.get("retrieval_cache_hit", False)),
        "response_cache_hit": bool(cache.get("response_cache_hit", False)),
        "cache_hit": bool(cache.get("hit", False)),
        "answer_source": body.get("answer_source"),
        "results_count": len(body.get("results", [])),
        "gold_ids": item.get("relevant_passage_ids", []),
        "retrieved_ids": [x.get("id") or x.get("doc_id") for x in body.get("results", [])],
        "is_cold": not bool(cache.get("response_cache_hit", False)) and not bool(cache.get("retrieval_cache_hit", False)),
    }


def start_backend(generate):
    env = os.environ.copy()
    env["HHG_CACHE_ENABLED"] = "true"
    env["CACHE_ENABLED"] = "true"
    env["HHG_RETRIEVAL_CACHE_ENABLED"] = "true"
    env["HHG_RESPONSE_CACHE_ENABLED"] = "true"
    env["HHG_SLM_ENABLED"] = "true" if generate else "false"
    env_file = ROOT / "backend" / ".env"
    # Starting a fresh backend process clears all in-memory caches (cold start)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000", "--env-file", str(env_file)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wait_ready(proc)
    return proc


def run_mode(workload, generate, seed):
    proc = start_backend(generate)
    rows = []
    errors = []
    try:
        # Use queries that cannot collide with the measured workload.
        for index, language in enumerate(LANGUAGES):
            for warmup_index in range(7):
                warmup = {
                    "query_id": f"warmup:{language}:{index}:{warmup_index}",
                    "language": language,
                    "query": f"benchmark warmup query {language} {index} {warmup_index} {seed}",
                    "workload_kind": "warmup",
                    "relevant_passage_ids": [],
                }
                try:
                    call(warmup, generate)
                except Exception:
                    pass
        for item in workload:
            try:
                row = call(item, generate)
                rows.append(row)
            except Exception as exc:
                errors.append({"query_id": item["query_id"], "error": str(exc)})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    # Count SLM calls (generation_ms > 0 means SLM was called)
    slm_calls = sum(1 for r in rows if r.get("generation_ms", 0) > 0)
    timeout_count = sum(1 for r in rows if r.get("answer_source") == "generated-unavailable")
    fallback_count = sum(1 for r in rows if r.get("answer_source") in ("extractive", "abstain") and r.get("workload_kind") != "out_of_dataset")
    ood_abstentions = sum(1 for r in rows if r.get("workload_kind") == "out_of_dataset" and r.get("answer_source") == "abstain")
    ood_total = sum(1 for r in rows if r.get("workload_kind") == "out_of_dataset")

    # Verify repeated-query invariant
    repeated_rows = [r for r in rows if r["workload_kind"] == "repeated"]
    invariant_violations = []
    for r in repeated_rows:
        if r["response_cache_hit"]:
            if r["generation_ms"] != 0.0:
                invariant_violations.append({"query_id": r["query_id"], "issue": "response_cache_hit but generation_ms > 0"})

    out = {
        "mode": "rag+slm" if generate else "rag_only",
        "seed": seed,
        "generate": generate,
        "requests": len(workload),
        "successful_requests": len(rows),
        "errors": errors,
        "slm_calls": slm_calls,
        "timeout_count": timeout_count,
        "fallback_count": fallback_count,
        "ood_abstentions": ood_abstentions,
        "ood_total": ood_total,
        "invariant_violations": invariant_violations,
        "by_workload": {},
        "by_language": {},
        "rows": rows,
    }
    for kind in ["unique", "repeated", "out_of_dataset"]:
        subset = [r for r in rows if r["workload_kind"] == kind]
        out["by_workload"][kind] = {
            "requests": len(subset),
            "response_cache_hits": sum(r["response_cache_hit"] for r in subset),
            "retrieval_cache_hits": sum(r["retrieval_cache_hit"] for r in subset),
            "slm_calls": sum(1 for r in subset if r.get("generation_ms", 0) > 0),
            "total_ms": stats(subset, "total_ms"),
            "rag_only_ms": stats(subset, "rag_only_ms"),
            "partial_ms": stats(subset, "partial_ms"),
            "generation_ms": stats(subset, "generation_ms"),
        }
    for lang in LANGUAGES:
        subset = [r for r in rows if r["language"] == lang]
        out["by_language"][lang] = {
            "requests": len(subset),
            "response_cache_hits": sum(r["response_cache_hit"] for r in subset),
            "retrieval_cache_hits": sum(r["retrieval_cache_hit"] for r in subset),
            "total_ms": stats(subset, "total_ms"),
            "rag_only_ms": stats(subset, "rag_only_ms"),
            "partial_ms": stats(subset, "partial_ms"),
            "generation_ms": stats(subset, "generation_ms"),
        }
    out["overall"] = {
        "requests": len(rows),
        "response_cache_hits": sum(r["response_cache_hit"] for r in rows),
        "retrieval_cache_hits": sum(r["retrieval_cache_hit"] for r in rows),
        "slm_calls": slm_calls,
        "timeout_count": timeout_count,
        "fallback_count": fallback_count,
        "total_ms": stats(rows, "total_ms"),
        "rag_only_ms": stats(rows, "rag_only_ms"),
        "partial_ms": stats(rows, "partial_ms"),
        "generation_ms": stats(rows, "generation_ms"),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", default="hhg_rag_artifacts/ground_truth.json")
    parser.add_argument("--per-language", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/final_60_30_10_benchmark.json")
    args = parser.parse_args()
    by_lang = load_ground_truth(args.ground_truth)
    workload = make_workload(by_lang, args.per_language, args.seed)
    manifest = {
        "seed": args.seed,
        "per_language": args.per_language,
        "total_requests": len(workload),
        "counts": {kind: sum(x["workload_kind"] == kind for x in workload) for kind in ["unique", "repeated", "out_of_dataset"]},
        "counts_by_language": {lang: {kind: sum(x["language"] == lang and x["workload_kind"] == kind for x in workload) for kind in ["unique", "repeated", "out_of_dataset"]} for lang in LANGUAGES},
    }
    output = {"workload": manifest, "modes": []}
    for generate in [False, True]:
        output["modes"].append(run_mode(workload, generate, args.seed))
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": args.output, "workload": manifest, "mode_summaries": [{"mode": x["mode"], "overall": x["overall"], "slm_calls": x.get("slm_calls", 0), "timeout_count": x.get("timeout_count", 0), "ood_abstentions": x.get("ood_abstentions", 0), "invariant_violations": len(x.get("invariant_violations", [])), "errors": len(x["errors"])} for x in output["modes"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
