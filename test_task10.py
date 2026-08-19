import sys
import os
import json

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import requests

url = "http://127.0.0.1:8000/api/query"

test_cases = [
    ("en", "who is the pm of india"),
    ("hi", "bharat ke pradhanmantri kaun hain"),
    ("bn", "bharater pradhanmantri ke"),
]

all_passed = True
for lang, query in test_cases:
    r = requests.post(url, json={"query": query, "language": lang, "top_k": 5})
    print(f"[{lang}] Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        answer = data.get('answer', '')
        sources = data.get('results', [])
        grounding = data.get('grounding', {})
        latency = data.get('latency', {})
        print(f"  Answer (50 chars): {answer[:50]}")
        print(f"  Sources: {len(sources)}")
        print(f"  Grounded: {grounding.get('grounded', 'N/A')}")
        print(f"  RAG_ONLY ms: {latency.get('rag_only_ms')}")
        print(f"  PARTIAL ms: {latency.get('partial_ms')}")
        print(f"  TOTAL ms: {latency.get('total_ms')}")
        if not answer or answer.startswith("I'm sorry, the generation service"):
            print(f"  WARN: SLM fallback answer")
        else:
            print(f"  OK: Real SLM answer")
    else:
        print(f"  FAIL: {r.text[:200]}")
        all_passed = False
    print("-" * 50)

if all_passed:
    print("\nAll queries returned HTTP 200 - PASS")
else:
    print("\nSome queries failed - FAIL")
