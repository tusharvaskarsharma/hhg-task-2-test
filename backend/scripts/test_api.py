import requests
import sys

BASE_URL = "http://localhost:8000"

def run_smoke_tests():
    print("Running API Smoke Tests...")
    
    # 1. Health
    try:
        res = requests.get(f"{BASE_URL}/api/health")
        print(f"GET /api/health -> {res.status_code}")
    except Exception as e:
        print(f"Server unreachable: {e}")
        sys.exit(1)
        
    # 2. Ready
    res = requests.get(f"{BASE_URL}/api/ready")
    print(f"GET /api/ready -> {res.status_code}")
    
    if res.status_code != 200:
        print("Backend not ready, skipping query test.")
        sys.exit(0)
        
    # 3. Query
    payload = {
        "query": "भारत की राजधानी क्या है?",
        "language": "hi",
        "top_k": 3
    }
    res = requests.post(f"{BASE_URL}/api/query", json=payload)
    print(f"POST /api/query -> {res.status_code}")
    if res.status_code == 200:
        data = res.json()
        print(f"Request ID: {res.headers.get('X-Request-ID')}")
        print(f"Results returned: {len(data['results'])}")
        print(f"Latency: {data['latency']['total_ms']}ms")
        print(f"Cache hit: {data['cache']['hit']}")
    else:
        print(res.text)
        
if __name__ == "__main__":
    run_smoke_tests()
