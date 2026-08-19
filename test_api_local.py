import httpx
import json

def run_tests():
    url = "http://127.0.0.1:8000/api/query"
    
    queries = [
        {"query": "भारत की राजधानी क्या है?", "language": "hi"},
        {"query": "বাংলাদেশের রাজধানী কী?", "language": "bn"},
        {"query": "What is the capital of India?", "language": "en"}
    ]
    
    with open("test_output.txt", "w", encoding="utf-8") as out:
        for q in queries:
            out.write(f"\n--- Testing query: {q['query']} ({q['language']}) ---\n")
            response = httpx.post(url, json={"query": q["query"], "language": q["language"], "top_k": 5}, timeout=60.0)
            if response.status_code != 200:
                out.write(f"FAILED! Status: {response.status_code}, Body: {response.text}\n")
                continue
                
            data = response.json()
            results = data.get("results", [])
            out.write(f"Got {len(results)} results. Cache hit: {data.get('cache', {}).get('hit')}\n")
            
            # Verify language filtering
            for i, res in enumerate(results[:2]):
                out.write(f"Result {i+1} [Lang: {res.get('language')}, Source: {res.get('source')}]: {res.get('text')[:100]}...\n")
                
if __name__ == "__main__":
    run_tests()
