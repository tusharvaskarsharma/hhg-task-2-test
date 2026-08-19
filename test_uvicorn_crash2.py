import requests
import time
import uuid

url_query = "http://127.0.0.1:8000/api/query"
url_voice = "http://127.0.0.1:8000/api/voice"

print("Sending 100 PARTIAL requests...")
for i in range(100):
    try:
        r = requests.post(url_query, json={"query": f"test {uuid.uuid4()}", "language": "en", "top_k": 10})
        if r.status_code != 200:
            print(f"PARTIAL {i+1} failed with {r.status_code}")
    except Exception as e:
        print(f"PARTIAL {i+1} exception: {e}")

print("Sending 10 TOTAL warmup requests...")
for i in range(10):
    try:
        with open('example.ogg', 'rb') as f:
            r = requests.post(url_voice, files={'audio': ('example.ogg', f, 'audio/ogg')}, data={'language': 'en'})
    except Exception as e:
        print(f"TOTAL warmup {i+1} exception: {e}")

print("Sending 100 TOTAL benchmark requests...")
for i in range(100):
    try:
        with open('example.ogg', 'rb') as f:
            r = requests.post(url_voice, files={'audio': ('example.ogg', f, 'audio/ogg')}, data={'language': 'en'})
            if r.status_code != 200:
                print(f"TOTAL {i+1} failed with {r.status_code}")
    except Exception as e:
        print(f"TOTAL {i+1} exception: {e}")
