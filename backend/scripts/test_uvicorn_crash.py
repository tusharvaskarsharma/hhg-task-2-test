import requests
import time

url = "http://127.0.0.1:8000/api/voice"

print("Sending 100 requests to /api/voice...")
for i in range(100):
    try:
        with open('example.ogg', 'rb') as f:
            r = requests.post(url, files={'audio': ('example.ogg', f, 'audio/ogg')}, data={'language': 'en'})
            print(f"Request {i+1} status: {r.status_code}")
    except Exception as e:
        print(f"Request {i+1} failed with {type(e).__name__}: {e}")
