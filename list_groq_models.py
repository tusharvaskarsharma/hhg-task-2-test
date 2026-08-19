import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), 'backend', '.env'))
from backend.config import settings
import requests

r = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {settings.SLM_API_KEY}"}
)
if r.status_code == 200:
    models = r.json().get("data", [])
    chat_models = [m["id"] for m in models if "whisper" not in m["id"].lower()]
    print("Available chat models:")
    for m in sorted(chat_models):
        print(" -", m)
else:
    print("Error:", r.status_code, r.text)
