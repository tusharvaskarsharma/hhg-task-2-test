import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), 'backend', '.env'))
from backend.config import settings
from backend.pipeline.saaras_client import transcribe_audio

def main():
    try:
        with open('example.ogg', 'rb') as f:
            audio_bytes = f.read()
    except Exception as e:
        print("Cannot read example.ogg")
        return
        
    for i in range(100):
        print(f"Request {i+1}...")
        try:
            res = transcribe_audio(audio_bytes, 'example.ogg')
            print(f" Success! text: {res.get('transcript', '')[:20]}...")
        except Exception as e:
            print(f" Failed: {type(e).__name__} - {e}")
            
if __name__ == "__main__":
    main()
