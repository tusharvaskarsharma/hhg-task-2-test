import os
import sys



import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_retrieval_service(monkeypatch):
    from backend.pipeline.retrieval_service import retrieval_service
    
    # Force initialize
    retrieval_service.initialized = True
    
    def mock_execute(query, lang, top_k):
        return {
            "query": query,
            "language": lang,
            "results": [
                {
                    "doc_id": "doc1",
                    "text": "content",
                    "score": 1.0,
                    "rank": 1,
                    "source": "mock"
                }
            ],
            "cache": {"hit": False},
            "latency": {"total_ms": 1.0},
            "latency_breakdown": {
                "language_detection_ms": 0.1,
                "tokenization_ms": 0.1,
                "embedding_ms": 0.1,
                "bm25_ms": 0.1,
                "hnsw_ms": 0.1,
                "rrf_ms": 0.1,
                "metadata_ms": 0.1
            },
            "rag_only_base_ms": 0.7
        }
    monkeypatch.setattr(retrieval_service, "execute_query", mock_execute)

def test_ready_endpoint():
    response = client.get("/api/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"

def test_health_endpoint():
    response = client.get("/api/health")
    # Health checks the loader status. In mock, it's not valid by default but it returns structured JSON safely.
    # It will return 503 if not fully valid, which is fine, but JSON structure must be solid.
    data = response.json()
    assert "status" in data
    assert "artifacts" in data

def test_query_valid():
    response = client.post("/api/query", json={"query": "test query", "language": "en", "top_k": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert data["language"] == "en"
    assert len(data["results"]) == 1
    assert "answer" in data
    assert "grounding" in data
    assert "X-Request-ID" in response.headers

def test_query_invalid_language():
    response = client.post("/api/query", json={"query": "test query", "language": "fr"})
    assert response.status_code == 422 # Pydantic validation error

def test_query_empty():
    response = client.post("/api/query", json={"query": "", "language": "en"})
    assert response.status_code == 422

def test_query_too_long():
    response = client.post("/api/query", json={"query": "a" * 600, "language": "en"})
    assert response.status_code == 422

def test_query_top_k_bounds():
    response = client.post("/api/query", json={"query": "test query", "language": "en", "top_k": 0})
    assert response.status_code == 422
    
    response2 = client.post("/api/query", json={"query": "test query", "language": "en", "top_k": 500})
    assert response2.status_code == 422

def test_request_id_preserved():
    response = client.post("/api/query", json={"query": "test query"}, headers={"X-Request-ID": "custom-id-123"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "custom-id-123"

def test_security_headers():
    response = client.get("/api/ready")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

def test_not_ready_behavior(monkeypatch):
    from backend.pipeline.retrieval_service import retrieval_service
    retrieval_service.initialized = False
    
    response = client.get("/api/ready")
    assert response.status_code == 503
    
    res2 = client.post("/api/query", json={"query": "test"})
    assert res2.status_code == 503
    assert res2.json()["error"]["code"] == "SERVICE_UNAVAILABLE"

def test_voice_endpoint_success(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "SAARAS_ENABLED", True)
    
    def mock_transcribe(audio_bytes, filename):
        return {"transcript": "voice query test", "language_code": "hi-IN"}
        
    monkeypatch.setattr("backend.pipeline.stt.transcribe_audio", mock_transcribe)
    
    # Needs to be a valid audio file mock payload
    files = {"audio": ("test.wav", b"fakeaudiobytes", "audio/wav")}
    data = {"top_k": 5}
    
    response = client.post("/api/voice", files=files, data=data)
    
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["query"] == "voice query test"
    assert res_data["language"] == "hi"
    assert "total_ms" in res_data["latency"]
    assert "answer" in res_data
    assert "grounding" in res_data
    assert "enabled" in res_data["grounding"]
    assert "generation_ms" in res_data["latency"]["breakdown"]
    assert len(res_data["results"]) == 1

def test_voice_endpoint_disabled(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "SAARAS_ENABLED", False)
    
    files = {"audio": ("test.wav", b"fakeaudiobytes", "audio/wav")}
    
    response = client.post("/api/voice", files=files)
    
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "STT_FAILURE"
