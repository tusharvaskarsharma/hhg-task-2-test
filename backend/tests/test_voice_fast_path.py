import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.pipeline.stt import stt_service
from backend.pipeline.retrieval_service import retrieval_service
from backend.pipeline.generator import generator_service
from backend.schemas.response import RetrievalResult
import io

@pytest.fixture
def test_client(monkeypatch):
    # Mock STT
    def mock_transcribe(audio_bytes, filename):
        return {"text": "what is the capital of india", "language": "en"}
    monkeypatch.setattr(stt_service, "transcribe", mock_transcribe)
    
    # Mock settings.SAARAS_ENABLED
    from backend.config import settings
    monkeypatch.setattr(settings, "SAARAS_ENABLED", True)

    # Mock Retrieval
    retrieval_service.initialized = True
    def mock_execute(*args, **kwargs):
        return {
            "query": "what is the capital of india",
            "language": "en",
            "results": [
                RetrievalResult(doc_id="doc1", text="The capital of India is New Delhi.", score=0.9, rank=1, source="bm25")
            ],
            "cache": {"hit": False},
            "latency": {"total_ms": 1.0},
            "latency_breakdown": {
                "language_detection_ms": 1.0,
                "tokenization_ms": 1.0,
                "embedding_ms": 1.0,
                "bm25_ms": 1.0,
                "hnsw_ms": 1.0,
                "rrf_ms": 1.0,
                "metadata_ms": 1.0
            },
            "rag_only_base_ms": 7.0
        }
    monkeypatch.setattr(retrieval_service, "execute_query", mock_execute)
    
    return TestClient(app)

def test_voice_generate_false(test_client, monkeypatch):
    generate_called = False
    def mock_generate(*args, **kwargs):
        nonlocal generate_called
        generate_called = True
        return {"answer": "fake", "grounding": {"grounded": True}}
    monkeypatch.setattr(generator_service, "generate", mock_generate)

    dummy_audio = io.BytesIO(b"fake audio data")
    dummy_audio.name = "test.wav"
    
    response = test_client.post(
        "/api/voice", 
        files={"audio": ("test.wav", dummy_audio, "audio/wav")},
        data={"language": "en", "generate": "false"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # 1. generate=false is accepted
    # 2. generator_service.generate is not called
    assert not generate_called
    
    # 3. response contains answer_source, sources, transcription, and timing breakdown
    assert "answer_source" in data
    assert data["answer_source"] in ["extractive", "abstain"]
    
    assert "sources" in data
    assert isinstance(data["sources"], list)
    
    assert "transcription" in data
    assert data["transcription"]["text"] == "what is the capital of india"
    
    assert "latency" in data
    assert "breakdown" in data["latency"]
    assert "stt_ms" in data["latency"]["breakdown"]

def test_voice_generate_true_ungrounded_fallback(test_client, monkeypatch):
    def mock_generate(*args, **kwargs):
        return {
            "answer": "I don't know the capital.",
            "answer_source": "generated",
            "grounding": {"grounded": False},
            "latency": {"grounding_ms": 1.0, "generation_ms": 500.0, "grounding_validation_ms": 1.0}
        }
    monkeypatch.setattr(generator_service, "generate", mock_generate)

    dummy_audio = io.BytesIO(b"fake audio data")
    dummy_audio.name = "test.wav"
    
    response = test_client.post(
        "/api/voice", 
        files={"audio": ("test.wav", dummy_audio, "audio/wav")},
        data={"language": "en", "generate": "true"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # 4. ungrounded generation falls back to extractive when generate=true
    assert data["answer_source"] == "extractive"
    # the answer should be the extracted sentence
    assert "New Delhi" in data["answer"]
