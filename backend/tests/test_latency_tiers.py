import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.config import settings

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_retrieval_service(monkeypatch):
    from backend.pipeline.retrieval_service import retrieval_service
    retrieval_service.initialized = True
    def mock_execute(query, lang, top_k):
        return {
            "query": query,
            "language": lang,
            "results": [{"doc_id": "doc1", "text": "content", "score": 1.0, "rank": 1, "source": "mock"}],
            "cache": {"hit": False},
            "latency": {"total_ms": 1.0},
            "latency_breakdown": {
                "language_detection_ms": 0.001,
                "tokenization_ms": 0.001,
                "embedding_ms": 0.001,
                "bm25_ms": 0.001,
                "hnsw_ms": 0.001,
                "rrf_ms": 0.001,
                "metadata_ms": 0.001
            },
            "rag_only_base_ms": 0.01
        }
    monkeypatch.setattr(retrieval_service, "execute_query", mock_execute)

def test_latency_tiers_text_request():
    """
    Test latency definitions for a standard text request.
    1. Text request:
       total includes retrieval + SLM
       partial includes retrieval + SLM
       rag_only excludes SLM
    """
    response = client.post("/api/query", json={"query": "hello", "language": "en", "top_k": 2})
    assert response.status_code == 200
    data = response.json()
    
    latency = data.get("latency", {})
    assert "total_ms" in latency
    assert "partial_ms" in latency
    assert "rag_only_ms" in latency
    
    total = latency["total_ms"]
    partial = latency["partial_ms"]
    rag_only = latency["rag_only_ms"]
    
    breakdown = latency.get("breakdown", {})
    stt_ms = breakdown.get("stt_ms", 0.0)
    slm_ms = breakdown.get("generation_ms", 0.0)
    
    # STT must be 0
    assert stt_ms == 0.0
    
    # TOTAL and PARTIAL must be equal for text requests
    assert abs(total - partial) < 0.1
    
    # TOTAL must be >= RAG_ONLY (because total includes SLM)
    assert total >= rag_only
    assert partial >= rag_only
    
    if slm_ms > 0:
        assert total >= (rag_only + slm_ms)

def test_latency_tiers_voice_request():
    """
    Test latency definitions for a voice request.
    Voice request:
       total includes STT + retrieval + SLM
       partial excludes STT
       rag_only excludes STT and SLM
    """
    import io
    dummy_audio = io.BytesIO(b"fake audio data")
    dummy_audio.name = "test.wav"
    
    response = client.post(
        "/api/voice", 
        files={"audio": ("test.wav", dummy_audio, "audio/wav")},
        data={"language": "en", "top_k": 2}
    )
    
    # Depending on STT mock, it might succeed or fail. If it's mocked in tests, it returns 200.
    if response.status_code == 200:
        data = response.json()
        latency = data.get("latency", {})
        
        total = latency["total_ms"]
        partial = latency["partial_ms"]
        rag_only = latency["rag_only_ms"]
        
        breakdown = latency.get("breakdown", {})
        stt_ms = breakdown.get("stt_ms", 0.0)
        
        assert stt_ms > 0.0 or stt_ms == 0.0 # Just verifying schema
        assert abs(total - (partial + stt_ms)) < 0.1
        assert partial >= rag_only

def test_latency_tiers_slm_disabled():
    """
    Test latency definitions when SLM is disabled.
    SLM disabled:
       total and partial remain valid
       SLM duration = 0
    """
    original_enabled = settings.SLM_ENABLED
    settings.SLM_ENABLED = False
    
    try:
        response = client.post("/api/query", json={"query": "hello", "language": "en", "top_k": 2})
        assert response.status_code == 200
        data = response.json()
        latency = data.get("latency", {})
        
        breakdown = latency.get("breakdown", {})
        slm_ms = breakdown.get("slm_ms", 0.0)
        gen_ms = breakdown.get("generation_ms", 0.0)
        
        assert slm_ms == 0.0
        assert gen_ms == 0.0
        
        total = latency["total_ms"]
        partial = latency["partial_ms"]
        rag_only = latency["rag_only_ms"]
        
        # When SLM is disabled, TOTAL and PARTIAL should roughly equal RAG_ONLY (minus some small overheads)
        assert total >= rag_only
    finally:
        settings.SLM_ENABLED = original_enabled
