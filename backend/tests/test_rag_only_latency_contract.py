import pytest
from fastapi.testclient import TestClient
from backend.main import app
import time

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_retrieval_service(monkeypatch):
    from backend.pipeline.retrieval_service import retrieval_service
    retrieval_service.initialized = True
    def mock_execute(query, lang, top_k):
        return {
            "query": query,
            "language": lang,
            "results": [{"doc_id": "doc1", "text": "Delhi is the capital of India.", "score": 1.0, "rank": 1, "source": "mock"}],
            "cache": {"hit": False},
            "latency": {"total_ms": 1.0},
            "latency_breakdown": {},
            "rag_only_base_ms": 0.7
        }
    monkeypatch.setattr(retrieval_service, "execute_query", mock_execute)

def test_rag_only_latency_contract_no_slm():
    """
    Test that when generate=False, the generation_ms and stt_ms are 0.0
    and the answer_source is either 'extractive' or 'abstain'.
    """
    payload = {
        "query": "what is the capital of india?",
        "language": "en",
        "top_k": 2,
        "generate": False
    }
    
    
    with TestClient(app) as client:
        t0 = time.perf_counter()
        response = client.post("/api/query", json=payload)
        elapsed = (time.perf_counter() - t0) * 1000.0
    
    assert response.status_code == 200
    
    data = response.json()
    assert data["answer_source"] in ["extractive", "abstain"]
    
    metrics = data.get("metrics", {})
    breakdown = data.get("latency_breakdown", {})
    
    # Must explicitly not call SLM or STT
    assert breakdown.get("generation_ms", 0.0) == 0.0
    assert breakdown.get("stt_ms", 0.0) == 0.0
    
    # Ensure RAG_ONLY path is actually fast in the test environment (e.g. < 100ms)
    # The actual < 50ms requirement is tested in the benchmark.
    assert elapsed < 500  # generous bound for CI
