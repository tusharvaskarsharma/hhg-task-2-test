"""
Tests for RAG+SLM optimisation integration at the API route level.
Covers: OOD early exit, RAG_ONLY SLM invariant, response cache integration,
        and schema backward compatibility.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.main import app
from backend.config import settings
from backend.pipeline.query_cache import cache_instance

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_retrieval_and_reset(monkeypatch):
    """Mock retrieval service and reset cache before each test."""
    from backend.pipeline.retrieval_service import retrieval_service
    from backend.schemas.response import RetrievalResult
    retrieval_service.initialized = True
    
    cache_instance.clear()
    
    def mock_execute(query, lang, top_k):
        return {
            "query": query,
            "language": lang,
            "results": [
                RetrievalResult(
                    doc_id="doc1",
                    text="The capital of India is New Delhi. It is the seat of the government.",
                    score=0.9,
                    rank=1,
                    source="rrf(bm25,hnsw)"
                )
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


def test_rag_only_never_invokes_slm(monkeypatch):
    """RAG_ONLY (generate=False) must never call the SLM, regardless of SLM_ENABLED."""
    monkeypatch.setattr(settings, "SLM_ENABLED", True)
    
    slm_calls = [0]
    original_generate = None
    
    with patch("backend.pipeline.generator.slm_client.generate") as mock_gen:
        mock_gen.side_effect = lambda prompt: (_ for _ in ()).throw(AssertionError("SLM should not be called for RAG_ONLY"))
        
        response = client.post("/api/query", json={"query": "What is the capital of India?", "language": "en", "top_k": 5, "generate": False})
        assert response.status_code == 200
        mock_gen.assert_not_called()
    
    data = response.json()
    assert data["answer_source"] in ["extractive", "abstain"]
    bd = data["latency"]["breakdown"]
    assert bd["generation_ms"] == 0.0
    assert bd["slm_ms"] == 0.0


def test_ood_early_exit_skips_slm(monkeypatch):
    """When extractive says ABSTAIN, SLM should be skipped (OOD early exit)."""
    from backend.pipeline.retrieval_service import retrieval_service
    monkeypatch.setattr(settings, "SLM_ENABLED", True)
    
    # Return results that won't match the query (simulating OOD)
    def mock_execute_ood(query, lang, top_k):
        return {
            "query": query,
            "language": lang,
            "results": [
                {
                    "doc_id": "doc_unrelated",
                    "text": "Photosynthesis is the process by which plants convert sunlight into energy.",
                    "score": 0.1,
                    "rank": 1,
                    "source": "rrf(bm25)"
                }
            ],
            "cache": {"hit": False},
            "latency_breakdown": {
                "language_detection_ms": 0.1, "tokenization_ms": 0.1,
                "embedding_ms": 0.1, "bm25_ms": 0.1, "hnsw_ms": 0.1,
                "rrf_ms": 0.1, "metadata_ms": 0.1
            },
            "rag_only_base_ms": 0.7
        }
    monkeypatch.setattr(retrieval_service, "execute_query", mock_execute_ood)
    
    with patch("backend.pipeline.generator.slm_client.generate") as mock_gen:
        mock_gen.side_effect = AssertionError("SLM should not be called for OOD query")
        
        response = client.post("/api/query", json={
            "query": "What is the quantum chromodynamics coupling constant at TeV scale?",
            "language": "en", "top_k": 5, "generate": True
        })
        assert response.status_code == 200
        mock_gen.assert_not_called()
    
    data = response.json()
    assert data["answer_source"] == "abstain"
    assert data["latency"]["breakdown"]["generation_ms"] == 0.0


@patch("backend.pipeline.generator.slm_client.generate")
def test_response_cache_hit_returns_generation_ms_zero(mock_gen, monkeypatch):
    """Repeated identical requests should hit response cache with generation_ms=0."""
    monkeypatch.setattr(settings, "SLM_ENABLED", True)
    mock_gen.return_value = "The capital of India is New Delhi."
    
    # First request
    r1 = client.post("/api/query", json={"query": "What is the capital of India?", "language": "en", "top_k": 5, "generate": True})
    assert r1.status_code == 200
    d1 = r1.json()
    
    # Second identical request
    r2 = client.post("/api/query", json={"query": "What is the capital of India?", "language": "en", "top_k": 5, "generate": True})
    assert r2.status_code == 200
    d2 = r2.json()
    
    # Second request should be a response cache hit
    assert d2["cache"]["response_cache_hit"] is True
    assert d2["cache"]["cache_layer"] == "response"
    assert d2["latency"]["breakdown"]["generation_ms"] == 0.0
    assert d2["latency"]["breakdown"]["slm_ms"] == 0.0
    
    # SLM should only be called once (first request)
    assert mock_gen.call_count == 1


def test_response_schema_backward_compatible():
    """Response must contain all expected fields for backward compatibility."""
    response = client.post("/api/query", json={"query": "test", "language": "en"})
    assert response.status_code == 200
    data = response.json()
    
    # Core fields
    assert "query" in data
    assert "language" in data
    assert "answer" in data
    assert "answer_source" in data
    assert "results" in data
    assert "grounding" in data
    
    # Cache fields
    cache = data["cache"]
    assert "hit" in cache
    assert "enabled" in cache
    assert "cache_layer" in cache
    assert "retrieval_cache_hit" in cache
    assert "response_cache_hit" in cache
    assert "cache_key_version" in cache
    assert "cache_lookup_ms" in cache
    
    # Latency fields
    latency = data["latency"]
    assert "total_ms" in latency
    assert "partial_ms" in latency
    assert "rag_only_ms" in latency
    assert "breakdown" in latency
    bd = latency["breakdown"]
    assert "generation_ms" in bd
    assert "slm_ms" in bd
    assert "extractive_ms" in bd
    assert "serialization_ms" in bd


def test_no_api_keys_in_response():
    """API keys must not appear in response data."""
    import json
    response = client.post("/api/query", json={"query": "test", "language": "en"})
    assert response.status_code == 200
    text = json.dumps(response.json())
    
    # Check that no API key patterns appear
    assert "Bearer" not in text
    assert settings.SLM_API_KEY not in text or settings.SLM_API_KEY == ""
    assert settings.SAARAS_API_KEY not in text or settings.SAARAS_API_KEY == ""
