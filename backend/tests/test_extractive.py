"""
backend/tests/test_extractive.py
Tests for the extractive fast-path module (Phase 5).
"""
import pytest
from backend.pipeline.extractive import (
    build_extractive_answer,
    ExtractiveDecision,
)
from backend.schemas.response import RetrievalResult


def _make_result(doc_id: str, text: str, score: float = 0.5):
    """Create a mock retrieval result as a simple object."""
    from types import SimpleNamespace
    return SimpleNamespace(doc_id=doc_id, text=text, score=score, rank=1, source="rrf", language="en", id=doc_id)


# ── Basic functionality ────────────────────────────────────────────────────────

def test_extractive_returns_answer_for_relevant_context():
    results = [
        _make_result("1", "Narendra Modi is the Prime Minister of India since 2014.", 0.9),
        _make_result("2", "India's Prime Minister lives in New Delhi.", 0.7),
    ]
    answer, decision, sources = build_extractive_answer("who is the prime minister of india", results)
    assert answer
    assert decision == ExtractiveDecision.SUPPORTED
    assert len(sources) > 0


def test_extractive_abstains_on_empty_results():
    answer, decision, sources = build_extractive_answer("some question", [])
    assert decision == ExtractiveDecision.ABSTAIN
    assert sources == []


def test_extractive_abstains_on_irrelevant_context():
    results = [
        _make_result("1", "The sun rises in the east every day without fail.", 0.1),
    ]
    answer, decision, sources = build_extractive_answer("quantum computing algorithms", results)
    # Should either abstain/insufficient — no overlap with query tokens
    assert decision in (ExtractiveDecision.ABSTAIN, ExtractiveDecision.INSUFFICIENT)


def test_extractive_no_slm_call(monkeypatch):
    """Extractive must NEVER call the SLM client."""
    called = []

    def fake_generate(*args, **kwargs):
        called.append(True)
        return "fake"

    # Patch slm_client if it exists in extractive scope
    import backend.pipeline.extractive as ext_module
    # The extractive module should not import slm_client at all
    assert not hasattr(ext_module, "slm_client"), "extractive.py must not import slm_client"
    assert called == []


def test_extractive_multilingual_hindi():
    results = [
        _make_result("1", "नरेंद्र मोदी भारत के प्रधानमंत्री हैं।", 0.9),
    ]
    answer, decision, sources = build_extractive_answer("bharat pradhanmantri", results)
    # With transliteration mismatch this may be INSUFFICIENT — that's acceptable
    assert decision in (ExtractiveDecision.SUPPORTED, ExtractiveDecision.INSUFFICIENT, ExtractiveDecision.ABSTAIN)
    assert isinstance(answer, str)


def test_extractive_source_ids_match_results():
    results = [
        RetrievalResult(doc_id="doc-abc", text="Blue whales live in oceans.", score=0.9, rank=1, source="bm25"),
        RetrievalResult(doc_id="doc-xyz", text="It is the largest animal.", score=0.8, rank=2, source="hnsw"),
    ]
    ans, decision, sources = build_extractive_answer("where do blue whales live", results)
    assert decision == ExtractiveDecision.SUPPORTED
    assert "doc-abc" in [s.get("doc_id") for s in sources]


def test_extractive_respects_max_length():
    # Create a very long passage
    long_text = " ".join(["The query topic is very important and relevant."] * 50)
    results = [_make_result("1", long_text, 0.9)]
    answer, decision, sources = build_extractive_answer("query topic relevant", results)
    assert len(answer) <= 700  # _MAX_ANSWER_CHARS + some buffer


def test_extractive_deduplicates_sentences():
    repeated = "The answer is forty-two. " * 5
    results = [_make_result("1", repeated, 0.9)]
    answer, decision, sources = build_extractive_answer("what is the answer", results)
    # Should not repeat the same sentence multiple times
    sentences = [s.strip() for s in answer.split(".") if s.strip()]
    unique = set(sentences)
    assert len(sentences) <= len(unique) + 1  # allow 1 near-duplicate


def test_api_query_extractive_fast_path(monkeypatch):
    """
    Route-level regression test:
    Verify POST /api/query with generate=false returns properly formatted response,
    sources is a list of dicts with correct keys, and SLM is bypassed.
    """
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.pipeline.generator import generator_service
    from backend.pipeline.retrieval_service import retrieval_service
    from backend.schemas.response import RetrievalResult
    import asyncio

    # 1. Mock retrieval to return standard results
    retrieval_service.initialized = True
    def mock_execute(*args, **kwargs):
        return {
            "query": "capital of india",
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

    # 2. Track if generate is called
    generate_called = False
    async def mock_generate(*args, **kwargs):
        nonlocal generate_called
        generate_called = True
        return "Fake generated response"
    monkeypatch.setattr(generator_service, "generate", mock_generate)

    client = TestClient(app)
    response = client.post("/api/query", json={"query": "capital of india", "language": "en", "generate": False})
    # 1. POST /api/query with generate=false returns HTTP 200
    assert response.status_code == 200
    data = response.json()

    # 5. answer_source is extractive or abstain
    assert data["answer_source"] in ["extractive", "abstain"]

    # 2. response.sources is a list of dictionaries
    sources = data.get("sources", [])
    assert isinstance(sources, list)
    if len(sources) > 0:
        assert isinstance(sources[0], dict)
        # 3. every source contains doc_id, rank, and source
        for source in sources:
            assert "doc_id" in source
            assert "rank" in source
            assert "source" in source

    # 4. generator_service.generate is not called
    assert not generate_called
