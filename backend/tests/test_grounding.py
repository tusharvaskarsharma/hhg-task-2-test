import pytest
from backend.schemas.response import RetrievalResult
from backend.pipeline.grounding import grounding_service
from backend.config import settings

def _mock_res(rank, text):
    return RetrievalResult(
        doc_id=f"doc_{rank}",
        text=text,
        score=0.9,
        rank=rank,
        source="wiki",
        language="hi"
    )

def test_empty_retrieval():
    ctx, sources = grounding_service.build_context([])
    assert ctx == ""
    assert sources == []

def test_single_passage():
    results = [_mock_res(1, "Hello world")]
    ctx, sources = grounding_service.build_context(results)
    
    assert "Hello world" in ctx
    assert "[Doc 1 | Source: wiki]" in ctx
    assert len(sources) == 1
    assert sources[0]["doc_id"] == "doc_1"

def test_top_k_truncation(monkeypatch):
    monkeypatch.setattr(settings, "GROUNDING_TOP_K", 2)
    results = [
        _mock_res(1, "One"),
        _mock_res(2, "Two"),
        _mock_res(3, "Three")
    ]
    
    # We must instantiate a new GroundingService to pick up the mocked setting
    from backend.pipeline.grounding import GroundingService
    svc = GroundingService()
    
    ctx, sources = svc.build_context(results)
    assert "One" in ctx
    assert "Two" in ctx
    assert "Three" not in ctx
    assert len(sources) == 2

def test_character_limit(monkeypatch):
    monkeypatch.setattr(settings, "MAX_CONTEXT_CHARS", 50)
    
    # "Doc 1" snippet will be around 25 chars. The first doc alone will fit, second might cross 50.
    results = [
        _mock_res(1, "A very long sentence here"),
        _mock_res(2, "Another very long sentence")
    ]
    
    from backend.pipeline.grounding import GroundingService
    svc = GroundingService()
    
    ctx, sources = svc.build_context(results)
    
    assert len(ctx) <= 50
    assert len(sources) == 1
    assert sources[0]["rank"] == 1
