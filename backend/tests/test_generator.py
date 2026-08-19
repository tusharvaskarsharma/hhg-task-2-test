import pytest
from unittest.mock import patch
from backend.pipeline.generator import generator_service
from backend.config import settings
from backend.pipeline.slm_client import SLMClientError
from backend.schemas.response import RetrievalResult

def _mock_res():
    return [RetrievalResult(
        doc_id="d1", text="The capital of India is New Delhi.", score=0.9, rank=1, source="wiki", language="en"
    )]

def test_slm_disabled(monkeypatch):
    monkeypatch.setattr(settings, "SLM_ENABLED", False)
    res = generator_service.generate("What is the capital of India?", "en", _mock_res())
    
    assert res["answer"] is None
    assert res["grounding"]["enabled"] is False
    assert res["grounding"]["grounded"] is False

@patch("backend.pipeline.generator.slm_client.generate")
def test_valid_generation(mock_generate, monkeypatch):
    monkeypatch.setattr(settings, "SLM_ENABLED", True)
    mock_generate.return_value = "The capital of India is New Delhi."
    
    res = generator_service.generate("What is the capital of India?", "en", _mock_res())
    
    assert res["answer"] == "The capital of India is New Delhi."
    assert res["grounding"]["enabled"] is True
    assert res["grounding"]["grounded"] is True
    assert len(res["grounding"]["sources"]) == 1

@patch("backend.pipeline.generator.slm_client.generate")
def test_insufficient_context(mock_generate, monkeypatch):
    monkeypatch.setattr(settings, "SLM_ENABLED", True)
    mock_generate.return_value = "I don't have enough information."
    
    res = generator_service.generate("What is the capital of France?", "en", _mock_res())
    
    assert "don't have enough information" in res["answer"]
    assert res["grounding"]["grounded"] is False
    assert len(res["grounding"]["sources"]) == 0

def test_empty_context(monkeypatch):
    monkeypatch.setattr(settings, "SLM_ENABLED", True)
    # With empty context, it shouldn't even call SLM
    res = generator_service.generate("query", "en", [])
    
    assert res["grounding"]["grounded"] is False
    assert "don't have enough information" in res["answer"]

@patch("backend.pipeline.generator.slm_client.generate")
def test_slm_failure(mock_generate, monkeypatch):
    monkeypatch.setattr(settings, "SLM_ENABLED", True)
    mock_generate.side_effect = SLMClientError("Provider timeout", status_code=504)
    
    res = generator_service.generate("query", "en", _mock_res())
    
    assert res["grounding"]["enabled"] is True
    assert "unavailable" in res["answer"]

@patch("backend.pipeline.generator.slm_client.generate")
def test_hallucination_detection(mock_generate, monkeypatch):
    monkeypatch.setattr(settings, "SLM_ENABLED", True)
    # The context is about New Delhi, but SLM hallucinates Tokyo
    mock_generate.return_value = "Tokyo is a very large city in Japan."
    
    res = generator_service.generate("query", "en", _mock_res())
    
    # Should be flagged as not grounded
    assert res["grounding"]["grounded"] is False
    assert len(res["grounding"]["sources"]) == 0
