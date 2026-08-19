import pytest
from backend.pipeline.extractive import build_extractive_answer, ExtractiveDecision
from backend.schemas.response import RetrievalResult
from backend.config import settings

def test_unrelated_passages_abstain():
    query = "who is the prime minister of india"
    results = [
        RetrievalResult(doc_id="1", text="The prime minister of Canada is Justin Trudeau.", score=0.9, rank=1, source="bm25", language="en"),
        RetrievalResult(doc_id="2", text="Peru has a president, not a prime minister.", score=0.8, rank=2, source="bm25", language="en")
    ]
    answer, decision, sources = build_extractive_answer(query, results)
    assert decision in (ExtractiveDecision.ABSTAIN, ExtractiveDecision.INSUFFICIENT)
    assert len(sources) == 0

def test_candidate_missing_india_abstain():
    query = "who is the prime minister of india"
    results = [
        RetrievalResult(doc_id="1", text="Narendra Modi is a powerful prime minister.", score=0.9, rank=1, source="bm25", language="en")
    ]
    answer, decision, sources = build_extractive_answer(query, results)
    assert decision in (ExtractiveDecision.ABSTAIN, ExtractiveDecision.INSUFFICIENT)

def test_candidate_with_both_subject_and_answer_extractive():
    query = "who is the prime minister of india"
    results = [
        RetrievalResult(doc_id="1", text="Narendra Modi is the current prime minister of India.", score=0.9, rank=1, source="bm25", language="en")
    ]
    answer, decision, sources = build_extractive_answer(query, results)
    assert decision == ExtractiveDecision.SUPPORTED
    assert "Modi" in answer

def test_empty_retrieval_abstain():
    query = "who is the prime minister of india"
    answer, decision, sources = build_extractive_answer(query, [])
    assert decision == ExtractiveDecision.ABSTAIN

def test_query_with_punctuation():
    query = "Who is the Prime Minister of India??"
    results = [
        RetrievalResult(doc_id="1", text="Narendra Modi is the current prime minister of India.", score=0.9, rank=1, source="bm25", language="en")
    ]
    answer, decision, sources = build_extractive_answer(query, results)
    assert decision == ExtractiveDecision.SUPPORTED
