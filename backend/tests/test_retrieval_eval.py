import pytest
import math
from backend.evaluation.retrieval_eval import hit_rate_at_k, recall_at_k, reciprocal_rank, ndcg_at_k, RetrievalEvaluator

def test_hit_rate_at_k():
    gold = {"doc1", "doc2"}
    retrieved = ["doc3", "doc1", "doc4"]
    
    assert hit_rate_at_k(retrieved, gold, k=1) == 0.0
    assert hit_rate_at_k(retrieved, gold, k=2) == 1.0
    assert hit_rate_at_k(retrieved, gold, k=5) == 1.0
    
    assert hit_rate_at_k(["doc3"], gold, k=1) == 0.0

def test_recall_at_k():
    gold = {"doc1", "doc2"}
    retrieved = ["doc3", "doc1", "doc4", "doc2"]
    
    assert recall_at_k(retrieved, gold, k=1) == 0.0
    assert recall_at_k(retrieved, gold, k=2) == 0.5
    assert recall_at_k(retrieved, gold, k=4) == 1.0
    
    # zero relevant docs?
    assert recall_at_k(retrieved, set(), k=4) == 0.0

def test_reciprocal_rank():
    gold = {"doc1", "doc2"}
    assert reciprocal_rank(["doc3", "doc1"], gold) == 0.5
    assert reciprocal_rank(["doc1", "doc3"], gold) == 1.0
    assert reciprocal_rank(["doc3", "doc4"], gold) == 0.0

def test_ndcg_at_k():
    gold = {"doc1"}
    retrieved = ["doc1", "doc2"]
    
    # rank 0 match
    actual = ndcg_at_k(retrieved, gold, k=1)
    # ideal is 1.0 / log2(2) = 1.0
    assert math.isclose(actual, 1.0)
    
    retrieved2 = ["doc2", "doc1"]
    # rank 1 match
    actual2 = ndcg_at_k(retrieved2, gold, k=2)
    # val is 1 / log2(3), ideal is 1 / log2(2)
    expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert math.isclose(actual2, expected)

def test_retrieval_evaluator():
    evaluator = RetrievalEvaluator(k_values=[1, 5])
    
    q_eval = evaluator.evaluate_query(
        "q1", "test query", {"doc1"}, 
        bm25_ids=["doc1", "doc2"], 
        hnsw_ids=["doc2", "doc3"], 
        rrf_ids=["doc2", "doc1"]
    )
    
    assert q_eval["bm25"]["hit_at_1"] is True
    assert q_eval["hnsw"]["hit_at_5"] is False
    assert q_eval["rrf"]["hit_at_1"] is False
    assert q_eval["rrf"]["hit_at_5"] is True
    
    agg = evaluator.aggregate_metrics([q_eval])
    assert agg["bm25"]["mrr"] == 1.0
    assert agg["hnsw"]["mrr"] == 0.0
    assert agg["rrf"]["mrr"] == 0.5
