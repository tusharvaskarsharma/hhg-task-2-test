import json
import pytest
from pathlib import Path
from backend.evaluation.ground_truth import load_ground_truth, GroundTruthError

@pytest.fixture
def temp_gt_file(tmp_path):
    def _create_gt(queries, schema="hhg-ground-truth-upload-v1", dataset="ai4bharat/MSMARCO-XI"):
        p = tmp_path / "ground_truth.json"
        data = {
            "schema_version": schema,
            "source": {"dataset": dataset},
            "queries": queries
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return p
    return _create_gt

def test_valid_load(temp_gt_file):
    queries = [
        {"query_id": "q1", "language": "hi", "query": "hello", "gold_answer": "a", "relevant_passage_ids": ["123", "456"]},
        {"query_id": "q2", "language": "en", "query": "world", "gold_answer": "b", "relevant_passage_ids": [789]},
        {"query_id": "q3", "language": "bn", "query": "bn", "gold_answer": "c", "relevant_passage_ids": []}
    ]
    p = temp_gt_file(queries)
    gt = load_ground_truth(p)
    
    assert len(gt.queries) == 3
    assert len(gt.supported_queries_by_language["hi"]) == 1
    assert len(gt.supported_queries_by_language["en"]) == 1
    assert len(gt.supported_queries_by_language["bn"]) == 0
    assert len(gt.unsupported_queries_by_language["bn"]) == 1
    
    # all IDs are strings
    assert all(isinstance(i, str) for i in gt.queries[1]["relevant_passage_ids"])
    assert gt.queries[1]["relevant_passage_ids"] == ["789"]

def test_duplicate_query_id_fails(temp_gt_file):
    queries = [
        {"query_id": "q1", "language": "hi", "query": "hello", "gold_answer": "a", "relevant_passage_ids": ["123"]},
        {"query_id": "q1", "language": "hi", "query": "hello2", "gold_answer": "b", "relevant_passage_ids": ["456"]}
    ]
    p = temp_gt_file(queries)
    with pytest.raises(GroundTruthError, match="Duplicate query ID"):
        load_ground_truth(p)

def test_unknown_language_fails(temp_gt_file):
    queries = [
        {"query_id": "q1", "language": "fr", "query": "hello", "gold_answer": "a", "relevant_passage_ids": ["123"]}
    ]
    p = temp_gt_file(queries)
    with pytest.raises(GroundTruthError, match="Unsupported language: fr"):
        load_ground_truth(p)

def test_missing_fields_fails(temp_gt_file):
    queries = [
        {"query_id": "q1", "language": "hi", "query": "hello", "relevant_passage_ids": ["123"]} # Missing gold_answer
    ]
    p = temp_gt_file(queries)
    with pytest.raises(GroundTruthError, match="Missing required query fields"):
        load_ground_truth(p)

def test_missing_relevant_ids_fails(temp_gt_file):
    queries = [
        {"query_id": "q1", "language": "hi", "query": "hello", "gold_answer": "a"} # Missing relevant_ids
    ]
    p = temp_gt_file(queries)
    with pytest.raises(GroundTruthError, match="Missing required query fields"):
        load_ground_truth(p)
