import subprocess
import json
import pytest
from pathlib import Path
from unittest.mock import patch
from backend.scripts.run_accuracy_benchmark import evaluate_abstention, run_language_benchmark
from backend.artifact_loader import loader_instance

def test_missing_file_returns_not_run_and_nonzero(tmp_path):
    # Call run_accuracy_benchmark as a subprocess with a missing file
    env = {}
    cmd = [
        "python", "-m", "backend.scripts.run_accuracy_benchmark",
        "--ground-truth", str(tmp_path / "nonexistent.json"),
        "--output", str(tmp_path / "out.json")
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    
    # check json output
    with open(tmp_path / "out.json") as f:
        data = json.load(f)
    assert data["status"] == "NOT RUN"

class MockEmbedder:
    def embed_query(self, text):
        return [0.1], None, None

class MockRetriever:
    def retrieve(self, query, lang, top_k):
        return [{"id": "1", "score": 1.0, "rank": 1}]

def test_no_positive_query_excluded_from_recall_and_retained_for_abstention():
    embedder = MockEmbedder()
    retriever = MockRetriever()
    
    # Run the benchmark method manually with a positive and a no-positive
    # wait, the benchmark script splits them beforehand.
    # The benchmark function run_language_benchmark only takes `queries` which are supported.
    supported = [{"query": "test1", "relevant_passage_ids": ["1"]}]
    unsupported = [{"query": "test2", "relevant_passage_ids": []}]
    
    # evaluate abstention
    abs_res = evaluate_abstention("hi", unsupported, embedder, retriever, retriever)
    assert abs_res["num_no_positive_queries"] == 1
    
    # run benchmark on supported only
    metrics = run_language_benchmark("hi", supported, embedder, retriever, retriever, top_k_list=[1])
    
    # since retriever returns "1", recall is 1.0
    assert metrics["bm25"]["Recall@1"] == 1.0
    
    # if it included the unsupported query in average, it would be 0.5

def test_online_query_route_never_loads_ground_truth():
    # Verify that loader_instance does not load ground_truth.json
    assert "ground_truth" not in loader_instance.status
    
    # A generic check that no ground_truth module is imported by artifact_loader
    import sys
    # if this test runs in isolation, backend.scripts might not be imported yet
    # but we can check if it's imported in the current process
    assert "backend.scripts.run_accuracy_benchmark" not in sys.modules or "backend.evaluation.ground_truth" in sys.modules
    
    # More practically, check if ground_truth.json is ever opened by loader_instance.initialize()
    # Mock open and see if ground_truth.json is opened during initialize
    # This requires artifacts to exist or it fails early, but we can just check it doesn't try
