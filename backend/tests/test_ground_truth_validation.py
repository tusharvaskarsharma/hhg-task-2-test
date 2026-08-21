import json
import pytest
import pandas as pd
from pathlib import Path
from backend.scripts.validate_artifacts import _check_ground_truth

@pytest.fixture
def mock_artifact_root(tmp_path):
    # Create mock artifact structure
    hi_dir = tmp_path / "hi" / "metadata"
    hi_dir.mkdir(parents=True)
    
    en_dir = tmp_path / "en" / "metadata"
    en_dir.mkdir(parents=True)
    
    bn_dir = tmp_path / "bn" / "metadata"
    bn_dir.mkdir(parents=True)
    
    # Create metadata parquets
    hi_df = pd.DataFrame({"passage_id": ["hi1", "hi2"], "language": ["hi", "hi"], "text": ["hi", "hi"]})
    en_df = pd.DataFrame({"passage_id": ["en1", "en2"], "language": ["en", "en"], "text": ["en", "en"]})
    bn_df = pd.DataFrame({"passage_id": ["bn1", "bn2"], "language": ["bn", "bn"], "text": ["bn", "bn"]})
    
    hi_df.to_parquet(hi_dir / "passage_metadata.parquet")
    en_df.to_parquet(en_dir / "passage_metadata.parquet")
    bn_df.to_parquet(bn_dir / "passage_metadata.parquet")
    
    # Create mapping report
    reports_dir = tmp_path.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    with open(reports_dir / "ground_truth_mapping_report.json", "w", encoding="utf-8") as f:
        json.dump({"counts": {"mapping_coverage_over_uploaded_rows": 1.0}}, f)
        
    return tmp_path

def test_valid_ids(mock_artifact_root):
    gt = {
        "schema_version": "hhg-ground-truth-upload-v1",
        "source": {"dataset": "ai4bharat/MSMARCO-XI"},
        "queries": [
            {"query_id": "q1", "language": "hi", "query": "hello", "gold_answer": "a", "relevant_passage_ids": ["hi1"]}
        ]
    }
    with open(mock_artifact_root / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f)
        
    res = _check_ground_truth(str(mock_artifact_root))
    assert res["present"] is True
    assert res["schema_valid"] is True
    assert res["ids_valid"] is True

def test_invalid_id_fails(mock_artifact_root):
    gt = {
        "schema_version": "hhg-ground-truth-upload-v1",
        "source": {"dataset": "ai4bharat/MSMARCO-XI"},
        "queries": [
            {"query_id": "q1", "language": "hi", "query": "hello", "gold_answer": "a", "relevant_passage_ids": ["invalid_hi"]}
        ]
    }
    with open(mock_artifact_root / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f)
        
    res = _check_ground_truth(str(mock_artifact_root))
    assert res["ids_valid"] is False

def test_wrong_language_id_fails(mock_artifact_root):
    gt = {
        "schema_version": "hhg-ground-truth-upload-v1",
        "source": {"dataset": "ai4bharat/MSMARCO-XI"},
        "queries": [
            {"query_id": "q1", "language": "hi", "query": "hello", "gold_answer": "a", "relevant_passage_ids": ["en1"]}
        ]
    }
    with open(mock_artifact_root / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f)
        
    res = _check_ground_truth(str(mock_artifact_root))
    assert res["ids_valid"] is False

def test_mapping_report_parsed(mock_artifact_root):
    gt = {
        "schema_version": "hhg-ground-truth-upload-v1",
        "source": {"dataset": "ai4bharat/MSMARCO-XI"},
        "queries": []
    }
    with open(mock_artifact_root / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f)
        
    reports_dir = mock_artifact_root.parent / "reports"
    with open(reports_dir / "ground_truth_mapping_report.json", "w", encoding="utf-8") as f:
        json.dump({"counts": {"mapping_coverage_over_uploaded_rows": 0.85}}, f)
        
    res = _check_ground_truth(str(mock_artifact_root))
    assert res["mapping_coverage"] == 0.85
