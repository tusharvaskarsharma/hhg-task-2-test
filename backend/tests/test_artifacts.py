import os
import json
import pytest
import pandas as pd
import pickle
import sys

# removed mock hnswlib
from backend.artifact_loader import ArtifactLoader
from backend.config import settings


@pytest.fixture
def mock_artifact_dir(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "hhg_rag_artifacts"
    artifact_dir.mkdir()
    
    monkeypatch.setattr(settings, 'HHG_ARTIFACT_DIR', str(artifact_dir))
    
    # Create manifest
    manifest_data = {
        "dataset": "ai4bharat/MSMARCO-XI",
        "language": "hi",
        "embedding_model": "intfloat/multilingual-e5-small",
        "embedding_dimension": 384,
        "normalized_embeddings": True,
        "onnx_quantization": "INT8",
        "hnsw_space": "cosine",
        "checksums": {
            "metadata.parquet": "dummy"
        }
    }
    with open(artifact_dir / "build_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)
        
    # Create bm25
    bm25_dir = artifact_dir / "bm25"
    bm25_dir.mkdir()
    with open(bm25_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"dummy": "bm25"}, f)
        
    # Create hnsw
    hnsw_dir = artifact_dir / "hnsw"
    hnsw_dir.mkdir()
    # Explicitly create the index.bin file since the mock hnswlib won't create it
    with open(hnsw_dir / "index.bin", "w") as f:
        f.write("dummy")
    
    # Create metadata
    metadata_dir = artifact_dir / "metadata"
    metadata_dir.mkdir()
    df = pd.DataFrame({"id": ["1"], "text": ["hello"], "lang": ["hi"]})
    df.to_parquet(metadata_dir / "passage_metadata.parquet")
    
    # Create model
    model_dir = artifact_dir / "model"
    model_dir.mkdir()
    with open(model_dir / "e5-small-int8.onnx", "wb") as f:
        f.write(b"dummy onnx")
        
    # Create validation summary
    val_dir = artifact_dir / "validation"
    val_dir.mkdir()
    with open(val_dir / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump({"dummy": "val"}, f)
        
    return artifact_dir

def test_artifact_loader_success(mock_artifact_dir, monkeypatch):
    import onnxruntime as ort
    
    class MockSession:
        def __init__(self, path, providers=None, **kwargs):
            pass
        def get_inputs(self):
            class MockInput:
                name = "input"
            return [MockInput()]
        def get_outputs(self):
            class MockOutput:
                name = "output"
                shape = [1, 384]
            return [MockOutput()]
            
    monkeypatch.setattr(ort, "InferenceSession", MockSession)
    
    loader = ArtifactLoader()
    loader.initialize()
    
    assert loader.status["valid"] is True
    lang_status = loader.get_language("hi").status
    assert lang_status["manifest"] is True
    assert lang_status["metadata"] is True
    assert loader.status["onnx"] is True
    assert lang_status["bm25"] is True
    assert lang_status["hnsw"] is True

def test_artifact_loader_missing_manifest(mock_artifact_dir, monkeypatch):
    import onnxruntime as ort
    class MockSession:
        def __init__(self, path, providers=None, **kwargs):
            pass
        def get_inputs(self):
            class MockInput: name = "input"
            return [MockInput()]
        def get_outputs(self):
            class MockOutput:
                name = "output"
                shape = [1, 384]
            return [MockOutput()]
    monkeypatch.setattr(ort, "InferenceSession", MockSession)

    os.remove(mock_artifact_dir / "build_manifest.json")
    loader = ArtifactLoader()
    loader.initialize()
    assert loader.status["valid"] is False
    assert any("Manifest not found" in e for e in loader.errors)

def test_validate_artifacts_missing_dir():
    from backend.scripts.validate_artifacts import validate
    res = validate("/invalid/path")
    assert "ARTIFACT_ROOT_MISSING" in res["overall_status"]

def test_validate_artifacts_missing_manifest(mock_artifact_dir):
    from backend.scripts.validate_artifacts import validate
    os.remove(mock_artifact_dir / "build_manifest.json")
    res = validate(str(mock_artifact_dir))
    assert res["overall_status"] == "FAIL"

def test_validate_artifacts_empty_text(mock_artifact_dir):
    from backend.scripts.validate_artifacts import validate
    df = pd.DataFrame({"id": ["1"], "text": [""], "lang": ["hi"]})
    df.to_parquet(mock_artifact_dir / "metadata" / "passage_metadata.parquet")
    res = validate(str(mock_artifact_dir))
    assert res["overall_status"] == "FAIL", f"res: {res}"
    assert any("empty or null" in e for e in res["errors"]), f"res: {res}"

def test_validate_artifacts_duplicate_ids(mock_artifact_dir):
    from backend.scripts.validate_artifacts import validate
    df = pd.DataFrame({"id": ["1", "1"], "text": ["a", "b"], "lang": ["hi", "hi"]})
    df.to_parquet(mock_artifact_dir / "metadata" / "passage_metadata.parquet")
    res = validate(str(mock_artifact_dir))
    assert res["overall_status"] == "FAIL", f"res: {res}"
    assert any("not unique" in e for e in res["errors"]), f"res: {res}"




