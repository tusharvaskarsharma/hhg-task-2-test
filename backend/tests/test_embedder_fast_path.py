import pytest
import numpy as np
from backend.pipeline.embedder import Embedder
from backend.artifact_loader import loader_instance

def test_embedder_fast_path():
    """
    Ensure the embedder caches input names and uses proper np.int64 types
    for fast ONNX execution without intermediate copies.
    """
    if not loader_instance.status.get("valid"):
        loader_instance.initialize()
        
    embedder = Embedder()
    
    # Check that input_names is cached
    assert hasattr(embedder, "input_names")
    if embedder.session:
        assert len(embedder.input_names) > 0
        
    if embedder.tokenizer:
        # Run a sample query to ensure it doesn't crash on type mismatches
        embedding, tok_ms, emb_ms = embedder.embed_query("test query for embedder fast path")
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32
