import pytest
import numpy as np
from backend.pipeline.sparse_retriever import BM25Retriever
from backend.pipeline.dense_retriever import HNSWRetriever
from backend.pipeline.embedder import Embedder

def test_bm25_retrieval(monkeypatch):
    class MockBM25:
        def get_scores(self, query):
            return np.array([0.1, 0.5, 0.2])
            
    class MockArtifactLoader:
        import pandas as pd
        metadata_df = pd.DataFrame({"id": ["doc1", "doc2", "doc3"], "text": ["a", "b", "c"]}).set_index("id")
        def get_bm25(self, lang): return MockBM25()
        def get_metadata(self, lang): return self.metadata_df
        
    monkeypatch.setattr("backend.pipeline.sparse_retriever.loader_instance", MockArtifactLoader())
    
    retriever = BM25Retriever()
    res = retriever.retrieve("hello", "en", top_k=2)
    
    assert len(res) == 2
    assert res[0]["id"] == "doc2"
    assert res[0]["score"] == 0.5
    assert res[0]["source"] == "bm25"
    assert res[1]["id"] == "doc3"
    
def test_hnsw_retrieval(monkeypatch):
    class MockIndex:
        def knn_query(self, q, k):
            return np.array([[1, 2]]), np.array([[0.1, 0.4]])
            
    class MockArtifactLoader:
        import pandas as pd
        metadata_df = pd.DataFrame({"id": ["doc1", "doc2", "doc3"], "text": ["a", "b", "c"]}).set_index("id")
        def get_hnsw(self, lang): return MockIndex()
        def get_metadata(self, lang): return self.metadata_df
        
    monkeypatch.setattr("backend.pipeline.dense_retriever.loader_instance", MockArtifactLoader())
    
    retriever = HNSWRetriever()
    res = retriever.retrieve(np.array([1,2,3]), "en", top_k=2)
    
    assert len(res) == 2
    assert res[0]["id"] == "doc2"
    assert res[0]["score"] == 0.9 # 1.0 - 0.1
    assert res[1]["id"] == "doc3"
    assert res[1]["score"] == 0.6
    
def test_embedder_mocked(monkeypatch):
    # Testing embedder requires mocking ONNX and Transformers
    class MockSession:
        def get_inputs(self):
            class I:
                name = "input_ids"
            return [I()]
        def run(self, _, inputs):
            return [np.ones((1, 5, 384))]
            
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            return {
                "input_ids": np.ones((1, 5)),
                "attention_mask": np.ones((1, 5))
            }
            
    class MockArtifactLoader:
        onnx_session = MockSession()
        
    monkeypatch.setattr("backend.pipeline.embedder.loader_instance", MockArtifactLoader())
    import backend.pipeline.embedder as emb
    monkeypatch.setattr(emb, "AutoTokenizer", lambda *args: MockTokenizer())
    
    # Needs to be re-instantiated since we mocked things
    class PatchedEmbedder(emb.Embedder):
        def __init__(self):
            self.session = MockArtifactLoader.onnx_session
            self.tokenizer = MockTokenizer()
            
    embedder = PatchedEmbedder()
    res, tok_ms, emb_ms = embedder.embed_query("test")
    
    assert res.shape == (384,)
    # check normalization
    assert np.isclose(np.linalg.norm(res), 1.0)
