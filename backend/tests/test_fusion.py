from backend.pipeline.fusion import rrf_fuse

def test_rrf_fuse():
    bm25_res = [
        {"id": "doc1", "rank": 1},
        {"id": "doc2", "rank": 2},
    ]
    hnsw_res = [
        {"id": "doc3", "rank": 1},
        {"id": "doc1", "rank": 2},
    ]
    
    fused = rrf_fuse(bm25_res, hnsw_res, k=60, top_k=10)
    
    # doc1 is rank 1 in bm25 and rank 2 in hnsw. Score: 1/61 + 1/62
    # doc3 is rank 1 in hnsw. Score: 1/61
    # doc2 is rank 2 in bm25. Score: 1/62
    
    # So doc1 should be rank 1
    assert fused[0]["id"] == "doc1"
    assert "bm25" in fused[0]["sources"] and "hnsw" in fused[0]["sources"]
    
    assert fused[1]["id"] == "doc3"
    assert fused[2]["id"] == "doc2"
    
    assert len(fused) == 3
    
def test_rrf_top_k():
    bm25_res = [{"id": f"doc{i}", "rank": i} for i in range(1, 20)]
    hnsw_res = [{"id": f"doc{i}", "rank": i} for i in range(1, 20)]
    
    fused = rrf_fuse(bm25_res, hnsw_res, k=60, top_k=5)
    
    assert len(fused) == 5
