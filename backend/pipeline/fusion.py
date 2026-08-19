from collections import defaultdict

def rrf_fuse(bm25_results, hnsw_results, k=60, top_k=10):
    rrf_scores = defaultdict(float)
    sources = defaultdict(set)
    
    for res in bm25_results:
        doc_id = res["id"]
        rank = res["rank"]
        rrf_scores[doc_id] += 1.0 / (k + rank)
        sources[doc_id].add("bm25")
        
    for res in hnsw_results:
        doc_id = res["id"]
        rank = res["rank"]
        rrf_scores[doc_id] += 1.0 / (k + rank)
        sources[doc_id].add("hnsw")
        
    fused_results = []
    for doc_id, score in rrf_scores.items():
        fused_results.append({
            "id": doc_id,
            "rrf_score": score,
            "sources": sorted(list(sources[doc_id]))
        })
        
    # Sort descending
    fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
    
    final_results = []
    for rank, res in enumerate(fused_results[:top_k]):
        res["rank"] = rank + 1
        final_results.append(res)
        
    return final_results
