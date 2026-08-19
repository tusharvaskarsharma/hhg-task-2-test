import numpy as np
from backend.artifact_loader import loader_instance
import logging

logger = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self):
        pass

    def retrieve(self, query: str, language: str, top_k: int):
        bm25 = loader_instance.get_bm25(language)
        metadata = loader_instance.get_metadata(language)
        if bm25 is None:
            raise ValueError(f"BM25 index not initialized for {language}")
            
        tokenized_query = query.split()
        
        try:
            # bm25s support
            if hasattr(bm25, 'retrieve'):
                import bm25s
                # bm25s expects a list of queries
                tokenized_queries = bm25s.tokenize([query])
                if len(tokenized_queries.ids) == 0 or len(tokenized_queries.ids[0]) == 0:
                    return []
                results_idx, scores = bm25.retrieve(tuple(tokenized_queries), corpus=None, k=top_k)
                top_k_indices = results_idx[0]
                scores = scores[0]
            else:
                scores = bm25.get_scores(tokenized_query)
                top_k_indices = np.argsort(scores)[::-1][:top_k]
                scores = scores[top_k_indices]
        except Exception as e:
            raise ValueError(f"Failed to get scores from BM25 object: {e}")
            
        results = []
        for rank, (idx, score) in enumerate(zip(top_k_indices, scores)):
            if score <= 0:
                continue
            doc_id = str(metadata.index[idx])
            results.append({
                "id": doc_id,
                "score": float(score),
                "rank": rank + 1,
                "source": "bm25"
            })
            
        return results
