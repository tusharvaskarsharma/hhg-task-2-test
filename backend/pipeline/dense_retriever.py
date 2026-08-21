import numpy as np
from backend.artifact_loader import loader_instance
import logging

logger = logging.getLogger(__name__)

class HNSWRetriever:
    def __init__(self):
        pass

    def retrieve(self, query_vector: np.ndarray, language: str, top_k: int):
        index = loader_instance.get_hnsw(language)
        metadata = loader_instance.get_metadata(language)
        
        if index is None:
            raise ValueError(f"HNSW index not initialized for {language}")
            
        metadata_count = len(metadata)
        index_count = (
            int(index.get_current_count())
            if hasattr(index, "get_current_count")
            else metadata_count
        )
        # The uploaded HI/EN indexes have declared/runtime count mismatches.
        # Query a modest candidate surplus and discard labels that cannot map to
        # the language metadata. This is bounded, keeps the parallel retrieval
        # path intact, and prevents invalid dense results from corrupting fusion.
        candidate_k = min(index_count, max(top_k, top_k * 2))
        try:
            labels, distances = index.knn_query(query_vector, k=candidate_k)
        except Exception as e:
            raise ValueError(f"HNSW query failed: {e}")
            
        results = []
        for label, dist in zip(labels[0], distances[0]):
            label = int(label)
            if label < 0 or label >= metadata_count:
                continue
            doc_id = str(metadata.index[label])
            results.append({
                "id": doc_id,
                "score": float(1.0 - dist),
                "rank": len(results) + 1,
                "source": "hnsw"
            })
            if len(results) >= top_k:
                break
            
        return results
