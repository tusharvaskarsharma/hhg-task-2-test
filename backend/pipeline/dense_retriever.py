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
            
        try:
            labels, distances = index.knn_query(query_vector, k=top_k)
        except Exception as e:
            raise ValueError(f"HNSW query failed: {e}")
            
        results = []
        for rank, (label, dist) in enumerate(zip(labels[0], distances[0])):
            try:
                doc_id = str(metadata.index[label])
            except IndexError:
                raise ValueError(f"HNSW label {label} is out of bounds for metadata size {len(metadata)}")
                
            results.append({
                "id": doc_id,
                "score": float(1.0 - dist),
                "rank": rank + 1,
                "source": "hnsw"
            })
            
        return results
