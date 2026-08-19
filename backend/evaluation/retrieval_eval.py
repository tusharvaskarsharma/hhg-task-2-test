import math
from typing import List, Dict, Any

def hit_rate_at_k(retrieved_ids: List[str], gold_ids: set, k: int) -> float:
    for idx in range(min(len(retrieved_ids), k)):
        if retrieved_ids[idx] in gold_ids:
            return 1.0
    return 0.0

def recall_at_k(retrieved_ids: List[str], gold_ids: set, k: int) -> float:
    if not gold_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in gold_ids)
    return float(hits) / len(gold_ids)

def reciprocal_rank(retrieved_ids: List[str], gold_ids: set) -> float:
    for rank, doc_id in enumerate(retrieved_ids):
        if doc_id in gold_ids:
            return 1.0 / (rank + 1)
    return 0.0

def dcg_at_k(retrieved_ids: List[str], gold_ids: set, k: int) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in gold_ids:
            # Assuming binary relevance (rel = 1)
            dcg += 1.0 / math.log2(rank + 2) # rank is 0-indexed
    return dcg

def ndcg_at_k(retrieved_ids: List[str], gold_ids: set, k: int) -> float:
    if not gold_ids:
        return 0.0
    actual_dcg = dcg_at_k(retrieved_ids, gold_ids, k)
    
    # Ideal DCG: top min(len(gold_ids), k) elements are relevant
    idcg = 0.0
    for rank in range(min(len(gold_ids), k)):
        idcg += 1.0 / math.log2(rank + 2)
        
    if idcg == 0.0:
        return 0.0
    return actual_dcg / idcg

class RetrievalEvaluator:
    def __init__(self, k_values: List[int] = [1, 5, 10, 20]):
        self.k_values = k_values

    def evaluate_query(self, query_id: str, query_text: str, gold_ids: set, bm25_ids: List[str], hnsw_ids: List[str], rrf_ids: List[str], language: str = "hi") -> Dict[str, Any]:
        result = {
            "query_id": query_id,
            "query": query_text,
            "language": language,
            "gold_ids": list(gold_ids),
            "bm25": self._calc_metrics(bm25_ids, gold_ids),
            "hnsw": self._calc_metrics(hnsw_ids, gold_ids),
            "rrf": self._calc_metrics(rrf_ids, gold_ids)
        }
        return result

    def _calc_metrics(self, retrieved_ids: List[str], gold_ids: set) -> Dict[str, Any]:
        metrics = {
            "retrieved_ids": retrieved_ids,
            "rr": reciprocal_rank(retrieved_ids, gold_ids)
        }
        for k in self.k_values:
            metrics[f"hit_at_{k}"] = bool(hit_rate_at_k(retrieved_ids, gold_ids, k))
            metrics[f"recall_at_{k}"] = recall_at_k(retrieved_ids, gold_ids, k)
            metrics[f"ndcg_at_{k}"] = ndcg_at_k(retrieved_ids, gold_ids, k)
        return metrics

    def aggregate_metrics(self, query_evals: List[Dict[str, Any]]) -> Dict[str, Any]:
        num_queries = len(query_evals)
        if num_queries == 0:
            return {}

        systems = ["bm25", "hnsw", "rrf"]
        agg = {sys: {} for sys in systems}
        
        for sys in systems:
            agg[sys]["mrr"] = sum(q[sys]["rr"] for q in query_evals) / num_queries
            for k in self.k_values:
                agg[sys][f"hit_rate@{k}"] = sum(1 for q in query_evals if q[sys][f"hit_at_{k}"]) / num_queries
                agg[sys][f"recall@{k}"] = sum(q[sys][f"recall_at_{k}"] for q in query_evals) / num_queries
                agg[sys][f"ndcg@{k}"] = sum(q[sys][f"ndcg_at_{k}"] for q in query_evals) / num_queries
                
        return agg
