import os
import sys
import json
import logging
import math
from typing import List, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import settings
from backend.artifact_loader import loader_instance
from backend.pipeline.embedder import Embedder
from backend.pipeline.sparse_retriever import BM25Retriever
from backend.pipeline.dense_retriever import HNSWRetriever
from backend.pipeline.fusion import rrf_fuse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def precision_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int) -> float:
    if not retrieved_ids:
        return 0.0
    k_actual = min(len(retrieved_ids), k)
    if k_actual == 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k_actual] if doc_id in gold_ids)
    return float(hits) / k

def recall_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in gold_ids)
    return float(hits) / len(gold_ids)

def mrr_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int) -> float:
    for rank, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in gold_ids:
            return 1.0 / (rank + 1)
    return 0.0

def ndcg_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in gold_ids:
            dcg += 1.0 / math.log2(rank + 2)
            
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(gold_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_retriever(results: List[dict], gold_ids: Set[str]):
    ids = [r["id"] for r in results]
    return {
        "Recall@1": recall_at_k(ids, gold_ids, 1),
        "Recall@5": recall_at_k(ids, gold_ids, 5),
        "Recall@10": recall_at_k(ids, gold_ids, 10),
        "Precision@5": precision_at_k(ids, gold_ids, 5),
        "MRR@10": mrr_at_k(ids, gold_ids, 10),
        "nDCG@10": ndcg_at_k(ids, gold_ids, 10),
    }

def run_language_benchmark(lang: str, queries: List[dict], embedder, bm25_retriever, hnsw_retriever):
    metrics = {
        "bm25": {"Recall@1": 0, "Recall@5": 0, "Recall@10": 0, "Precision@5": 0, "MRR@10": 0, "nDCG@10": 0},
        "hnsw": {"Recall@1": 0, "Recall@5": 0, "Recall@10": 0, "Precision@5": 0, "MRR@10": 0, "nDCG@10": 0},
        "rrf": {"Recall@1": 0, "Recall@5": 0, "Recall@10": 0, "Precision@5": 0, "MRR@10": 0, "nDCG@10": 0}
    }
    
    total = len(queries)
    if total == 0:
        return metrics
        
    for i, q in enumerate(queries):
        if i % 10 == 0:
            logger.info(f"Processing query {i+1}/{total} for {lang}")
            
        q_text = q["query"]
        gold_ids = set(q["gold_doc_ids"])
        
        q_vec, _, _ = embedder.embed_query(q_text)
        
        bm25_res = bm25_retriever.retrieve(q_text, lang, top_k=10)
        hnsw_res = hnsw_retriever.retrieve(q_vec, lang, top_k=10)
        rrf_res = rrf_fuse(bm25_res, hnsw_res, k=60, top_k=10)
        
        b_mets = evaluate_retriever(bm25_res, gold_ids)
        h_mets = evaluate_retriever(hnsw_res, gold_ids)
        r_mets = evaluate_retriever(rrf_res, gold_ids)
        
        for k in metrics["bm25"].keys():
            metrics["bm25"][k] += b_mets[k]
            metrics["hnsw"][k] += h_mets[k]
            metrics["rrf"][k] += r_mets[k]
            
    for sys_name in metrics:
        for k in metrics[sys_name]:
            metrics[sys_name][k] /= total
            
    return metrics

def main():
    print("=" * 60)
    print("HHG RAG ACCURACY BENCHMARK")
    print("=" * 60)
    
    gt_path = os.path.join(settings.HHG_ARTIFACT_DIR, "ground_truth.json")
    if not os.path.exists(gt_path):
        print("ground_truth.json not found.")
        print("Status: NOT RUN")
        
        report_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "reports", "rag_accuracy_results.md"
        )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# RAG Accuracy Results\n\n**STATUS: NOT RUN**\n\nNo `ground_truth.json` file provided for production IDs.\n")
        return
        
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth_data = json.load(f)
        
    loader_instance.initialize()
    if not loader_instance.status.get("valid"):
        print("ERROR: Artifacts not valid.")
        sys.exit(1)
        
    embedder = Embedder()
    bm25_retriever = BM25Retriever()
    hnsw_retriever = HNSWRetriever()
    
    base_report = "# RAG Accuracy Results\n\nThis report evaluates the accuracy of the production HHG RAG pipeline.\n\n"
    
    for lang, queries in ground_truth_data.items():
        if lang not in ["hi", "bn", "en"]:
            continue
            
        logger.info(f"Running benchmark for {lang} ({len(queries)} queries)...")
        metrics = run_language_benchmark(lang, queries, embedder, bm25_retriever, hnsw_retriever)
        
        base_report += f"## Language: {lang.upper()}\n"
        base_report += f"- **Evaluated Queries:** {len(queries)}\n\n"
        base_report += "### Metrics\n"
        base_report += "| Pipeline | Recall@1 | Recall@5 | Recall@10 | Precision@5 | MRR@10 | nDCG@10 |\n"
        base_report += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        
        for sys_name in ["bm25", "hnsw", "rrf"]:
            m = metrics[sys_name]
            base_report += (
                f"| **{sys_name.upper()}** | "
                f"{m['Recall@1']:.4f} | "
                f"{m['Recall@5']:.4f} | "
                f"{m['Recall@10']:.4f} | "
                f"{m['Precision@5']:.4f} | "
                f"{m['MRR@10']:.4f} | "
                f"{m['nDCG@10']:.4f} |\n"
            )
        base_report += "\n---\n\n"
        
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "reports", "rag_accuracy_results.md"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(base_report)
            
    print(f"\nBenchmark completed. Results saved to {report_path}")

if __name__ == "__main__":
    main()
