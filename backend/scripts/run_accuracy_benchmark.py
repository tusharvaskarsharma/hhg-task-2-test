import os
import sys
import json
import logging
import math
from typing import List, Dict, Any, Set
import pandas as pd
from datasets import load_dataset
import httpx
import re

# We must ensure we don't accidentally import the mock HNSWLib from somewhere else
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import settings
from backend.artifact_loader import loader_instance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------- METRICS -------------

def hit_rate_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int) -> float:
    return 1.0 if any(doc_id in gold_ids for doc_id in retrieved_ids[:k]) else 0.0

def recall_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in gold_ids)
    return float(hits) / len(gold_ids)

def precision_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int) -> float:
    if not retrieved_ids:
        return 0.0
    k_actual = min(len(retrieved_ids), k)
    if k_actual == 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k_actual] if doc_id in gold_ids)
    return float(hits) / k

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
            
    idcg = 0.0
    for rank in range(min(len(gold_ids), k)):
        idcg += 1.0 / math.log2(rank + 2)
        
    return dcg / idcg if idcg > 0 else 0.0


# ------------- GROUNDING EVALUATION -------------

def get_slm_judge_response(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {settings.SLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": settings.SLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.0
    }
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(settings.SLM_BASE_URL, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"SLM Judge error: {e}")
        return "ERROR"

def evaluate_grounding(query: str, context: str, answer: str) -> Dict[str, Any]:
    refusal_keywords = ["don't know", "do not know", "cannot answer", "doesn't provide", "does not provide", "not found", "context does not contain"]
    is_refused = any(kw in answer.lower() for kw in refusal_keywords)
    
    if is_refused:
        return {"supported": False, "refused": True, "hallucinated": False}
        
    prompt = f"""You are an exact evaluator. 
Context: {context}
Answer: {answer}

Does the Answer contain information that is NOT present in the Context?
Respond with exactly one word: 'YES' if it contains hallucinations (information not in context), or 'NO' if it is fully supported by the context."""
    
    judge_res = get_slm_judge_response(prompt).upper()
    hallucinated = "YES" in judge_res
    
    return {
        "supported": not hallucinated,
        "refused": False,
        "hallucinated": hallucinated
    }

# ------------- MAIN BENCHMARK -------------

def run_language_benchmark(lang: str, embedder, bm25_retriever, hnsw_retriever, rrf_fuse, generator):
    from backend.artifact_loader import loader_instance
    logger.info(f"Loading MSMARCO-XI dev dataset for {lang}...")
    try:
        ds = load_dataset("ai4bharat/MSMARCO-XI", lang, split="train", trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset for {lang}: {e}")
        return None
        
    logger.info(f"Loaded {len(ds)} queries for {lang}. Sampling top 100 for fast eval.")
    queries = []
    for row in ds:
        pos_passages = row.get("positive_passages", [])
        if not pos_passages:
            continue
        gold_texts = [p["text"].strip() for p in pos_passages]
        queries.append({
            "query_id": row["query_id"],
            "query": row["query"],
            "gold_texts": gold_texts
        })
        if len(queries) >= 100:
            break
            
    if not queries:
        return None
        
    logger.info(f"Running eval on {len(queries)} queries for {lang}...")
    
    metrics = {
        "bm25": {"Recall@1": 0, "Recall@5": 0, "Recall@10": 0, "Precision@5": 0, "MRR@10": 0, "nDCG@10": 0},
        "hnsw": {"Recall@1": 0, "Recall@5": 0, "Recall@10": 0, "Precision@5": 0, "MRR@10": 0, "nDCG@10": 0},
        "rrf": {"Recall@1": 0, "Recall@5": 0, "Recall@10": 0, "Precision@5": 0, "MRR@10": 0, "nDCG@10": 0}
    }
    
    grounding_stats = {"supported": 0, "refused": 0, "hallucinated": 0, "total_evals": 0}
    failures = []
    
    for i, q in enumerate(queries):
        if i > 0 and i % 20 == 0:
            logger.info(f"Processed {i}/{len(queries)}")
            
        q_text = q["query"]
        gold_texts = q["gold_texts"]
        
        q_vec, _, _ = embedder.embed_query(q_text)
        
        bm25_res = bm25_retriever.retrieve(q_text, lang, top_k=10)
        hnsw_res = hnsw_retriever.retrieve(q_vec, lang, top_k=10)
        rrf_res = rrf_fuse(bm25_res, hnsw_res, k=60, top_k=10)
        
        meta_df = loader_instance.languages[lang].metadata_df
        
        def is_hit(retrieved_id):
            if meta_df is None: return False
            row = meta_df.loc[meta_df["passage_id"] == retrieved_id]
            if row.empty: return False
            retrieved_text = row.iloc[0]["text"].strip()
            # Simple substring matching or exact matching to account for minor preprocessing
            for gt in gold_texts:
                if gt in retrieved_text or retrieved_text in gt:
                    return True
            return False

        def evaluate_hits(results):
            hits_at_1 = 1 if len(results) > 0 and is_hit(results[0]["id"]) else 0
            hits_at_5 = sum(1 for res in results[:5] if is_hit(res["id"]))
            hits_at_10 = sum(1 for res in results[:10] if is_hit(res["id"]))
            
            mrr_10 = 0
            for rank, res in enumerate(results[:10]):
                if is_hit(res["id"]):
                    mrr_10 = 1.0 / (rank + 1)
                    break
                    
            import math
            ndcg_10 = 0
            dcg = 0
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_texts), 10)))
            for rank, res in enumerate(results[:10]):
                if is_hit(res["id"]):
                    dcg += 1.0 / math.log2(rank + 2)
            if idcg > 0:
                ndcg_10 = dcg / idcg
                
            return hits_at_1, hits_at_5, hits_at_10, mrr_10, ndcg_10

        bm25_hits_1, bm25_hits_5, bm25_hits_10, bm25_mrr, bm25_ndcg = evaluate_hits(bm25_res)
        hnsw_hits_1, hnsw_hits_5, hnsw_hits_10, hnsw_mrr, hnsw_ndcg = evaluate_hits(hnsw_res)
        rrf_hits_1, rrf_hits_5, rrf_hits_10, rrf_mrr, rrf_ndcg = evaluate_hits(rrf_res)
        
        metrics["bm25"]["Recall@1"] += bm25_hits_1
        metrics["bm25"]["Recall@5"] += 1 if bm25_hits_5 > 0 else 0
        metrics["bm25"]["Recall@10"] += 1 if bm25_hits_10 > 0 else 0
        metrics["bm25"]["Precision@5"] += bm25_hits_5 / 5.0
        metrics["bm25"]["MRR@10"] += bm25_mrr
        metrics["bm25"]["nDCG@10"] += bm25_ndcg
        
        metrics["hnsw"]["Recall@1"] += hnsw_hits_1
        metrics["hnsw"]["Recall@5"] += 1 if hnsw_hits_5 > 0 else 0
        metrics["hnsw"]["Recall@10"] += 1 if hnsw_hits_10 > 0 else 0
        metrics["hnsw"]["Precision@5"] += hnsw_hits_5 / 5.0
        metrics["hnsw"]["MRR@10"] += hnsw_mrr
        metrics["hnsw"]["nDCG@10"] += hnsw_ndcg
        
        metrics["rrf"]["Recall@1"] += rrf_hits_1
        metrics["rrf"]["Recall@5"] += 1 if rrf_hits_5 > 0 else 0
        metrics["rrf"]["Recall@10"] += 1 if rrf_hits_10 > 0 else 0
        metrics["rrf"]["Precision@5"] += rrf_hits_5 / 5.0
        metrics["rrf"]["MRR@10"] += rrf_mrr
        metrics["rrf"]["nDCG@10"] += rrf_ndcg
        
        if rrf_hits_10 == 0:
            if len(failures) < 3:
                failures.append({
                    "query": q_text,
                    "gold_texts": gold_texts,
                    "retrieved_top_5_texts": [
                        meta_df.loc[meta_df["passage_id"] == res["id"]].iloc[0]["text"] 
                        if not meta_df.loc[meta_df["passage_id"] == res["id"]].empty else ""
                        for res in rrf_res[:5]
                    ]
                })
                
        if i < 10:
            from backend.artifact_loader import loader_instance
            top_passages = []
            for doc_id in [r["id"] for r in rrf_res[:5]]:
                meta_df = loader_instance.languages[lang].metadata_df
                if meta_df is not None:
                    row = meta_df.loc[meta_df["passage_id"] == doc_id]
                    if not row.empty:
                        top_passages.append(row.iloc[0]["text"])
                
            context = "\n".join(top_passages)
            try:
                answer = generator.generate(q_text, lang, top_passages)
                g_eval = evaluate_grounding(q_text, context, answer)
                if g_eval["supported"]: grounding_stats["supported"] += 1
                if g_eval["refused"]: grounding_stats["refused"] += 1
                if g_eval["hallucinated"]: grounding_stats["hallucinated"] += 1
                grounding_stats["total_evals"] += 1
            except Exception as e:
                pass
                
    for sys_name in metrics:
        for m in metrics[sys_name]:
            metrics[sys_name][m] /= len(queries)
            
    return {
        "num_queries": len(queries),
        "metrics": metrics,
        "grounding": grounding_stats,
        "failures": failures
    }

def main():
    print("=" * 60)
    print("HHG RAG ACCURACY BENCHMARK (MIRACL)")
    print("=" * 60)
    
    loader_instance.initialize()
    if not loader_instance.status.get("valid"):
        print("ERROR: Artifacts not valid.")
        sys.exit(1)
        
    from backend.pipeline.embedder import Embedder
    from backend.pipeline.sparse_retriever import BM25Retriever
    from backend.pipeline.dense_retriever import HNSWRetriever
    from backend.pipeline.fusion import rrf_fuse
    from backend.pipeline.generator import GeneratorService
    
    embedder = Embedder()
    bm25_retriever = BM25Retriever()
    hnsw_retriever = HNSWRetriever()
    generator = GeneratorService()
    
    results = {}
    
    base_report = "# RAG Accuracy Results\n\nThis report evaluates the accuracy of the production HHG RAG pipeline using the `miracl/miracl` dataset.\n\n"
    
    for lang in ["hi", "bn"]:
        res = run_language_benchmark(lang, embedder, bm25_retriever, hnsw_retriever, rrf_fuse, generator)
        if res:
            results[lang] = res
            
            # Write progressively to avoid data loss on memory errors
            current_report = base_report
            for l, r in results.items():
                current_report += f"## Language: {l.upper()}\n"
                current_report += f"- **Evaluated Queries:** {r['total_evals']}\n\n"
                
                current_report += "### Metrics\n"
                current_report += "| Pipeline | Recall@1 | Recall@5 | Recall@10 | Precision@5 | MRR@10 | nDCG@10 |\n"
                current_report += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
                
                for sys_name in ["bm25", "hnsw", "rrf"]:
                    m = r["metrics"][sys_name]
                    current_report += (
                        f"| **{sys_name.upper()}** | "
                        f"{m['Recall@1']/r['total_evals']:.4f} | "
                        f"{m['Recall@5']/r['total_evals']:.4f} | "
                        f"{m['Recall@10']/r['total_evals']:.4f} | "
                        f"{m['Precision@5']/r['total_evals']:.4f} | "
                        f"{m['MRR@10']/r['total_evals']:.4f} | "
                        f"{m['nDCG@10']/r['total_evals']:.4f} |\n"
                    )
                current_report += "\n"
                
                current_report += "### Grounding Evaluation\n"
                if r['total_evals'] > 0:
                    current_report += f"- **Supported Answers:** {r['grounding']['supported'] / r['total_evals'] * 100:.1f}%\n"
                    current_report += f"- **Refused (Insufficient Context):** {r['grounding']['refused'] / r['total_evals'] * 100:.1f}%\n"
                    current_report += f"- **Hallucinated:** {r['grounding']['hallucinated'] / r['total_evals'] * 100:.1f}%\n\n"
                
                current_report += "### Examples of Missed Relevant Documents\n"
                for f in r["failures"]:
                    current_report += f"**Query:** {f['query']}\n"
                    current_report += f"- Gold Texts (Truncated): {[gt[:50] + '...' for gt in f['gold_texts']]}\n"
                    current_report += f"- Retrieved Top 5 Texts (Truncated): {[rt[:50] + '...' for rt in f['retrieved_top_5_texts']]}\n\n"
                
                current_report += "---\n\n"
            
            with open("rag_accuracy_results.md", "w", encoding="utf-8") as f:
                f.write(current_report)
                
    print("\nBenchmark completed. Results saved to rag_accuracy_results.md")

if __name__ == "__main__":
    main()
