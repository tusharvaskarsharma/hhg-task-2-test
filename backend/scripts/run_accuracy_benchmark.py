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
    logger.info(f"Loading MIRACL dev dataset for {lang}...")
    try:
        ds = load_dataset("miracl/miracl", lang, split="dev", trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset for {lang}: {e}")
        return None
        
    logger.info(f"Loaded {len(ds)} queries for {lang}. Sampling top 100 for fast eval.")
    queries = []
    for row in ds:
        pos_passages = row.get("positive_passages", [])
        if not pos_passages:
            continue
        gold_ids = set([p["docid"] for p in pos_passages])
        queries.append({
            "query_id": row["query_id"],
            "query": row["query"],
            "gold_ids": gold_ids
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
    
    from backend.pipeline.tokenizer import preprocess_query
    
    for i, q in enumerate(queries):
        if i % 20 == 0:
            logger.info(f"Processed {i}/{len(queries)}")
            
        q_text = q["query"]
        gold_ids = q["gold_ids"]
        
        # 1. Pipeline
        prep_text = preprocess_query(q_text)
        
        bm25_res = bm25_retriever.retrieve(prep_text, lang, top_k=20)
        bm25_ids = [r["id"] for r in bm25_res]
        
        emb, _, _ = embedder.embed_query(q_text)
        hnsw_res = hnsw_retriever.retrieve(emb, lang, top_k=20)
        hnsw_ids = [r["id"] for r in hnsw_res]
        
        rrf_res = rrf_fuse(bm25_res, hnsw_res, top_k=20)
        rrf_ids = [r["id"] for r in rrf_res]
        
        for sys_name, ids in [("bm25", bm25_ids), ("hnsw", hnsw_ids), ("rrf", rrf_ids)]:
            metrics[sys_name]["Recall@1"] += recall_at_k(ids, gold_ids, 1)
            metrics[sys_name]["Recall@5"] += recall_at_k(ids, gold_ids, 5)
            metrics[sys_name]["Recall@10"] += recall_at_k(ids, gold_ids, 10)
            metrics[sys_name]["Precision@5"] += precision_at_k(ids, gold_ids, 5)
            metrics[sys_name]["MRR@10"] += mrr_at_k(ids, gold_ids, 10)
            metrics[sys_name]["nDCG@10"] += ndcg_at_k(ids, gold_ids, 10)
            
        if mrr_at_k(rrf_ids, gold_ids, 10) == 0.0:
            if len(failures) < 3:
                failures.append({
                    "query": q_text,
                    "gold_ids": list(gold_ids),
                    "retrieved_top_5": rrf_ids[:5]
                })
                
        if i < 10:
            from backend.artifact_loader import loader_instance
            top_passages = []
            for doc_id in rrf_ids[:5]:
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
    
    for lang in ["hi", "bn", "en"]:
        res = run_language_benchmark(lang, embedder, bm25_retriever, hnsw_retriever, rrf_fuse, generator)
        if res:
            results[lang] = res
            
    report = "# RAG Accuracy Results\n\n"
    report += "This report evaluates the accuracy of the production HHG RAG pipeline using the `miracl/miracl` dataset.\n\n"
    
    for lang in ["hi", "bn", "en"]:
        if lang not in results: continue
        res = results[lang]
        
        report += f"## Language: {lang.upper()}\n"
        report += f"- **Evaluated Queries:** {res['num_queries']}\n\n"
        
        report += "### Metrics\n"
        report += "| Pipeline | Recall@1 | Recall@5 | Recall@10 | Precision@5 | MRR@10 | nDCG@10 |\n"
        report += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        
        for sys_name in ["bm25", "hnsw", "rrf"]:
            m = res['metrics'][sys_name]
            report += f"| **{sys_name.upper()}** | {m['Recall@1']:.4f} | {m['Recall@5']:.4f} | {m['Recall@10']:.4f} | {m['Precision@5']:.4f} | {m['MRR@10']:.4f} | {m['nDCG@10']:.4f} |\n"
            
        report += "\n### Grounding Evaluation\n"
        g = res['grounding']
        if g['total_evals'] > 0:
            report += f"- Total evaluated for grounding: {g['total_evals']}\n"
            report += f"- **Supported Answers:** {g['supported']} ({(g['supported']/g['total_evals'])*100:.1f}%)\n"
            report += f"- **Correct Refusals (Insufficient Context):** {g['refused']} ({(g['refused']/g['total_evals'])*100:.1f}%)\n"
            report += f"- **Hallucinations Detected:** {g['hallucinated']} ({(g['hallucinated']/g['total_evals'])*100:.1f}%)\n\n"
            
        report += "### Examples of Missed Relevant Documents\n"
        for i, f in enumerate(res['failures']):
            report += f"**Query:** {f['query']}\n"
            report += f"- Gold IDs: {f['gold_ids']}\n"
            report += f"- Retrieved Top 5: {f['retrieved_top_5']}\n\n"
            
        report += "---\n\n"
        
    with open("rag_accuracy_results.md", "w", encoding="utf-8") as f:
        f.write(report)
        
    print("\nBenchmark completed. Results saved to rag_accuracy_results.md")

if __name__ == "__main__":
    main()
