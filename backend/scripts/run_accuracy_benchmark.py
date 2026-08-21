import os
import sys
import json
import logging
import math
import argparse
import hashlib
import subprocess
from typing import List, Set, Dict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import settings
from backend.artifact_loader import loader_instance
from backend.pipeline.embedder import Embedder
from backend.pipeline.sparse_retriever import BM25Retriever
from backend.pipeline.dense_retriever import HNSWRetriever
from backend.pipeline.fusion import rrf_fuse
from backend.evaluation.ground_truth import load_ground_truth, GroundTruthError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

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

def hit_rate_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    for doc_id in retrieved_ids[:k]:
        if doc_id in gold_ids:
            return 1.0
    return 0.0

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

def evaluate_retriever(results: List[dict], gold_ids: Set[str], top_k_list: List[int]):
    ids = [r["id"] for r in results]
    metrics = {}
    for k in top_k_list:
        metrics[f"Recall@{k}"] = recall_at_k(ids, gold_ids, k)
        metrics[f"Precision@{k}"] = precision_at_k(ids, gold_ids, k)
        metrics[f"HitRate@{k}"] = hit_rate_at_k(ids, gold_ids, k)
        metrics[f"MRR@{k}"] = mrr_at_k(ids, gold_ids, k)
        metrics[f"nDCG@{k}"] = ndcg_at_k(ids, gold_ids, k)
    return metrics

def run_language_benchmark(lang: str, queries: List[dict], embedder, bm25_retriever, hnsw_retriever, top_k_list: List[int]):
    metrics = {
        "bm25": {},
        "hnsw": {},
        "rrf": {}
    }
    
    for k in top_k_list:
        for sys_name in metrics:
            metrics[sys_name][f"Recall@{k}"] = 0.0
            metrics[sys_name][f"Precision@{k}"] = 0.0
            metrics[sys_name][f"HitRate@{k}"] = 0.0
            metrics[sys_name][f"MRR@{k}"] = 0.0
            metrics[sys_name][f"nDCG@{k}"] = 0.0
            
    total = len(queries)
    if total == 0:
        return metrics
        
    for i, q in enumerate(queries):
        if i % 100 == 0:
            logger.info(f"Processing query {i}/{total} for {lang}")
            
        q_text = q["query"]
        gold_ids = set(q["relevant_passage_ids"])
        
        q_vec, _, _ = embedder.embed_query(q_text)
        
        max_k = max(top_k_list)
        bm25_res = bm25_retriever.retrieve(q_text, lang, top_k=max_k)
        hnsw_res = hnsw_retriever.retrieve(q_vec, lang, top_k=max_k)
        rrf_res = rrf_fuse(bm25_res, hnsw_res, k=60, top_k=max_k)
        
        b_mets = evaluate_retriever(bm25_res, gold_ids, top_k_list)
        h_mets = evaluate_retriever(hnsw_res, gold_ids, top_k_list)
        r_mets = evaluate_retriever(rrf_res, gold_ids, top_k_list)
        
        for k_met in b_mets:
            metrics["bm25"][k_met] += b_mets[k_met]
            metrics["hnsw"][k_met] += h_mets[k_met]
            metrics["rrf"][k_met] += r_mets[k_met]
            
    for sys_name in metrics:
        for k_met in metrics[sys_name]:
            metrics[sys_name][k_met] /= total
            
    return metrics

def evaluate_abstention(lang: str, queries: List[dict], embedder, bm25_retriever, hnsw_retriever):
    # For no-positive queries, an abstention metric could be the rate at which we return empty results
    # Since we always return top_k, abstention metrics might just track how many no-positive queries exist
    # and maybe the scores of the returned items. Here we just return the count and empty dict as requested.
    return {
        "num_no_positive_queries": len(queries),
        "abstention_metrics": {}
    }

def generate_markdown_report(report_data: dict, output_path: str, args: argparse.Namespace):
    md = f"# RAG Accuracy Results\n\n**STATUS: {report_data.get('status', 'NOT RUN')}**\n\n"
    
    if report_data.get("status") == "NOT RUN":
        md += f"Reason: {report_data.get('reason', 'Unknown error')}\n"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)
        return
        
    source = report_data.get("source", {})
    md += f"- **Commit SHA:** `{source.get('commit_sha', 'unknown')}`\n"
    md += f"- **Artifact Manifest SHA:** `{source.get('artifact_manifest_sha256', 'unknown')}`\n"
    md += f"- **Ground Truth SHA256:** `{source.get('ground_truth_sha256', 'unknown')}`\n"
    md += f"- **Dataset:** `{source.get('dataset', 'unknown')}` (Revision: `{source.get('revision', 'unknown')}`)\n"
    md += f"- **Selection Profile:** `{source.get('selection_profile', 'default')}`\n"
    
    overall = report_data.get("overall", {})
    md += f"- **Mapping Coverage:** `{overall.get('mapping_coverage', 1.0):.4f}`\n\n"
    
    md += f"**Exact Command Used:** `python -m backend.scripts.run_accuracy_benchmark " + " ".join(sys.argv[1:]) + "`\n\n"
    
    for lang in args.languages:
        lang_data = report_data.get("by_language", {}).get(lang)
        if not lang_data:
            continue
            
        md += f"## Language: {lang.upper()}\n"
        md += f"- **Evaluated Queries (with gold):** {lang_data.get('num_queries_with_gold', 0)}\n\n"
        
        md += "### Metrics\n"
        
        top_k_list = args.top_k
        headers = ["Pipeline"] + [f"{m}@{k}" for m in ["Recall", "Precision", "HitRate", "MRR", "nDCG"] for k in top_k_list]
        md += "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join([":---"] * len(headers)) + " |\n"
        
        for sys_name in ["bm25", "hnsw", "rrf"]:
            sys_metrics = lang_data.get(sys_name, {})
            row = [f"**{sys_name.upper()}**"]
            for m in ["Recall", "Precision", "HitRate", "MRR", "nDCG"]:
                for k in top_k_list:
                    val = sys_metrics.get(f"{m}@{k}", 0.0)
                    row.append(f"{val:.4f}")
            md += "| " + " | ".join(row) + " |\n"
            
        md += "\n---\n\n"
        
    abstention = report_data.get("abstention", {})
    md += "## Abstention / Unsupported Queries\n"
    md += f"- **No-Positive Queries:** {abstention.get('num_no_positive_queries', 0)}\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

def main():
    parser = argparse.ArgumentParser(description="HHG RAG ACCURACY BENCHMARK")
    parser.add_argument("--artifact-root", default=settings.HHG_ARTIFACT_DIR)
    parser.add_argument("--ground-truth")
    parser.add_argument("--languages", nargs="+", default=["hi", "en", "bn"])
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--output")
    args = parser.parse_args()
    
    gt_path_str = args.ground_truth or os.path.join(args.artifact_root, "ground_truth.json")
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "reports", "accuracy_results.json"
    )
    report_md_path = os.path.join(
        os.path.dirname(output_path) if output_path else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "accuracy_report.md"
    )
    if not args.output:
        report_md_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "reports", "accuracy_report.md"
        )
    else:
        report_md_path = os.path.join(os.path.dirname(args.output), "accuracy_report.md")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    def fail_not_run(reason):
        print(f"Status: NOT RUN. Reason: {reason}")
        res = {"status": "NOT RUN", "reason": reason}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        generate_markdown_report(res, report_md_path, args)
        sys.exit(1)
        
    if not os.path.exists(gt_path_str):
        fail_not_run("ground_truth.json not found")
        
    try:
        gt = load_ground_truth(Path(gt_path_str), artifact_root=Path(args.artifact_root))
    except GroundTruthError as e:
        fail_not_run(f"Ground truth validation failed: {e}")
        
    mapping_report_path = os.path.join(os.path.dirname(output_path), "ground_truth_mapping_report.json")
    mapping_coverage = 1.0
    if os.path.exists(mapping_report_path):
        try:
            with open(mapping_report_path, "r", encoding="utf-8") as f:
                mapping_report = json.load(f)
            mapping_coverage = mapping_report.get("counts", {}).get("mapping_coverage_over_uploaded_rows", 1.0)
        except Exception:
            pass
            
    if mapping_coverage < 1.0:
        fail_not_run(f"Mapping coverage {mapping_coverage} is below 1.0")

    loader_instance.initialize()
    if not loader_instance.status.get("valid"):
        fail_not_run("Artifacts not valid.")
        
    manifest_path = os.path.join(args.artifact_root, "build_manifest.json")
    if not os.path.exists(manifest_path):
        manifest_path = os.path.join(args.artifact_root, "config.json")
        
    embedder = Embedder()
    bm25_retriever = BM25Retriever()
    hnsw_retriever = HNSWRetriever()
    
    report_data = {
        "status": "MEASURED",
        "source": {
            "dataset": "ai4bharat/MSMARCO-XI",
            "ground_truth_sha256": _sha256_file(gt_path_str),
            "artifact_manifest_sha256": _sha256_file(manifest_path) if os.path.exists(manifest_path) else "",
            "commit_sha": get_git_commit(),
            "revision": gt.queries[0].get("revision", "unknown") if gt.queries else "unknown",
            "selection_profile": "default"
        },
        "overall": {
            "mapping_coverage": mapping_coverage
        },
        "by_language": {},
        "abstention": {
            "num_no_positive_queries": 0,
            "abstention_metrics": {}
        }
    }
    
    total_no_positive = 0
    
    for lang in args.languages:
        if lang not in gt.by_language:
            continue
            
        supported = gt.supported_queries_by_language.get(lang, [])
        unsupported = gt.unsupported_queries_by_language.get(lang, [])
        
        logger.info(f"Running benchmark for {lang} ({len(supported)} supported queries)...")
        metrics = run_language_benchmark(lang, supported, embedder, bm25_retriever, hnsw_retriever, args.top_k)
        
        report_data["by_language"][lang] = {
            "num_queries": len(supported) + len(unsupported),
            "num_queries_with_gold": len(supported),
            **metrics
        }
        
        total_no_positive += len(unsupported)
        
    report_data["abstention"]["num_no_positive_queries"] = total_no_positive
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
        
    generate_markdown_report(report_data, report_md_path, args)
    print(f"\nBenchmark completed. Results saved to {output_path} and {report_md_path}")

if __name__ == "__main__":
    main()
