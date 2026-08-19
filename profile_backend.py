import time
import json
import numpy as np
from collections import defaultdict
from backend.main import app
from backend.artifact_loader import loader_instance
from backend.pipeline.tokenizer import preprocess_query
from backend.pipeline.language import resolve_language
from backend.pipeline.query_cache import cache_instance
from backend.pipeline.fusion import rrf_fuse
from backend.schemas.query import QueryRequest
from backend.schemas.response import RetrievalResult, QueryResponse
from backend.pipeline.retrieval_service import retrieval_service
from backend.pipeline.grounding import grounding_service
from backend.config import settings

def profile():
    loader_instance.initialize()
    if not loader_instance.status.get("valid"):
        print("Artifact loader not valid!")
        return

    retrieval_service.initialize()
    
    query_text = "भारत की राजधानी क्या है?"
    lang = "hi"
    top_k = 5
    
    # Disable cache for these tests to measure full retrieval path
    settings.CACHE_ENABLED = False
    
    stats = defaultdict(list)
    
    print("Warming up...")
    for _ in range(5):
        retrieval_service.execute_query(query_text, lang, top_k)
        
    print("Running 100 iterations for profiling...")
    
    for _ in range(100):
        # 1. request parsing
        t0 = time.perf_counter()
        req = QueryRequest(query=query_text, language=lang, top_k=top_k)
        t1 = time.perf_counter()
        stats["1. request parsing"].append((t1 - t0) * 1000)
        
        # 2. language detection (+ preprocessing)
        t0 = time.perf_counter()
        processed_query = preprocess_query(req.query)
        detected_lang = resolve_language(processed_query, req.language)
        t1 = time.perf_counter()
        stats["2. language detection"].append((t1 - t0) * 1000)
        
        # 3. cache lookup
        t0 = time.perf_counter()
        _ = cache_instance.get(processed_query, detected_lang, req.top_k)
        t1 = time.perf_counter()
        stats["3. cache lookup"].append((t1 - t0) * 1000)
        
        # 4 & 5. tokenization & ONNX embedding
        # We need to split this manually since embedder.py combines them
        embedder = retrieval_service.embedder
        
        t0 = time.perf_counter()
        model_inputs = embedder.tokenizer(
            processed_query,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512
        )
        t1 = time.perf_counter()
        stats["4. tokenization"].append((t1 - t0) * 1000)
        
        t0 = time.perf_counter()
        inputs_onnx = {k: v.astype(np.int64) for k, v in model_inputs.items()}
        if "token_type_ids" not in inputs_onnx:
            inputs_onnx["token_type_ids"] = np.zeros_like(inputs_onnx["input_ids"])
            
        outputs = embedder.session.run(None, inputs_onnx)
        embeddings = outputs[0]
        q_emb = embeddings[0, 0, :]
        q_emb = q_emb / np.linalg.norm(q_emb) # just normalize it by default
        t1 = time.perf_counter()
        stats["5. ONNX embedding"].append((t1 - t0) * 1000)
        
        fusion_k = max(60, req.top_k * 2)
        
        # 6. BM25 retrieval
        t0 = time.perf_counter()
        bm25_res = retrieval_service.bm25.retrieve(processed_query, detected_lang, top_k=fusion_k)
        t1 = time.perf_counter()
        stats["6. BM25 retrieval"].append((t1 - t0) * 1000)
        
        # 7. HNSW retrieval
        t0 = time.perf_counter()
        hnsw_res = retrieval_service.hnsw.retrieve(q_emb, detected_lang, top_k=fusion_k)
        t1 = time.perf_counter()
        stats["7. HNSW retrieval"].append((t1 - t0) * 1000)
        
        # 8. RRF fusion
        t0 = time.perf_counter()
        fused = rrf_fuse(bm25_res, hnsw_res, k=60, top_k=req.top_k)
        t1 = time.perf_counter()
        stats["8. RRF fusion"].append((t1 - t0) * 1000)
        
        # 9. metadata lookup
        t0 = time.perf_counter()
        df = loader_instance.get_metadata(detected_lang)
        id_col = next((c for c in ["id", "doc_id", "passage_id"] if c in df.columns), df.columns[0])
        text_col = "text" if "text" in df.columns else "passage"
        lang_col = "lang" if "lang" in df.columns else ("language" if "language" in df.columns else None)
        
        final_results = []
        for res in fused:
            try:
                row = df.loc[str(res["id"])]
            except KeyError:
                continue
                
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            
            doc_text = row[text_col]
            doc_lang = row[lang_col] if lang_col else None
            source_str = "rrf(" + ",".join(res["sources"]) + ")"
            final_results.append(RetrievalResult(
                doc_id=res["id"],
                text=doc_text,
                score=res["rrf_score"],
                rank=res["rank"],
                source=source_str,
                language=doc_lang
            ))
        t1 = time.perf_counter()
        stats["9. metadata lookup"].append((t1 - t0) * 1000)
        
        # 10. grounding/context construction
        t0 = time.perf_counter()
        context_str, sources = grounding_service.build_context(final_results)
        t1 = time.perf_counter()
        stats["10. grounding/context construction"].append((t1 - t0) * 1000)
        
        # 11. response serialization
        t0 = time.perf_counter()
        resp = QueryResponse(
            query=req.query,
            language=detected_lang,
            answer=None,
            grounding={"enabled": True, "grounded": False, "sources": sources},
            results=final_results,
            cache={"hit": False},
            latency={"total_ms": 0}
        )
        _ = resp.model_dump_json()
        t1 = time.perf_counter()
        stats["11. response serialization"].append((t1 - t0) * 1000)
        
        # 12. total retrieval latency (summing for simplicity, roughly equivalent to endpoint without network overhead)
        total = sum([
            stats["1. request parsing"][-1],
            stats["2. language detection"][-1],
            stats["3. cache lookup"][-1],
            stats["4. tokenization"][-1],
            stats["5. ONNX embedding"][-1],
            stats["6. BM25 retrieval"][-1],
            stats["7. HNSW retrieval"][-1],
            stats["8. RRF fusion"][-1],
            stats["9. metadata lookup"][-1],
            stats["10. grounding/context construction"][-1],
            stats["11. response serialization"][-1]
        ])
        stats["12. total retrieval latency"].append(total)

    print("\n--- PROFILING RESULTS (100 iterations) ---")
    print(f"{'Stage':<35} | {'p50 (ms)':>8} | {'p95 (ms)':>8} | {'Min (ms)':>8} | {'Max (ms)':>8}")
    print("-" * 75)
    for k in sorted(stats.keys(), key=lambda x: int(x.split(".")[0])):
        arr = np.array(stats[k])
        p50 = np.percentile(arr, 50)
        p95 = np.percentile(arr, 95)
        m_min = np.min(arr)
        m_max = np.max(arr)
        print(f"{k:<35} | {p50:8.2f} | {p95:8.2f} | {m_min:8.2f} | {m_max:8.2f}")

if __name__ == "__main__":
    profile()
