import time
import pandas as pd
from backend.pipeline.tokenizer import preprocess_query
from backend.pipeline.language import resolve_language
from backend.pipeline.embedder import Embedder
from backend.pipeline.sparse_retriever import BM25Retriever
from backend.pipeline.dense_retriever import HNSWRetriever
from backend.pipeline.fusion import rrf_fuse
from backend.pipeline.query_cache import cache_instance
from backend.schemas.response import RetrievalResult
from backend.artifact_loader import loader_instance
import logging
import pandas as pd
logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self):
        # We assume the loader is already initialized by main.py
        self.embedder = None
        self.bm25 = None
        self.hnsw = None
        self.initialized = False

    def initialize(self):
        if not self.initialized and loader_instance.status.get("valid"):
            self.embedder = Embedder()
            self.bm25 = BM25Retriever()
            self.hnsw = HNSWRetriever()
            
            # Pre-index metadata dataframes for O(1) lookup
            for lang in loader_instance.SUPPORTED_LANGUAGES:
                df = loader_instance.get_metadata(lang)
                if df is not None:
                    id_col = next((c for c in ["id", "doc_id", "passage_id"] if c in df.columns), df.columns[0])
                    if df.index.name != id_col:
                        df.set_index(id_col, inplace=True)
                        df.index = df.index.astype(str)
                        
            self.initialized = True
            logger.info("Retrieval service initialized. Running warmup query...")
            try:
                # Force lazy initialization (AutoTokenizer, ONNX, pandas index)
                self.execute_query("warmup", "hi", 1)
                logger.info("Warmup query completed successfully.")
            except Exception as e:
                logger.warning(f"Warmup query failed (non-fatal): {e}")

    def execute_query(self, query: str, language: str, top_k: int) -> dict:
        start_time = time.perf_counter()
        breakdown = {}
        
        # 1. Preprocess & Language Detection
        t0 = time.perf_counter()
        processed_query = preprocess_query(query)
        lang = resolve_language(processed_query, language)
        breakdown["language_detection_ms"] = (time.perf_counter() - t0) * 1000.0
        
        # 2. Cache Lookup
        cached_results = cache_instance.get(processed_query, lang, top_k)
        if cached_results is not None:
            # For cache hit, we mock out the retrieval breakdown to 0.0
            breakdown.update({
                "tokenization_ms": 0.0, "embedding_ms": 0.0, "bm25_ms": 0.0,
                "hnsw_ms": 0.0, "rrf_ms": 0.0, "metadata_ms": 0.0
            })
            total_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "query": query,
                "language": lang,
                "results": cached_results,
                "cache": {"hit": True},
                "latency_breakdown": breakdown,
                "rag_only_base_ms": total_ms
            }
            
        # 3. Retrieve
        if not self.initialized:
            raise RuntimeError("RetrievalService not initialized")
            
        q_emb, tok_ms, emb_ms = self.embedder.embed_query(processed_query)
        breakdown["tokenization_ms"] = tok_ms
        breakdown["embedding_ms"] = emb_ms
        
        fusion_k = max(60, top_k * 2)
        
        t1 = time.perf_counter()
        bm25_res = self.bm25.retrieve(processed_query, lang, top_k=fusion_k)
        breakdown["bm25_ms"] = (time.perf_counter() - t1) * 1000.0
        
        t2 = time.perf_counter()
        hnsw_res = self.hnsw.retrieve(q_emb, lang, top_k=fusion_k)
        breakdown["hnsw_ms"] = (time.perf_counter() - t2) * 1000.0
        
        t3 = time.perf_counter()
        fused = rrf_fuse(bm25_res, hnsw_res, k=60, top_k=top_k)
        breakdown["rrf_ms"] = (time.perf_counter() - t3) * 1000.0
        
        # 4. Metadata Lookup
        t4 = time.perf_counter()
        df = loader_instance.get_metadata(lang)
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
            
        breakdown["metadata_ms"] = (time.perf_counter() - t4) * 1000.0
            
        # 5. Cache Store
        cache_instance.set(processed_query, lang, top_k, final_results)
        
        total_ms = (time.perf_counter() - start_time) * 1000.0
        
        return {
            "query": query,
            "language": lang,
            "results": final_results,
            "cache": {"hit": False},
            "latency_breakdown": breakdown,
            "rag_only_base_ms": total_ms
        }

# Singleton instance
retrieval_service = RetrievalService()
