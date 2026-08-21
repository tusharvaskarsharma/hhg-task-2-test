import uuid
import time
import logging
import json
import hashlib
from fastapi import APIRouter, Request, Response
from backend.schemas.query import QueryRequest
from backend.schemas.response import QueryResponse, ErrorResponse, APIErrorDetail
from backend.pipeline.retrieval_service import retrieval_service
from backend.pipeline.generator import generator_service
from backend.pipeline.extractive import build_extractive_answer, ExtractiveDecision
from backend.pipeline.language import normalize_language, detect_text_language
from backend.pipeline.tokenizer import preprocess_query
from backend.pipeline.query_cache import cache_instance
from backend.config import settings
from backend.artifact_loader import loader_instance

router = APIRouter()
logger = logging.getLogger(__name__)

def _get_artifact_version(lang: str) -> str:
    lang_artifacts = loader_instance.languages.get(lang)
    if lang_artifacts and lang_artifacts.manifest_data:
        manifest_str = json.dumps(lang_artifacts.manifest_data, sort_keys=True)
        return hashlib.sha256(manifest_str.encode('utf-8')).hexdigest()
    return "unknown"

@router.post("/api/query", response_model=QueryResponse, responses={
    400: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
})
async def query_endpoint(req: QueryRequest, request: Request, response: Response):
    # ── Request ID tracking ────────────────────────────────────────────────
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    response.headers["X-Request-ID"] = req_id

    # ── Readiness check ────────────────────────────────────────────────────
    if not retrieval_service.initialized:
        err = APIErrorDetail(code="SERVICE_UNAVAILABLE", message="Backend not ready")
        return Response(
            content=ErrorResponse(error=err).model_dump_json(),
            status_code=503,
            media_type="application/json",
        )

    start_time = time.perf_counter()
    cache_lookup_ms = 0.0
    
    # ── Language Detection & Preprocessing ──────────────────────────────────
    user_lang = normalize_language(req.language)
    if user_lang:
        final_lang = user_lang
    else:
        detected = detect_text_language(req.query)
        final_lang = detected or "hi"

    processed_query = preprocess_query(req.query)

    logger.info(
        f"ReqID: {req_id} | POST /api/query | lang_req: {req.language} | final_lang: {final_lang} | top_k: {req.top_k} | generate: {req.generate}"
    )

    try:
        # ── Response Cache Check ──────────────────────────────────────────────
        # Full-response entries are only valid for generated requests. RAG_ONLY
        # must always execute retrieval/extractive handling and must never read
        # or write the generated-response cache.
        response_cache_key = None
        should_compute = False
        cache_lookup_ms = 0.0
        if req.generate and cache_instance.response_enabled:
            t_cache = time.perf_counter()
            artifact_version = _get_artifact_version(final_lang)
            response_cache_key = cache_instance.generate_response_key(
                normalized_query=processed_query,
                language=final_lang,
                top_k=req.top_k,
                generate=True,
                slm_provider=settings.SLM_PROVIDER,
                slm_model=settings.SLM_MODEL,
                prompt_version=settings.PROMPT_VERSION,
                grounding_version=settings.GROUNDING_VERSION,
                artifact_version=artifact_version
            )
            cached_response, should_compute = cache_instance.get_or_wait_response(response_cache_key)
            cache_lookup_ms = (time.perf_counter() - t_cache) * 1000.0
        else:
            cached_response = None

        if cached_response is not None:
            # Response Cache Hit
            total_latency_ms = (time.perf_counter() - start_time) * 1000.0
            resp_dict = cached_response
            
            cache_meta = {
                "hit": True,
                "enabled": True,
                "cache_layer": "response",
                "retrieval_cache_hit": False,
                "response_cache_hit": True,
                "cache_key_version": cache_instance.schema_version,
                "cache_lookup_ms": round(cache_lookup_ms, 2)
            }
            
            t_pre_ser = time.perf_counter()
            resp = QueryResponse(
                query=req.query,
                language=final_lang,
                answer=resp_dict["answer"],
                extractive_answer=resp_dict["extractive_answer"],
                generated_answer=resp_dict["generated_answer"],
                answer_source=resp_dict["answer_source"],
                grounding=resp_dict["grounding"],
                results=resp_dict["results"],
                cache=cache_meta,
                latency={
                    "total_ms": 0.0,
                    "partial_ms": 0.0,
                    "rag_only_ms": 0.0,
                    "breakdown": {
                        "generation_ms": 0.0,
                        "slm_ms": 0.0,
                        "stt_ms": 0.0,
                        "language_detection_ms": 0.0,
                        "tokenization_ms": 0.0,
                        "embedding_ms": 0.0,
                        "bm25_ms": 0.0,
                        "hnsw_ms": 0.0,
                        "rrf_ms": 0.0,
                        "metadata_ms": 0.0,
                        "grounding_ms": 0.0,
                        "extractive_ms": 0.0,
                        "validation_ms": 0.0,
                        "serialization_ms": 0.0
                    }
                },
                sources=resp_dict["sources"],
                retrieval=resp_dict["retrieval"],
                metrics={
                    "cache_hit": True,
                    "retrieval_latency_ms": 0.0,
                    "total_latency_ms": 0.0,
                },
            )
            
            _ = resp.model_dump_json()
            serialization_ms = (time.perf_counter() - t_pre_ser) * 1000.0
            total_latency_ms += serialization_ms
            
            resp.latency.total_ms = round(total_latency_ms, 2)
            resp.latency.partial_ms = round(total_latency_ms, 2)
            resp.latency.rag_only_ms = round(total_latency_ms, 2)
            resp.latency.breakdown.serialization_ms = round(serialization_ms, 2)
            resp.metrics["total_latency_ms"] = round(total_latency_ms, 2)
            
            logger.info(
                f"ReqID: {req_id} | Success (Response Cache Hit) | lang: {final_lang} | "
                f"total_ms: {total_latency_ms:.2f}"
            )
            return resp

        # ── Execution Path (Response Cache Miss) ──────────────────────────
        try:
            # ── Retrieval ──────────────────────────────────────────────────────
            ret_res = retrieval_service.execute_query(req.query, final_lang, req.top_k)
            retrieval_done_ms = (time.perf_counter() - start_time) * 1000.0
            bd = ret_res["latency_breakdown"]
            stt_ms = 0.0

            # ── Extractive fast path (always run) ──────────────────────────────
            t_ext = time.perf_counter()
            ext_answer, ext_decision, ext_sources = build_extractive_answer(
                req.query, ret_res["results"]
            )
            extractive_ms = (time.perf_counter() - t_ext) * 1000.0

            # ── Optional SLM generation ────────────────────────────────────────
            generation_ms = 0.0
            grounding_ms = 0.0
            grounding_validation_ms = 0.0
            ood_early_exit = False

            if req.generate:
                # ── OOD Early Exit: skip SLM if extractive says no support ──────
                if ext_decision != ExtractiveDecision.SUPPORTED:
                    # Out-of-dataset or insufficient context — skip SLM to save latency
                    ood_early_exit = True
                    gen_answer = None
                    answer_source = "extractive" if ext_decision == ExtractiveDecision.SUPPORTED else "abstain"
                    final_answer = ext_answer
                    grounding_obj = {
                        "enabled": True,
                        "grounded": ext_decision == ExtractiveDecision.SUPPORTED,
                        "status": ext_decision,
                        "sources": ext_sources,
                        "validated": ext_decision == ExtractiveDecision.SUPPORTED,
                        "ood_early_exit": True,
                    }
                else:
                    # Extractive found support — try SLM generation
                    gen_res = generator_service.generate(req.query, final_lang, ret_res["results"])
                    gen_answer = gen_res.get("answer")
                    answer_source = gen_res.get("answer_source", "generated")
                    gen_grounding = gen_res.get("grounding", {})
                    gen_bd = gen_res.get("latency", {})
                    grounding_ms = gen_bd.get("grounding_ms", 0.0)
                    generation_ms = gen_bd.get("generation_ms", 0.0)
                    grounding_validation_ms = gen_bd.get("grounding_validation_ms", 0.0)

                    if gen_answer and gen_grounding.get("grounded"):
                        final_answer = gen_answer
                        grounding_obj = gen_grounding
                    else:
                        final_answer = ext_answer
                        if answer_source == "generated-unavailable" and ext_decision != ExtractiveDecision.SUPPORTED:
                            pass
                        else:
                            answer_source = "extractive" if ext_decision == ExtractiveDecision.SUPPORTED else "abstain"
                        grounding_obj = {
                            "enabled": True,
                            "grounded": ext_decision == ExtractiveDecision.SUPPORTED,
                            "status": ext_decision,
                            "sources": ext_sources,
                            "validated": ext_decision == ExtractiveDecision.SUPPORTED,
                        }
            else:
                gen_answer = None
                final_answer = ext_answer
                answer_source = "extractive" if ext_decision == ExtractiveDecision.SUPPORTED else "abstain"
                grounding_obj = {
                    "enabled": True,
                    "grounded": ext_decision == ExtractiveDecision.SUPPORTED,
                    "status": ext_decision,
                    "sources": ext_sources,
                    "validated": ext_decision == ExtractiveDecision.SUPPORTED,
                }

            if not req.generate and answer_source not in ["extractive", "abstain"]:
                answer_source = "extractive" if ext_decision == ExtractiveDecision.SUPPORTED else "abstain"

            # ── Response Cache Store ──────────────────────────────────────────
            if req.generate and response_cache_key is not None:
                # Cache conditions: generated+grounded OR extractive+supported (for repeat OOD avoidance)
                should_cache_generated = (
                    answer_source == "generated" and 
                    grounding_obj.get("grounded", False)
                )
                should_cache_extractive = (
                    answer_source == "extractive" and
                    ext_decision == ExtractiveDecision.SUPPORTED and
                    grounding_obj.get("grounded", False)
                )
                
                if should_cache_generated or should_cache_extractive:
                    def get_source(r):
                        return r.source if hasattr(r, "source") else r.get("source", "")
                    cache_payload = {
                        "answer": final_answer,
                        "extractive_answer": ext_answer if ext_decision == ExtractiveDecision.SUPPORTED else None,
                        "generated_answer": gen_answer,
                        "answer_source": answer_source,
                        "grounding": grounding_obj,
                        "results": ret_res["results"],
                        "sources": grounding_obj.get("sources", []),
                        "retrieval": {
                            "bm25": sum(1 for r in ret_res["results"] if "bm25" in get_source(r)),
                            "hnsw": sum(1 for r in ret_res["results"] if "hnsw" in get_source(r)),
                            "rrf": len(ret_res["results"])
                        }
                    }
                    cache_instance.set_response(response_cache_key, cache_payload)
                else:
                    cache_instance.release_response_lock(response_cache_key)
            # RAG_ONLY deliberately does not populate the full-response cache.

        except Exception as e:
            if response_cache_key is not None:
                cache_instance.release_response_lock(response_cache_key)
            raise e

        # ── Finalize Response ──────────────────────────────────────────────
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        cache_layer = "retrieval" if ret_res["cache"]["hit"] else "none"
        if not cache_instance.response_enabled and not cache_instance.retrieval_enabled:
            cache_layer = "none"

        cache_meta = {
            "hit": ret_res["cache"]["hit"],
            "enabled": cache_instance.retrieval_enabled or cache_instance.response_enabled,
            "cache_layer": cache_layer,
            "retrieval_cache_hit": ret_res["cache"]["hit"],
            "response_cache_hit": False,
            "cache_key_version": cache_instance.schema_version,
            "cache_lookup_ms": round(cache_lookup_ms, 2)
        }

        breakdown_dict = {
            "stt_ms": stt_ms,
            "language_detection_ms": bd.get("language_detection_ms", 0.0),
            "tokenization_ms": bd.get("tokenization_ms", 0.0),
            "embedding_ms": bd.get("embedding_ms", 0.0),
            "bm25_ms": bd.get("bm25_ms", 0.0),
            "hnsw_ms": bd.get("hnsw_ms", 0.0),
            "rrf_ms": bd.get("rrf_ms", 0.0),
            "metadata_ms": bd.get("metadata_ms", 0.0),
            "grounding_ms": grounding_ms,
            "extractive_ms": extractive_ms,
            "slm_ms": generation_ms,
            "generation_ms": generation_ms,
            "validation_ms": grounding_validation_ms,
        }

        rag_only_ms = ret_res["rag_only_base_ms"] + extractive_ms
        partial_ms = total_latency_ms

        def get_source2(r):
            return r.source if hasattr(r, "source") else r.get("source", "")
        bm25_count = sum(1 for r in ret_res["results"] if "bm25" in get_source2(r))
        hnsw_count = sum(1 for r in ret_res["results"] if "hnsw" in get_source2(r))
        retrieval_obj = {"bm25": bm25_count, "hnsw": hnsw_count, "rrf": len(ret_res["results"])}

        t_pre_ser = time.perf_counter()
        resp = QueryResponse(
            query=req.query,
            language=final_lang,
            answer=final_answer,
            extractive_answer=ext_answer if ext_decision == ExtractiveDecision.SUPPORTED else None,
            generated_answer=gen_answer if req.generate else None,
            answer_source=answer_source,
            grounding=grounding_obj,
            results=ret_res["results"],
            cache=cache_meta,
            latency={
                "total_ms": 0.0,
                "partial_ms": 0.0,
                "rag_only_ms": 0.0,
                "breakdown": breakdown_dict,
            },
            sources=grounding_obj.get("sources", []),
            retrieval=retrieval_obj,
            metrics={
                "cache_hit": ret_res["cache"]["hit"],
                "retrieval_latency_ms": round(ret_res["rag_only_base_ms"], 2),
                "total_latency_ms": round(total_latency_ms, 2),
            },
        )

        _ = resp.model_dump_json()
        serialization_ms = (time.perf_counter() - t_pre_ser) * 1000.0

        breakdown_dict["serialization_ms"] = serialization_ms
        rag_only_ms += serialization_ms
        partial_ms += serialization_ms
        total_latency_ms += serialization_ms

        resp.latency.total_ms = round(total_latency_ms, 2)
        resp.latency.partial_ms = round(partial_ms, 2)
        resp.latency.rag_only_ms = round(rag_only_ms, 2)
        resp.latency.breakdown.serialization_ms = round(serialization_ms, 2)

        logger.info(
            f"ReqID: {req_id} | Success | lang: {final_lang} | "
            f"cache_layer: {cache_layer} | answer_source: {answer_source} | "
            f"ood_early_exit: {ood_early_exit} | "
            f"total_ms: {total_latency_ms:.2f}"
        )
        return resp

    except Exception as e:
        logger.error(f"ReqID: {req_id} | Error: {str(e)}", exc_info=True)
        err = APIErrorDetail(code="INTERNAL_ERROR", message="An unexpected internal error occurred")
        return Response(
            content=ErrorResponse(error=err).model_dump_json(),
            status_code=500,
            media_type="application/json",
        )
