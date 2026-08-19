import uuid
import time
import logging
from fastapi import APIRouter, Request, Response
from backend.schemas.query import QueryRequest
from backend.schemas.response import QueryResponse, ErrorResponse, APIErrorDetail
from backend.pipeline.retrieval_service import retrieval_service
from backend.pipeline.generator import generator_service
from backend.pipeline.extractive import build_extractive_answer, ExtractiveDecision
from backend.pipeline.language import normalize_language, detect_text_language

router = APIRouter()
logger = logging.getLogger(__name__)


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

    # ── Language Detection ──────────────────────────────────────────────────
    user_lang = normalize_language(req.language)
    if user_lang:
        final_lang = user_lang
    else:
        detected = detect_text_language(req.query)
        final_lang = detected or "hi"

    logger.info(
        f"ReqID: {req_id} | POST /api/query | lang_req: {req.language} | final_lang: {final_lang} | top_k: {req.top_k} | generate: {req.generate}"
    )

    try:
        start_time = time.perf_counter()

        # ── Retrieval ──────────────────────────────────────────────────────
        ret_res = retrieval_service.execute_query(req.query, final_lang, req.top_k)
        retrieval_done_ms = (time.perf_counter() - start_time) * 1000.0

        bd = ret_res["latency_breakdown"]
        stt_ms = 0.0  # text query — no STT

        # ── Extractive fast path (always run) ──────────────────────────────
        t_ext = time.perf_counter()
        ext_answer, ext_decision, ext_sources = build_extractive_answer(
            req.query, ret_res["results"]
        )
        extractive_ms = (time.perf_counter() - t_ext) * 1000.0

        # ── Optional SLM generation (only when generate=True) ─────────────
        generation_ms = 0.0
        grounding_ms = 0.0
        grounding_validation_ms = 0.0

        if req.generate:
            gen_res = generator_service.generate(req.query, final_lang, ret_res["results"])
            gen_answer = gen_res.get("answer")
            answer_source = gen_res.get("answer_source", "generated")
            gen_grounding = gen_res.get("grounding", {})
            gen_bd = gen_res.get("latency", {})
            grounding_ms = gen_bd.get("grounding_ms", 0.0)
            generation_ms = gen_bd.get("generation_ms", 0.0)
            grounding_validation_ms = gen_bd.get("grounding_validation_ms", 0.0)

            # If SLM produced a grounded answer, use it; otherwise fall back to extractive
            if gen_answer and gen_grounding.get("grounded"):
                final_answer = gen_answer
                grounding_obj = gen_grounding
            else:
                # SLM failed or ungrounded — fall back to extractive
                final_answer = ext_answer
                
                if answer_source == "generated-unavailable" and ext_decision != ExtractiveDecision.SUPPORTED:
                    pass # keep generated-unavailable
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
            # STRICT INVARIANT: RAG_ONLY MUST NOT call generator or SLM
            generation_ms = 0.0
            stt_ms = 0.0
            
            # Default: extractive only, no SLM
            final_answer = ext_answer
            answer_source = (
                "extractive" if ext_decision == ExtractiveDecision.SUPPORTED
                else "abstain"
            )
            grounding_obj = {
                "enabled": True,
                "grounded": ext_decision == ExtractiveDecision.SUPPORTED,
                "status": ext_decision,
                "sources": ext_sources,
                "validated": ext_decision == ExtractiveDecision.SUPPORTED,
            }

        # STRICT INVARIANT: answer_source MUST be extractive or abstain when generate is False
        if not req.generate:
            if answer_source not in ["extractive", "abstain"]:
                logger.warning(f"Invariant violation: answer_source={answer_source} but generate=False")
                answer_source = "extractive" if ext_decision == ExtractiveDecision.SUPPORTED else "abstain"

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            f"ReqID: {req_id} | Success | lang: {final_lang} | "
            f"cache_hit: {ret_res['cache']['hit']} | answer_source: {answer_source} | "
            f"total_ms: {total_latency_ms:.2f}"
        )

        # ── Latency breakdown ──────────────────────────────────────────────
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

        # RAG_ONLY = retrieval + extractive (no STT, no SLM)
        rag_only_ms = ret_res["rag_only_base_ms"] + extractive_ms

        # PARTIAL = total time (since there is no STT in text query)
        partial_ms = total_latency_ms

        # ── Retrieval breakdown ────────────────────────────────────────────
        def get_source(r):
            return r.source if hasattr(r, "source") else r.get("source", "")

        bm25_count = sum(1 for r in ret_res["results"] if "bm25" in get_source(r))
        hnsw_count = sum(1 for r in ret_res["results"] if "hnsw" in get_source(r))
        rrf_count = len(ret_res["results"])

        retrieval_obj = {"bm25": bm25_count, "hnsw": hnsw_count, "rrf": rrf_count}

        # ── Serialization timing ───────────────────────────────────────────
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
            cache=ret_res["cache"],
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

        return resp

    except Exception as e:
        logger.error(f"ReqID: {req_id} | Error: {str(e)}", exc_info=True)
        err = APIErrorDetail(code="INTERNAL_ERROR", message="An unexpected internal error occurred")
        return Response(
            content=ErrorResponse(error=err).model_dump_json(),
            status_code=500,
            media_type="application/json",
        )
