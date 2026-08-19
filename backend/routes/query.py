import uuid
import time
import logging
from fastapi import APIRouter, HTTPException, status, Request, Response
from backend.schemas.query import QueryRequest
from backend.schemas.response import QueryResponse, ErrorResponse, APIErrorDetail
from backend.pipeline.retrieval_service import retrieval_service
from backend.pipeline.generator import generator_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/api/query", response_model=QueryResponse, responses={
    400: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse}
})
async def query_endpoint(req: QueryRequest, request: Request, response: Response):
    # Request ID tracking
    req_id = request.headers.get("X-Request-ID")
    if not req_id:
        req_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = req_id

    # Readiness check
    if not retrieval_service.initialized:
        err = APIErrorDetail(code="SERVICE_UNAVAILABLE", message="Backend not ready")
        return Response(content=ErrorResponse(error=err).model_dump_json(), status_code=503, media_type="application/json")

    logger.info(f"ReqID: {req_id} | POST /api/query | lang: {req.language} | top_k: {req.top_k}")

    try:
        start_time = time.perf_counter()
        
        # Retrieval
        ret_res = retrieval_service.execute_query(req.query, req.language, req.top_k)
        
        # Generation
        gen_res = generator_service.generate(req.query, req.language, ret_res["results"])
        
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        logger.info(f"ReqID: {req_id} | Success | lang: {req.language} | cache_hit: {ret_res['cache']['hit']} | total_ms: {total_latency_ms:.2f}")
        
        # Breakdown Mapping
        bd = ret_res["latency_breakdown"]
        gen_bd = gen_res["latency"]
        
        stt_ms = 0.0 # Not used in text query
        
        # Construct pre-serialization breakdown
        breakdown_dict = {
            "stt_ms": stt_ms,
            "language_detection_ms": bd.get("language_detection_ms", 0.0),
            "tokenization_ms": bd.get("tokenization_ms", 0.0),
            "embedding_ms": bd.get("embedding_ms", 0.0),
            "bm25_ms": bd.get("bm25_ms", 0.0),
            "hnsw_ms": bd.get("hnsw_ms", 0.0),
            "rrf_ms": bd.get("rrf_ms", 0.0),
            "metadata_ms": bd.get("metadata_ms", 0.0),
            "grounding_ms": gen_bd.get("grounding_ms", 0.0),
            "slm_ms": 0.0,
            "generation_ms": gen_bd.get("generation_ms", 0.0),
            "validation_ms": gen_bd.get("grounding_validation_ms", 0.0)
        }
        
        # RAG_ONLY calculation (everything except SLM/STT, plus overhead logic)
        rag_only_ms = ret_res["rag_only_base_ms"] + breakdown_dict["grounding_ms"]

        # PARTIAL calculation (everything except STT)
        partial_ms = total_latency_ms - stt_ms

        # Metrics requested in Task 5
        metrics_obj = {
            "cache_hit": ret_res['cache']['hit'],
            "retrieval_latency_ms": round(ret_res["rag_only_base_ms"], 2),
            "total_latency_ms": round(total_latency_ms, 2)
        }
        
        # Retrieval breakdown requested in Task 5
        def get_source(r):
            return r.source if hasattr(r, 'source') else r.get('source', '')
            
        bm25_count = sum(1 for r in ret_res["results"] if "bm25" in get_source(r))
        hnsw_count = sum(1 for r in ret_res["results"] if "hnsw" in get_source(r))
        rrf_count = len(ret_res["results"])
        
        retrieval_obj = {
            "bm25": bm25_count,
            "hnsw": hnsw_count,
            "rrf": rrf_count
        }

        # Validate format for grounding dict
        grounding_obj = gen_res["grounding"].copy()
        grounding_obj["validated"] = grounding_obj.get("grounded", False)
        
        # We need to time serialization, so we first build the latency struct minus serialization
        # then we measure JSON dump time and update it before returning.
        
        t_pre_ser = time.perf_counter()
        resp = QueryResponse(
            query=req.query,
            language=req.language,
            answer=gen_res["answer"],
            grounding=grounding_obj,
            results=ret_res["results"],
            cache=ret_res["cache"],
            latency={
                "total_ms": 0.0,
                "partial_ms": 0.0,
                "rag_only_ms": 0.0,
                "breakdown": breakdown_dict
            },
            sources=grounding_obj.get("sources", []),
            retrieval=retrieval_obj,
            metrics=metrics_obj
        )
        
        # Mock serialization time for latency injection
        _ = resp.model_dump_json()
        serialization_ms = (time.perf_counter() - t_pre_ser) * 1000.0
        
        breakdown_dict["serialization_ms"] = serialization_ms
        rag_only_ms += serialization_ms
        partial_ms += serialization_ms
        total_latency_ms += serialization_ms
        
        # Now update actual values
        resp.latency.total_ms = round(total_latency_ms, 2)
        resp.latency.partial_ms = round(partial_ms, 2)
        resp.latency.rag_only_ms = round(rag_only_ms, 2)
        resp.latency.breakdown.serialization_ms = round(serialization_ms, 2)
        
        # We can't re-assign the response's fields as simply via model_dump_json because it's a pydantic model, 
        # so returning the updated `resp` will use FastAPI's serialization automatically. The measured serialization_ms
        # is an approximation of what FastAPI does right after we return.
        
        return resp
        
    except Exception as e:
        logger.error(f"ReqID: {req_id} | Error: {str(e)}")
        err = APIErrorDetail(code="INTERNAL_ERROR", message="An unexpected internal error occurred")
        return Response(content=ErrorResponse(error=err).model_dump_json(), status_code=500, media_type="application/json")
