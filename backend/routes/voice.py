import uuid
import time
import logging
from fastapi import APIRouter, UploadFile, File, Form, Request, Response
from typing import Optional
from backend.schemas.response import QueryResponse, ErrorResponse, APIErrorDetail
from backend.pipeline.stt import stt_service, STTServiceError
from backend.pipeline.retrieval_service import retrieval_service
from backend.pipeline.generator import generator_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/api/voice", response_model=QueryResponse, responses={
    400: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
    503: {"model": ErrorResponse}
})
async def voice_endpoint(
    request: Request,
    response: Response,
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    top_k: int = Form(10)
):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    response.headers["X-Request-ID"] = req_id
    
    if not retrieval_service.initialized:
        err = APIErrorDetail(code="SERVICE_UNAVAILABLE", message="Backend not ready")
        return Response(content=ErrorResponse(error=err).model_dump_json(), status_code=503, media_type="application/json")
        
    logger.info(f"ReqID: {req_id} | POST /api/voice | file: {audio.filename}")
    
    start_time = time.perf_counter()
    
    try:
        # 1. Read Audio
        audio_bytes = await audio.read()
        await audio.close()
        
        # 2. Transcribe
        stt_start = time.perf_counter()
        stt_res = stt_service.transcribe(audio_bytes, audio.filename)
        stt_latency_ms = (time.perf_counter() - stt_start) * 1000.0
        
        transcript_text = stt_res["text"]
        
        # 3. Resolve Language (Form override takes precedence over STT detection)
        final_lang = language or stt_res.get("language") or "hi"
        
        # 4. Retrieval
        ret_res = retrieval_service.execute_query(transcript_text, final_lang, top_k)
        
        # 5. Generation
        gen_res = generator_service.generate(transcript_text, final_lang, ret_res["results"])
        
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        logger.info(f"ReqID: {req_id} | STT Success | lang: {final_lang} | stt_ms: {stt_latency_ms:.2f} | cache_hit: {ret_res['cache']['hit']}")
        
        # Breakdown Mapping
        bd = ret_res["latency_breakdown"]
        gen_bd = gen_res["latency"]
        
        # Construct pre-serialization breakdown
        breakdown_dict = {
            "stt_ms": stt_latency_ms,
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
        partial_ms = total_latency_ms - stt_latency_ms
        
        t_pre_ser = time.perf_counter()
        resp = QueryResponse(
            query=transcript_text,
            language=final_lang,
            answer=gen_res["answer"],
            grounding=gen_res["grounding"],
            results=ret_res["results"],
            cache=ret_res["cache"],
            latency={
                "total_ms": 0.0,
                "partial_ms": 0.0,
                "rag_only_ms": 0.0,
                "breakdown": breakdown_dict
            },
            transcription={
                "text": transcript_text,
                "detected_language": stt_res.get("language")
            }
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
        
        return resp
        
    except STTServiceError as e:
        logger.error(f"ReqID: {req_id} | STT Error {e.status_code}: {str(e)}")
        err = APIErrorDetail(code="STT_FAILURE", message=str(e))
        return Response(content=ErrorResponse(error=err).model_dump_json(), status_code=e.status_code, media_type="application/json")
    except Exception as e:
        logger.error(f"ReqID: {req_id} | Unexpected Error: {str(e)}")
        err = APIErrorDetail(code="INTERNAL_ERROR", message="An unexpected internal error occurred")
        return Response(content=ErrorResponse(error=err).model_dump_json(), status_code=500, media_type="application/json")
