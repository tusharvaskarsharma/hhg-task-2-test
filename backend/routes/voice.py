import uuid
import time
import logging
from fastapi import APIRouter, UploadFile, File, Form, Request, Response
from typing import Optional
from backend.schemas.response import QueryResponse, ErrorResponse, APIErrorDetail
from backend.pipeline.stt import stt_service, STTServiceError
from backend.pipeline.retrieval_service import retrieval_service
from backend.pipeline.generator import generator_service
from backend.pipeline.extractive import build_extractive_answer, ExtractiveDecision
from backend.pipeline.language import normalize_language
from backend.config import settings

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
    top_k: int = Form(10),
    generate: bool = Form(False)
):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    response.headers["X-Request-ID"] = req_id
    
    if not retrieval_service.initialized:
        err = APIErrorDetail(code="SERVICE_UNAVAILABLE", message="Backend not ready")
        return Response(content=ErrorResponse(error=err).model_dump_json(), status_code=503, media_type="application/json")
        
    if not settings.SAARAS_ENABLED:
        err = APIErrorDetail(code="STT_FAILURE", message="Speech-to-text service is disabled.")
        return Response(content=ErrorResponse(error=err).model_dump_json(), status_code=503, media_type="application/json")
        
    logger.info(f"ReqID: {req_id} | POST /api/voice | file: {audio.filename} | generate: {generate}")
    
    start_time = time.perf_counter()
    
    try:
        # 1. Read Audio and Validate
        audio_bytes = await audio.read()
        await audio.close()
        
        # Validation
        valid_magic_bytes = [
            b'OggS', # Ogg
            b'RIFF', # Wav
            b'ID3',  # MP3
            b'\x1a\x45\xdf\xa3', # WebM/MKV
            b'fLaC', # FLAC
            b'0&\xb2u\x8ef\xcf\x11' # WMA (ASF)
        ]
        
        is_audio = False
        for magic in valid_magic_bytes:
            if audio_bytes.startswith(magic):
                is_audio = True
                break
                
        # Also check content type loosely
        is_valid_mime = audio.content_type and (audio.content_type.startswith('audio/') or audio.content_type in ['video/webm', 'video/ogg'])
        
        if not (is_audio or is_valid_mime):
            err = APIErrorDetail(code="INVALID_AUDIO", message="Invalid audio file format")
            return Response(content=ErrorResponse(error=err).model_dump_json(), status_code=400, media_type="application/json")
        
        # 2. Transcribe
        stt_start = time.perf_counter()
        stt_res = stt_service.transcribe(audio_bytes, audio.filename)
        stt_latency_ms = (time.perf_counter() - stt_start) * 1000.0
        
        transcript_text = stt_res["text"]
        
        # 3. Resolve Language (Form override takes precedence over STT detection)
        user_lang = normalize_language(language)
        stt_lang = normalize_language(stt_res.get("language"))
        
        if user_lang:
            final_lang = user_lang
            lang_source = "user_override"
        elif stt_lang:
            final_lang = stt_lang
            lang_source = "stt"
        else:
            final_lang = "hi"
            lang_source = "fallback"
            
        transcription_obj = {
            "text": transcript_text,
            "detected_language": final_lang,
            "language_source": lang_source,
            "confidence": 0.0
        }
        
        # 4. Retrieval
        ret_res = retrieval_service.execute_query(transcript_text, final_lang, top_k)
        
        # 5. Extractive fast path (always run)
        t_ext = time.perf_counter()
        ext_answer, ext_decision, ext_sources = build_extractive_answer(
            transcript_text, ret_res["results"]
        )
        extractive_ms = (time.perf_counter() - t_ext) * 1000.0

        # 6. Optional SLM generation
        generation_ms = 0.0
        grounding_ms = 0.0
        grounding_validation_ms = 0.0

        if generate:
            gen_res = generator_service.generate(transcript_text, final_lang, ret_res["results"])
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
                
                if answer_source == "generated-unavailable":
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
        
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        logger.info(
            f"ReqID: {req_id} | STT Success | lang: {final_lang} | stt_ms: {stt_latency_ms:.2f} | "
            f"cache_hit: {ret_res['cache']['hit']} | answer_source: {answer_source}"
        )
        
        # Breakdown Mapping
        bd = ret_res["latency_breakdown"]
        
        breakdown_dict = {
            "stt_ms": stt_latency_ms,
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
        
        # RAG_ONLY calculation (retrieval + extractive, NO STT, NO SLM)
        rag_only_ms = ret_res["rag_only_base_ms"] + extractive_ms

        # PARTIAL calculation (everything except STT)
        partial_ms = total_latency_ms - stt_latency_ms
        
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

        t_pre_ser = time.perf_counter()
        resp = QueryResponse(
            query=transcript_text,
            language=final_lang,
            answer=final_answer,
            extractive_answer=ext_answer if ext_decision == ExtractiveDecision.SUPPORTED else None,
            generated_answer=gen_answer if generate else None,
            answer_source=answer_source,
            grounding=grounding_obj,
            results=ret_res["results"],
            cache=ret_res["cache"],
            latency={
                "total_ms": 0.0,
                "partial_ms": 0.0,
                "rag_only_ms": 0.0,
                "breakdown": breakdown_dict
            },
            transcription=transcription_obj,
            sources=grounding_obj.get("sources", []),
            retrieval=retrieval_obj,
            metrics={
                "cache_hit": ret_res['cache']['hit'],
                "retrieval_latency_ms": round(ret_res["rag_only_base_ms"], 2),
                "total_latency_ms": round(total_latency_ms, 2)
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
