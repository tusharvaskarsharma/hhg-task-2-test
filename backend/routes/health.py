from fastapi import APIRouter, Response, status
from fastapi.responses import JSONResponse
from backend.artifact_loader import loader_instance
from backend.pipeline.query_cache import cache_instance
from backend.config import settings

router = APIRouter()

@router.get("/api/health", summary="Legacy health check with full artifact status")
def health_check(response: Response):
    is_valid = loader_instance.status.get("valid", False)
    # Return 200 as long as the process is alive. Artifact state is handled by /api/ready.
    if not is_valid:
        return {
            "status": "error",
            "artifacts": loader_instance.status,
            "cache": cache_instance.stats(),
            "saaras": {"enabled": settings.SAARAS_ENABLED},
            "slm": {
                "enabled": settings.SLM_ENABLED,
                "provider": settings.SLM_PROVIDER,
                "model": settings.SLM_MODEL,
            },
        }
        
    return {
        "status": "ok",
        "artifacts": {
            "manifest": loader_instance.status.get("manifest"),
            "bm25": loader_instance.status.get("bm25"),
            "hnsw": loader_instance.status.get("hnsw"),
            "metadata": loader_instance.status.get("metadata"),
            "onnx": loader_instance.status.get("onnx"),
            "validation_summary": loader_instance.status.get("validation_summary"),
            "checksums": loader_instance.status.get("checksums"),
        },
        "embedding_dimension": loader_instance.manifest_data.get("embedding_dimension", 384),
        "language": loader_instance.manifest_data.get("language", "hi"),
        "hnsw_space": loader_instance.manifest_data.get("hnsw_space", "cosine"),
        "cache": cache_instance.stats(),
        "saaras": {"enabled": settings.SAARAS_ENABLED},
        "slm": {
            "enabled": settings.SLM_ENABLED,
            "provider": settings.SLM_PROVIDER,
            "model": settings.SLM_MODEL,
        },
    }


# ── Phase 12 — Standard liveness and readiness probes ────────────────────────

@router.get("/health", summary="Liveness probe — always 200 while process is alive")
async def liveness():
    """Always returns 200 while the process is running."""
    return {"status": "ok"}


@router.get("/ready", summary="Readiness probe — 200 when all languages are loaded")
async def readiness():
    """
    Returns 200 if the retrieval service is fully initialized for all required languages.
    Returns 503 if initialization is still in progress or failed.
    """
    from backend.pipeline.retrieval_service import retrieval_service

    if retrieval_service.initialized:
        return {"status": "ready", "languages": ["hi", "bn", "en"]}

    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "reason": "Retrieval service not initialized"},
    )
