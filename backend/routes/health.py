from fastapi import APIRouter, Response, status
from backend.artifact_loader import loader_instance
from backend.pipeline.query_cache import cache_instance
from backend.config import settings

router = APIRouter()

@router.get("/api/health")
def health_check(response: Response):
    is_valid = loader_instance.status.get("valid", False)
    if not is_valid:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "error",
            "artifacts": loader_instance.status,
            "cache": cache_instance.stats(),
            "saaras": {"enabled": settings.SAARAS_ENABLED},
            "slm": {
                "enabled": settings.SLM_ENABLED,
                "provider": settings.SLM_PROVIDER,
                "model": settings.SLM_MODEL
            }
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
            "checksums": loader_instance.status.get("checksums")
        },
        "embedding_dimension": loader_instance.manifest_data.get("embedding_dimension", 384),
        "language": loader_instance.manifest_data.get("language", "hi"),
        "hnsw_space": loader_instance.manifest_data.get("hnsw_space", "cosine"),
        "cache": cache_instance.stats(),
        "saaras": {"enabled": settings.SAARAS_ENABLED},
        "slm": {
            "enabled": settings.SLM_ENABLED,
            "provider": settings.SLM_PROVIDER,
            "model": settings.SLM_MODEL
        }
    }
