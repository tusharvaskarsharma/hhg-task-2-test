from fastapi import APIRouter, Response, status
from backend.pipeline.retrieval_service import retrieval_service

router = APIRouter()

@router.get("/api/ready")
def readiness_check(response: Response):
    if retrieval_service.initialized:
        return {"status": "ready"}
    
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "not_ready", "reason": "Backend is running but RAG artifacts are not ready"}
