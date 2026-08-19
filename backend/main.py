from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from backend.config import settings
from backend.artifact_loader import loader_instance
from backend.pipeline.retrieval_service import retrieval_service
from backend.routes import health, ready, query, voice

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting HHG Backend...")
    try:
        loader_instance.initialize()
        if loader_instance.status.get("valid"):
            retrieval_service.initialize()
            logger.info("Application Ready.")
        else:
            logger.error(f"Artifact initialization failed: {loader_instance.errors}")
    except Exception as e:
        logger.error(f"Failed during startup: {e}")
        
    yield
    # Shutdown
    logger.info("Shutting down HHG Backend...")

app = FastAPI(
    title="HHG Retrieval API",
    description="Backend API for multilingual HHG retrieval",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

app.include_router(health.router)
app.include_router(ready.router)
app.include_router(query.router)
app.include_router(voice.router)
