from pydantic import BaseModel, Field
from typing import List, Optional

class RetrievalResult(BaseModel):
    doc_id: str
    text: str
    score: float
    rank: int
    source: str
    language: Optional[str] = None

class LatencyBreakdown(BaseModel):
    stt_ms: float = 0.0
    language_detection_ms: float = 0.0
    tokenization_ms: float = 0.0
    embedding_ms: float = 0.0
    bm25_ms: float = 0.0
    hnsw_ms: float = 0.0
    rrf_ms: float = 0.0
    metadata_ms: float = 0.0
    grounding_ms: float = 0.0
    slm_ms: float = 0.0
    generation_ms: float = 0.0
    validation_ms: float = 0.0
    serialization_ms: float = 0.0

class LatencyMetrics(BaseModel):
    total_ms: float = 0.0
    partial_ms: float = 0.0
    rag_only_ms: float = 0.0
    breakdown: LatencyBreakdown = Field(default_factory=LatencyBreakdown)

class QueryResponse(BaseModel):
    query: str
    language: str
    answer: Optional[str] = Field(default=None, description="Generated answer from SLM")
    grounding: dict = Field(default={"enabled": False, "grounded": False, "sources": []}, description="Grounding metadata and citations")
    results: List[RetrievalResult] = Field(description="List of ranked retrieval results")
    cache: dict = Field(default={"hit": False}, description="Metadata indicating cache status")
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics, description="Latency metadata in milliseconds")
    transcription: Optional[dict] = Field(default=None, description="STT transcription metadata if voice was used")
    
    # Extended fields for strict Task 5 schema alignment
    sources: Optional[List[dict]] = Field(default=None, description="List of source citations")
    retrieval: Optional[dict] = Field(default=None, description="Retrieval source breakdown")
    metrics: Optional[dict] = Field(default=None, description="Key performance metrics")

class APIErrorDetail(BaseModel):
    code: str
    message: str

class ErrorResponse(BaseModel):
    error: APIErrorDetail
