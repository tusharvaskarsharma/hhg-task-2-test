from pydantic import BaseModel, Field
from typing import Optional, Literal

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="The search query text. Must not be empty.")
    language: str = Field(default="hi", pattern="^(hi|en|bn)$", description="The language of the query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to retrieve")
