import logging
from typing import List, Tuple
from backend.schemas.response import RetrievalResult
from backend.config import settings

logger = logging.getLogger(__name__)

class GroundingService:
    def __init__(self):
        self.top_k = settings.GROUNDING_TOP_K
        self.max_chars = settings.MAX_CONTEXT_CHARS

    def build_context(self, results: List[RetrievalResult]) -> Tuple[str, List[dict]]:
        """
        Deterministically builds a context string and extracts source metadata.
        Returns (context_string, sources_list).
        """
        if not results:
            return "", []

        # Enforce top-K restriction
        grounding_results = results[:self.top_k]
        
        context_parts = []
        sources = []
        current_chars = 0
        
        for res in grounding_results:
            # Format passage snippet
            snippet = f"[Doc {res.rank} | Source: {res.source}]\n{res.text}"
            snippet_len = len(snippet)
            
            # Check character limit
            if current_chars + snippet_len > self.max_chars:
                if current_chars == 0:
                    # If even the first snippet is too large, truncate it to fit
                    snippet = snippet[:self.max_chars]
                    context_parts.append(snippet)
                    sources.append({
                        "doc_id": res.doc_id,
                        "rank": res.rank,
                        "source": res.source
                    })
                break
                
            context_parts.append(snippet)
            current_chars += snippet_len
            
            # Persist source metadata securely
            sources.append({
                "doc_id": res.doc_id,
                "rank": res.rank,
                "source": res.source
            })
            
        context_str = "\n\n".join(context_parts)
        return context_str, sources

grounding_service = GroundingService()
