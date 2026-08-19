"""
backend/pipeline/extractive.py
Deterministic extractive fast-path answer builder.

This module produces grounded answers directly from retrieved passages
WITHOUT calling any external SLM or LLM. It is the primary answer source
for RAG_ONLY mode (generate=False).
"""
from __future__ import annotations

import re
import unicodedata
from enum import Enum
from typing import List, Tuple

from backend.schemas.response import RetrievalResult


class ExtractiveDecision(str, Enum):
    SUPPORTED = "SUPPORTED"
    ABSTAIN = "ABSTAIN"          # Not enough context to answer
    INSUFFICIENT = "INSUFFICIENT"  # Context found but no confident span


from backend.config import settings

# Stop-words to ignore when scoring sentence relevance (EN + HI + BN function words)
_STOP_WORDS = frozenset({
    # EN
    "the", "is", "at", "which", "on", "in", "a", "an", "and", "or", "of",
    "to", "for", "with", "it", "this", "that", "was", "are", "be", "been",
    "by", "from", "as", "but", "not", "so", "its", "their", "has", "have",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    # HI transliterated common function words
    "hai", "hain", "ka", "ki", "ke", "ko", "se", "mein", "par", "aur",
    "ek", "yeh", "jo", "toh", "bhi", "ne",
    # BN transliterated
    "hoy", "ebong", "kintu", "ba", "ei", "oi",
})

_QUESTION_WORDS = frozenset({
    "who", "what", "where", "when", "why", "how", "whose", "whom", "which",
    "kya", "kaun", "kahan", "kab", "kyon", "kaise", "kis", "kisne", "kisko",
    "ke", "ki", "ka"
})

_GENERIC_WORDS = frozenset({
    "government", "minister", "country", "president", "state", "city", "people",
    "name", "year", "date", "time", "day", "month", "person", "place", "prime",
    "world", "nation", "leader"
})

_MIN_SENTENCE_LEN = 15      # characters — skip very short sentences
_MAX_SENTENCES = 3          # max sentences to concatenate


def _normalize(text: str) -> str:
    """Unicode NFC normalization + lowercase."""
    return unicodedata.normalize("NFC", text).lower()


def _tokenize(text: str, ignore_question_words: bool = True) -> set[str]:
    """Simple whitespace+punctuation tokenizer."""
    text = re.sub(r'[^\w\s]', ' ', _normalize(text))
    tokens = re.findall(r"\b\w+\b", text)
    filtered = set()
    for t in tokens:
        if t in _STOP_WORDS:
            continue
        if ignore_question_words and t in _QUESTION_WORDS:
            continue
        if len(t) > 1:
            filtered.add(t)
    return filtered


def _split_sentences(text: str) -> List[str]:
    """Split text on sentence boundaries (handles Hindi/Bengali too)."""
    # Split on ., !, ?, ।, ॥, ।। and newlines
    sentences = re.split(r"(?<=[.!?।॥\n])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) >= _MIN_SENTENCE_LEN]


def _score_sentence(sentence: str, query_tokens: set[str]) -> Tuple[float, bool]:
    """
    Score a sentence by fraction of query tokens it contains.
    Returns: (coverage, passes_generic_check)
    """
    if not query_tokens:
        return 0.0, False
    sent_tokens = _tokenize(sentence, ignore_question_words=False)
    overlap = query_tokens & sent_tokens
    
    if not overlap:
        return 0.0, False
        
    coverage = len(overlap) / len(query_tokens)
    
    # 8. Reject candidates with only generic overlap such as government, minister, country, or president.
    only_generic = all(t in _GENERIC_WORDS for t in overlap)
    
    # 4. Require the candidate sentence to contain the core subject terms.
    # Find non-generic (core) terms in the query. If there are core terms,
    # the overlap MUST contain at least one core term.
    core_query_terms = query_tokens - _GENERIC_WORDS
    if core_query_terms:
        if not (overlap & core_query_terms):
            return 0.0, False
            
    return coverage, not only_generic


def build_extractive_answer(
    query: str,
    retrieval_results: List[RetrievalResult],
) -> Tuple[str, str, List[dict]]:
    """
    Build a grounded extractive answer from retrieved passages.

    Returns:
        (answer_text, decision, source_ids)
        decision is one of ExtractiveDecision values as a string.
    """
    if not retrieval_results:
        return (
            "The available information is insufficient to answer this question.",
            ExtractiveDecision.ABSTAIN,
            [],
        )

    query_tokens = _tokenize(query)
    if not query_tokens:
        # Very short / non-content query (greetings, etc.)
        return (
            "Please provide a more specific question.",
            ExtractiveDecision.ABSTAIN,
            [],
        )

    # Collect (score, sentence, source_id) tuples across all results
    candidates: List[Tuple[float, str, str]] = []
    
    min_coverage = settings.EXTRACTIVE_MIN_QUERY_COVERAGE
    min_score = settings.EXTRACTIVE_MIN_SCORE
    
    for result in retrieval_results:
        text = result.text if hasattr(result, "text") else result.get("text", "")
        source_id = result.id if hasattr(result, "id") else result.get("id", "")
        rrf_score = result.score if hasattr(result, "score") else result.get("score", 0.0)

        for sentence in _split_sentences(text):
            if len(sentence) > settings.EXTRACTIVE_MAX_SENTENCE_CHARS:
                continue
                
            coverage_score, passes_generic = _score_sentence(sentence, query_tokens)
            
            # Combine coverage score with RRF rank score
            combined = coverage_score * 0.7 + min(rrf_score, 1.0) * 0.3
            
            if passes_generic and coverage_score >= min_coverage and combined >= min_score:
                candidates.append((combined, sentence, str(source_id)))

    if not candidates:
        return (
            "The available information is insufficient to answer this question.",
            ExtractiveDecision.INSUFFICIENT,
            [],
        )

    # Sort by combined score descending, pick top sentences
    candidates.sort(key=lambda x: x[0], reverse=True)

    selected_sentences: List[str] = []
    selected_sources: List[dict] = []
    seen_sentences: set[str] = set()
    total_chars = 0

    for score, sentence, source_id in candidates:
        norm_sent = _normalize(sentence)
        if norm_sent in seen_sentences:
            continue
        if total_chars + len(sentence) > settings.EXTRACTIVE_MAX_SENTENCE_CHARS:
            break
        selected_sentences.append(sentence)
        seen_sentences.add(norm_sent)
        
        # Build dict for sources array to match schema
        if not any(s.get("doc_id") == source_id for s in selected_sources):
            # Find the original result to grab rank and source
            orig_result = next((r for r in retrieval_results if (r.id if hasattr(r, "id") else r.get("id")) == source_id), None)
            if orig_result:
                rank = orig_result.rank if hasattr(orig_result, "rank") else orig_result.get("rank", 0)
                src = orig_result.source if hasattr(orig_result, "source") else orig_result.get("source", "")
                selected_sources.append({"doc_id": source_id, "rank": rank, "source": src})
            else:
                selected_sources.append({"doc_id": source_id, "rank": 0, "source": ""})
                
        total_chars += len(sentence)
        if len(selected_sentences) >= _MAX_SENTENCES:
            break

    if not selected_sentences:
        return (
            "The available information is insufficient to answer this question.",
            ExtractiveDecision.INSUFFICIENT,
            [],
        )

    answer = " ".join(selected_sentences)
    return answer, ExtractiveDecision.SUPPORTED, selected_sources
