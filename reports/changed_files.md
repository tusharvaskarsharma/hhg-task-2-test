# HHG Changed Files Report

Generated: 2026-08-19 (commit: 8332784 → HEAD)

## Files Modified

| File | Reason |
|------|--------|
| `backend/config.py` | Phase 2: Added `parse_bool()`, fixed `HHG_HNSW_EF_SEARCH` prefix, added `ONNX_INTRA_THREADS`/`ONNX_INTER_THREADS` |
| `backend/schemas/query.py` | Phase 5: Added `generate: bool = False` field — makes SLM opt-in |
| `backend/schemas/response.py` | Phase 5/6: Added `answer_source` field to `QueryResponse`, `extractive_ms` to `LatencyBreakdown`, `id` property to `RetrievalResult` |
| `backend/pipeline/generator.py` | Phase 6: Added `GroundingDecision` enum, fixed `_validate_answer` return type, added `answer_source` field |
| `backend/routes/query.py` | Phase 5: Rewrote to use extractive fast path by default; SLM only called when `generate=True` |
| `backend/routes/health.py` | Phase 12: Added `/health` (liveness) and `/ready` (readiness) endpoints |
| `backend/scripts/validate_artifacts.py` | Phase 12: Rewritten to output machine-readable JSON, per-language checks, `--json` flag |
| `backend/.env` | Fix: Updated `HHG_SLM_MODEL` from decommissioned `mixtral-8x7b-32768` to `qwen/qwen3.6-27b` |

## Files Created

| File | Reason |
|------|--------|
| `backend/pipeline/extractive.py` | Phase 5: New deterministic extractive fast-path module (no SLM) |
| `backend/tests/test_extractive.py` | Phase 5: Tests for extractive module |
| `backend/tests/test_health.py` | Phase 12: Tests for /health and /ready endpoints |
| `reports/baseline_environment.json` | Phase 0 deliverable |
| `reports/artifact_validation.json` | Phase 12 deliverable |
| `reports/latency_report.md` | Phase 10 deliverable |
| `reports/accuracy_report.md` | Phase 9 deliverable |
| `reports/changed_files.md` | This file |
| `reports/rollback.md` | Rollback instructions |
| `reports/known_limitations.md` | Known limitations |

## Files NOT Changed (intentionally preserved)

| File | Reason |
|------|--------|
| `backend/artifact_loader.py` | Already implements strict per-language loading with manifest validation |
| `backend/pipeline/retrieval_service.py` | Already implements per-stage timing and metadata dict |
| `backend/pipeline/embedder.py` | ONNX int8 embedder working correctly (P50=17ms) |
| `backend/pipeline/slm_client.py` | Working with retry/backoff logic |
| `backend/pipeline/grounding.py` | Functional grounding context builder |
| All test files (except new ones) | 61/61 tests pass — preserved |
| All production artifacts | Never modified (HNSW, BM25, metadata, ONNX) |
