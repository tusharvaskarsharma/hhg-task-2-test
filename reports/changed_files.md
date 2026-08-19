# HHG Changed Files Report

Generated: 2026-08-19 (commit: b5dad0db5d577f66b874cf8d88a0ccd57565de9f → HEAD)

## Files Modified

| File | Reason |
|------|--------|
| `backend/config.py` | Phase 2: Added `parse_bool()`, fixed `HHG_HNSW_EF_SEARCH` prefix, added SLM defaults |
| `backend/schemas/query.py` | Phase 5: Added `generate: bool = False` field — makes SLM opt-in |
| `backend/schemas/response.py` | Phase 5/6: Added `answer_source` field to `QueryResponse`, `extractive_ms` to `LatencyBreakdown`, `id` property to `RetrievalResult` |
| `backend/pipeline/generator.py` | Phase 6: Added `GroundingDecision` enum, fixed `_validate_answer` return type, added `answer_source` field |
| `backend/routes/query.py` | Phase 5: Rewrote to use extractive fast path by default; SLM only called when `generate=True` |
| `backend/routes/health.py` | Phase 12: Added `/health` (liveness) and `/ready` (readiness) endpoints |
| `backend/scripts/validate_artifacts.py` | Phase 3/12: Deep artifact validation for metadata uniqueness, HNSW counts, BM25 counts, missing manifests. |
| `backend/scripts/run_accuracy_benchmark.py` | Phase 4: Strict accuracy benchmark falling back gracefully when `ground_truth.json` is missing. |
| `run_final_benchmark.py` | Phase 5: Added latency gates (P50/P95/P99 <= 50ms) to fail tests if violated. |
| `backend/.env.example` | Phase 2: Added strict Groq llama-3.1-8b-instant constants. |
| `backend/tests/test_artifacts.py` | Phase 3: Added tests for validation scripts missing manifests and duplicate IDs. |
| `backend/tests/test_rag_only_latency_contract.py` | Phase 1: Fixed incorrect response JSON parsing, enforced 0 mock calls to generators. |

## Files Created

| File | Reason |
|------|--------|
| `backend/pipeline/extractive.py` | Phase 5: New deterministic extractive fast-path module (no SLM) |
| `backend/tests/test_extractive.py` | Phase 5: Tests for extractive module |
| `backend/tests/test_health.py` | Phase 12: Tests for /health and /ready endpoints |
| `backend/tests/test_benchmark_gates.py` | Phase 5: Ensure strict benchmark gating. |
| `backend/tests/test_slm_client_payload.py` | Phase 2: Ensure strictly compliant payload (temp/tokens/model) sent to SLM endpoint. |
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
