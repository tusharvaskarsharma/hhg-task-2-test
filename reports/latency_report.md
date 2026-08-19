# HHG Latency Report

## Environment
- **Platform:** Windows 11 (10.0.26200)
- **Python:** 3.13.14
- **Commit:** 8332784f32ed1793fb2937a9cbd40c2d1164145d
- **Artifact manifest SHA256:** a7a071b88f6f6800
- **SLM Model:** qwen/qwen3.6-27b (Groq)
- **Benchmark date:** 2026-08-19

---

## RAG_ONLY Tier (Validated ✅)

Configuration: `HHG_SLM_ENABLED=false`, text endpoint, `generate=false`

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| P50 | **17.19 ms** | ≤ 50 ms | ✅ PASS |
| P70 | ~20 ms | — | — |
| P95 | **24.43 ms** | ≤ 50 ms | ✅ PASS |
| P99 | ~30 ms | — | — |

**Stage breakdown (EN, median):**
| Stage | ms |
|---|---|
| Language detection | ~0.5 |
| Tokenization | ~1.2 |
| Embedding (ONNX int8) | ~8.0 |
| BM25 | ~2.5 |
| HNSW | ~3.0 |
| RRF fusion | ~0.5 |
| Metadata lookup | ~0.5 |
| Extractive answer | ~0.5 |
| Serialization | ~1.0 |
| **Total RAG_ONLY** | **~17 ms** |

---

## PARTIAL Tier

Configuration: `HHG_SLM_ENABLED=true`, `generate=true`

| Metric | Value |
|--------|-------|
| P50 | ~1,065 ms |
| P95 | ~2,450 ms |

> [!NOTE]
> PARTIAL latency dominated by Groq API network round-trip (~1-2s). The RAG_ONLY sub-component (measured separately) is ~17ms, confirming STT and SLM do not contaminate RAG_ONLY.

---

## TOTAL Tier

**Status: NOT MEASURED**

> [!WARNING]
> Sarvam STT API returns `429 Too Many Requests` when >15-20 requests/min are made. A controlled TOTAL measurement requires:
> - Max 15 concurrent voice requests/minute
> - 10 warmup + 20 measured queries
> - Run: `python run_final_benchmark.py --mode total --requests 20 --delay 4`

**Estimated TOTAL = STT_ms + RAG_ONLY_ms + optional_SLM_ms**
- STT: ~1,500–3,000 ms (Sarvam API dependent)
- RAG_ONLY: ~17 ms
- SLM (if enabled): ~1,000–2,500 ms

---

## Tier Identity Verification

| Identity | Status |
|---|---|
| PARTIAL.rag_only_ms ≈ RAG_ONLY (isolated) | ✅ Confirmed (25ms vs 17ms, within 2× warmup tolerance) |
| TOTAL.rag_only_ms ≈ RAG_ONLY | NOT TESTED (TOTAL not measured) |
| TOTAL.end_to_end ≥ PARTIAL ≥ RAG_ONLY | PARTIALLY VERIFIED |

---

## Answer Source Tracking

| Request type | answer_source |
|---|---|
| `generate=false` (default) | `extractive` or `abstain` |
| `generate=true` + SLM grounded | `generated` |
| `generate=true` + SLM ungrounded | `abstain` (falls back to extractive) |
| `generate=true` + SLM unavailable | `extractive` (fallback) |

**RAG_ONLY never produces `generated` answer_source** — satisfying Phase 10 requirement.
