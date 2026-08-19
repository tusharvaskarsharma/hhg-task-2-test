# HHG Latency Report

## Environment
- **Platform:** Windows-11-10.0.26200-SP0
- **Python:** 3.13.14
- **Commit:** b5dad0db5d577f66b874cf8d88a0ccd57565de9f
- **Artifact manifest SHA256:** a7a071b88f6f6800
- **Benchmark date:** 2026-08-19T21:02:28.950406Z

## RAG_ONLY Tier
Configuration: `slm_enabled=False`, `saaras_enabled=False`

| stage | avg | p50 | p95 | p99 | min | max |
|---|---|---|---|---|---|---|
| language detection | 0.01 | 0.01 | 0.02 | 0.02 | 0.01 | 0.02 |
| tokenization | 0.51 | 0.52 | 0.61 | 0.62 | 0.39 | 0.63 |
| embedding | 12.38 | 11.47 | 16.60 | 17.93 | 10.82 | 18.27 |
| BM25 | 0.94 | 0.92 | 1.15 | 1.22 | 0.75 | 1.24 |
| HNSW | 0.12 | 0.11 | 0.15 | 0.16 | 0.11 | 0.16 |
| RRF | 0.24 | 0.23 | 0.27 | 0.27 | 0.21 | 0.27 |
| metadata lookup | 0.25 | 0.24 | 0.29 | 0.30 | 0.23 | 0.30 |
| extractive | 0.14 | 0.13 | 0.16 | 0.16 | 0.10 | 0.16 |
| serialization | 0.12 | 0.12 | 0.15 | 0.16 | 0.09 | 0.16 |
| total_rag-only | 14.81 | 13.75 | 19.36 | 20.58 | 13.10 | 20.88 |

## PARTIAL Tier
Configuration: `slm_enabled=True`, `saaras_enabled=False`

| stage | avg | p50 | p95 | p99 | min | max |
|---|---|---|---|---|---|---|
| language detection | 0.02 | 0.02 | 0.03 | 0.03 | 0.02 | 0.03 |
| tokenization | 0.65 | 0.67 | 0.76 | 0.77 | 0.52 | 0.78 |
| embedding | 12.28 | 11.76 | 13.46 | 13.61 | 11.45 | 13.65 |
| BM25 | 0.99 | 0.99 | 1.01 | 1.01 | 0.96 | 1.01 |
| HNSW | 0.12 | 0.12 | 0.13 | 0.13 | 0.11 | 0.13 |
| RRF | 0.23 | 0.24 | 0.25 | 0.25 | 0.21 | 0.25 |
| metadata lookup | 0.25 | 0.23 | 0.30 | 0.30 | 0.22 | 0.30 |
| extractive | 0.23 | 0.24 | 0.26 | 0.26 | 0.18 | 0.26 |
| serialization | 0.12 | 0.12 | 0.13 | 0.13 | 0.10 | 0.13 |
| grounding | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| generation | 3525.25 | 1901.98 | 7405.45 | 7894.65 | 656.83 | 8016.95 |
| validation | 3.05 | 2.97 | 3.26 | 3.28 | 2.90 | 3.29 |
| total_partial | 3543.37 | 1919.33 | 7424.85 | 7914.23 | 674.20 | 8036.58 |

## TOTAL Tier
Configuration: `slm_enabled=True`, `saaras_enabled=True`

| stage | avg | p50 | p95 | p99 | min | max |
|---|---|---|---|---|---|---|
| language detection | 0.07 | 0.07 | 0.09 | 0.09 | 0.06 | 0.09 |
| tokenization | 0.40 | 0.40 | 0.49 | 0.50 | 0.30 | 0.50 |
| embedding | 9.84 | 9.84 | 10.52 | 10.58 | 9.08 | 10.60 |
| BM25 | 1.23 | 1.23 | 1.38 | 1.40 | 1.07 | 1.40 |
| HNSW | 0.11 | 0.11 | 0.11 | 0.11 | 0.11 | 0.11 |
| RRF | 0.25 | 0.25 | 0.27 | 0.27 | 0.23 | 0.27 |
| metadata lookup | 0.63 | 0.63 | 0.74 | 0.75 | 0.51 | 0.75 |
| extractive | 0.76 | 0.76 | 0.87 | 0.88 | 0.65 | 0.88 |
| serialization | 0.21 | 0.21 | 0.25 | 0.26 | 0.15 | 0.26 |
| grounding | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| generation | 8595.12 | 8595.12 | 9035.69 | 9074.85 | 8105.61 | 9084.64 |
| validation | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| STT | 798.83 | 798.83 | 825.25 | 827.59 | 769.47 | 828.18 |
| total_total | 9407.67 | 9407.67 | 9822.74 | 9859.64 | 8946.49 | 9868.86 |
