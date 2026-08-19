# Benchmark Results

- **Timestamp:** 2026-08-19T21:21:21.579018
- **Number of Requests:** 100 per tier
- **Warmup Status:** Completed
- **Cache-Bypass Status:** UUIDs used
- **Cold-start Latency (SLM OFF):** 36.23 s
- **Cold-start Latency (SLM ON):** 26.64 s

## RAG_ONLY Statistics
| stage | avg | p50 | p95 | p99 | min | max |
|---|---|---|---|---|---|---|
| language detection | 0.02 | 0.02 | 0.03 | 0.05 | 0.01 | 0.10 |
| tokenization | 0.60 | 0.53 | 1.09 | 1.51 | 0.35 | 1.54 |
| embedding | 16.60 | 15.50 | 21.15 | 30.52 | 13.14 | 43.50 |
| BM25 | 1.28 | 1.25 | 1.57 | 1.84 | 0.91 | 1.87 |
| HNSW | 0.15 | 0.15 | 0.20 | 0.23 | 0.08 | 0.30 |
| RRF | 0.35 | 0.30 | 0.46 | 1.43 | 0.16 | 1.55 |
| metadata lookup | 0.50 | 0.48 | 0.67 | 0.98 | 0.34 | 0.98 |
| grounding | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| serialization | 0.18 | 0.11 | 0.51 | 1.42 | 0.07 | 1.69 |
| total RAG_ONLY | 19.79 | 18.65 | 25.10 | 34.44 | 16.12 | 46.74 |

## PARTIAL Statistics
| stage | avg | p50 | p95 | p99 | min | max |
|---|---|---|---|---|---|---|
| language detection | 0.02 | 0.02 | 0.05 | 0.06 | 0.01 | 0.07 |
| tokenization | 0.47 | 0.41 | 0.92 | 1.63 | 0.33 | 1.79 |
| embedding | 13.58 | 13.38 | 16.07 | 17.19 | 11.20 | 17.56 |
| BM25 | 1.09 | 1.07 | 1.36 | 1.55 | 0.80 | 1.66 |
| HNSW | 0.14 | 0.12 | 0.17 | 0.51 | 0.08 | 0.80 |
| RRF | 0.30 | 0.25 | 0.45 | 1.12 | 0.19 | 1.17 |
| metadata lookup | 0.42 | 0.41 | 0.54 | 0.59 | 0.32 | 0.96 |
| grounding | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.03 |
| serialization | 0.12 | 0.10 | 0.15 | 0.20 | 0.08 | 0.92 |
| SLM/generation | 374.91 | 238.10 | 709.59 | 2888.24 | 189.46 | 3273.62 |
| validation | 0.13 | 0.12 | 0.17 | 0.21 | 0.09 | 0.25 |
| total PARTIAL | 391.34 | 254.04 | 725.12 | 2904.22 | 208.14 | 3292.11 |

## TOTAL Statistics
| stage | avg | p50 | p95 | p99 | min | max |
|---|---|---|---|---|---|---|
| STT | 1616.21 | 1515.03 | 2666.48 | 3814.31 | 908.99 | 4685.75 |
| language detection | 0.04 | 0.03 | 0.09 | 0.15 | 0.01 | 0.15 |
| tokenization | 0.83 | 0.66 | 2.14 | 3.30 | 0.32 | 3.36 |
| embedding | 22.26 | 19.39 | 43.19 | 79.33 | 7.78 | 137.38 |
| BM25 | 2.31 | 1.87 | 4.87 | 10.50 | 0.84 | 22.17 |
| HNSW | 0.23 | 0.18 | 0.31 | 1.57 | 0.08 | 1.64 |
| RRF | 2.54 | 0.39 | 1.52 | 4.94 | 0.17 | 203.59 |
| metadata lookup | 2.44 | 1.48 | 6.43 | 18.72 | 0.76 | 31.10 |
| grounding | 0.04 | 0.03 | 0.08 | 0.15 | 0.01 | 0.55 |
| serialization | 0.37 | 0.23 | 0.63 | 1.60 | 0.11 | 10.08 |
| SLM/generation | 375.78 | 251.83 | 911.01 | 1597.35 | 186.21 | 3851.13 |
| validation | 0.88 | 0.46 | 1.19 | 4.46 | 0.20 | 29.91 |
| total TOTAL | 2024.27 | 1833.20 | 3621.97 | 5060.52 | 1157.78 | 5481.53 |

## Final Verdict
**RAG_ONLY Target (<=50ms):** PASS
**Logic Check:** TOTAL >= PARTIAL >= RAG_ONLY: PASS
