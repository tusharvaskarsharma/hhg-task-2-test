
RAG_ONLY
slm_enabled=false
saaras_enabled=false
cache_enabled=false
requests=300
warmups=100
language=en
p50_ms=14.91
p95_ms=17.98
p99_ms=23.58
slm_calls=0
status=PASS

| stage | avg | p50 | p95 | p99 | min | max |
|---|---|---|---|---|---|---|
| language detection | 0.02 | 0.01 | 0.02 | 0.04 | 0.01 | 0.05 |
| tokenization | 0.43 | 0.42 | 0.58 | 0.65 | 0.29 | 0.83 |
| embedding | 13.20 | 12.74 | 15.93 | 21.64 | 9.38 | 61.69 |
| BM25 | 0.95 | 0.94 | 1.23 | 1.37 | 0.68 | 1.52 |
| HNSW | 0.09 | 0.09 | 0.12 | 0.14 | 0.06 | 0.27 |
| RRF | 0.22 | 0.16 | 0.25 | 1.33 | 0.10 | 1.58 |
| metadata lookup | 0.04 | 0.03 | 0.05 | 0.05 | 0.02 | 0.05 |
| extractive | 0.15 | 0.15 | 0.19 | 0.21 | 0.10 | 0.42 |
| serialization | 0.13 | 0.12 | 0.17 | 0.20 | 0.09 | 0.27 |
| total_rag-only | 15.30 | 14.91 | 17.98 | 23.58 | 11.53 | 64.81 |

RAG_ONLY
slm_enabled=false
saaras_enabled=false
cache_enabled=false
requests=300
warmups=100
language=hi
p50_ms=14.27
p95_ms=17.80
p99_ms=31.44
slm_calls=0
status=PASS

| stage | avg | p50 | p95 | p99 | min | max |
|---|---|---|---|---|---|---|
| language detection | 0.02 | 0.01 | 0.02 | 0.04 | 0.01 | 0.06 |
| tokenization | 0.44 | 0.42 | 0.59 | 0.66 | 0.31 | 0.98 |
| embedding | 12.78 | 12.21 | 15.62 | 29.50 | 9.80 | 38.67 |
| BM25 | 0.92 | 0.91 | 1.21 | 1.38 | 0.65 | 1.58 |
| HNSW | 0.09 | 0.08 | 0.12 | 0.15 | 0.05 | 0.21 |
| RRF | 0.17 | 0.12 | 0.20 | 1.22 | 0.08 | 1.81 |
| metadata lookup | 0.03 | 0.03 | 0.05 | 0.06 | 0.02 | 0.08 |
| extractive | 0.23 | 0.23 | 0.30 | 0.32 | 0.15 | 0.44 |
| serialization | 0.13 | 0.13 | 0.18 | 0.20 | 0.09 | 0.34 |
| total_rag-only | 14.87 | 14.27 | 17.80 | 31.44 | 11.81 | 41.63 |

RAG_ONLY
slm_enabled=false
saaras_enabled=false
cache_enabled=false
requests=300
warmups=100
language=bn
p50_ms=14.54
p95_ms=16.78
p99_ms=19.32
slm_calls=0
status=PASS

| stage | avg | p50 | p95 | p99 | min | max |
|---|---|---|---|---|---|---|
| language detection | 0.02 | 0.01 | 0.02 | 0.04 | 0.01 | 0.04 |
| tokenization | 0.48 | 0.47 | 0.62 | 0.72 | 0.34 | 0.76 |
| embedding | 12.59 | 12.43 | 14.54 | 16.44 | 9.42 | 28.19 |
| BM25 | 0.91 | 0.91 | 1.13 | 1.34 | 0.65 | 1.55 |
| HNSW | 0.09 | 0.08 | 0.12 | 0.13 | 0.05 | 0.15 |
| RRF | 0.16 | 0.11 | 0.16 | 1.49 | 0.07 | 1.66 |
| metadata lookup | 0.03 | 0.03 | 0.05 | 0.06 | 0.02 | 0.10 |
| extractive | 0.27 | 0.27 | 0.34 | 0.37 | 0.16 | 0.76 |
| serialization | 0.13 | 0.13 | 0.18 | 0.22 | 0.09 | 0.25 |
| total_rag-only | 14.74 | 14.54 | 16.78 | 19.32 | 11.68 | 30.52 |
