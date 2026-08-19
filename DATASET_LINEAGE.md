# Dataset Lineage

This document outlines the complete lineage of the production RAG artifacts stored in `hhg_rag_artifacts/`.

## 1. Source Data
**Dataset:** `ai4bharat/MSMARCO-XI`
**Split:** `train`
**Configuration:** `hi`, `bn`, `en`
**Selection Rule:** The first 10,000 eligible queries per native language (in source order), retaining all associated positive passages for these queries.

## 2. Preprocessing & Deduplication
- Extracted passages were deduplicated based on the composite key: `language + NFC-normalized text`.
- Casefolding was disabled.
- This resulted in approximately 300,869 unique passages across the 3 languages.

## 3. Chunking
- **Mode:** `auto_preserve_atomic`
- **Max Tokens Approx:** 128
- **Overlap Approx:** 16
- **Min Tokens Approx:** 8
- **Token Count Definition:** Whitespace approximation

## 4. Metadata Compilation
- Generated unique identifier `passage_id` for each chunk.
- Saved to `metadata/passage_metadata.parquet`.

## 5. Embeddings Generation
- **Model:** `intfloat/multilingual-e5-small`
- Exported model to INT8-quantized ONNX.
- Computed 384-dimensional dense vectors for all 300,869 chunks.

## 6. Indexing (HNSW & BM25)
- **HNSW:** Computed using `hnswlib` with Cosine distance.
- **BM25:** Computed using `rank_bm25` (BM25Okapi).

## 7. Evaluation Target
Because the production artifacts were strictly generated from the MSMARCO-XI corpus, **all accuracy benchmarks must be executed against MSMARCO-XI queries**. Running evaluations against alternate datasets (e.g. `miracl/miracl`) will result in mathematical failure (Recall = 0.0) as the target passages do not physically exist in the indexed database.
