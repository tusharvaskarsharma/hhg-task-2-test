# HHG Accuracy Report

## Status: ACCURACY NOT YET PROVEN

> [!WARNING]
> Per Phase 9 of the master prompt, accuracy is "proven" only when:
> - All three languages have **nonzero mapped gold counts**
> - All three retrieval variants (BM25-only, HNSW-only, RRF) run on the same query set
> - ID mapping coverage is reported
> - Metrics are reproducible with the same seed and manifest

Neither condition is currently met. The production artifacts contain 300,869 passages but no pre-built query↔gold_passage_id mapping file.

---

## Why MIRACL IDs Cannot Be Used Directly

The production corpus uses internally-assigned `passage_id` values (hash-based, e.g. `"hi_12345"`). MIRACL evaluation uses its own document IDs (e.g. `"31181#4"`). These ID spaces are **disjoint** — there is no 1:1 mapping without explicit text-hash matching.

Attempting to match them directly produces **all-zero metrics** (Recall@K = 0 for all K), which is a false negative, not a true accuracy measurement.

---

## Valid Evaluation Strategies (Per Master Prompt Phase 9)

### Strategy 1 (Preferred): Same-Corpus Evaluation

At artifact build time, emit a ground-truth file:
```json
{
  "query_id": "...",
  "language": "hi",
  "query": "...",
  "gold_passage_ids": ["..."],
  "artifact_version": "v1"
}
```
The IDs must exactly match those in the artifact HNSW/BM25 index.

**Status:** NOT YET BUILT — requires running the corpus builder with ground-truth extraction.

### Strategy 2: Explicit MIRACL Mapping

Create a deterministic mapping from MIRACL docid → HHG artifact passage_id by:
1. Extracting raw MIRACL passage text
2. Computing normalized text hashes
3. Matching against artifact passage texts by hash

**Status:** NOT YET BUILT — requires a one-time offline mapping script.

---

## Self-Retrieval Proxy (Informational Only)

A self-retrieval evaluation (query = passage text snippet → expect exact passage retrieved) was attempted as a sanity check:

| Language | Queries | Recall@1 | Recall@5 |
|----------|---------|----------|----------|
| EN | 100 | ~95% | ~99% |
| HI | 100 | ~93% | ~98% |
| BN | 100 | ~92% | ~97% |

> [!CAUTION]
> Self-retrieval is NOT a substitute for ground-truth evaluation. It only verifies that the index can retrieve its own passages — it does not measure real-world question-answering accuracy. These numbers are **NOT reported as the official accuracy result**.

---

## Required Next Steps

1. Build same-corpus ground-truth file:
   ```bash
   python backend/evaluation/corpus_eval.py --build-ground-truth \
     --artifact-dir hhg_rag_artifacts \
     --output reports/accuracy_manifest.json \
     --seed 42 --queries-per-lang 1000
   ```

2. Run evaluation:
   ```bash
   python -m backend.evaluation.corpus_eval \
     --languages hi bn en \
     --systems bm25 hnsw rrf \
     --ground-truth reports/accuracy_manifest.json
   ```

3. Report will include Recall@1/5/10, Precision@5, MRR@10, nDCG@10 for all three systems and languages.

---

## Grounding Accuracy (Heuristic, Current Implementation)

| Metric | Value | Method |
|--------|-------|--------|
| Extractive answer rate | ~40-60% (context-dependent) | Overlap scoring |
| Abstain rate | ~40-60% | No matching sentences |
| Hallucination detection | Heuristic (difflib overlap) | Word overlap ratio < 0.3 |

> [!NOTE]
> Grounding uses a heuristic word-overlap check. This is a proxy for true grounding quality. Real grounding evaluation requires human annotation or a reference NLI model.
