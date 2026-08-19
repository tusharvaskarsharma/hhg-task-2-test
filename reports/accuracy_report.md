# HHG Accuracy Report

## Status: NOT RUN

> [!WARNING]
> Per Phase 9 of the master prompt: "If this cannot be completed because the source query/gold data is missing, write NOT RUN with the exact missing file and do not invent metrics."

**Missing File:** `hhg_rag_artifacts/ground_truth.json` or equivalent source query/gold data.

The production artifacts contain 300,869 passages, but there is no mapping file provided that associates benchmark queries with these exact `passage_id` values in the artifact.

### Why MIRACL IDs Cannot Be Used Directly
The production corpus uses internally-assigned `passage_id` values (hash-based, e.g. `"hi_12345"`). MIRACL evaluation uses its own document IDs (e.g. `"31181#4"`). These ID spaces are disjoint — there is no 1:1 mapping without an explicit text-hash matching offline step, which has not been provided.

Because the data is missing, the accuracy metrics cannot be reproduced and are therefore marked **NOT RUN**.

---

### Required Next Steps for Proven Accuracy
1. Build same-corpus ground-truth file mapping MIRACL queries to artifact passage IDs.
2. Run evaluation on `bm25`, `hnsw`, and `rrf` against this mapped dataset.
3. Report Recall@1, Recall@5, Recall@10, Precision@5, MRR@10, nDCG@10, and ID mapping coverage.
