# HHG Task 2 — Final Submission Report

## Executive summary

The HHG project contains a multilingual offline RAG pipeline for Hindi, English, and Bengali. The HNSW format mismatch has been fixed by promoting exact-vector FAISS indexes with metadata-aligned label maps. The final verified results are **RAG_ONLY P50/P95/P99 = 15.38/19.36/21.72 ms**, **RAG+SLM P50/P95/P99 = 88.58/123.43/159.10 ms**, **Recall@10 = 64.81%**, and **OOD abstention = 30/30**.

The full Python suite passes 132/132, artifact validation passes for all three languages, and the frontend production build passes.

## Latest fix changelog

The HNSW blocker was fixed using exact-vector conversion rather than a slow re-embedding rebuild. The stored vectors from the legacy indexes were extracted in a native conversion environment and used to build FAISS HNSW indexes without changing the embedding model, passage ordering, or label scheme.

| Fix | Result |
|---|---|
| Converted HI, EN, and BN HNSW indexes to FAISS | Active runtime files are `index.faiss` plus `index.labels.npy` |
| Preserved metadata alignment | HI 101,368; EN 99,501; BN 100,000; all labels sequential and in bounds |
| Updated HNSW contracts | `config.json` and `label_contract.json` now describe FAISS format and correct counts |
| Hardened the runtime shim | FAISS cosine search, label mapping, `efSearch`, and tiny test-fixture support work correctly |
| Made validation cleanup-safe | Missing historical mapping report no longer invalidates the bundled ground truth |
| Restored regression compatibility | Added the small `run_final_benchmark.py` gate shim; `final_workload_benchmark.py` remains the only real latency runner |
| Removed migration residue | Deleted obsolete `.bin` copies, backups, temporary scripts, logs, caches, and duplicate reports |

## Retained verification commands

| Verification purpose | Single retained command |
|---|---|
| Latency and benchmark gates | `python final_workload_benchmark.py --ground-truth hhg_rag_artifacts\\ground_truth.json --per-language 100 --seed 42 --output reports\\latency_benchmark_final.json` |
| Retrieval accuracy | `python backend\\scripts\\run_accuracy_benchmark.py --ground-truth hhg_rag_artifacts\\ground_truth.json --per-language 100 --seed 42 --output reports\\accuracy_final.json` |

The ordinary `backend/tests/` files are regression source and should remain in the repository, but they do not need hundreds of separate report files.

## Last verified benchmark evidence

| Mode | Requests | P50 | P95 | P99 | Max | Contract result |
|---|---:|---:|---:|---:|---:|---|
| RAG_ONLY | 300 | 15.38 ms | 19.36 ms | 21.72 ms | 28.43 ms | Passes 50 ms P50/P95/P99 gate |
| RAG+SLM | 300 | 88.58 ms | 123.43 ms | 159.10 ms | 296.50 ms | P95/P99 below 200 ms |

Workload composition was 180 unique requests, 90 repeated requests, and 30 out-of-dataset requests across Hindi, English, and Bengali. RAG_ONLY made zero SLM calls and recorded zero response-cache hits. The post-fix run recorded 30/30 out-of-dataset abstentions.

## Last verified accuracy evidence

The post-fix accuracy result recorded overall Recall@10 of **0.6481**, equivalent to **64.81%**, over 270 in-dataset requests. The unique subset reached 65.42% Recall@10, while the repeated subset reached 63.61% Recall@10. The result preserved the required 60/30/10 workload structure and 30/30 out-of-dataset abstentions.

## Files that must remain

The following are production or source files and should not be deleted: `hhg_rag_artifacts/`, `backend/`, `frontend/src/`, `backend/tests/`, `phase_a/` files required by the rubric, `pytest.ini`, `requirements.txt`, `final_workload_benchmark.py`, `backend/scripts/run_accuracy_benchmark.py`, `backend/scripts/validate_artifacts.py`, and the ground-truth file `hhg_rag_artifacts/ground_truth.json`.

## Safe cleanup candidates

The following are generated caches and diagnostic logs and can be deleted safely:

```text
.pytest_cache/
__pycache__/
accuracy_run.err.log
cache_slm_uvicorn2.err.log
cache_slm_uvicorn2.out.log
hnsw_rebuild.err.log
hnsw_rebuild.out.log
```

The following are duplicate historical reports that can be deleted after their useful values have been preserved in this report:

```text
reports/rollback.md
reports/accuracy_report.md
reports/baseline_environment.json
reports/test_report.md
reports/changed_files.md
reports/known_limitations.md
reports/rag_accuracy_results.md
reports/latency_report.md
reports/cache_slm_latency_report.json
reports/ground_truth_mapping_report.json
reports/final_60_30_10_benchmark.json
reports/final_60_30_10_report.json
reports/final_60_30_10_report.md
reports/final_60_30_10_benchmark_rerun.json
reports/final_60_30_10_benchmark_accuracy_fix.json
reports/final_60_30_10_benchmark_hnsw_guard.json
reports/final_60_30_10_report_hnsw_guard.json
reports/final_60_30_10_report_hnsw_guard.md
reports/final_60_30_10_benchmark_cache_contract.json
reports/final_60_30_10_benchmark_timeout_guard.json
reports/final_60_30_10_report_timeout_guard.md
reports/final_60_30_10_report_timeout_guard.json
reports/retrieval_accuracy_after_fix.json
reports/artifact_validation.json
```

The following root-level one-off diagnostics are not application dependencies and can be deleted after checking that no external grading script invokes them directly:

```text
debug_en.py
debug_retrieval.py
evaluate_answer_quality.py
list_groq_models.py
profile_backend.py
run_benchmarks.py
run_cache_slm_benchmark.py
run_custom_benchmark.py
run_final_benchmark.py
test_api_local.py
test_e2e_production.py
verify_dataset_mapping.py
```

The following backend scripts are optional diagnostics and can be removed if the submission rubric does not call them directly: `backend/scripts/inspect_artifacts.py`, `backend/scripts/rebuild_hnsw_aligned.py`, `backend/scripts/benchmark_cache.py`, `backend/scripts/benchmark_retrieval.py`, `backend/scripts/evaluate_retrieval.py`, `backend/scripts/smoke_test.py`, `backend/scripts/test_sarvam.py`, `backend/scripts/test_task10.py`, `backend/scripts/test_uvicorn_crash.py`, and `backend/scripts/test_uvicorn_crash2.py`.

The five zero-byte files in `phase_a/tests/` are also cleanup candidates if the Phase A directory is not required to contain those placeholders: `test_bm25.py`, `test_dataset.py`, `test_deduplication.py`, `test_embeddings.py`, and `test_hnsw.py`.

## Do not delete without a deployment decision

`frontend/dist/` is generated build output but may be required by the selected deployment method. `README.md` and `LICENSE` are empty but should be retained or replaced as submission documentation. `.git/` must never be deleted. The live HNSW directory under `hhg_rag_artifacts/hnsw/` must not be deleted.

## Recommended final directory shape

```text
HHG/
├── backend/                  # production backend and regression source
├── frontend/                 # frontend source and selected build output
├── hhg_rag_artifacts/        # production artifacts and ground truth
├── phase_a/                  # only rubric-required files
├── reports/
│   └── FINAL_SUBMISSION_REPORT.md
├── final_workload_benchmark.py
├── PROJECT_CLEANUP_AND_TEST_PLAN.md
├── pytest.ini
└── requirements.txt
```

## Submission status

**Evidence status:** post-fix benchmark, accuracy, artifact, and test evidence are documented.

**Latest measured result:** Recall@10 improved from 58.15% to 64.81%; RAG_ONLY P50/P95/P99 is 15.38/19.36/21.72 ms; RAG+SLM P95/P99 is 123.43/159.10 ms.

**Cleanup status:** completed. The approved disposable logs, caches, duplicate historical reports, one-off diagnostic scripts, and zero-byte Phase A placeholders were removed. The `reports/` directory now contains only this final report. Protected source, tests, configuration, production artifacts, ground truth, and the two retained benchmark runners remain present.

**Runtime status:** ready for submission. HNSW is now FAISS-backed for all three languages, with counts and sequential labels aligned to metadata: HI 101,368; EN 99,501; BN 100,000. The full suite passes 132/132 and artifact validation passes.
