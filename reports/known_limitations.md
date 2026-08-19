# HHG Known Limitations

Generated: 2026-08-19

## 1. RAG Accuracy Benchmark Dependency
The official accuracy evaluation (`run_accuracy_benchmark.py`) requires a precise `ground_truth.json` mapping production artifact IDs to queries. 
- **Impact:** Without this file, the benchmark cannot verify accuracy against the loaded `hhg_rag_artifacts` corpus and will output `NOT RUN`.
- **Status:** Mitigated by strict `NOT RUN` fallback.

## 2. TOTAL Latency Benchmark Blocked
The Sarvam STT API rate-limits aggressively (~15-20 requests/minute), causing `429 Too Many Requests` during the 100-query benchmark.
- **Impact:** We cannot prove the end-to-end (STT + RAG + SLM) latency is within SLA under load without throttling.
- **Status:** Mitigated by `--delay` parameter in `run_final_benchmark.py`.

## 3. Grounding Validation is Heuristic
The current grounding validation uses a simple word-overlap heuristic (`difflib` with a 0.3 threshold) for generation evaluation during extraction.
- **Impact:** It may falsely accept hallucinations that reuse words in different contexts.
- **Mitigation needed:** Implement an NLI (Natural Language Inference) model or an LLM-as-a-judge for robust real-time grounding validation.

## 4. Extractive Path is Rigid
The extractive fast-path uses deterministic sentence boundary detection and token overlap scoring.
- **Impact:** It may struggle with complex reasoning questions or queries where the answer requires synthesizing information across multiple non-adjacent sentences.
- **Mitigation needed:** Consider a lightweight extractive QA model (e.g., BERT-SQuAD style) for the fast path.
