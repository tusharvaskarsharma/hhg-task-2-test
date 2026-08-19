# HHG Known Limitations

Generated: 2026-08-19

## 1. RAG Accuracy is NOT Proven
The system currently relies on self-retrieval (querying the exact text of a passage) for validation. A true evaluation requires independent queries mapped to gold-standard passages from the same corpus.
- **Impact:** We cannot guarantee the system retrieves the best answer for real-world questions.
- **Mitigation needed:** Generate a same-corpus ground-truth dataset (`query` -> `gold_passage_ids`) and evaluate using `Recall@K`.

## 2. TOTAL Latency Benchmark Blocked
The Sarvam STT API rate-limits aggressively (~15-20 requests/minute), causing `429 Too Many Requests` during the 100-query benchmark.
- **Impact:** We cannot prove the end-to-end (STT + RAG + SLM) latency is within SLA under load.
- **Mitigation needed:** Implement request throttling (`--delay` parameter) in the benchmark script, or switch to a higher-capacity STT provider.

## 3. Grounding Validation is Heuristic
The current grounding validation uses a simple word-overlap heuristic (`difflib` with a 0.3 threshold).
- **Impact:** It may falsely accept hallucinations that reuse words in different contexts, or falsely reject valid paraphrases.
- **Mitigation needed:** Implement an NLI (Natural Language Inference) model or an LLM-as-a-judge for robust grounding validation.

## 4. Extractive Path is Rigid
The extractive fast-path uses deterministic sentence boundary detection and token overlap scoring.
- **Impact:** It may struggle with complex reasoning questions or queries where the answer requires synthesizing information across multiple non-adjacent sentences.
- **Mitigation needed:** Consider a lightweight extractive QA model (e.g., BERT-SQuAD style) for the fast path.
