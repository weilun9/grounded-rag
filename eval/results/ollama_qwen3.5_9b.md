# Evaluation report (2026-10-27 14:16)

- Corpus: 29 documents, 1571 chunks
- Embeddings: `BAAI/bge-small-en-v1.5` · Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Retrieval (35 answerable questions, doc-level)

| Mode | Hit@1 | Hit@5 | MRR@5 | Misses |
|---|---|---|---|---|
| vector | 0.857 | 0.943 | 0.890 | q09, q32 |
| bm25 | 0.800 | 1.000 | 0.883 | - |
| hybrid | 0.886 | 1.000 | 0.930 | - |
| hybrid_rerank | 0.914 | 0.971 | 0.936 | q16 |

## End-to-end (LLM: `ollama:qwen3.5:9b`, mode: `hybrid_rerank`)

| Metric | Value |
|---|---|
| Answer accuracy (35 answerable) | 0.886 |
| Citation precision | 0.965 |
| False-refusal rate | 0.057 |
| Refusal rate on 10 unanswerable | 1.0 |
| Latency p50 / p95 | 3233 ms / 5027 ms |

### Failures

- **q16** (refused): "I don't know based on the provided documents."
- **q32** (refused): "I don't know based on the provided documents."
- **q33** (answered): 'You can assign a value to a variable inside an `if` condition using assignment expressions with the syntax `if re.search(pat, text) as match:` [1]. This approach works effectively when the condition is based on the truth'
- **q35** (answered): 'A Linux distribution can stop pip from installing into the system Python by shipping a marker file in the ``stdlib`` directory and placing distro-shipped packages in a location other than the default ``sysconfig`` scheme'
