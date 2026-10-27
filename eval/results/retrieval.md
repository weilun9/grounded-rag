# Evaluation report (2026-10-27 14:04)

- Corpus: 29 documents, 1571 chunks
- Embeddings: `BAAI/bge-small-en-v1.5` · Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Retrieval (35 answerable questions, doc-level)

| Mode | Hit@1 | Hit@5 | MRR@5 | Misses |
|---|---|---|---|---|
| vector | 0.857 | 0.943 | 0.890 | q09, q32 |
| bm25 | 0.800 | 1.000 | 0.883 | - |
| hybrid | 0.886 | 1.000 | 0.930 | - |
| hybrid_rerank | 0.914 | 0.971 | 0.936 | q16 |
