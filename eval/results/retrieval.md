# Evaluation report (2026-10-27 07:32)

- Corpus: 29 documents, 1571 chunks
- Embeddings: `BAAI/bge-small-en-v1.5` · Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Retrieval (35 answerable questions, doc-level)

Rejected-proposal penalty: 2.0. *Top from rejected*: questions whose first passage comes from a rejected-ideas section.

| Mode | Hit@1 | Hit@5 | MRR@5 | Misses | Top from rejected |
|---|---|---|---|---|---|
| vector | 0.857 | 0.943 | 0.890 | q09, q32 | q32 |
| bm25 | 0.800 | 1.000 | 0.883 | - | q18, q19 |
| hybrid | 0.914 | 1.000 | 0.949 | - | - |
| hybrid_rerank | 0.914 | 0.971 | 0.943 | q16 | - |
| hybrid_rerank (no rejected penalty) | 0.914 | 0.971 | 0.936 | q16 | q01, q09, q13, q19, q33 |
