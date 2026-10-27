# Grounded RAG

Question answering over a document corpus that **cites its sources or says "I don't know"**.
Hybrid retrieval (dense + BM25, fused with Reciprocal Rank Fusion), a cross-encoder relevance gate,
a LangGraph pipeline with query rewriting and citation verification, a FastAPI service with a web UI,
and an evaluation harness that measures every one of those choices.

The demo corpus is 29 Python Enhancement Proposals (public domain). Point `RAG_CORPUS_DIR` at your own
`.md / .rst / .txt / .pdf` files to use it on anything else.

```
question ──► retrieve ──► grade ──(relevant)──► generate ──► verify ──► answer + citations
              ▲   hybrid    │ cross-encoder                  │ every claim must cite [n];
              │   (RRF)     │ score ≥ threshold              │ no valid citation ⇒ refuse
              │             │
              └─ rewrite ◄──┤ (nothing relevant, 1 retry)
                            └──► refuse  (still nothing: no LLM call is made)
```

## Results

45 questions (35 answerable, 10 unanswerable) over 29 documents / 1,571 chunks, run on a laptop
(RTX 4060, 8 GB). Full reports in [`eval/results/`](eval/results).

**Retrieval ablation** (doc-level, top 5; embeddings `bge-small-en-v1.5`, reranker `ms-marco-MiniLM-L-6-v2`)

| Mode | Hit@1 | Hit@5 | MRR@5 |
|---|---|---|---|
| Dense vector only | 0.857 | 0.943 | 0.890 |
| BM25 only | 0.800 | **1.000** | 0.883 |
| Hybrid (RRF) | 0.886 | **1.000** | 0.930 |
| **Hybrid + cross-encoder rerank** (default) | **0.914** | 0.971 | **0.936** |

**End to end** (hybrid + rerank, local models via Ollama)

| | qwen3.5:4b | qwen3.5:9b |
|---|---|---|
| Answer accuracy (35 answerable) | 0.857 | **0.886** |
| Citation precision | **0.974** | 0.965 |
| False-refusal rate | 0.086 | **0.057** |
| Refusal rate on 10 unanswerable | **1.0** | **1.0** |
| Latency p50 / p95 | 3.1 s / 4.3 s | 3.2 s / 5.0 s |

In the run above, the relevance gate rejected 8 of the 10 unanswerable questions on the first retrieval
pass. The query rewrite then pulled 4 of those back in: the rewrite prompt steered queries toward Python
vocabulary, so "PostgreSQL replication slot" came back as a Python-flavoured query whose passages cleared
the gate. In the end 4 of 10 were refused with no generation call and 6 by the model's own "I don't know".
All 10 ended up refused, but the deterministic guard was leaking.

Two causes, both fixed since that run: in `hybrid_rerank` mode the passages kept the cross-encoder scores
they got against the *rewritten* query, so the gate never judged them against the user's actual question;
and the rewrite prompt named the corpus domain. Now, after a rewrite, passages are re-scored against the
original question and must clear a stricter threshold (`RAG_REWRITE_THRESHOLD`, default 0.0 versus -2.0
on the first pass), and the rewrite prompt is domain-neutral. Every refusal records who made it
(`refused_by`: `gate` / `model` / `verifier`, plus the gate scores in the trace). The table above has not
been re-run with this change yet; `python -m grounded_rag eval --refusals` produces the gate-versus-model
breakdown for the 10 unanswerable questions.

**What still fails, and why**

- *"How can I assign a value inside an if condition?"* The top passage is from PEP 572's
  **rejected alternatives** section (`if ... as match:`), and the model presents a rejected idea as the
  answer. Retrieval is right at the document level but wrong about *status*; tagging chunks from
  "Rejected Ideas" sections and down-weighting them is the next fix.
- *"Which PEP allows `list[int]`?"* Reranking promoted PEP 604 (which also mentions `list[int]`)
  above PEP 585, and the model correctly refused rather than guess from the wrong passages.
- Two answers are right in substance but miss the exact key term (e.g. "a marker file" instead of
  `EXTERNALLY-MANAGED`), which the strict key-term metric counts as wrong.

## Design decisions

**Contextual chunk headers.** Chunks are split on document sections, and each chunk is embedded as
`"<doc title> > <section>\n<text>"`. A passage that says "The default is 79 characters" never names
the style guide; its header does.

**Hybrid retrieval with RRF.** Dense vectors catch paraphrases ("assign a value inside an if condition"
→ assignment expressions); BM25 catches exact identifiers (`pyvenv.cfg`, `__annotations__`) that small
embedding models blur. Reciprocal Rank Fusion combines the two rankings without having to calibrate
their incompatible score scales.

**The relevance gate decides whether the model is called at all.** A cross-encoder scores
(question, passage) pairs; if nothing clears the threshold the graph tries one query rewrite, then
refuses. Clearly off-topic questions are rejected deterministically and cheaply, instead of relying on
the prompt instruction "answer only from context". The gate always scores passages against the user's
original question; after a rewrite it applies a stricter threshold, because a rewrite that drifts toward
the corpus vocabulary can otherwise turn an off-topic question into an on-topic search (see Results).

**The verifier fails closed.** The model must cite `[n]` for every claim. Citations pointing at passages
that don't exist are stripped; an answer left with no valid citation is replaced by a refusal.
A wrong refusal costs the user a rephrase; a confident uncited answer costs trust.

**Provider-agnostic.** The same graph runs on a local model through Ollama (default, free, offline),
Anthropic Claude, or OpenAI, selected by an environment variable. With `RAG_LLM_PROVIDER=none` it
returns the best passages verbatim, which is also how the unit tests run without any model.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
cp .env.example .env                                    # choose the LLM provider
ollama pull qwen3.5:4b                                  # default local model

python -m grounded_rag ingest                           # build the index
python -m grounded_rag ask "How can I assign a value inside an if condition?"
python -m grounded_rag search "pyvenv.cfg" --mode bm25  # inspect retrieval only
python -m grounded_rag eval --e2e                       # full evaluation report
python -m grounded_rag eval --refusals                  # unanswerable questions: refused by gate vs model
python -m grounded_rag serve                            # API + UI at http://127.0.0.1:8000
```

Docker (talks to Ollama on the host by default):

```bash
docker compose up --build
```

## API

| Method | Path | |
|---|---|---|
| `POST` | `/ask` `{"question": "..."}` | answer, status (`answered` / `refused` / `ungrounded`), `refused_by` (`gate` / `model` / `verifier`), best gate score per pass, citations, retrieved passages, graph trace, latency |
| `GET` | `/search?q=...&mode=hybrid_rerank&k=5` | retrieval only, any mode |
| `GET` | `/health` | corpus size, retrieval mode, active LLM |
| `GET` | `/` | web UI |

## Evaluation

`eval/questions.jsonl` holds 45 hand-written questions:

- **35 answerable**, each labelled with the PEP(s) that contain the answer and the key terms a correct
  answer must contain. Ten of them deliberately avoid the vocabulary of their source
  ("assign a value inside an if condition") to test semantic retrieval.
- **10 unanswerable**, including near misses that look in-domain ("switch statement in Java",
  "What does PEP 9999 propose?").

Metrics: doc-level Hit@1 / Hit@5 / MRR@5 per retrieval mode; end-to-end answer accuracy, citation
precision, false-refusal rate, refusal rate on unanswerable questions (split into refused by the gate,
by the gate after a rewrite, by the model, or by the citation verifier), and p50 / p95 latency.
`eval --refusals` runs only the unanswerable questions and reports that split with the gate scores; with
`RAG_LLM_PROVIDER=none` there is no rewrite and no model, so it measures the first-pass gate alone.
A script checks that every gold key term really occurs in its labelled source document.

CI runs lint and unit tests (no model downloads), then builds the real index and publishes the
retrieval report and the first-pass gate refusal report (no LLM) to the job summary, and builds the
Docker image.

## Layout

```
src/grounded_rag/
  ingest.py      loading, PEP header parsing, section-aware chunking
  retrieval.py   embeddings, Chroma, BM25, RRF, cross-encoder reranker
  graph.py       LangGraph pipeline, prompts, citation verifier
  llm.py         Ollama / Anthropic / OpenAI behind one interface
  api.py         FastAPI app + web/index.html
  evaluate.py    retrieval ablation and end-to-end metrics
eval/questions.jsonl, eval/results/
tests/           unit tests with fake embeddings, reranker and LLM
```

## Limitations

- Answer correctness is scored by key-term matching, which is strict about wording and blind to extra
  wrong claims; an LLM-as-judge faithfulness score would complement it.
- 45 questions written by one person over one corpus: enough to compare configurations, not to claim
  general accuracy.
- The relevance threshold was tuned on the same questions it is evaluated on. The post-rewrite threshold
  (0.0) sits below the top score of most answerable questions, but it has not yet been validated against
  the post-rewrite scores of the questions that leaked.

## License

Code: MIT. Corpus: the PEPs in `data/corpus` are placed in the public domain by their authors.
