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

45 questions (35 answerable, 10 unanswerable) over 29 documents / 1,571 chunks. End-to-end runs on a laptop
(RTX 4060, 8 GB); the retrieval table comes from the CI runner (CPU), which reproduced the laptop's
retrieval numbers exactly before the rejected-proposal change. Full reports in [`eval/results/`](eval/results).

**Retrieval ablation** (doc-level, top 5; embeddings `bge-small-en-v1.5`, reranker `ms-marco-MiniLM-L-6-v2`;
the two hybrid modes demote passages from rejected-proposal sections, see Design decisions)

| Mode | Hit@1 | Hit@5 | MRR@5 | Top passage from a rejected section |
|---|---|---|---|---|
| Dense vector only | 0.857 | 0.943 | 0.890 | 1 |
| BM25 only | 0.800 | **1.000** | 0.883 | 2 |
| Hybrid (RRF) | **0.914** | **1.000** | **0.949** | 0 |
| **Hybrid + cross-encoder rerank** (default) | **0.914** | 0.971 | 0.943 | 0 |
| Hybrid + rerank, no rejected-proposal penalty | **0.914** | 0.971 | 0.936 | 5 |

The penalty leaves Hit@1 and Hit@5 unchanged and lifts MRR@5 (0.936 → 0.943 with rerank, 0.930 → 0.949 for
plain hybrid, whose Hit@1 also rises from 0.886). Its main effect is in the last column: without it, 5 of 35
questions get a rejected idea as their top passage, including "How can I assign a value to a variable inside
an if condition?". With it, none do. Plain hybrid now edges out the reranked mode on Hit@5 and MRR@5.
The default has not been switched: 35 questions are too few to separate them, and that change would need
its own end-to-end run.

**End to end** (hybrid + rerank, local models via Ollama; measured before the rejected-proposal penalty was
added and not yet re-run)

| | qwen3.5:4b | qwen3.5:9b |
|---|---|---|
| Answer accuracy (35 answerable) | 0.857 | **0.886** |
| Citation precision | **0.974** | 0.965 |
| False-refusal rate | 0.086 | **0.057** |
| Refusal rate on 10 unanswerable | **1.0** | **1.0** |
| Latency p50 / p95 | 3.1 s / 4.3 s | 3.2 s / 5.0 s |

On the first retrieval pass the relevance gate rejects 8 of the 10 unanswerable questions. The query
rewrite then pulls 4 of those back in: the rewrite prompt steers queries toward Python vocabulary, so
"PostgreSQL replication slot" comes back as a Python-flavoured query whose passages clear the gate.
In the end 4 of 10 are refused with no generation call and 6 are refused by the model's own
"I don't know". All 10 end up refused, but the rewrite step costs one extra LLM call and weakens the
cheap deterministic guard. Making the gate stricter after a rewrite is a planned change.

**What still fails, and why**

- *"How can I assign a value inside an if condition?"* In the end-to-end runs above, the top passage was
  from PEP 572's **rejected alternatives** section (`if ... as match:`) and the model presented a rejected
  idea as the answer. Chunks under "Rejected Ideas"-style headings are now tagged and demoted, so a
  normative PEP 572 passage ranks first (retrieval table above), and any rejected passage that is still
  retrieved reaches the model labelled "(rejected proposal)". Whether the answer itself is now correct
  has not been re-measured end to end yet.
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

**Rejected proposals are tagged, not deleted.** PEPs spend whole sections on ideas they turned down, and
those passages often match a question's wording better than the specification does. At ingest, every chunk
records its heading path; a chunk whose heading or any ancestor heading reads "Rejected …", "Abandoned …",
"Alternatives …" or "Deferred ideas" gets `status="rejected"` (135 of 1,571 chunks). In the hybrid modes these
chunks are demoted by `RAG_REJECTED_PENALTY` (default 2.0): that many cross-encoder logits after reranking,
that many rank places in plain RRF. Stored scores, and so the relevance gate, are untouched. The chunks stay
retrievable for questions that are *about* rejected ideas, and the prompt labels them "(rejected proposal)".

**The relevance gate decides whether the model is called at all.** A cross-encoder scores
(question, passage) pairs; if nothing clears the threshold the graph tries one query rewrite, then
refuses. Clearly off-topic questions are rejected deterministically and cheaply, instead of relying on
the prompt instruction "answer only from context" (see Results for where this guard leaks).

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
python -m grounded_rag serve                            # API + UI at http://127.0.0.1:8000
```

Docker (talks to Ollama on the host by default):

```bash
docker compose up --build
```

## API

| Method | Path | |
|---|---|---|
| `POST` | `/ask` `{"question": "..."}` | answer, status (`answered` / `refused` / `ungrounded`), citations, retrieved passages, graph trace, latency |
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
precision, false-refusal rate, refusal rate on unanswerable questions, and p50 / p95 latency.
A script checks that every gold key term really occurs in its labelled source document.

CI runs lint and unit tests (no model downloads), then builds the real index and publishes the
retrieval report to the job summary, and builds the Docker image.

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
- The relevance threshold and the rejected-proposal penalty were tuned on the same questions they are
  evaluated on.
- Rejected sections are found by heading text, a heuristic. A bare "alternatives" is too broad: PEP 318's
  "Syntax Alternatives" section also explains the syntax that was adopted. So "Alternative(s)" only
  counts as a heading's first word. A different corpus may need a different rule.

## License

Code: MIT. Corpus: the PEPs in `data/corpus` are placed in the public domain by their authors.
