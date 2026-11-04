# Interview notes: design trade-offs

Ten questions an interviewer is likely to ask, with short answers that point at the code. Numbers come from
`eval/results/` (measured at commit `ffed046`; see the provenance note in the README).

---

### 1. Why hybrid retrieval instead of just a vector database?

Small embedding models blur exact identifiers. In vector-only mode, q09 (`pyvenv.cfg`) and q32 miss the top 5
entirely, while BM25 gets Hit@5 = 1.000 on the same 35 questions. BM25 has the opposite weakness: Hit@1 is only
0.800, because paraphrased questions share few terms with the answer. Fusing the two gives Hit@1 0.886 and
Hit@5 1.000. It costs one extra in-memory BM25 scoring pass (`rank_bm25` over 1,571 chunks), which is negligible
next to the LLM call. Code: `retrieval.py:HybridIndex.search`.

### 2. Why Reciprocal Rank Fusion rather than a weighted sum of scores?

Cosine similarity and BM25 scores live on unrelated scales, and the BM25 scale shifts with the query and the corpus.
A weighted sum needs normalisation and a tuned weight, and both drift. RRF (`score = Σ 1/(60 + rank)`,
`retrieval.py:rrf_fuse`) uses only ranks, has a single constant that the literature sets to 60, and rewards agreement
between the retrievers. The downside is that it throws away score *magnitude*: a chunk that is overwhelmingly the best
lexical match counts the same as one that barely ranked first.

### 3. What does the cross-encoder add, and what does it cost?

Two jobs. **Reranking**: a cross-encoder reads question and passage together, so it judges relevance better than
comparing two independent embeddings. Hit@1 goes 0.886 → 0.914. **Gating**: its logit is an absolute relevance
score, so a threshold (`RAG_RELEVANCE_THRESHOLD=-2.0`) can decide whether to call the LLM at all. The cost is 20
(question, passage) forward passes per query and one regression: q16 drops out of the top 5 (Hit@5 1.000 → 0.971),
because the reranker overrides RRF completely. Blending the two scores is the proposed fix
(`docs/error_analysis.md`, retrieval failure 2).

### 4. How do you stop the model from answering questions the documents can't answer?

There are three layers, and the cheapest ones run first:
1. The **relevance gate** (`graph.py:_grade`). If no passage clears the cross-encoder threshold, the graph rewrites the
   query once, then refuses **without a generation call**.
2. The **prompt** tells the model to answer only from the numbered passages, or reply with the exact refusal string.
3. The **verifier** (`verify_citations`) fails closed: an answer with no valid `[n]` becomes a refusal (`ungrounded`).

All 10 unanswerable questions were refused in both runs. While writing the error analysis I found that layer 1 leaked
after a rewrite: the gate reused scores computed against the rewritten query. That's fixed now, and a regression test
covers it.

### 5. Why a LangGraph state machine instead of a simple chain?

The pipeline has a loop (grade → rewrite → retrieve) and two exits (generate or refuse), and both are data-dependent.
A linear chain can't express that cleanly. In the graph each decision is a named node with a conditional edge
(`_route_after_grade`), and every node appends to a `trace` that the API returns. That makes failures debuggable:
"grade → 0/5 relevant, rewrite, grade → 5/5" is exactly how the gate leak showed up. The trade-off is a framework
dependency for what is roughly 40 lines of control flow. Without LangGraph it would be a `while` loop. The
graph earns its place mainly through the trace and readability.

### 6. Is "it has a citation" the same as "it's grounded"?

No, and the README says so. The verifier checks that citations *exist* and point at real passages. It doesn't check
that the passage *supports* the sentence, or that every sentence is cited. q33 shows the gap: the answer cited
PEP 572 correctly (citation precision 1.0) but presented a *rejected* proposal as the answer. Closing it needs an
entailment check, such as an NLI model or an LLM judge per (sentence, cited passage) pair. That's next on the roadmap.
The current design catches the cheap, common failure (no citations, or invented `[7]`) deterministically.

### 7. How did you choose chunk size and chunking strategy?

Chunks are split on document **sections** first (`ingest.py:split_sections_with_path`), then recursively to 1,000
characters with 150 overlap. Section-first splitting keeps a passage about one topic together. The 1,000/150 setting
is a common default; I didn't run a chunk-size ablation, and I'd say that plainly. The bigger lever was
**contextual chunk headers**: each chunk is embedded as `"<doc title> > <section>\n<text>"`, so "The default is 79
characters" still retrieves for "style guide line length". The heading *path* is now also used to flag chunks
under "Rejected …" headings (116 of 1,571).

### 8. How do you evaluate it, and how much should anyone trust the numbers?

There are 45 hand-written questions (35 answerable, each labelled with gold PEPs and required key terms; 10
unanswerable). Retrieval is scored at document level (Hit@1, Hit@5, MRR@5) for every mode. End to end, I measure answer
accuracy (all key-term groups present), citation precision, false-refusal rate, refusal rate on unanswerable
questions, and p50/p95 latency (`evaluate.py`). Caveats I'd raise myself: at n = 35, one question is about 2.9 points,
so 4b vs 9b (0.857 vs 0.886) is a single question. One person wrote all the questions. The threshold was tuned on the
same set. Key-term matching is strict about wording and blind to extra wrong claims. Tests check that the data is
internally consistent: gold terms occur in the sources, and the metrics recompute from the rows.

### 9. How do you test an LLM pipeline without an API key or GPU?

Swap the models, keep everything else. `tests/conftest.py` provides hash embeddings, a token-overlap stand-in for the
cross-encoder, and a scripted chat model that records every prompt. `test_pipeline_e2e.py` then runs the real chunker,
BM25, RRF, LangGraph wiring, verifier, FastAPI app and eval harness over the real 29-PEP corpus. The 98 tests run in
about 4 s with no network. Prompt recording means tests can assert on *what the model was shown*: for example, that a
rejected passage is labelled, or that no generation prompt was sent for an off-topic question. Model quality itself is
measured by the eval, not by unit tests. CI also builds the real index and publishes the retrieval report.

### 10. What would you change to run this in production?

- **Measure first:** re-run the e2e eval after the two fixes, add a held-out split for the threshold, and add a
  faithfulness metric.
- **Ingest:** incremental, hash-per-document re-indexing (today `HybridIndex.build` deletes and rebuilds everything),
  and BM25 in a real search engine once the corpus no longer fits in memory.
- **Serving:** the pipeline is synchronous and single-process, and latency is dominated by generation
  (p50 about 3.1 to 3.2 s on a local 4b/9b model). Production would need async model calls, streaming,
  caching of rerank scores, and a timeout/fallback on the provider. The provider is already pluggable
  (`llm.py`: Ollama, Anthropic, OpenAI, or none).
- **Observability:** log the trace that the API already returns, plus gate scores, so threshold drift and refusal rates
  can be monitored.
- **Security:** retrieved text goes into the prompt verbatim, which is a prompt-injection surface if the corpus isn't
  trusted. The UI already escapes model output (there is a Selenium test for that).
