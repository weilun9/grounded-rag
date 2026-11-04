# Error analysis

Four concrete failures: two in retrieval and two in answering. Each one gives the query, what came back, the
root cause, the fix, and evidence.

**Where the evidence comes from.** Every quote is labelled with one of these sources:

| label | source | how real |
|---|---|---|
| **[eval run]** | `eval/results/*.json`, the runs at commit `ffed046` with `bge-small-en-v1.5`, `ms-marco-MiniLM-L-6-v2` and Ollama `qwen3.5:4b` / `:9b` | real model outputs |
| **[BM25 live]** | BM25 over the real 1,571-chunk index, re-run for this document | real; BM25 has no model, so it is identical to the production BM25 leg |
| **[offline sim]** | the real pipeline code and corpus with the models swapped for the test fakes (hash embeddings, token-overlap reranker, scripted LLM) | shows the *mechanism* deterministically; the numbers are not model behaviour |

The Hugging Face models and Ollama could not be reached from the environment where this document was written,
so no "after" numbers come from real models. Each case says what still needs re-measuring.

---

## Retrieval failure 1: dense retrieval misses an exact identifier (q09)

**Query.** *q09: "What configuration file marks the root of a virtual environment?"* Gold: PEP 405, must say
`pyvenv.cfg`.

**What was retrieved.**
- **[eval run]** `retrieval.json`: in `vector` mode q09 is a **miss at k=5** (PEP 405 is not in the top 5 at
  all). q32 is the other vector-only miss.
- **[BM25 live]** BM25's top hit is PEP 405 › API, *"``create_configuration`` – creates the ``pyvenv.cfg``
  configuration file in the environment…"*. Ranks 2, 3 and 5 are PEP 668's "Marker file" sections. They are
  lexically close ("file", "marks"), but PEP 405 is rank 1.

**Root cause.** `bge-small-en-v1.5` is a small (33M-parameter) embedding model. The question is phrased in
general terms ("configuration file", "root", "virtual environment"), and several PEPs talk about environments
and marker files, so the dense neighbourhood is crowded with PEP 668 (externally-managed environments) and
packaging PEPs. The decisive evidence is the literal string `pyvenv.cfg`, and that sits in the lexical channel.

**Fix (applied, part of the original design).** Hybrid retrieval: run the dense and BM25 rankings and fuse them with
Reciprocal Rank Fusion (`retrieval.py:rrf_fuse`). RRF only uses ranks, so the two incompatible score scales never
have to be calibrated against each other.

**Before / after.** **[eval run]**

| mode | Hit@5 (all 35) | q09 |
|---|---|---|
| vector | 0.943 | miss |
| bm25 | 1.000 | hit |
| hybrid (RRF) | 1.000 | hit |
| hybrid + rerank | 0.971 | hit |

**Residual.** End to end, `qwen3.5:4b` still refused q09 **[eval run]** while `:9b` answered *"…is
``pyvenv.cfg`` [4]"*. The 9b answer cites passage **[4]**, so the right chunk was low in the reranked top 5.
That points at the small model ignoring lower-ranked passages, not at retrieval.

---

## Retrieval failure 2: the reranker demotes the right document (q16)

**Query.** *q16: "Which PEP allows writing list[int] instead of typing.List[int]?"* Gold: PEP 585.

**What was retrieved.**
- **[eval run]** `hybrid` (RRF, no rerank) has PEP 585 in the top 5. `hybrid_rerank` does **not**: q16 is the
  rerank mode's only miss. Both models answered *"I don't know based on the provided documents."* That is
  the correct behaviour given the passages, but it is a false refusal.
- **[BM25 live]** top 5:

  | # | doc | section | note |
  |---|---|---|---|
  | 1 | PEP 604 | Simplified Syntax | `def f(list: List[Union[int, str]], …)` |
  | 2 | PEP 604 | isinstance and issubclass | |
  | 3 | PEP 585 | Generics erasure | under PEP 585's *Rejected alternatives* heading |
  | 4 | PEP 636 | Matching builtin classes | |
  | 5 | PEP 526 | Abstract | |

**Root cause.** Two effects stack:
1. *Lexical:* the tokenizer splits `list[int]` and `typing.List[int]` into `list`, `int` and `typing`. PEP 604's examples
   repeat `List[...]`, `int` and `Union` constantly, so it wins on term frequency. PEP 585's Abstract says
   "generics syntax in all standard collections" and barely uses those tokens.
2. *Rerank:* the cross-encoder scores each passage on its own against the question. A PEP 604 passage full
   of `List[int]`-style code looks very relevant to a question about `list[int]`. The question is really about
   *which document* introduced something, and a passage-level scorer can't see that. Reranking replaces the
   RRF order entirely (`hits.sort(key=rerank_score)`), so the agreement signal from the two retrievers is thrown
   away.

**Fix (proposed, not applied).** Either
- **blend the scores**: order by `α·z(rerank) + (1-α)·z(rrf)` instead of the rerank score alone, so a document that both
  retrievers like can't be pushed out by one cross-encoder judgment; or
- **keep the top 5 document-diverse** (at most 2 chunks per doc, MMR-style), so PEP 585 isn't crowded out by three PEP 604
  chunks.

It isn't applied because both change the retrieval table, and choosing α needs the real models to measure. It's
the next experiment. The acceptance criterion is that q16 is a hit with no loss in Hit@1 on the other 34
questions.

---

## Generation failure 1: the model presents a rejected idea as the answer (q33)

**Query.** *q33: "How can I assign a value to a variable inside an if condition?"* Gold: PEP 572, must contain
`:=`.

**What was answered.** **[eval run]** with both models, citing PEP 572 (citation precision 1.0, retrieval
correct):

> You can assign a value to a variable inside an `if` condition using assignment expressions with the syntax
> `if re.search(pat, text) as match:` [1]. This approach works effectively when the condition is based on the
> truthiness of the captured value… *(qwen3.5:9b)*

`if … as match:` is not Python syntax. PEP 572 considered it and **rejected** it.

**Root cause.** The cited chunk (`PEP572-039`) comes from this part of the document:

```
Rejected alternative proposals        <- level-1 heading
  Special-casing conditional statements   <- level-2 heading, the only one the chunk recorded
```

The passage text itself says *"…this proposal enhances the syntax of these two statements to add a means of
capturing the compared value"*, so read in isolation it is a proposal. The old chunker kept only the immediate
heading ("Special-casing conditional statements"), so neither the reranker nor the model could see that the
passage sits in a rejected section. This isn't a hallucination. The model faithfully summarised a passage that
was missing its status. The citation verifier passes it because the citation is valid. A key-term metric catches
it only because `:=` happens to be missing.

**Fix (applied).** `ingest.py` now tracks the full heading path (RST levels follow the order in which underline
characters first appear, as reStructuredText assigns them) and sets `Chunk.rejected = True` when any heading in the path
matches `rejected | abandoned | postponed | withdrawn`. `graph.format_context` labels those passages, and the system
prompt says never to present them as the answer:

**Before / after: the prompt the model sees for this chunk** **[offline sim, real chunk]**

```
before: [1] (PEP 572 – Assignment Expressions > Special-casing conditional statements)
after:  [1] (PEP 572 – Assignment Expressions > Special-casing conditional statements; REJECTED IDEA, not part of the final proposal)
```

116 of the 1,571 chunks (15 of 29 PEPs) are flagged. `embed_text` is deliberately unchanged, so retrieval rankings and
the published retrieval metrics are unaffected. Rejected chunks are still retrievable, because "why wasn't X chosen?"
is a legitimate question.

Tests: `test_ingest.py::test_chunks_under_rejected_headings_are_flagged`,
`::test_demo_corpus_matches_the_numbers_in_the_readme` (the `as match` chunk is flagged), and
`test_pipeline_e2e.py::test_rejected_section_reaches_the_model_labelled`.

**Still to measure.** Whether `qwen3.5:4b/9b` actually follow the label on q33. The obvious risk is over-correction:
refusing "why was `as` rejected?" questions. If the label isn't enough, the next step is a small retrieval penalty
for rejected chunks when the question doesn't contain "why not", "rejected" or "alternative".

---

## Generation failure 2: the query rewrite smuggles off-topic questions past the relevance gate (u02 and similar)

**Query.** *u02: "How do I configure a PostgreSQL replication slot?"* It is unanswerable, so the expected outcome is a
refusal **without** a generation call.

**What happened.** **[eval run]** u02 and all 10 unanswerable questions end with *"I don't know based on the
provided documents."*, so the refusal rate is 1.0. The README originally said that only 4 of the 10 were
refused by the gate and that 6 reached the model, which then declined on its own. Those runs didn't store
the graph trace, so that split **can't be checked from the result files**. The mechanism behind it, though, is
a real bug, and it reproduces deterministically.

**Root cause (a code bug).** In `hybrid_rerank` mode, `HybridIndex.search()` already fills `hit.rerank_score`,
but it scores against the **search query**. `_grade` then called `score_relevance(question, hits)`, which only
fills scores that are *missing*. On the first pass query == question, so nothing goes wrong. After a rewrite,
the hits carry scores against the **rewritten** query. That query is LLM output, and the rewrite prompt
explicitly steers it toward PEP vocabulary. The gate meant to judge "are these passages relevant to what the
user asked?" was really judging "are they relevant to what the model turned it into?" That leaves the
model's own "I don't know" as the only guard, and it's the guard the gate exists to replace.

**Fix (applied).** `_grade` clears the scores and re-scores against the original question whenever
`query != question` (`graph.py`, `_grade`). The cost is one extra cross-encoder pass over 5 passages, and only on the
rewrite path.

**Before / after** **[offline sim]**: real corpus, token-overlap reranker as the cross-encoder (threshold 1.0),
scripted rewrite *"configure Python package installation environment"*:

```
before (ffed046): refused   generation calls: 1
  retrieve('How do I configure a PostgreSQL replication slot?') -> 5
  grade -> 0/5 relevant
  rewrite -> 'configure Python package installation environment'
  retrieve('configure Python package installation environment') -> 5
  grade -> 5/5 relevant          <- judged against the rewrite
  generate                       <- the model is asked to answer an off-topic question
  verify -> refused              <- saved only because the scripted model declined

after:            refused   generation calls: 0
  ... same first four steps ...
  grade -> 0/5 relevant          <- judged against the user's question
  refuse
```

Regression test: `test_refusal.py::test_rewritten_query_does_not_let_irrelevant_passages_through_the_gate`. It
fails on the old `graph.py` and passes on the new one (checked by swapping the file).

**Still to measure.** With the real cross-encoder, (a) how many unanswerable questions are now refused with zero
generation calls (expected: all 10), and (b) whether any *answerable* question that relied on the rewrite to pass the
gate is now falsely refused. The eval rows now store `trace` and `retrieved_docs`, so both can be read straight from
the next `eval --e2e` report.

---

## Other failures seen in the eval runs (not analysed in depth)

- **q35** **[eval run]**: *"…by shipping a marker file in the ``stdlib`` directory…"*. That's right in substance, but
  it doesn't name `EXTERNALLY-MANAGED`. Partly a metric artefact (strict key terms) and partly a prompt-adherence
  issue ("quote identifiers exactly"). The fix is a faithfulness/LLM-judge metric. Loosening the gold terms to match
  the model would be grading on a curve.
- **q32** **[eval run]**: refused by both models even though `hybrid_rerank` has PEP 484/604 in the top 5 at document
  level. Not diagnosed, because these runs didn't record which passages were shown. The new trace fields make that
  possible.
