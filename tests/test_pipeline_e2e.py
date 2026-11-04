"""The whole pipeline over the real 29-PEP corpus, offline.

Real chunking, real BM25 and RRF, real LangGraph wiring, real verifier and API. Only the models are
replaced: hash embeddings for the dense side, a token-overlap scorer for the cross-encoder, and a
scripted chat model that answers from the passages it is shown. No API key, no downloads.
"""

import re

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from grounded_rag.api import create_app
from grounded_rag.config import ROOT, Settings
from grounded_rag.evaluate import eval_e2e, eval_retrieval, load_questions
from grounded_rag.graph import REFUSAL, REJECTED_TAG, RAGPipeline
from grounded_rag.ingest import chunk_documents, load_corpus
from grounded_rag.retrieval import HybridIndex

from .conftest import OverlapReranker, ScriptedLLM


@pytest.fixture(scope="module")
def corpus_index():
    chunks = chunk_documents(load_corpus(ROOT / "data" / "corpus"), 1000, 150)
    return HybridIndex(chunks, DeterministicFakeEmbedding(size=64), reranker=OverlapReranker())


@pytest.fixture
def e2e_settings():
    return Settings(llm_provider="mock", retrieval_mode="hybrid_rerank", top_k=5, candidate_k=20,
                    relevance_threshold=0.0, max_rewrites=1)


def cite_first_passage(prompt: str) -> str:
    """A 'model' that answers with the first sentence of passage [1] and cites it; refuses if there is none."""
    if not prompt.startswith("You answer questions"):
        return prompt.split("Question:")[-1].strip()  # rewrite: echo the question
    m = re.search(r"\[1\] \([^\n]*\)\n(.+?)(?:\n\n\[2\]|\n\nQuestion:)", prompt, re.DOTALL)
    return f"{m.group(1).split('.')[0].strip()} [1]." if m else REFUSAL


def test_answerable_question_is_answered_from_the_right_document(corpus_index, e2e_settings):
    llm = ScriptedLLM(cite_first_passage)
    out = RAGPipeline(corpus_index, llm, e2e_settings).ask("Which file marks the root of a virtual environment pyvenv.cfg?")
    assert out["status"] == "answered"
    assert out["citations"][0]["doc_id"] == "PEP 405"
    assert [re.split(r"[ (]", t)[0] for t in out["trace"]] == ["retrieve", "grade", "generate", "verify"]


def test_off_topic_question_is_refused_on_the_real_corpus(corpus_index, e2e_settings):
    llm = ScriptedLLM(cite_first_passage)
    out = RAGPipeline(corpus_index, llm, e2e_settings).ask("Who won the 2022 FIFA World Cup?")
    assert out["status"] == "refused" and llm.generation_prompts == []


def test_rejected_section_reaches_the_model_labelled(corpus_index, e2e_settings):
    """docs/error_analysis.md case 3: PEP 572's rejected `if ... as match:` spelling is labelled in the prompt."""
    llm = ScriptedLLM(cite_first_passage)
    RAGPipeline(corpus_index, llm, e2e_settings).ask("special-casing conditional statements if re.search as match")
    prompt = llm.generation_prompts[0]
    block = next(b for b in prompt.split("\n\n[") if "as match" in b)
    assert "Special-casing conditional statements" in block and REJECTED_TAG in block


def test_api_serves_the_pipeline_end_to_end(corpus_index, e2e_settings):
    app = create_app(RAGPipeline(corpus_index, ScriptedLLM(cite_first_passage), e2e_settings))
    with TestClient(app) as c:
        health = c.get("/health").json()
        body = c.post("/ask", json={"question": "Which file declares build system requirements pyproject.toml?"}).json()
        search = c.get("/search", params={"q": "pyvenv.cfg", "mode": "bm25", "k": 3}).json()
    assert health["documents"] == 29 and health["chunks"] == 1571
    assert body["status"] == "answered" and body["citations"] and body["latency_ms"] >= 0
    assert search[0]["doc_id"] == "PEP 405"


def test_evaluation_harness_runs_offline(corpus_index, e2e_settings):
    questions = [q for q in load_questions() if q["id"] in {"q09", "q13", "u01", "u04"}]
    retrieval = eval_retrieval(corpus_index, questions, e2e_settings)
    assert retrieval["n_questions"] == 2 and retrieval["modes"]["bm25"]["hit@5"] == 1.0
    e2e = eval_e2e(RAGPipeline(corpus_index, ScriptedLLM(cite_first_passage), e2e_settings), questions)
    assert e2e["n_answerable"] == 2 and e2e["n_unanswerable"] == 2
    assert e2e["refusal_rate_unanswerable"] == 1.0
    assert all("trace" in r and "retrieved_docs" in r for r in e2e["rows"])
