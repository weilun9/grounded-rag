"""When the context is insufficient the pipeline must refuse, and say why in the trace."""

from grounded_rag.config import Settings
from grounded_rag.graph import REFUSAL, RAGPipeline

from .conftest import ScriptedLLM


def test_off_topic_question_never_reaches_generation(index, settings):
    llm = ScriptedLLM(["capital city New Zealand"])  # the rewrite; a generation call would raise IndexError
    out = RAGPipeline(index, llm, settings).ask("What is the capital of New Zealand?")
    assert out["status"] == "refused" and out["answer"] == REFUSAL and out["citations"] == []
    assert len(llm.prompts) == 1 and llm.generation_prompts == []
    assert out["trace"][-1] == "refuse"


def test_no_rewrite_budget_refuses_immediately(index):
    s = Settings(llm_provider="none", top_k=3, candidate_k=10, relevance_threshold=0.0, max_rewrites=0)
    llm = ScriptedLLM([])
    out = RAGPipeline(index, llm, s).ask("What is the capital of New Zealand?")
    assert out["status"] == "refused" and llm.prompts == []
    assert not any(t.startswith("rewrite") for t in out["trace"])


def test_rewritten_query_does_not_let_irrelevant_passages_through_the_gate(index, settings):
    """Regression test for the gate leak described in docs/error_analysis.md.

    The rewrite turns an off-topic question into corpus vocabulary. In hybrid_rerank mode the
    retrieved hits are already scored against that rewritten query; before the fix the gate reused those
    scores, so the passages passed and the model was asked to answer. They must be re-scored
    against the user's original question.
    """
    llm = ScriptedLLM(["walrus operator assignment expression notation", "Use := [1]."])
    out = RAGPipeline(index, llm, settings).ask("How do I configure a PostgreSQL replication slot?")
    assert out["status"] == "refused"
    assert llm.generation_prompts == []
    assert "grade -> 0/" in out["trace"][-2]


def test_model_saying_it_does_not_know_is_reported_as_refused(index, settings):
    llm = ScriptedLLM(["I don't know based on the provided documents."])
    out = RAGPipeline(index, llm, settings).ask("How many spaces per indentation level?")
    assert out["status"] == "refused" and out["citations"] == []
    assert len(llm.generation_prompts) == 1  # it was asked, and declined


def test_extractive_mode_refuses_off_topic_questions_without_rewrite(index, settings):
    out = RAGPipeline(index, None, settings).ask("What is the capital of New Zealand?")
    assert out["status"] == "refused"
    assert not any(t.startswith("rewrite") for t in out["trace"])
