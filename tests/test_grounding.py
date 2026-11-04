"""Citation verification and how passages are presented to the model."""

from grounded_rag.graph import REFUSAL, REJECTED_TAG, RAGPipeline, format_context, verify_citations
from grounded_rag.ingest import Chunk
from grounded_rag.retrieval import Hit

from .conftest import ScriptedLLM, make_pipeline


def _hit(text, section="S", rejected=False):
    return Hit(Chunk("id-" + section, "PEP 1", "PEP 1 – T", section, text, "x.rst", 0, rejected), 0.1)


def test_duplicate_and_adjacent_citations_are_collected_once_in_order():
    _, cited, status = verify_citations("A [2][1]. B [2]. C [1][3].", n_passages=3)
    assert status == "answered" and cited == [2, 1, 3]


def test_zero_and_out_of_range_citations_are_invalid():
    answer, cited, status = verify_citations("A [0]. B [4].", n_passages=3)
    assert (answer, cited, status) == (REFUSAL, [], "ungrounded")


def test_refusal_wins_even_if_the_model_adds_citations():
    assert verify_citations("I don't know based on the provided documents [1].", 3) == (REFUSAL, [], "refused")


def test_context_numbers_passages_and_names_their_source():
    ctx = format_context([_hit("first", "Abstract"), _hit("second", "Rationale")])
    assert ctx.startswith("[1] (PEP 1 – T > Abstract)\nfirst")
    assert "[2] (PEP 1 – T > Rationale)\nsecond" in ctx


def test_rejected_passages_are_labelled_for_the_model():
    ctx = format_context([_hit("use if x as y", "Using as", rejected=True), _hit("use :=", "Spec")])
    first, second = ctx.split("\n\n")
    assert REJECTED_TAG in first and REJECTED_TAG not in second


def test_citations_point_at_the_passage_the_model_cited(index, settings):
    # passage [2] in the prompt must come back as citation n=2 with that passage's text
    llm = ScriptedLLM(lambda p: "Lines are limited to 79 characters [2].")
    out = RAGPipeline(index, llm, settings).ask("maximum line length characters indentation spaces")
    prompt = llm.generation_prompts[0]
    second = prompt.split("[2] (")[1].split("\n", 1)[1].split("\n\n")[0]
    assert out["status"] == "answered"
    assert [c["n"] for c in out["citations"]] == [2]
    assert out["citations"][0]["text"].startswith(second.strip()[:40])


def test_citations_beyond_the_relevant_passages_are_dropped(index, settings):
    # only passages that passed the gate are numbered; [9] cannot point at anything
    p = make_pipeline(index, settings, ["Use 4 spaces [1]. Also tabs [9]."])
    out = p.ask("How many spaces per indentation level?")
    assert "[9]" not in out["answer"] and [c["n"] for c in out["citations"]] == [1]
