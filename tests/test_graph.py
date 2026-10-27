from grounded_rag.graph import REFUSAL, format_context, verify_citations
from grounded_rag.ingest import chunk_documents
from grounded_rag.retrieval import Hit

from .conftest import REJECTED_DOC, make_pipeline


def test_verify_keeps_valid_citations_and_strips_invalid_ones():
    answer, cited, status = verify_citations("It uses := [1]. Also see [7].", n_passages=3)
    assert status == "answered" and cited == [1]
    assert "[7]" not in answer and "[1]" in answer


def test_verify_fails_closed_without_citations():
    assert verify_citations("It uses :=.", n_passages=3) == (REFUSAL, [], "ungrounded")


def test_verify_recognises_model_refusal():
    assert verify_citations("I don't know based on the provided documents", 3)[2] == "refused"


def test_answer_with_citation_is_returned(index, settings):
    p = make_pipeline(index, settings, ["Use the := operator, the walrus operator [1]."])
    out = p.ask("What is the walrus operator notation?")
    assert out["status"] == "answered"
    assert out["citations"][0]["doc_id"] == "PEP 572"
    assert out["trace"][-1] == "verify -> answered"


def test_uncited_answer_is_replaced_by_refusal(index, settings):
    p = make_pipeline(index, settings, ["The walrus operator is :=."])
    out = p.ask("What is the walrus operator notation?")
    assert out["status"] == "ungrounded" and out["answer"] == REFUSAL


def test_off_topic_question_is_refused_without_generation(index, settings):
    # first reply is the rewritten query, which still matches nothing; no generation reply is scripted
    p = make_pipeline(index, settings, ["capital city New Zealand"])
    out = p.ask("What is the capital of New Zealand?")
    assert out["status"] == "refused"
    assert any(t.startswith("rewrite") for t in out["trace"])
    assert "generate" not in out["trace"]


def test_extractive_mode_works_without_an_llm(index, settings):
    out = make_pipeline(index, settings, None).ask("How many spaces per indentation level?")
    assert out["status"] == "answered"
    assert "4 spaces" in out["answer"]


def test_context_labels_rejected_proposals():
    chunks = chunk_documents([REJECTED_DOC], chunk_size=300, chunk_overlap=0)
    context = format_context([Hit(c, 0.0) for c in chunks])
    assert "[1] (PEP 9001 – Inline Assignment > Specification)\n" in context
    assert "[2] (PEP 9001 – Inline Assignment > Using ``as`` in conditions) (rejected proposal)\n" in context
    assert context.count("(rejected proposal)") == 1
