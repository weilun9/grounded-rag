from grounded_rag.graph import REFUSAL, verify_citations
from grounded_rag.retrieval import Hit, HybridIndex

from .conftest import make_pipeline


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


class ScriptedIndex:
    """Retrieval routed by query (doc_id per query), relevance from a per-question table of phrase -> score.

    Mimics hybrid_rerank: search() fills rerank_score against the query it was given, like the real index does.
    """

    def __init__(self, chunks, routes: dict[str, str], relevance: dict[str, dict[str, float]]):
        self.chunks = chunks
        self.routes = routes
        self.relevance = relevance
        self.reranker = self

    def score(self, query: str, passages: list[str]) -> list[float]:
        table = self.relevance.get(query, {})
        return [max((v for k, v in table.items() if k in p), default=-6.0) for p in passages]

    def search(self, query, mode, top_k, candidate_k):
        hits = [Hit(c, 1.0) for c in self.chunks if c.doc_id == self.routes.get(query)][:top_k]
        for h, s in zip(hits, self.score(query, [h.chunk.embed_text for h in hits])):
            h.rerank_score = s
        return hits

    def score_relevance(self, query, hits):
        return HybridIndex.score_relevance(self, query, hits)


def test_rewrite_cannot_pull_an_off_topic_question_past_the_gate(index, settings):
    # The rewrite drifts into corpus vocabulary. Its passages score high against the rewritten query, but the gate
    # judges them against the original question, so nothing is generated.
    p = make_pipeline(index, settings, ["maximum line length 79 characters indentation spaces"])
    out = p.ask("How do I configure a PostgreSQL replication slot?")
    assert out["status"] == "refused" and out["refused_by"] == "gate"
    assert "generate" not in out["trace"]
    assert len(out["gate_best"]) == 2 and out["gate_best"][1] < settings.rewrite_threshold
    assert out["trace"][-1].startswith("refuse (gate:") and "after 1 rewrite" in out["trace"][-1]
    assert all(r["rerank_score"] < settings.rewrite_threshold for r in out["retrieved"])


def test_post_rewrite_gate_is_stricter_than_the_first_pass(chunks, settings):
    question, rewrite = "Where do variables get names?", "naming variables expression"
    # after the rewrite the best passage scores 1.0 against the question: enough for the first-pass gate (0.0),
    # not for the post-rewrite gate (2.0)
    idx = ScriptedIndex(chunks, routes={question: "PEP 8", rewrite: "PEP 572"},
                        relevance={question: {"Naming the result": 1.0}})
    out = make_pipeline(idx, settings, [rewrite]).ask(question)
    assert out["status"] == "refused" and out["refused_by"] == "gate"
    assert out["gate_best"] == [-6.0, 1.0]
    assert "generate" not in out["trace"]


def test_paraphrased_in_domain_question_is_still_answered_after_rewrite(chunks, settings):
    question, rewrite = "How wide may a line of code be?", "maximum line length characters"
    # first-pass retrieval misses the style guide; the rewrite finds it, and the passage is relevant to the question
    idx = ScriptedIndex(chunks, routes={question: "PEP 572", rewrite: "PEP 8"},
                        relevance={question: {"79 characters": 4.0}})
    out = make_pipeline(idx, settings, [rewrite, "At most 79 characters [1]."]).ask(question)
    assert out["status"] == "answered" and out["refused_by"] is None
    assert out["citations"][0]["doc_id"] == "PEP 8"
    assert out["gate_best"] == [-6.0, 4.0]
    assert [t.split(" ")[0] for t in out["trace"]] == ["retrieve('How", "grade", "rewrite", "retrieve('maximum",
                                                       "grade", "generate", "verify"]


def test_model_refusal_is_attributed_to_the_model(index, settings):
    p = make_pipeline(index, settings, ["I don't know based on the provided documents."])
    out = p.ask("What is the walrus operator notation?")
    assert out["status"] == "refused" and out["refused_by"] == "model"
    assert out["trace"][-1] == "verify -> refused (by model)"
