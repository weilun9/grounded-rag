import pytest

from grounded_rag.retrieval import MODES, rrf_fuse, tokenize


def test_rrf_rewards_agreement_between_rankers():
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == fused["a"]  # symmetric ranks, same score
    assert fused["c"] == pytest.approx(1 / 63)
    assert max(fused, key=fused.get) in {"a", "b"}
    assert fused["d"] < fused["a"]


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is the walrus := operator?") == ["walrus", "operator"]


def test_bm25_finds_the_lexical_match(index):
    top = index.search("maximum line length characters", mode="bm25", top_k=1)[0]
    assert top.chunk.section == "Maximum Line Length"


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_returns_at_most_top_k(index, mode):
    hits = index.search("walrus operator assignment", mode=mode, top_k=2, candidate_k=5)
    assert 0 < len(hits) <= 2


def test_rerank_orders_by_cross_encoder_score(index):
    hits = index.search("walrus operator notation expression", mode="hybrid_rerank", top_k=3, candidate_k=10)
    scores = [h.rerank_score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0].chunk.doc_id == "PEP 572"


def test_unknown_mode_is_rejected(index):
    with pytest.raises(ValueError):
        index.search("x", mode="magic")
