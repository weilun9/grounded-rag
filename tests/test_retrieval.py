import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from grounded_rag.retrieval import MODES, HybridIndex, rrf_fuse, tokenize

from .conftest import make_index


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


QUESTION = "How can I assign a value inside an if condition?"


def test_without_penalty_the_rejected_idea_ranks_first(rejected_chunks):
    top = make_index(rejected_chunks, 0.0).search(QUESTION, mode="hybrid_rerank", top_k=3)[0]
    assert top.chunk.status == "rejected"


def test_penalty_demotes_rejected_chunks_after_rerank(rejected_chunks):
    hits = make_index(rejected_chunks, 3.0).search(QUESTION, mode="hybrid_rerank", top_k=3)
    assert hits[0].chunk.section == "Specification"
    rejected = next(h for h in hits if h.chunk.status == "rejected")
    # the stored score stays raw, so the relevance gate is unaffected by the penalty
    assert rejected.rerank_score > hits[0].rerank_score


def test_small_penalty_keeps_a_clearly_better_rejected_chunk_on_top(rejected_chunks):
    top = make_index(rejected_chunks, 0.5).search(QUESTION, mode="hybrid_rerank", top_k=3)[0]
    assert top.chunk.status == "rejected"


def test_penalty_in_rank_places_without_reranker(rejected_chunks):
    idx = make_index(rejected_chunks, 0.0)
    raw = [h.chunk.chunk_id for h in idx.search(QUESTION, mode="hybrid", top_k=5)]
    idx.rejected_penalty = 1.0
    demoted = [h.chunk.chunk_id for h in idx.search(QUESTION, mode="hybrid", top_k=5)]
    rejected_id = next(c.chunk_id for c in rejected_chunks if c.status == "rejected")
    i = raw.index(rejected_id)
    assert i < len(raw) - 1  # it has someone to drop below
    assert demoted.index(rejected_id) == i + 1
    assert sorted(demoted) == sorted(raw)


@pytest.mark.parametrize("mode", ["vector", "bm25"])
def test_single_retriever_modes_are_not_penalised(rejected_chunks, mode):
    raw = [h.chunk.chunk_id for h in make_index(rejected_chunks, 0.0).search(QUESTION, mode=mode, top_k=5)]
    penalised = [h.chunk.chunk_id for h in make_index(rejected_chunks, 5.0).search(QUESTION, mode=mode, top_k=5)]
    assert penalised == raw


def test_status_survives_build_and_load_and_reaches_chroma(rejected_chunks, tmp_path):
    emb = DeterministicFakeEmbedding(size=64)
    HybridIndex.build(rejected_chunks, emb, tmp_path / "index")
    loaded = HybridIndex.load(tmp_path / "index", emb, rejected_penalty=2.0)
    assert [c.status for c in loaded.chunks] == [c.status for c in rejected_chunks]
    assert loaded.rejected_penalty == 2.0
    metas = loaded.vectorstore.get(include=["metadatas"])["metadatas"]
    assert sorted(m["status"] for m in metas) == sorted(c.status for c in rejected_chunks)
