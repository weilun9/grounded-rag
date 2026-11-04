"""Retrieval behaviour beyond the basics in test_retrieval.py: fusion, the gate's scoring helper, persistence."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from grounded_rag.retrieval import HybridIndex, rrf_fuse

from .conftest import OverlapReranker


def test_rrf_keeps_items_found_by_only_one_ranker():
    fused = [cid for cid, _ in rrf_fuse([["a", "b"], ["c"]])]
    assert set(fused) == {"a", "b", "c"}
    assert fused[-1] == "b"  # rank 2 in one list scores below rank 1 in either list


def test_bm25_ignores_chunks_with_zero_overlap(index):
    assert index.bm25_ranking("zzzz qqqq", k=5) == []


def test_bm25_matches_exact_identifiers(index):
    top = index.search("NAME := expr notation", mode="bm25", top_k=1)[0]
    assert top.chunk.doc_id == "PEP 572" and top.chunk.section == "Abstract"


def test_hybrid_returns_union_of_both_rankers(index):
    hits = index.search("indentation spaces", mode="hybrid", top_k=10, candidate_k=10)
    ids = {h.chunk.chunk_id for h in hits}
    assert set(index.bm25_ranking("indentation spaces", 10)) <= ids
    assert all(h.rerank_score is None for h in hits)


def test_rerank_mode_without_reranker_is_an_error(chunks):
    idx = HybridIndex(chunks, DeterministicFakeEmbedding(size=16), reranker=None)
    with pytest.raises(RuntimeError):
        idx.search("walrus", mode="hybrid_rerank")


def test_score_relevance_only_fills_missing_scores(index):
    hits = index.search("walrus operator", mode="hybrid", top_k=2, candidate_k=5)
    hits[0].rerank_score = 99.0
    index.score_relevance("walrus operator", hits)
    assert hits[0].rerank_score == 99.0 and hits[1].rerank_score is not None


def test_index_survives_save_and_load(chunks, tmp_path):
    emb = DeterministicFakeEmbedding(size=32)
    built = HybridIndex.build(chunks, emb, tmp_path / "idx", reranker=OverlapReranker())
    loaded = HybridIndex.load(tmp_path / "idx", emb, reranker=OverlapReranker())
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in built.chunks]
    q = "maximum line length"
    assert [h.chunk.chunk_id for h in loaded.search(q, mode="vector", top_k=3)] == \
           [h.chunk.chunk_id for h in built.search(q, mode="vector", top_k=3)]


def test_loading_a_missing_index_explains_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest"):
        HybridIndex.load(tmp_path, DeterministicFakeEmbedding(size=8))
