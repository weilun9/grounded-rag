"""Hybrid retrieval: dense vectors (Chroma) + BM25, fused with Reciprocal Rank Fusion,
optionally re-ordered by a cross-encoder.

Modes, used by the evaluation ablation:
    vector         dense only
    bm25           lexical only
    hybrid         RRF(vector, bm25)
    hybrid_rerank  RRF candidates re-scored by a cross-encoder

Chunks from "Rejected Ideas"-style sections (Chunk.status == "rejected") can be demoted in the
final ranking by `rejected_penalty`; see HybridIndex.search.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from .ingest import Chunk

MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")
RRF_K = 60
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_TOKEN = re.compile(r"[a-z0-9_]+")
_STOP = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "how", "i", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "what", "when", "which", "who", "why", "will", "with", "does", "do", "can", "should", "would"]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class SentenceTransformerEmbeddings(Embeddings):
    """bge-style embeddings: queries get an instruction prefix, passages do not."""

    def __init__(self, model_name: str, query_prefix: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.query_prefix = BGE_QUERY_PREFIX if query_prefix is None and "bge" in model_name.lower() else (query_prefix or "")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True).tolist()


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self.model.predict([(query, p) for p in passages])]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rerank_score: float | None = None


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d)), rank starting at 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridIndex:
    COLLECTION = "chunks"

    def __init__(self, chunks: list[Chunk], embeddings: Embeddings, persist_dir: Path | None = None,
                 reranker: Reranker | None = None, rejected_penalty: float = 0.0, _vectorstore=None):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.embeddings = embeddings
        self.reranker = reranker
        self.rejected_penalty = rejected_penalty
        self.bm25 = BM25Okapi([tokenize(c.embed_text) for c in chunks])
        self.vectorstore = _vectorstore or self._build_vectorstore(persist_dir)

    # ---- construction / persistence -------------------------------------------------

    def _build_vectorstore(self, persist_dir: Path | None):
        from langchain_chroma import Chroma

        store = Chroma(
            # persisted indexes use a fixed name; in-memory ones get a unique name because
            # Chroma shares one ephemeral client per process
            collection_name=self.COLLECTION if persist_dir else f"{self.COLLECTION}-{uuid.uuid4().hex[:8]}",
            embedding_function=self.embeddings,
            persist_directory=str(persist_dir / "chroma") if persist_dir else None,
            collection_metadata={"hnsw:space": "cosine"},
        )
        ids = [c.chunk_id for c in self.chunks]
        for i in range(0, len(ids), 256):
            batch = self.chunks[i : i + 256]
            store.add_texts(
                texts=[c.embed_text for c in batch],
                metadatas=[{"doc_id": c.doc_id, "section": c.section, "status": c.status} for c in batch],
                ids=[c.chunk_id for c in batch],
            )
        return store

    @classmethod
    def build(cls, chunks: list[Chunk], embeddings: Embeddings, index_dir: Path,
              reranker: Reranker | None = None, rejected_penalty: float = 0.0) -> HybridIndex:
        if index_dir.exists():
            shutil.rmtree(index_dir)
        index_dir.mkdir(parents=True)
        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        return cls(chunks, embeddings, persist_dir=index_dir, reranker=reranker, rejected_penalty=rejected_penalty)

    @classmethod
    def load(cls, index_dir: Path, embeddings: Embeddings, reranker: Reranker | None = None,
             rejected_penalty: float = 0.0) -> HybridIndex:
        from langchain_chroma import Chroma

        path = index_dir / "chunks.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"No index at {index_dir}. Run `python -m grounded_rag ingest` first.")
        chunks = [Chunk(**json.loads(line)) for line in path.open(encoding="utf-8")]
        store = Chroma(collection_name=cls.COLLECTION, embedding_function=embeddings,
                       persist_directory=str(index_dir / "chroma"))
        return cls(chunks, embeddings, reranker=reranker, rejected_penalty=rejected_penalty, _vectorstore=store)

    # ---- search ---------------------------------------------------------------------

    def vector_ranking(self, query: str, k: int) -> list[str]:
        docs = self.vectorstore.similarity_search(query, k=k)
        return [d.id for d in docs if d.id in self.by_id]

    def bm25_ranking(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i].chunk_id for i in order if scores[i] > 0]

    def search(self, query: str, mode: str = "hybrid_rerank", top_k: int = 5, candidate_k: int = 20) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "vector":
            ranked = [(cid, 1.0 / (RRF_K + r)) for r, cid in enumerate(self.vector_ranking(query, top_k), 1)]
        elif mode == "bm25":
            ranked = [(cid, 1.0 / (RRF_K + r)) for r, cid in enumerate(self.bm25_ranking(query, top_k), 1)]
        else:
            ranked = rrf_fuse([self.vector_ranking(query, candidate_k), self.bm25_ranking(query, candidate_k)])
        hits = [Hit(self.by_id[cid], score) for cid, score in ranked]

        if mode == "hybrid_rerank":
            if self.reranker is None:
                raise RuntimeError("hybrid_rerank mode needs a reranker")
            hits = hits[:candidate_k]
            for hit, s in zip(hits, self.reranker.score(query, [h.chunk.embed_text for h in hits])):
                hit.rerank_score = s
            hits.sort(key=lambda h: h.rerank_score, reverse=True)
        return self._demote_rejected(hits)[:top_k]

    def _demote_rejected(self, hits: list[Hit]) -> list[Hit]:
        """Re-order so rejected-proposal chunks lose `rejected_penalty` against normative ones.

        After reranking the penalty is subtracted from the cross-encoder logit. RRF and single-retriever
        scores are rank-derived, so there it is applied in rank units (a rejected chunk drops `penalty`
        places, rounded up). Stored scores are left untouched: the relevance gate still sees the raw logit.
        """
        if self.rejected_penalty <= 0 or not hits:
            return hits
        p = self.rejected_penalty
        if all(h.rerank_score is not None for h in hits):
            def key(i_h):
                return -(i_h[1].rerank_score - (p if i_h[1].chunk.status == "rejected" else 0.0))
        else:
            def key(i_h):
                return i_h[0] + (math.ceil(p) + 0.5 if i_h[1].chunk.status == "rejected" else 0.0)
        return [h for _, h in sorted(enumerate(hits), key=key)]

    def score_relevance(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Fill rerank_score for hits that don't have one yet (used by the relevance gate)."""
        missing = [h for h in hits if h.rerank_score is None]
        if missing and self.reranker is not None:
            for hit, s in zip(missing, self.reranker.score(query, [h.chunk.embed_text for h in missing])):
                hit.rerank_score = s
        return hits
