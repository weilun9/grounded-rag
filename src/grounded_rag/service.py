"""Wire settings -> models -> index -> pipeline. Shared by the CLI, the API and the evaluator."""

from __future__ import annotations

from functools import cache

from .config import Settings, get_settings
from .graph import RAGPipeline
from .ingest import chunk_documents, load_corpus
from .llm import get_chat_model
from .retrieval import CrossEncoderReranker, HybridIndex, SentenceTransformerEmbeddings


@cache
def _embeddings(name: str) -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(name)


@cache
def _reranker(name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(name)


def build_index(settings: Settings | None = None) -> tuple[HybridIndex, dict]:
    s = settings or get_settings()
    docs = load_corpus(s.corpus_dir)
    if not docs:
        raise SystemExit(f"No documents found in {s.corpus_dir}")
    chunks = chunk_documents(docs, s.chunk_size, s.chunk_overlap)
    index = HybridIndex.build(chunks, _embeddings(s.embed_model), s.index_dir, reranker=_reranker(s.rerank_model))
    return index, {"documents": len(docs), "chunks": len(chunks), "index_dir": str(s.index_dir)}


def load_index(settings: Settings | None = None) -> HybridIndex:
    s = settings or get_settings()
    return HybridIndex.load(s.index_dir, _embeddings(s.embed_model), reranker=_reranker(s.rerank_model))


def load_pipeline(settings: Settings | None = None) -> RAGPipeline:
    s = settings or get_settings()
    return RAGPipeline(load_index(s), get_chat_model(s), s)
