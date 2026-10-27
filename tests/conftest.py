"""Fixtures that need no model downloads: hash embeddings, a token-overlap reranker and scripted LLM replies."""

from __future__ import annotations

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from grounded_rag.config import Settings
from grounded_rag.graph import RAGPipeline
from grounded_rag.ingest import Document, chunk_documents
from grounded_rag.retrieval import HybridIndex, tokenize

DOCS = [
    Document("PEP 572", "PEP 572 – Assignment Expressions", """Abstract
========

This is a proposal for creating a way to assign to variables within an expression using
the notation ``NAME := expr``. The new operator is informally known as the walrus operator.

Rationale
=========

Naming the result of an expression is an important part of programming, allowing a descriptive
name to be used in place of a longer expression, and permitting reuse.
""", "pep-0572.rst"),
    Document("PEP 8", "PEP 8 – Style Guide for Python Code", """Indentation
===========

Use 4 spaces per indentation level. Continuation lines should align wrapped elements vertically.

Maximum Line Length
===================

Limit all lines to a maximum of 79 characters. For docstrings or comments the limit is 72 characters.
""", "pep-0008.rst"),
]


class OverlapReranker:
    """Score = number of shared query tokens minus 2, so a passage with no overlap scores below the default gate."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = set(tokenize(query))
        return [float(len(q & set(tokenize(p)))) - 2.0 for p in passages]


@pytest.fixture
def chunks():
    return chunk_documents(DOCS, chunk_size=300, chunk_overlap=0)


@pytest.fixture
def index(chunks):
    return HybridIndex(chunks, DeterministicFakeEmbedding(size=64), persist_dir=None, reranker=OverlapReranker())


@pytest.fixture
def settings():
    return Settings(llm_provider="none", top_k=3, candidate_k=10, relevance_threshold=0.0, rewrite_threshold=2.0,
                    max_rewrites=1)


def make_pipeline(index, settings, replies: list[str] | None):
    llm = FakeListChatModel(responses=replies) if replies is not None else None
    return RAGPipeline(index, llm, settings)
