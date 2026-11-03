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


# The answer lives in "Specification"; a rejected spelling sits under "Rejected Ideas" one heading level down
# and repeats the question's wording more closely, so it wins on raw relevance scores.
REJECTED_DOC = Document("PEP 9001", "PEP 9001 – Inline Assignment", """Specification
=============

Use the walrus operator ``:=`` to assign a value to a name inside an expression, such as a while loop test.

Rejected Ideas
==============

Using ``as`` in conditions
--------------------------

Writing ``if value as name:`` to assign a value inside an if condition was proposed and turned down.

Deferred Reference Counting
===========================

A design term for reference counting that is postponed until collection, which is part of the proposal.
""", "pep-9001.rst")


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
def rejected_chunks():
    return chunk_documents([REJECTED_DOC, *DOCS], chunk_size=300, chunk_overlap=0)


def make_index(chunks, rejected_penalty: float = 0.0) -> HybridIndex:
    return HybridIndex(chunks, DeterministicFakeEmbedding(size=64), persist_dir=None, reranker=OverlapReranker(),
                       rejected_penalty=rejected_penalty)


@pytest.fixture
def settings():
    return Settings(llm_provider="none", top_k=3, candidate_k=10, relevance_threshold=0.0, max_rewrites=1)


def make_pipeline(index, settings, replies: list[str] | None):
    llm = FakeListChatModel(responses=replies) if replies is not None else None
    return RAGPipeline(index, llm, settings)
