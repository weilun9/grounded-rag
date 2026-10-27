"""Runtime settings, read from environment variables (a .env file is honoured)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path = field(default_factory=lambda: Path(_env("RAG_CORPUS_DIR", str(ROOT / "data" / "corpus"))))
    index_dir: Path = field(default_factory=lambda: Path(_env("RAG_INDEX_DIR", str(ROOT / "data" / "index"))))

    embed_model: str = field(default_factory=lambda: _env("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    rerank_model: str = field(default_factory=lambda: _env("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

    chunk_size: int = field(default_factory=lambda: int(_env("RAG_CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("RAG_CHUNK_OVERLAP", "150")))

    top_k: int = field(default_factory=lambda: int(_env("RAG_TOP_K", "5")))
    candidate_k: int = field(default_factory=lambda: int(_env("RAG_CANDIDATE_K", "20")))
    retrieval_mode: str = field(default_factory=lambda: _env("RAG_RETRIEVAL_MODE", "hybrid_rerank"))
    # Cross-encoder logit below which a passage is treated as irrelevant.
    relevance_threshold: float = field(default_factory=lambda: float(_env("RAG_RELEVANCE_THRESHOLD", "-2.0")))
    # Demotion for chunks from rejected/alternative-proposal sections in the hybrid modes, in cross-encoder
    # logits (in rank places for plain hybrid). 0 disables it.
    rejected_penalty: float = field(default_factory=lambda: float(_env("RAG_REJECTED_PENALTY", "2.0")))
    max_rewrites: int = field(default_factory=lambda: int(_env("RAG_MAX_REWRITES", "1")))

    # ollama | anthropic | openai | none (extractive, no LLM)
    llm_provider: str = field(default_factory=lambda: _env("RAG_LLM_PROVIDER", "ollama"))
    llm_model: str = field(default_factory=lambda: _env("RAG_LLM_MODEL", ""))
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))


DEFAULT_MODELS = {
    "ollama": "qwen3.5:4b",
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
}


def get_settings() -> Settings:
    return Settings()
