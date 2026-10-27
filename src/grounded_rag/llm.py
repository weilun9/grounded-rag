"""Chat model factory. Every provider is behind the same LangChain chat-model interface."""

from __future__ import annotations

import re

from langchain_core.language_models.chat_models import BaseChatModel

from .config import DEFAULT_MODELS, Settings

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def get_chat_model(settings: Settings) -> BaseChatModel | None:
    provider = settings.llm_provider.lower()
    model = settings.llm_model or DEFAULT_MODELS.get(provider, "")
    if provider == "none":
        return None
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, base_url=settings.ollama_base_url, temperature=0, reasoning=False)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=0, max_tokens=800)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=0)
    raise ValueError(f"Unknown RAG_LLM_PROVIDER: {settings.llm_provider!r}")


def text_of(message) -> str:
    """Plain text from a chat response, with any <think> block removed."""
    content = message.content if hasattr(message, "content") else message
    if isinstance(content, list):  # Anthropic content blocks
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return _THINK.sub("", str(content)).strip()
