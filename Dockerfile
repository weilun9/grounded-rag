FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    TOKENIZERS_PARALLELISM=false \
    RAG_CORPUS_DIR=/app/data/corpus \
    RAG_INDEX_DIR=/app/data/index

WORKDIR /app

# CPU-only torch keeps the image ~1.5 GB instead of ~6 GB.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[anthropic,openai]"

# Bake the corpus, the models and the index into the image so the container starts ready.
COPY data/corpus ./data/corpus
RUN python -m grounded_rag ingest

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "grounded_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
