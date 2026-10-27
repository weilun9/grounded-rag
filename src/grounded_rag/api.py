"""FastAPI service: POST /ask, GET /search, GET /health, and a single-page UI at /."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import DEFAULT_MODELS
from .graph import RAGPipeline
from .retrieval import MODES

WEB_DIR = Path(__file__).resolve().parent / "web"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    state: dict = {"pipeline": pipeline}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["pipeline"] is None:
            from .service import load_pipeline

            state["pipeline"] = load_pipeline()
        yield

    app = FastAPI(title="Grounded RAG", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        p: RAGPipeline = state["pipeline"]
        s = p.s
        return {
            "status": "ok",
            "chunks": len(p.index.chunks),
            "documents": len({c.doc_id for c in p.index.chunks}),
            "retrieval_mode": s.retrieval_mode,
            "llm": "none" if p.llm is None else f"{s.llm_provider}:{s.llm_model or DEFAULT_MODELS.get(s.llm_provider, '')}",
        }

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        try:
            return state["pipeline"].ask(req.question.strip())
        except Exception as exc:  # surface model/back-end outages as 503, not a stack trace
            raise HTTPException(status_code=503, detail=f"{type(exc).__name__}: {exc}") from exc

    @app.get("/search")
    def search(q: str = Query(..., min_length=2), mode: str = "hybrid_rerank", k: int = Query(5, ge=1, le=20)) -> list[dict]:
        if mode not in MODES:
            raise HTTPException(status_code=422, detail=f"mode must be one of {MODES}")
        p: RAGPipeline = state["pipeline"]
        hits = p.index.search(q, mode=mode, top_k=k, candidate_k=p.s.candidate_k)
        return [{"doc_id": h.chunk.doc_id, "title": h.chunk.title, "section": h.chunk.section,
                 "score": round(h.score, 4), "rerank_score": h.rerank_score, "text": h.chunk.text} for h in hits]

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    return app


app = create_app()  # the pipeline is loaded on startup, so importing this module stays cheap
