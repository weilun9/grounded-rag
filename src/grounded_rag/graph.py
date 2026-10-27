"""The answer pipeline as a LangGraph state machine.

    retrieve -> grade --(relevant passages)--> generate -> verify -> END
                  |
                  +--(none relevant, rewrite budget left)--> rewrite -> retrieve
                  +--(none relevant, budget spent)---------> refuse  -> END

Two guardrails sit outside the model:
  * the relevance gate (cross-encoder score) decides whether the model is asked at all,
    so off-corpus questions are refused without a generation call;
  * the verifier fails closed: an answer with no valid [n] citation is replaced by a refusal.
"""

from __future__ import annotations

import re
import time
from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .config import Settings
from .llm import text_of
from .retrieval import Hit, HybridIndex

REFUSAL = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered passages provided.
Rules:
- Every sentence that states a fact must end with a citation like [1] or [2][3], pointing at the passage(s) that support it.
- Do not use outside knowledge. If the passages do not contain the answer, reply exactly: {REFUSAL}
- Be concise: at most 4 sentences. Quote code or identifiers exactly as they appear."""

REWRITE_PROMPT = """Rewrite the user's question into a short search query for a keyword + semantic search engine
over Python Enhancement Proposals (PEPs). Expand abbreviations, add likely technical terms, and drop filler words.
Return only the query, nothing else.

Question: {question}"""

_CITATION = re.compile(r"\[(\d+)\]")


class RAGState(TypedDict, total=False):
    question: str
    query: str
    hits: list[Hit]
    relevant: list[Hit]
    rewrites: int
    answer: str
    citations: list[dict]
    status: str  # answered | refused | ungrounded
    trace: list[str]


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.chunk.title} > {h.chunk.section})\n{h.chunk.text}" for i, h in enumerate(hits, 1))


def verify_citations(answer: str, n_passages: int) -> tuple[str, list[int], str]:
    """Return (cleaned_answer, cited_indices, status). Invalid [n] markers are stripped."""
    if REFUSAL.lower().rstrip(".") in answer.lower():
        return REFUSAL, [], "refused"
    cited: list[int] = []
    for m in _CITATION.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= n_passages and n not in cited:
            cited.append(n)
    cleaned = _CITATION.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_passages else "", answer).strip()
    if not cited:
        return REFUSAL, [], "ungrounded"
    return cleaned, cited, "answered"


class RAGPipeline:
    def __init__(self, index: HybridIndex, llm: BaseChatModel | None, settings: Settings):
        self.index = index
        self.llm = llm
        self.s = settings
        self.graph = self._build()

    # ---- nodes ----------------------------------------------------------------------

    def _retrieve(self, state: RAGState) -> RAGState:
        hits = self.index.search(state["query"], mode=self.s.retrieval_mode,
                                 top_k=self.s.top_k, candidate_k=self.s.candidate_k)
        return {"hits": hits, "trace": state.get("trace", []) + [f"retrieve({state['query']!r}) -> {len(hits)}"]}

    def _grade(self, state: RAGState) -> RAGState:
        hits = state["hits"]
        if self.index.reranker is None:
            relevant = hits
        else:
            self.index.score_relevance(state["question"], hits)
            relevant = [h for h in hits if h.rerank_score is not None and h.rerank_score >= self.s.relevance_threshold]
        return {"relevant": relevant, "trace": state["trace"] + [f"grade -> {len(relevant)}/{len(hits)} relevant"]}

    def _route_after_grade(self, state: RAGState) -> str:
        if state["relevant"]:
            return "generate"
        if self.llm is not None and state.get("rewrites", 0) < self.s.max_rewrites:
            return "rewrite"
        return "refuse"

    def _rewrite(self, state: RAGState) -> RAGState:
        query = text_of(self.llm.invoke(REWRITE_PROMPT.format(question=state["question"]))).splitlines()[0].strip()
        return {"query": query or state["question"], "rewrites": state.get("rewrites", 0) + 1,
                "trace": state["trace"] + [f"rewrite -> {query!r}"]}

    def _generate(self, state: RAGState) -> RAGState:
        hits = state["relevant"]
        if self.llm is None:  # extractive fallback: the best passages, verbatim, with citations
            answer = " ".join(f"{h.chunk.text[:400].strip()} [{i}]" for i, h in enumerate(hits[:2], 1))
        else:
            messages = [SystemMessage(SYSTEM_PROMPT),
                        HumanMessage(f"Passages:\n\n{format_context(hits)}\n\nQuestion: {state['question']}")]
            answer = text_of(self.llm.invoke(messages))
        return {"answer": answer, "trace": state["trace"] + ["generate"]}

    def _verify(self, state: RAGState) -> RAGState:
        hits = state["relevant"]
        answer, cited, status = verify_citations(state["answer"], len(hits))
        citations = [
            {"n": n, "doc_id": hits[n - 1].chunk.doc_id, "title": hits[n - 1].chunk.title,
             "section": hits[n - 1].chunk.section, "chunk_id": hits[n - 1].chunk.chunk_id,
             "text": hits[n - 1].chunk.text}
            for n in cited
        ]
        return {"answer": answer, "citations": citations, "status": status,
                "trace": state["trace"] + [f"verify -> {status}"]}

    def _refuse(self, state: RAGState) -> RAGState:
        return {"answer": REFUSAL, "citations": [], "status": "refused", "trace": state["trace"] + ["refuse"]}

    # ---- wiring ---------------------------------------------------------------------

    def _build(self):
        g = StateGraph(RAGState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade", self._grade)
        g.add_node("rewrite", self._rewrite)
        g.add_node("generate", self._generate)
        g.add_node("verify", self._verify)
        g.add_node("refuse", self._refuse)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self._route_after_grade,
                                {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"})
        g.add_edge("rewrite", "retrieve")
        g.add_edge("generate", "verify")
        g.add_edge("verify", END)
        g.add_edge("refuse", END)
        return g.compile()

    def ask(self, question: str) -> dict:
        t0 = time.perf_counter()
        out = self.graph.invoke({"question": question, "query": question, "rewrites": 0, "trace": []})
        return {
            "question": question,
            "answer": out["answer"],
            "status": out["status"],
            "citations": out.get("citations", []),
            "retrieved": [
                {"doc_id": h.chunk.doc_id, "section": h.chunk.section, "chunk_id": h.chunk.chunk_id,
                 "score": round(h.score, 4),
                 "rerank_score": None if h.rerank_score is None else round(h.rerank_score, 3)}
                for h in out.get("hits", [])
            ],
            "trace": out["trace"],
            "latency_ms": round((time.perf_counter() - t0) * 1000),
        }
