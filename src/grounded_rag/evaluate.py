"""Offline evaluation.

retrieval  - doc-level Hit@1, Hit@5 and MRR@5 for every retrieval mode (no LLM needed).
e2e        - runs the full graph and scores answer correctness (required key terms present),
             citation precision (cited passages come from an expected document),
             refusal on unanswerable questions, false refusals on answerable ones, and latency.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path

from .config import DEFAULT_MODELS, ROOT, Settings
from .graph import RAGPipeline
from .retrieval import MODES, HybridIndex

QUESTIONS = ROOT / "eval" / "questions.jsonl"
RESULTS = ROOT / "eval" / "results"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _first_rank(doc_ids: list[str], expected: set[str]) -> int | None:
    for rank, d in enumerate(doc_ids, 1):
        if d in expected:
            return rank
    return None


def eval_retrieval(index: HybridIndex, questions: list[dict], settings: Settings, k: int = 5) -> dict:
    answerable = [q for q in questions if q["answerable"]]
    penalty = index.rejected_penalty
    runs = [(mode, mode, penalty) for mode in MODES]
    if penalty > 0:  # ablation row: the default mode with the rejected-proposal penalty switched off
        runs.append(("hybrid_rerank (no rejected penalty)", "hybrid_rerank", 0.0))
    report: dict[str, dict] = {}
    try:
        for label, mode, index.rejected_penalty in runs:
            ranks, rejected_top = [], []
            for q in answerable:
                hits = index.search(q["question"], mode=mode, top_k=k, candidate_k=settings.candidate_k)
                ranks.append(_first_rank([h.chunk.doc_id for h in hits], set(q["expected_docs"])))
                if hits and hits[0].chunk.status == "rejected":
                    rejected_top.append(q["id"])
            n = len(ranks)
            report[label] = {
                "hit@1": round(sum(r == 1 for r in ranks) / n, 3),
                f"hit@{k}": round(sum(r is not None for r in ranks) / n, 3),
                f"mrr@{k}": round(sum(1 / r for r in ranks if r) / n, 3),
                "misses": [q["id"] for q, r in zip(answerable, ranks) if r is None],
                # questions whose top passage comes from a rejected-proposal section
                "rejected@1": rejected_top,
            }
    finally:
        index.rejected_penalty = penalty
    return {"n_questions": len(answerable), "k": k, "rejected_penalty": penalty, "modes": report}


def eval_e2e(pipeline: RAGPipeline, questions: list[dict]) -> dict:
    rows = []
    for q in questions:
        out = pipeline.ask(q["question"])
        answer = out["answer"].lower()
        cited_docs = [c["doc_id"] for c in out["citations"]]
        row = {"id": q["id"], "answerable": q["answerable"], "status": out["status"],
               "latency_ms": out["latency_ms"], "answer": out["answer"], "cited_docs": cited_docs}
        if q["answerable"]:
            # must_include is a list of groups; every group needs at least one of its alternatives
            row["correct"] = out["status"] == "answered" and all(
                any(alt.lower() in answer for alt in group) for group in q["must_include"]
            )
            row["citation_precision"] = (
                sum(d in q["expected_docs"] for d in cited_docs) / len(cited_docs) if cited_docs else None
            )
        rows.append(row)

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    precisions = [r["citation_precision"] for r in ans if r["citation_precision"] is not None]
    latencies = sorted(r["latency_ms"] for r in rows)
    return {
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "answer_accuracy": round(sum(r["correct"] for r in ans) / len(ans), 3),
        "citation_precision": round(statistics.mean(precisions), 3) if precisions else None,
        "false_refusal_rate": round(sum(r["status"] != "answered" for r in ans) / len(ans), 3),
        "refusal_rate_unanswerable": round(sum(r["status"] != "answered" for r in unans) / len(unans), 3) if unans else None,
        "latency_ms_p50": latencies[len(latencies) // 2],
        "latency_ms_p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        "rows": rows,
    }


def to_markdown(retrieval: dict | None, e2e: dict | None, meta: dict) -> str:
    lines = [f"# Evaluation report ({meta['timestamp']})", "", f"- Corpus: {meta['corpus']}",
             f"- Embeddings: `{meta['embed_model']}` · Reranker: `{meta['rerank_model']}`", ""]
    if retrieval:
        k = retrieval["k"]
        lines += [f"## Retrieval ({retrieval['n_questions']} answerable questions, doc-level)", "",
                  f"Rejected-proposal penalty: {retrieval.get('rejected_penalty', 0.0)}. *Top from rejected*: questions "
                  + "whose first passage comes from a rejected-ideas section.", "",
                  f"| Mode | Hit@1 | Hit@{k} | MRR@{k} | Misses | Top from rejected |", "|---|---|---|---|---|---|"]
        for mode, m in retrieval["modes"].items():
            lines.append(f"| {mode} | {m['hit@1']:.3f} | {m[f'hit@{k}']:.3f} | {m[f'mrr@{k}']:.3f} | "
                         f"{', '.join(m['misses']) or '-'} | {', '.join(m.get('rejected@1', [])) or '-'} |")
        lines.append("")
    if e2e:
        lines += [f"## End-to-end (LLM: `{meta['llm']}`, mode: `{meta['retrieval_mode']}`)", "",
                  "| Metric | Value |", "|---|---|",
                  f"| Answer accuracy ({e2e['n_answerable']} answerable) | {e2e['answer_accuracy']:.3f} |",
                  f"| Citation precision | {e2e['citation_precision']} |",
                  f"| False-refusal rate | {e2e['false_refusal_rate']:.3f} |",
                  f"| Refusal rate on {e2e['n_unanswerable']} unanswerable | {e2e['refusal_rate_unanswerable']} |",
                  f"| Latency p50 / p95 | {e2e['latency_ms_p50']} ms / {e2e['latency_ms_p95']} ms |", "",
                  "### Failures", ""]
        for r in e2e["rows"]:
            bad = (r["answerable"] and not r["correct"]) or (not r["answerable"] and r["status"] == "answered")
            if bad:
                lines.append(f"- **{r['id']}** ({r['status']}): {r['answer'][:220]!r}")
        lines.append("")
    return "\n".join(lines)


def run(settings: Settings, index: HybridIndex, pipeline: RAGPipeline | None, do_retrieval: bool, do_e2e: bool,
        questions_path: Path = QUESTIONS) -> Path:
    questions = load_questions(questions_path)
    retrieval = eval_retrieval(index, questions, settings) if do_retrieval else None
    e2e = eval_e2e(pipeline, questions) if do_e2e and pipeline else None
    meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "corpus": f"{len({c.doc_id for c in index.chunks})} documents, {len(index.chunks)} chunks",
        "embed_model": settings.embed_model,
        "rerank_model": settings.rerank_model,
        "retrieval_mode": settings.retrieval_mode,
        "llm": "none" if pipeline is None or pipeline.llm is None
        else f"{settings.llm_provider}:{settings.llm_model or DEFAULT_MODELS.get(settings.llm_provider, '')}",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = "retrieval" if not do_e2e else meta["llm"].replace(":", "_").replace("/", "_")
    (RESULTS / f"{tag}.json").write_text(json.dumps({"meta": meta, "retrieval": retrieval, "e2e": e2e}, indent=2,
                                                    ensure_ascii=False), encoding="utf-8")
    md = RESULTS / f"{tag}.md"
    md.write_text(to_markdown(retrieval, e2e, meta), encoding="utf-8")
    return md
