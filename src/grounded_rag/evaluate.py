"""Offline evaluation.

retrieval  - doc-level Hit@1, Hit@5 and MRR@5 for every retrieval mode (no LLM needed).
e2e        - runs the full graph and scores answer correctness (required key terms present),
             citation precision (cited passages come from an expected document),
             refusal on unanswerable questions, false refusals on answerable ones, and latency.
refusals   - unanswerable questions only: who refused each one (relevance gate on the first pass, gate after a
             rewrite, the model's own "I don't know", or the citation verifier) and the gate scores behind it.
             Runs with RAG_LLM_PROVIDER=none (first-pass gate only: no LLM means no rewrite and no model refusal).
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
    report: dict[str, dict] = {}
    for mode in MODES:
        ranks = []
        for q in answerable:
            hits = index.search(q["question"], mode=mode, top_k=k, candidate_k=settings.candidate_k)
            ranks.append(_first_rank([h.chunk.doc_id for h in hits], set(q["expected_docs"])))
        n = len(ranks)
        report[mode] = {
            "hit@1": round(sum(r == 1 for r in ranks) / n, 3),
            f"hit@{k}": round(sum(r is not None for r in ranks) / n, 3),
            f"mrr@{k}": round(sum(1 / r for r in ranks if r) / n, 3),
            "misses": [q["id"] for q, r in zip(answerable, ranks) if r is None],
        }
    return {"n_questions": len(answerable), "k": k, "modes": report}


def eval_e2e(pipeline: RAGPipeline, questions: list[dict]) -> dict:
    rows = []
    for q in questions:
        out = pipeline.ask(q["question"])
        answer = out["answer"].lower()
        cited_docs = [c["doc_id"] for c in out["citations"]]
        row = {"id": q["id"], "answerable": q["answerable"], "status": out["status"],
               "refused_by": out.get("refused_by"), "gate_best": out.get("gate_best", []),
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
        "unanswerable_outcomes": refusal_breakdown(unans),
        "latency_ms_p50": latencies[len(latencies) // 2],
        "latency_ms_p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        "rows": rows,
    }


def refusal_outcome(row: dict) -> str:
    """gate | gate_after_rewrite | model | verifier | answered"""
    if row["status"] == "answered":
        return "answered"
    if row.get("refused_by") == "gate":
        return "gate_after_rewrite" if len(row.get("gate_best", [])) > 1 else "gate"
    return row.get("refused_by") or "model"


OUTCOMES = ("gate", "gate_after_rewrite", "model", "verifier", "answered")


def refusal_breakdown(rows: list[dict]) -> dict[str, int]:
    counts = dict.fromkeys(OUTCOMES, 0)
    for r in rows:
        counts[refusal_outcome(r)] += 1
    return counts


def eval_refusals(pipeline: RAGPipeline, questions: list[dict]) -> dict:
    rows = []
    for q in questions:
        if q["answerable"]:
            continue
        out = pipeline.ask(q["question"])
        row = {"id": q["id"], "question": q["question"], "status": out["status"], "refused_by": out["refused_by"],
               "gate_best": out["gate_best"], "trace": out["trace"]}
        row["outcome"] = refusal_outcome(row)
        rows.append(row)
    return {"n_unanswerable": len(rows), "relevance_threshold": pipeline.s.relevance_threshold,
            "rewrite_threshold": pipeline.s.rewrite_threshold, "outcomes": refusal_breakdown(rows), "rows": rows}


def refusals_markdown(report: dict, meta: dict) -> str:
    def fmt(v: float | None) -> str:
        return "-" if v is None else f"{v:.2f}"

    o = report["outcomes"]
    lines = [f"# Refusal attribution ({meta['timestamp']})", "",
             (f"- LLM: `{meta['llm']}` · mode: `{meta['retrieval_mode']}` · gate threshold "
              f"{report['relevance_threshold']} (first pass) / {report['rewrite_threshold']} (after rewrite)"), "",
             f"| Outcome ({report['n_unanswerable']} unanswerable) | Count |", "|---|---|",
             f"| Refused by gate, first pass (no LLM call) | {o['gate']} |",
             f"| Refused by gate after rewrite (rewrite call only) | {o['gate_after_rewrite']} |",
             f"| Refused by the model | {o['model']} |",
             f"| Refused by the citation verifier | {o['verifier']} |",
             f"| **Answered (leak)** | {o['answered']} |", "",
             "| Id | Question | Best score, first pass | Best score, after rewrite | Outcome |",
             "|---|---|---|---|---|"]
    for r in report["rows"]:
        g = r["gate_best"] + [None] * (2 - len(r["gate_best"]))
        lines.append(f"| {r['id']} | {r['question']} | {fmt(g[0])} | {fmt(g[1])} | {r['outcome']} |")
    lines.append("")
    return "\n".join(lines)


def to_markdown(retrieval: dict | None, e2e: dict | None, meta: dict) -> str:
    lines = [f"# Evaluation report ({meta['timestamp']})", "", f"- Corpus: {meta['corpus']}",
             f"- Embeddings: `{meta['embed_model']}` · Reranker: `{meta['rerank_model']}`", ""]
    if retrieval:
        k = retrieval["k"]
        lines += [f"## Retrieval ({retrieval['n_questions']} answerable questions, doc-level)", "",
                  f"| Mode | Hit@1 | Hit@{k} | MRR@{k} | Misses |", "|---|---|---|---|---|"]
        for mode, m in retrieval["modes"].items():
            lines.append(f"| {mode} | {m['hit@1']:.3f} | {m[f'hit@{k}']:.3f} | {m[f'mrr@{k}']:.3f} | {', '.join(m['misses']) or '-'} |")
        lines.append("")
    if e2e:
        lines += [f"## End-to-end (LLM: `{meta['llm']}`, mode: `{meta['retrieval_mode']}`)", "",
                  "| Metric | Value |", "|---|---|",
                  f"| Answer accuracy ({e2e['n_answerable']} answerable) | {e2e['answer_accuracy']:.3f} |",
                  f"| Citation precision | {e2e['citation_precision']} |",
                  f"| False-refusal rate | {e2e['false_refusal_rate']:.3f} |",
                  f"| Refusal rate on {e2e['n_unanswerable']} unanswerable | {e2e['refusal_rate_unanswerable']} |",
                  "| Unanswerable refused by gate / gate after rewrite / model / verifier | "
                  + " / ".join(str(e2e["unanswerable_outcomes"][k]) for k in OUTCOMES[:4]) + " |",
                  f"| Latency p50 / p95 | {e2e['latency_ms_p50']} ms / {e2e['latency_ms_p95']} ms |", "",
                  "### Failures", ""]
        for r in e2e["rows"]:
            bad = (r["answerable"] and not r["correct"]) or (not r["answerable"] and r["status"] == "answered")
            if bad:
                lines.append(f"- **{r['id']}** ({r['status']}): {r['answer'][:220]!r}")
        lines.append("")
    return "\n".join(lines)


def _meta(settings: Settings, index: HybridIndex, pipeline: RAGPipeline | None) -> dict:
    return {
        "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"),
        "corpus": f"{len({c.doc_id for c in index.chunks})} documents, {len(index.chunks)} chunks",
        "embed_model": settings.embed_model,
        "rerank_model": settings.rerank_model,
        "retrieval_mode": settings.retrieval_mode,
        "llm": "none" if pipeline is None or pipeline.llm is None
        else f"{settings.llm_provider}:{settings.llm_model or DEFAULT_MODELS.get(settings.llm_provider, '')}",
    }


def run(settings: Settings, index: HybridIndex, pipeline: RAGPipeline | None, do_retrieval: bool, do_e2e: bool,
        questions_path: Path = QUESTIONS) -> Path:
    questions = load_questions(questions_path)
    retrieval = eval_retrieval(index, questions, settings) if do_retrieval else None
    e2e = eval_e2e(pipeline, questions) if do_e2e and pipeline else None
    meta = _meta(settings, index, pipeline)
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = "retrieval" if not do_e2e else meta["llm"].replace(":", "_").replace("/", "_")
    (RESULTS / f"{tag}.json").write_text(json.dumps({"meta": meta, "retrieval": retrieval, "e2e": e2e}, indent=2,
                                                    ensure_ascii=False), encoding="utf-8")
    md = RESULTS / f"{tag}.md"
    md.write_text(to_markdown(retrieval, e2e, meta), encoding="utf-8")
    return md


def run_refusals(settings: Settings, pipeline: RAGPipeline, questions_path: Path = QUESTIONS) -> Path:
    report = eval_refusals(pipeline, load_questions(questions_path))
    meta = _meta(settings, pipeline.index, pipeline)
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = "refusals_" + meta["llm"].replace(":", "_").replace("/", "_")
    (RESULTS / f"{tag}.json").write_text(json.dumps({"meta": meta, "refusals": report}, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    md = RESULTS / f"{tag}.md"
    md.write_text(refusals_markdown(report, meta), encoding="utf-8")
    return md
