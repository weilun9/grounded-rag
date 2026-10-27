"""CLI:  python -m grounded_rag {ingest | ask "question" | search "query" | eval [--e2e | --refusals] | serve}"""

from __future__ import annotations

import argparse
import json
import sys

from .config import get_settings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="grounded_rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="(re)build the index from the corpus directory")
    a = sub.add_parser("ask", help="answer a question with citations")
    a.add_argument("question")
    a.add_argument("--json", action="store_true", help="print the full response, including the graph trace")
    s = sub.add_parser("search", help="show retrieved passages only")
    s.add_argument("query")
    s.add_argument("--mode", default=None)
    s.add_argument("-k", type=int, default=5)
    e = sub.add_parser("eval", help="run the evaluation set")
    e.add_argument("--e2e", action="store_true", help="also run the full pipeline with the configured LLM")
    e.add_argument("--retrieval-only", action="store_true")
    e.add_argument("--refusals", action="store_true",
                   help="only the unanswerable questions: count refusals by the gate vs the model")
    v = sub.add_parser("serve", help="start the API + web UI")
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--host", default="127.0.0.1")
    args = p.parse_args(argv)
    settings = get_settings()

    if args.cmd == "ingest":
        from .service import build_index

        _, stats = build_index(settings)
        print(f"Indexed {stats['documents']} documents into {stats['chunks']} chunks -> {stats['index_dir']}")
    elif args.cmd == "ask":
        from .service import load_pipeline

        out = load_pipeline(settings).ask(args.question)
        if args.json:
            print(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            print(out["answer"], "\n")
            for c in out["citations"]:
                print(f"  [{c['n']}] {c['title']} > {c['section']}")
            print(f"\n  ({out['status']}, {out['latency_ms']} ms)")
    elif args.cmd == "search":
        from .service import load_index

        for h in load_index(settings).search(args.query, mode=args.mode or settings.retrieval_mode, top_k=args.k,
                                             candidate_k=settings.candidate_k):
            rr = "" if h.rerank_score is None else f" rerank={h.rerank_score:.2f}"
            print(f"{h.chunk.doc_id:>9} | {h.chunk.section[:40]:<40} | rrf={h.score:.4f}{rr}")
    elif args.cmd == "eval":
        from . import evaluate
        from .service import load_index, load_pipeline

        if args.refusals:
            print(evaluate.run_refusals(settings, load_pipeline(settings)).read_text(encoding="utf-8"))
            return 0
        do_e2e = args.e2e and not args.retrieval_only
        pipeline = load_pipeline(settings) if do_e2e else None
        index = pipeline.index if pipeline else load_index(settings)
        report = evaluate.run(settings, index, pipeline, do_retrieval=True, do_e2e=do_e2e)
        print(report.read_text(encoding="utf-8"))
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("grounded_rag.api:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
