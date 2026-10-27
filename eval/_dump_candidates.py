"""TEMPORARY: dump reranked candidates per question so the rejected penalty can be tuned offline."""
import json

from grounded_rag.config import get_settings
from grounded_rag.evaluate import load_questions
from grounded_rag.service import load_index

s = get_settings()
idx = load_index(s)
idx.rejected_penalty = 0.0
for q in load_questions():
    if not q["answerable"]:
        continue
    row = {"id": q["id"], "exp": q["expected_docs"]}
    for mode, k in (("vector", 5), ("bm25", 5), ("hybrid", 20), ("hybrid_rerank", 20)):
        hits = idx.search(q["question"], mode=mode, top_k=k, candidate_k=s.candidate_k)
        row[mode] = [[h.chunk.doc_id, h.chunk.status[0], round(h.rerank_score if h.rerank_score is not None else h.score, 4),
                      h.chunk.section[:30]] for h in hits]
    print("DUMP " + json.dumps(row))
