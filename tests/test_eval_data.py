"""The evaluation set and the published results are internally consistent."""

import json
import statistics

import pytest

from grounded_rag.config import ROOT
from grounded_rag.evaluate import load_questions
from grounded_rag.ingest import load_corpus

QUESTIONS = load_questions()
DOCS = {d.doc_id: d.body.lower() for d in load_corpus(ROOT / "data" / "corpus")}


def test_question_set_shape():
    assert len(QUESTIONS) == 45
    assert sum(q["answerable"] for q in QUESTIONS) == 35
    assert len({q["id"] for q in QUESTIONS}) == 45
    for q in QUESTIONS:
        assert bool(q["expected_docs"]) == q["answerable"] == bool(q["must_include"])
        assert set(q["expected_docs"]) <= set(DOCS)


@pytest.mark.parametrize("q", [q for q in QUESTIONS if q["answerable"]], ids=lambda q: q["id"])
def test_every_gold_key_term_occurs_in_a_labelled_source(q):
    for group in q["must_include"]:
        assert any(alt.lower() in DOCS[d] for alt in group for d in q["expected_docs"]), group


@pytest.mark.parametrize("name", ["ollama_qwen3.5_4b", "ollama_qwen3.5_9b"])
def test_published_e2e_metrics_match_their_rows(name):
    e = json.loads((ROOT / "eval" / "results" / f"{name}.json").read_text(encoding="utf-8"))["e2e"]
    ans = [r for r in e["rows"] if r["answerable"]]
    unans = [r for r in e["rows"] if not r["answerable"]]
    assert e["answer_accuracy"] == round(sum(r["correct"] for r in ans) / len(ans), 3)
    assert e["false_refusal_rate"] == round(sum(r["status"] != "answered" for r in ans) / len(ans), 3)
    assert e["refusal_rate_unanswerable"] == round(sum(r["status"] != "answered" for r in unans) / len(unans), 3)
    precisions = [r["citation_precision"] for r in ans if r["citation_precision"] is not None]
    assert e["citation_precision"] == round(statistics.mean(precisions), 3)
