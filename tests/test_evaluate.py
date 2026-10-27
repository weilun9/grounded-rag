from grounded_rag.evaluate import eval_refusals, refusal_breakdown, refusals_markdown

from .conftest import make_pipeline


def test_refusal_breakdown_attributes_each_outcome():
    rows = [
        {"status": "refused", "refused_by": "gate", "gate_best": [-5.0]},
        {"status": "refused", "refused_by": "gate", "gate_best": [-5.0, -1.0]},
        {"status": "refused", "refused_by": "model", "gate_best": [1.2]},
        {"status": "ungrounded", "refused_by": "verifier", "gate_best": [0.5]},
        {"status": "answered", "refused_by": None, "gate_best": [3.5]},
    ]
    assert refusal_breakdown(rows) == {"gate": 1, "gate_after_rewrite": 1, "model": 1, "verifier": 1, "answered": 1}


def test_eval_refusals_runs_without_an_llm(index, settings):
    questions = [
        {"id": "u1", "answerable": False, "question": "What is the capital of New Zealand?"},
        {"id": "u2", "answerable": False, "question": "What is the maximum line length in Java?"},
        {"id": "a1", "answerable": True, "question": "How many spaces per indentation level?"},
    ]
    report = eval_refusals(make_pipeline(index, settings, None), questions)
    assert report["n_unanswerable"] == 2
    # no LLM: no rewrite and no model refusal, so the gate either refuses on the first pass or the question leaks
    assert report["outcomes"] == {"gate": 1, "gate_after_rewrite": 0, "model": 0, "verifier": 0, "answered": 1}
    md = refusals_markdown(report, {"timestamp": "t", "llm": "none", "retrieval_mode": "hybrid_rerank"})
    assert "| u1 |" in md and "**Answered (leak)** | 1" in md
