from grounded_rag.evaluate import eval_retrieval

from .conftest import make_index
from .test_retrieval import QUESTION

QUESTIONS = [{"id": "q1", "answerable": True, "question": QUESTION, "expected_docs": ["PEP 9001"]}]


def test_retrieval_eval_reports_rejected_top_passages_with_and_without_penalty(rejected_chunks, settings):
    index = make_index(rejected_chunks, 3.0)
    report = eval_retrieval(index, QUESTIONS, settings)
    assert report["rejected_penalty"] == 3.0 and index.rejected_penalty == 3.0  # restored after the ablation row
    assert report["modes"]["hybrid_rerank"]["rejected@1"] == []
    assert report["modes"]["hybrid_rerank (no rejected penalty)"]["rejected@1"] == ["q1"]


def test_no_ablation_row_when_penalty_is_off(rejected_chunks, settings):
    report = eval_retrieval(make_index(rejected_chunks, 0.0), QUESTIONS, settings)
    assert "hybrid_rerank (no rejected penalty)" not in report["modes"]
