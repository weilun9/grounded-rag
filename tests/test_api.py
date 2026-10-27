from fastapi.testclient import TestClient

from grounded_rag.api import create_app

from .conftest import make_pipeline


def client(index, settings, replies):
    return TestClient(create_app(make_pipeline(index, settings, replies)))


def test_health_reports_corpus_size(index, settings):
    with client(index, settings, None) as c:
        body = c.get("/health").json()
    assert body["documents"] == 2 and body["llm"] == "none"


def test_ask_returns_answer_citations_and_trace(index, settings):
    with client(index, settings, ["Use 4 spaces per indentation level [1]."]) as c:
        body = c.post("/ask", json={"question": "How many spaces per indentation level?"}).json()
    assert body["status"] == "answered"
    assert body["citations"][0]["section"] == "Indentation"
    assert body["trace"]


def test_ask_validates_input(index, settings):
    with client(index, settings, None) as c:
        assert c.post("/ask", json={"question": "x"}).status_code == 422


def test_search_rejects_unknown_mode(index, settings):
    with client(index, settings, None) as c:
        assert c.get("/search", params={"q": "spaces", "mode": "nope"}).status_code == 422


def test_ui_is_served(index, settings):
    with client(index, settings, None) as c:
        r = c.get("/")
    assert r.status_code == 200 and "Grounded RAG" in r.text
