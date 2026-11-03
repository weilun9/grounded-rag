"""End-to-end checks of the web UI in a real browser: what a user sees for each pipeline outcome.

The API tests already cover the JSON; these cover what only a browser shows: the JS rendering, HTML escaping of
model output, the in-flight state, deep links and the mobile layout.
"""

import pytest

pytest.importorskip("selenium")  # the unit-test CI job does not install the e2e extra

from selenium.webdriver.common.by import By

from grounded_rag.graph import REFUSAL

from .conftest import BrokenLLM, SlowFakeLLM
from .pages import AskPage

pytestmark = pytest.mark.e2e


def test_health_line_is_filled_from_the_api(driver, serve, index):
    page = AskPage(driver, serve()).open()
    assert page.health_text == f"2 documents · {len(index.chunks)} chunks · retrieval: hybrid_rerank · LLM: none"


def test_answer_shows_badge_and_expandable_citations(driver, serve):
    page = AskPage(driver, serve()).open()
    r = page.ask("How many spaces per indentation level does PEP 8 recommend?", submit_with_enter=True).result()
    assert r.status == "answered"
    assert "Use 4 spaces per indentation level" in r.answer
    first = r.citations[0]
    assert first.find_element(By.TAG_NAME, "summary").text == "[1] PEP 8 – Style Guide for Python Code › Indentation"
    passage = first.find_element(By.TAG_NAME, "p")
    assert not passage.is_displayed()  # collapsed until clicked
    first.find_element(By.TAG_NAME, "summary").click()
    assert passage.is_displayed() and "4 spaces" in passage.text
    assert "verify -> answered" in r.trace


def test_off_corpus_example_is_refused_without_citations(driver, serve):
    page = AskPage(driver, serve()).open()
    r = page.click_example("What is the capital of New Zealand?").result()
    assert r.status == "refused"
    assert r.answer == REFUSAL
    assert r.citations == []
    assert "generate" not in r.trace  # the relevance gate refused before any generation


def test_uncited_model_answer_is_shown_as_ungrounded_refusal(driver, serve):
    page = AskPage(driver, serve(SlowFakeLLM(responses=["Use four spaces."]))).open()
    r = page.ask("How many spaces per indentation level does PEP 8 recommend?").result()
    assert r.status == "ungrounded"
    assert r.answer == REFUSAL
    assert r.citations == []


def test_model_output_is_rendered_as_text_not_html(driver, serve):
    payload = '<img src=x onerror="window.__xss=1"><b>bold</b> Use 4 spaces [1].'
    page = AskPage(driver, serve(SlowFakeLLM(responses=[payload]))).open()
    r = page.ask("How many spaces per indentation level does PEP 8 recommend?").result()
    assert r.status == "answered"
    assert '<img src=x onerror="window.__xss=1">' in r.answer  # shown literally
    out = driver.find_element(By.ID, "out")
    assert out.find_elements(By.TAG_NAME, "img") == [] and out.find_elements(By.TAG_NAME, "b") == []
    assert driver.execute_script("return window.__xss") is None


def test_button_is_disabled_while_the_request_is_in_flight(driver, serve):
    page = AskPage(driver, serve(SlowFakeLLM(responses=["Use 4 spaces [1]."], delay_s=1.5))).open()
    page.ask("How many spaces per indentation level does PEP 8 recommend?")
    assert not page.ask_button.is_enabled()
    assert driver.find_element(By.ID, "out").text == "Thinking…"
    assert page.result().status == "answered"
    assert page.ask_button.is_enabled()


def test_backend_failure_shows_an_error_and_recovers(driver, serve):
    page = AskPage(driver, serve(BrokenLLM(responses=["unused"]))).open()
    r = page.ask("How many spaces per indentation level does PEP 8 recommend?").result()
    assert r.status == "error"
    assert "ConnectionError: model server unreachable" in r.answer
    assert page.ask_button.is_enabled()


def test_query_parameter_asks_immediately(driver, serve):
    page = AskPage(driver, serve()).open(query="What does the walrus operator := do?")
    r = page.result()
    assert driver.find_element(By.ID, "q").get_attribute("value") == "What does the walrus operator := do?"
    assert r.status == "answered"
    assert "PEP 572" in r.citations[0].text


def test_too_short_question_is_blocked_by_the_browser(driver, serve):
    page = AskPage(driver, serve()).open()
    page.ask("ab")
    is_valid = driver.execute_script("return document.getElementById('q').checkValidity()")
    assert is_valid is False
    assert driver.find_element(By.ID, "out").text == ""  # nothing was sent


def test_mobile_width_has_no_horizontal_scroll(driver, serve):
    page = AskPage(driver, serve())
    driver.set_window_size(375, 812)
    try:
        page.open()
        page.ask("How many spaces per indentation level does PEP 8 recommend?").result()
        overflow = driver.execute_script(
            "return document.documentElement.scrollWidth - document.documentElement.clientWidth")
        assert overflow <= 0
    finally:
        driver.set_window_size(1280, 900)
