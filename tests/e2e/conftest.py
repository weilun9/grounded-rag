"""Browser tests: a real uvicorn server on a free port, driven by headless Chrome through Selenium.

The pipeline is the same model-free one the unit tests use (hash embeddings, token-overlap reranker), so the
browser tests need no downloads and the answers are deterministic. A scripted LLM can be plugged in per test.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
import uvicorn
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from grounded_rag.api import create_app
from grounded_rag.graph import RAGPipeline


class SlowFakeLLM(FakeListChatModel):
    """Scripted replies with a delay, so the test can observe the UI while a request is in flight."""

    delay_s: float = 0.0

    def _call(self, *args, **kwargs):
        time.sleep(self.delay_s)
        return super()._call(*args, **kwargs)


class BrokenLLM(FakeListChatModel):
    """Simulates the model back-end being down."""

    def _call(self, *args, **kwargs):
        raise ConnectionError("model server unreachable")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def _serve(pipeline: RAGPipeline) -> Iterator[str]:
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(pipeline), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture
def serve(index, settings) -> Iterator[Callable[..., str]]:
    """serve(llm=None) starts the app with the given chat model and returns its base URL."""
    stack: list = []

    def start(llm=None) -> str:
        cm = _serve(RAGPipeline(index, llm, settings))
        stack.append(cm)
        return cm.__enter__()

    yield start
    for cm in reversed(stack):
        cm.__exit__(None, None, None)


@pytest.fixture(scope="session")
def driver():
    webdriver = pytest.importorskip("selenium.webdriver")
    opts = webdriver.ChromeOptions()
    if os.environ.get("E2E_HEADED") != "1":
        opts.add_argument("--headless=new")
    for arg in ("--no-sandbox", "--disable-dev-shm-usage", "--window-size=1280,900"):
        opts.add_argument(arg)
    d = webdriver.Chrome(options=opts)  # Selenium Manager resolves a matching chromedriver
    yield d
    d.quit()


@pytest.fixture(autouse=True)
def _screenshot_on_failure(request, driver):
    yield
    rep = getattr(request.node, "rep_call", None)
    if rep is not None and rep.failed:
        os.makedirs("e2e-artifacts", exist_ok=True)
        driver.save_screenshot(f"e2e-artifacts/{request.node.name}.png")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    rep = yield
    if rep.when == "call":
        item.rep_call = rep
    return rep
