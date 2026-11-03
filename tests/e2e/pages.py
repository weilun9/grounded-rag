"""Page object for the single-page UI. Tests talk to this, not to CSS selectors."""

from __future__ import annotations

from dataclasses import dataclass

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

TIMEOUT_S = 10


@dataclass
class Result:
    status: str
    answer: str
    citations: list[WebElement]
    trace: str


class AskPage:
    def __init__(self, driver: WebDriver, base_url: str):
        self.d = driver
        self.base_url = base_url
        self.wait = WebDriverWait(driver, TIMEOUT_S)

    def open(self, query: str | None = None) -> AskPage:
        self.d.get(self.base_url + ("/?q=" + query if query else "/"))
        self.wait.until(EC.visibility_of_element_located((By.ID, "q")))
        return self

    @property
    def health_text(self) -> str:
        # The line starts as a static tagline and is replaced once GET /health returns.
        self.wait.until(lambda d: "chunks" in d.find_element(By.ID, "health").text)
        return self.d.find_element(By.ID, "health").text

    @property
    def ask_button(self) -> WebElement:
        return self.d.find_element(By.ID, "go")

    def ask(self, question: str, submit_with_enter: bool = False) -> AskPage:
        box = self.d.find_element(By.ID, "q")
        box.clear()
        box.send_keys(question)
        if submit_with_enter:
            box.send_keys(Keys.ENTER)
        else:
            self.ask_button.click()
        return self

    def click_example(self, text: str) -> AskPage:
        self.d.find_element(By.XPATH, f"//div[@class='examples']/button[normalize-space()={text!r}]").click()
        return self

    def result(self) -> Result:
        """Wait until the request settles (status badge rendered, button re-enabled) and read the result."""
        self.wait.until(lambda d: d.find_elements(By.CSS_SELECTOR, "#out .badge") and self.ask_button.is_enabled())
        out = self.d.find_element(By.ID, "out")
        answers = out.find_elements(By.CSS_SELECTOR, ".answer")
        traces = out.find_elements(By.TAG_NAME, "code")
        return Result(
            status=out.find_element(By.CSS_SELECTOR, ".badge").text,
            answer=answers[0].text if answers else out.text,
            citations=out.find_elements(By.TAG_NAME, "details"),
            trace=traces[0].text if traces else "",
        )

    def out_html(self) -> str:
        return self.d.find_element(By.ID, "out").get_attribute("innerHTML")
