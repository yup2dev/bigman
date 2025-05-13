import re
import time
from datetime import datetime
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from webdriver_manager.chrome import ChromeDriverManager
from crawler.temp import UtteranceScorer
from utils.constants import EXCLUDED_KEYWORDS, PEOPLE_CONFIG_PATH, DEFAULT_HEADERS
from utils.util import load_site


class RollCallCrawler:
    DEFAULT_WAIT_TIME = 3
    DEFAULT_BUTTON_TEXT = "View Transcript"
    DEFAULT_ANCHOR_SELECTOR = "a[href*='/factbase/trump/transcript/']"

    def __init__(self, site_key: str = "rollcall", limit: int = 5) -> None:
        self.site_key = site_key
        self.limit = limit
        self.config = self._load_config()
        self.base_url = self.config["base_url"].rstrip("/")
        self.search_url = self._build_search_url()
        self.wait_time = self.config.get("wait_time", self.DEFAULT_WAIT_TIME)
        self.anchor_selector = self.config.get("anchor_selector", self.DEFAULT_ANCHOR_SELECTOR)
        self.button_text = self.config.get("button_text", self.DEFAULT_BUTTON_TEXT)
        self.driver = self._init_driver()
        self.utterance_scorer = UtteranceScorer()
        self.interview_extractor = InterviewExtractor(self.utterance_scorer, limit=self.limit)

    def _load_config(self) -> Dict:
        config = load_site(PEOPLE_CONFIG_PATH).get(self.site_key)
        if not config:
            raise ValueError(f"Config for '{self.site_key}' not found")
        return config

    def _build_search_url(self) -> str:
        search_path = self.config.get("search_path", self.config.get("search_url", "/factbase/trump/search/"))
        return f"{self.base_url}{search_path}"

    def _init_driver(self) -> webdriver.Chrome:
        options = Options()
        options.headless = True
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        return webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )

    def get_urls(self) -> List[str]:
        self.driver.get(self.search_url)
        time.sleep(self.wait_time)
        self._handle_sort_dropdown()
        self._scroll_to_bottom()
        return self._extract_urls()

    def _handle_sort_dropdown(self) -> None:
        try:
            dropdown = Select(self.driver.find_element(By.TAG_NAME, "select"))
            dropdown.select_by_value("desc")
            time.sleep(self.wait_time)
            dropdown.select_by_value("asc")
            time.sleep(self.wait_time)
        except Exception:
            pass

    def _scroll_to_bottom(self, max_attempts: int = 2) -> None:
        previous_height = 0
        for _ in range(max_attempts):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(self.wait_time)
            current_height = self.driver.execute_script("return document.body.scrollHeight")
            if current_height == previous_height:
                break
            previous_height = current_height

    def _extract_urls(self) -> List[str]:
        urls = []
        for anchor in self.driver.find_elements(By.CSS_SELECTOR, self.anchor_selector):
            href = anchor.get_attribute("href") or ""
            if (href.startswith("http") and
                self.button_text in anchor.text and
                not any(keyword in href for keyword in EXCLUDED_KEYWORDS) and
                href not in urls):
                urls.append(href)
                if len(urls) >= self.limit:
                    break
        print(f"Total URLs found: {len(urls)}")
        return urls

    @staticmethod
    def _fetch_soup(url: str) -> Optional[BeautifulSoup]:
        try:
            response = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
            return BeautifulSoup(response.content, "html.parser") if response.ok else None
        except requests.RequestException:
            return None

    def get_title(self, url: str) -> str:
        soup = self._fetch_soup(url)
        if not soup:
            return ""
        h1 = soup.select_one("h1.not-italic.font-semibold.leading-normal")
        return h1.get_text(strip=True) if h1 else ""

    def get_date(self, url: str) -> Optional[str]:
        title = self.get_title(url)
        if not title:
            return None
        match = re.search(r'([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})$', title)
        if not match:
            return None
        month_str, day, year = match.groups()
        month_map = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        month_str = month_str.lower()
        for full_month, month_num in month_map.items():
            if month_str.startswith(full_month[:3]):
                month = month_num
                break
        else:
            return None
        day = day.zfill(2)
        try:
            date_obj = datetime(int(year), int(month), int(day))
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            return None

    def get_text(self, url: str) -> str:
        soup = self._fetch_soup(url)
        if not soup:
            return ""
        blocks = []
        current_speaker = None
        for tag in soup.select("h2.text-md.inline, div.flex-auto.text-md.text-gray-600.leading-loose"):
            if tag.name == "h2":
                current_speaker = tag.get_text(strip=True)
            elif tag.name == "div" and current_speaker:
                speech = tag.get_text(strip=True)
                if speech:
                    blocks.append(f"{current_speaker}: {speech}")
        return "\n\n".join(blocks)

    @staticmethod
    def get_document_type(title: str) -> str:
        return title.split(":", 1)[0] if ":" in title else "article"

    def extract_interviews(self, urls: List[str], existing: List[Dict]) -> List[Dict]:
        return self.interview_extractor.extract(
            urls,
            existing,
            self.get_text,
            self.get_title,
            self.get_date,
            self.get_document_type
        )

    def close(self) -> None:
        self.driver.quit()


class InterviewExtractor:
    def __init__(self, utterance_scorer: UtteranceScorer, limit: int = 10):
        self.utterance_scorer = utterance_scorer
        self.limit = limit

    def extract(self, urls: List[str], existing: List[Dict], get_text, get_title, get_date, get_document_type) -> List[Dict]:
        seen_urls = {article["url"] for article in existing}
        new_articles = []

        for url in urls:
            if url in seen_urls:
                continue

            text = get_text(url).strip()
            if not text:
                continue

            title = get_title(url)
            doc_type = get_document_type(title)
            date = get_date(url)

            blocks = []
            for i, block in enumerate(text.split("\n\n")):
                if not block.strip():
                    continue
                scores = self.utterance_scorer.score(block.strip())
                blocks.append({
                    "id": i + 1,
                    "text": block.strip(),
                    "emotion_arousal": scores.get("emotion_arousal", 0.0),
                    "keyword_rarity": scores.get("keyword_rarity", 0.0),
                    "structural_emphasis": scores.get("structural_emphasis", 0.0),
                    "sentiment": None,
                    "importance": None
                })

            new_articles.append({
                "url": url,
                "date": date,
                "title": title,
                "text": blocks,
                "source": url.split("/")[2],
                "doc_type": doc_type
            })

            seen_urls.add(url)
            if len(new_articles) >= self.limit:
                break

        return new_articles