import time, re
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

from crawler.SpeechBlockSegmenter import SpeechBlockSegmenter
from crawler.gpt_inferncer import GPTUtteranceInterpreter
from crawler.scrap_summary import UtteranceScorer, EmpathAnalyzer
from utils.constants import EXCLUDED_KEYWORDS, PEOPLE_CONFIG_PATH, DEFAULT_HEADERS
from utils.util import load_site


class RollCallCrawler:
    DEFAULT_WAIT_TIME = 3
    DEFAULT_BUTTON_TEXT = "View Transcript"
    DEFAULT_ANCHOR_SELECTOR = "a[href*='/factbase/trump/transcript/']"

    def __init__(self, site_key: str = "rollcall", start_year: int = 1000, limit: int = 10) -> None:
        self.site_key = site_key
        self.start_year = start_year
        self.limit = limit
        self.config = self._load_config()
        self.base_url = self.config["base_url"].rstrip("/")
        self.search_url = self._build_search_url()
        self.wait_time = self.config.get("wait_time", self.DEFAULT_WAIT_TIME)
        self.anchor_selector = self.config.get("anchor_selector", self.DEFAULT_ANCHOR_SELECTOR)
        self.button_text = self.config.get("button_text", self.DEFAULT_BUTTON_TEXT)
        self.driver = self._init_driver()
        self.utterance_scorer = UtteranceScorer()
        self.empath_analyzer = EmpathAnalyzer(threshold=0.05)
        self.gpt_interpreter = GPTUtteranceInterpreter()
        self.block_segmentor = SpeechBlockSegmenter()
        self.interview_extractor = InterviewExtractor(self.utterance_scorer, self.empath_analyzer, self.gpt_interpreter, self.block_segmentor, limit=self.limit)

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

    def get_urls(self, start_year: Optional[int] = None) -> List[str]:
        """
        검색 페이지에서 URL을 추출하며, start_year를 지정하면 해당 연도 이후만 필터링함.
        """
        self.driver.get(self.search_url)
        time.sleep(self.wait_time)
        self._handle_sort_dropdown()
        self._scroll_to_bottom()

        all_urls = self._extract_urls()
        if not start_year:
            return all_urls[:self.limit]

        filtered_urls = []
        for url in all_urls:
            date_str = self.get_date(url)
            if not date_str:
                continue
            try:
                year = int(date_str.split("-")[0])
                if year >= start_year:
                    filtered_urls.append(url)
            except ValueError:
                continue

            if len(filtered_urls) >= self.limit:
                break

        print(f"📅 {start_year}년 이후 URL 수: {len(filtered_urls)}")
        return filtered_urls

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
    def __init__(self, utterance_scorer, empath_analyzer, gpt_interpreter, block_segmenter, limit: int = 10):
        self.utterance_scorer = utterance_scorer
        self.empath_analyzer = empath_analyzer
        self.gpt_interpreter = gpt_interpreter
        self.block_segmenter = block_segmenter
        self.limit = limit

    def extract(self, urls, existing, get_text, get_title, get_date, get_document_type):
        seen_urls = {article["url"] for article in existing}
        results = []

        for url in urls:
            if url in seen_urls:
                continue

            try:
                full_text = get_text(url).strip()
            except Exception as e:
                print(f"Error fetching text for {url}: {e}")
                continue

            if not full_text:
                continue

            try:
                title = get_title(url)
                date = get_date(url)
                doc_type = get_document_type(title)
            except Exception as e:
                print(f"Error fetching metadata for {url}: {e}")
                continue

            segments = self._parse_segments(full_text)

            if doc_type.lower() == "interview":
                blocks = self.block_segmenter.segment_as_qa_pairs(segments)
            else:
                blocks = [[seg] for seg in segments if seg.get("text")]

            for i, block in enumerate(blocks):
                if len(results) >= self.limit:
                    break
                result = self._process_block(block, url, date, doc_type, i)
                if result:
                    results.append(result)

            seen_urls.add(url)
            if len(results) >= self.limit:
                break

        return results

    def _parse_segments(self, full_text):
        segments = []
        for block in full_text.split("\n\n"):
            if not block.strip():
                continue
            match = re.match(r"^(.*?):\s*(.*)", block.strip(), re.DOTALL)
            if match:
                speaker, text = match.groups()
            else:
                speaker, text = None, block.strip()
            segments.append({"speaker": speaker, "text": text})
        return segments

    def _process_block(self, block, url, date, doc_type, i):
        speaker = block[0].get("speaker", "Unknown")
        combined_text = " ".join(seg["text"] for seg in block).strip()
        if not combined_text:
            return None

        scores = self.utterance_scorer.score(combined_text)
        empath = self.empath_analyzer.analyze(combined_text)
        sentiment = self.gpt_interpreter.analyze_sentiment(combined_text)
        tone_label = self.gpt_interpreter.classify_tone(combined_text)
        arousal = scores.get("emotion_arousal", 0.0)

        conditions, conclusion = self.gpt_interpreter.extract_conditions_and_conclusion(combined_text)
        persona = self.gpt_interpreter.infer_persona(conditions, conclusion, sentiment["label"], arousal, empath)
        policy_result = self.gpt_interpreter.extract_policies_and_effects(combined_text)

        predicted_policies = policy_result.get("predicted_policies", [])
        expected_effects = policy_result.get("expected_effects", [])
        # TODO: Implement a better selection criterion for best_policy
        best_policy = predicted_policies[0] if predicted_policies else None

        result = {
            "id": f"{url.split('/')[-1]}_{i+1}",
            "datetime": date,
            "doc_type": doc_type,
            "context": [{"speaker": seg["speaker"], "text": seg["text"]} for seg in block],
            "parsed": {
                "conditions": conditions,
                "conclusion": conclusion,
                "answer_type": "conditional" if "if" in combined_text.lower() else "assertive"
            },
            "empath": empath,
            "emotion_arousal": arousal,
            "tone": tone_label,
            "persona": persona,
            "predicted_policies": predicted_policies,
            "expected_effects": expected_effects,
            "labels": {
                "predicted_best_policy": best_policy,
                "historical_alignment": self.gpt_interpreter.validate_policy(best_policy) if best_policy else 0.0
            }
        }
        return result