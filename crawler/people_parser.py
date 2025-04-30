import time
from typing import List, Dict
import re
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from utils.constants import EXCLUDED_KEYWORDS, PEOPLE_CONFIG_PATH, DEFAULT_HEADERS
from crawler.util import load_site


def get_transcript_urls(site_key: str, limit: int = 10) -> List[str]:
    # 1) site_config 로드
    configs = load_site(PEOPLE_CONFIG_PATH)
    site_config = configs.get(site_key)
    if not site_config:
        raise ValueError(f"Site config for '{site_key}' not found.")

    base_url = site_config["base_url"].rstrip("/")
    search_path = site_config.get("search_path",
                                 site_config.get("search_url", "/factbase/trump/search/"))
    wait_time = site_config.get("wait_time", 3)
    anchor_sel = site_config.get("anchor_selector",
                                 "a[href*='/factbase/trump/transcript/']")
    button_text = site_config.get("button_text", "View Transcript")

    # 2) Selenium headless 브라우저 설정
    opts = Options()
    opts.headless = True
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    service = Service(ChromeDriverManager().install())
    driver  = webdriver.Chrome(service=service, options=opts)

    try:
        driver.get(base_url + search_path)
        time.sleep(wait_time)  # JS 렌더링 대기

        elems = driver.find_elements(By.CSS_SELECTOR, anchor_sel)
        urls = []
        for a in elems:
            href = a.get_attribute("href") or ""
            text = a.text.strip()
            # 필터링
            if not href.startswith("http"):
                continue
            if any(k in href for k in EXCLUDED_KEYWORDS):
                continue
            if button_text and button_text not in text:
                continue

            if href not in urls:
                urls.append(href)
            if len(urls) >= limit:
                break

        return urls

    finally:
        driver.quit()


def extract_full_text_rollcall_with_speaker(url: str) -> str:
    """Rollcall 스타일 인터뷰에서 발언자 + 발언 내용을 정리해서 가져온다."""
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)

        if resp.status_code != 200:
            print(f"❌ Failed to fetch URL ({resp.status_code}): {url}")
            return ""

        if not resp.content.strip():
            print(f"❌ Empty response received from: {url}")
            return ""

        soup = BeautifulSoup(resp.content, "html.parser")

        content_blocks = soup.select('h2.text-md.inline, div.flex-auto.text-md.text-gray-600.leading-loose')

        current_speaker = None
        dialogue = []

        for block in content_blocks:
            if block.name == "h2":
                current_speaker = block.get_text(strip=True)
            elif block.name == "div":
                if current_speaker:
                    speech = block.get_text(strip=True)
                    if speech:  # 빈 발언 제거
                        dialogue.append(f"{current_speaker}: {speech}")

        full_text = "\n\n".join(dialogue)
        return full_text

    except requests.RequestException as e:
        print(f"HTTP error while extracting: {url}, error: {e}")
        return ""
    except Exception as e:
        print(f"Failed to extract full text (parse error): {url}, error: {e}")
        return ""

def guess_type(t: str) -> str:
    t = t.strip()
    if re.search(r'^(q:|question:)', t, re.I|re.M):
        return "interview"
    if re.search(r'^[A-Z][a-z]+\\s[A-Z][a-z]+:', t):    # 화자: 텍스트
        lines = t.splitlines()
        speaker_lines = sum(':' in l and l.split(':',1)[0].istitle() for l in lines[:20])
        if speaker_lines > 3:
            return "speech"
    return "article"


def extract_rollcall_interview(urls: List[str], existing_articles: List[Dict]) -> List[Dict]:
    """주어진 Rollcall 인터뷰 URL 리스트를 파싱하고 중복을 제거하여 신규 기사 리스트 반환"""
    seen_urls = set(a["url"] for a in existing_articles)
    existing_texts = [a["text"] for a in existing_articles if a.get("text")]

    new_articles = []

    for url in urls:
        if not isinstance(url, str) or not url.startswith("http"):
            print(f"Skipping invalid URL: {url}")
            continue
        if url in seen_urls:
            print(f"Duplicate URL skipped: {url}")
            continue

        full_text = extract_full_text_rollcall_with_speaker(url)

        if not full_text.strip():
            print(f"Skipping empty article: {url}")
            continue

        # 중복 텍스트 검사 (앞부분 500자 또는 전체 20% 기준 비교)
        is_duplicate = any(full_text[:500] in text or full_text[:int(len(full_text)*0.2)] in text for text in existing_texts)
        if is_duplicate:
            print(f"Skipping duplicate article based on text: {url}")
            continue

        new_article = {
            "url": url,
            "title": url.split("/")[-1].replace('-', ' ').capitalize(),
            "text": full_text.strip(),
            "published": None,
            "source": url.split("/")[2],
            "doc_type": guess_type(full_text)
        }

        new_articles.append(new_article)
        seen_urls.add(url)
        existing_texts.append(full_text)

    return new_articles


if __name__ == "__main__":
    import sys

    key = sys.argv[1] if len(sys.argv) > 1 else "rollcall"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print(f"\n '{key}' URL {limit}건 수집 시작")
    for idx, u in enumerate(get_transcript_urls(key, limit), start=1):
        print(f"{idx:2d}. {u}")
