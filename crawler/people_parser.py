import time
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from webdriver_manager.chrome import ChromeDriverManager

from utils.constants import EXCLUDED_KEYWORDS, PEOPLE_CONFIG_PATH, DEFAULT_HEADERS
from utils.util import load_site, similarity_check


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
    driver = webdriver.Chrome(service=service, options=opts)

    try:
        driver.get(base_url + search_path)
        time.sleep(wait_time)  # JS 렌더링 대기

        driver.find_element(By.XPATH, '// *[ @ id = "main"] / div[2] / div / div / form / div[3] / div[2] / div[1] / div[2] / select').click()
        dropdown = Select(driver.find_element(By.XPATH, '// *[ @ id = "main"] / div[2] / div / div / form / div[3] / div[2] / div[1] / div[2] / select'))
        time.sleep(wait_time)  # JS 렌더링 대기
        dropdown.select_by_value('desc')
        time.sleep(wait_time)  # JS 렌더링 대기
        dropdown.select_by_value('asc')
        time.sleep(wait_time)

        # 스크롤 처리
        last_height = driver.execute_script("return document.body.scrollHeight")
        for _ in range(2):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(wait_time)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

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
        print(f"총 URL 수: {len(urls)}")
        return urls

    finally:
        driver.quit()


def get_rollcall_title(url: str) -> str:
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)

        if resp.status_code != 200:
            print(f"❌ Failed to fetch URL ({resp.status_code}): {url}")
            return ""

        if not resp.content.strip():
            print(f"❌ Failed to fetch URL ({resp.status_code}): {url}")
            return ""

        soup = BeautifulSoup(resp.content, "html.parser")
        content_blocks = soup.select('h1.not-italic.font-semibold.leading-normal')
        title = None

        for block in content_blocks:
            if block.name == "h1":
                title = block.get_text(strip=True)

        return title

    except requests.RequestException as e:
        print(f"HTTP error while extracting: {url}, error: {e}")
        return ""
    except Exception as e:
        print(f"Failed to extract full text (parse error): {url}, error: {e}")
        return ""


def get_rollcall_text(url: str) -> str:
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
                    if speech:
                        dialogue.append(f"{current_speaker}: {speech}")

        full_text = "\n\n".join(dialogue)
        return full_text

    except requests.RequestException as e:
        print(f"HTTP error while extracting: {url}, error: {e}")
        return ""
    except Exception as e:
        print(f"Failed to extract full text (parse error): {url}, error: {e}")
        return ""


def get_type(t: str) -> str:
    return t.strip().split(":")[0]


def extract_rollcall_interview(urls: List[str], existing_articles: List[Dict]) -> List[Dict]:
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

        full_text = get_rollcall_text(url).strip()
        doc_type = get_type(get_rollcall_title(url))
        text = []

        # enumerate 사용 시 2개만 반환 (index, "text")
        for i, block in enumerate(full_text.split("\n\n")):
            block = block.strip()
            if not block:
                continue
            text.append({
                "id": i+1,
                "text": block,
                "sentiment": None,
                "importance": None
            })

        # 변수 2개임
        # similarity_check(
        #     any(
        #         full_text[:200] in text for text in existing_texts)
        # )

        new_article = {
            "url": url,
            "title": get_rollcall_title(url),
            "text": text,
            "source": url.split("/")[2],
            "doc_type": doc_type
        }

        new_articles.append(new_article)
        seen_urls.add(url)
        existing_texts.append(full_text)

    return new_articles