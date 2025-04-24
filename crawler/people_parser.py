import time
from typing import List
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from utils.constants import DEFAULT_HEADERS, EXCLUDED_KEYWORDS, PEOPLE_CONFIG_PATH
from crawler.util import load_site


def get_transcript_urls(site_key: str, limit: int = 10) -> List[str]:
    # 1) site_config 로드
    configs     = load_site(PEOPLE_CONFIG_PATH)
    site_config = configs.get(site_key)
    if not site_config:
        raise ValueError(f"Site config for '{site_key}' not found.")

    base_url    = site_config["base_url"].rstrip("/")
    search_path = site_config.get("search_path",
                                 site_config.get("search_url", "/factbase/trump/search/"))
    wait_time   = site_config.get("wait_time", 3)
    anchor_sel  = site_config.get("anchor_selector",
                                 "a[href*='/factbase/trump/transcript/']")
    button_text = site_config.get("button_text", "View Transcript")

    # 2) Selenium headless 브라우저 설정
    opts = Options()
    opts.headless       = True
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    service = Service(ChromeDriverManager().install())
    driver  = webdriver.Chrome(service=service, options=opts)

    try:
        driver.get(base_url + search_path)
        time.sleep(wait_time)  # JS 렌더링 대기

        elems = driver.find_elements(By.CSS_SELECTOR, anchor_sel)
        urls  = []
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


if __name__ == "__main__":
    import sys

    key   = sys.argv[1] if len(sys.argv) > 1 else "rollcall"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print(f"\n▶️ '{key}' URL {limit}건 수집 시작")
    for idx, u in enumerate(get_transcript_urls(key, limit), start=1):
        print(f"{idx:2d}. {u}")
