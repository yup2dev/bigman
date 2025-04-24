import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from utils.constants import DEFAULT_HEADERS, EXCLUDED_KEYWORDS, SITES_CONFIG_PATH
from crawler.util import load_site

def is_valid_article_url(href: str, base_url: str, keywords: list) -> bool:
    if not href or any(k in href for k in EXCLUDED_KEYWORDS):
        return False
    full_url = urljoin(base_url, href)
    return any(keyword.lower() in full_url.lower() for keyword in keywords)

def collect_urls(site_key: str, keywords: list, limit: int = 10) -> list:
    configs = load_site(SITES_CONFIG_PATH)
    site_config = configs.get(site_key)

    if not site_config:
        raise ValueError(f"Site config for '{site_key}' not found.")

    collected_links = set()

    for seed_url in site_config.get("seed_urls", []):
        print(f" Visiting seed: {seed_url}")
        try:
            response = requests.get(seed_url, headers=DEFAULT_HEADERS, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"❌ Failed to fetch {seed_url}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        anchors = soup.find_all("a", href=True)

        for a in anchors:
            href = a["href"]
            full_url = urljoin(site_config['base_url'], href)

            if is_valid_article_url(href, site_config['base_url'], keywords):
                collected_links.add(full_url)

            if len(collected_links) >= limit:
                break

        if len(collected_links) >= limit:
            break

    result_links = list(collected_links)
    print(f" 관련 URL 수집 완료: {len(result_links)}건")
    return result_links
