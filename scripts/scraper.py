# bigman/scripts/scraper.py

import os
import json
from crawler.article_parser import ArticleParser
from crawler.utils import load_json, save_json
from config.constants import RAW_DATA_DIR, PROCESSED_DATA_DIR, JSON_ENCODING

def scrape_articles(site_key: str):
    input_path = os.path.join(RAW_DATA_DIR, f"{site_key}_urls.json")
    output_path = os.path.join(PROCESSED_DATA_DIR, f"{site_key}_articles.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ URL list not found: {input_path}")

    urls = load_json(input_path)
    parser = ArticleParser()
    parsed_articles = []

    for i, url in enumerate(urls):
        print(f"📄 ({i+1}/{len(urls)}) Parsing: {url}")
        result = parser.parse(url)
        if result:
            parsed_articles.append(result)

    save_json(parsed_articles, output_path)
    print(f"✅ 총 {len(parsed_articles)}개의 기사 저장 완료 → {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python scraper.py [site_key]")
    else:
        scrape_articles(sys.argv[1])
