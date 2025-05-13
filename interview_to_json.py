import os
import json
from collections import defaultdict
from datetime import datetime
from crawler.people_parser import RollCallCrawler
from crawler.article_parser import save_articles_with_date
from crawler.temp import DataSetProcessor, LIWCAnalyzer, StructuralEmphasisScorer
from utils.constants import PEOPLE_CONFIG_PATH
from utils.util import load_site


def load_existing_articles(site_key: str) -> list:
    today = datetime.today().strftime('%Y-%m-%d')
    file_path = os.path.join("scripts/data", "processed", today, f"articles_{site_key}_{today}.json")

    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f" 기존 기사 로드 실패: {e}")
    return []


def process_site(site_key: str, limit: int = 5) -> None:
    print(f"🔍 Starting article collection for {site_key}")
    crawler = None
    try:
        crawler = RollCallCrawler(site_key=site_key, limit=limit)
        liwc = StructuralEmphasisScorer()
        urls = crawler.get_urls()

        for url in urls:
            text = crawler.get_text(url)
            print(liwc.analyze(text))

        # if not urls:
        #     print(f"⚠️ No URLs collected from {site_key}")
        #     return
        #
        # existing_articles = load_existing_articles(site_key)
        #
        # # 날짜별로 기사를 그룹화
        # date_groups = defaultdict(list)
        # for url in urls:
        #     date_str = crawler.get_date(url)
        #     if date_str:
        #         date_groups[date_str].append(url)
        #     else:
        #         print(f"⚠️ Could not extract date from title for URL: {url}")
        #
        # # 각 날짜별로 처리
        # for date_str, date_urls in date_groups.items():
        #     articles = crawler.extract_interviews(date_urls, existing_articles)
        #     if articles:
        #         save_articles_with_date(articles, site_key, date_str)
        #         print(f"✅ Saved {len(articles)} new articles for {site_key} on {date_str}")
        #     else:
        #         print(f"ℹ️ No new articles for {site_key} on {date_str}")

    except Exception as e:
        print(f"❌ Error processing {site_key}: {e}")
        raise
    finally:
        if crawler:
            crawler.close()
            print(f"🛑 Closed crawler for {site_key}")

def main():
    sites_config = load_site(PEOPLE_CONFIG_PATH)

    for site_key in sites_config:
        process_site(site_key, limit=10)


if __name__ == "__main__":
    main()