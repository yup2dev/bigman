# 📁 scripts/run_site_pipeline.py
import os
import json
from collections import defaultdict
from crawler.people_parser import RollCallCrawler
from crawler.article_parser import save_articles_with_date
from utils.constants import PEOPLE_CONFIG_PATH
from utils.util import load_site


def load_existing_articles(site_key: str, date_str: str) -> list:
    file_path = os.path.join("scripts/data", "processed", date_str, f"articles_{site_key}_{date_str}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 기존 기사 로드 실패: {e}")
    return []


def process_site(site_key: str, limit: int = 5) -> None:
    print(f"🔍 Starting article collection for {site_key}")
    crawler = None

    try:
        crawler = RollCallCrawler(site_key=site_key, limit=limit)
        urls = crawler.get_urls()

        if not urls:
            print(f"⚠️ No URLs collected from {site_key}")
            return

        # 날짜별로 기사 URL 그룹핑
        date_groups = defaultdict(list)
        for url in urls:
            date_str = crawler.get_date(url)
            if date_str:
                date_groups[date_str].append(url)
            else:
                print(f"⚠️ 날짜 추출 실패: {url}")

        # 날짜별로 인터뷰 데이터 추출 및 저장
        for date_str, date_urls in date_groups.items():
            existing_articles = load_existing_articles(site_key, date_str)
            new_articles = crawler.extract_interviews(date_urls, existing_articles)

            if new_articles:
                save_articles_with_date(new_articles, site_key, date_str)
                print(f"✅ Saved {len(new_articles)} new articles for {site_key} on {date_str}")
            else:
                print(f"ℹ️ No new articles for {site_key} on {date_str}")

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