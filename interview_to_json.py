import os
import json
from datetime import datetime
from crawler.people_parser import get_transcript_urls, extract_rollcall_interview
from crawler.article_parser import parse_articles, save_articles
from utils.constants import PEOPLE_CONFIG_PATH
from crawler.util import load_site


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


def process_site(site_key: str, limit: int = 3):
    print(f"\n🔍 {site_key} 사이트 기사 수집 시작")
    try:
        # URL 수집
        urls = get_transcript_urls(site_key=site_key, limit=limit)

        if not urls:
            print(f" {site_key}에서 수집된 URL이 없습니다.")
            return

        # 기존 기사 로드
        existing_articles = load_existing_articles(site_key)

        # 기사 파싱 및 중복 제거
        articles = extract_rollcall_interview(urls, existing_articles)

        if articles:
            save_articles(articles, site_key)  # site_key 전달
        else:
            print(f" {site_key}에서 저장할 신규 기사가 없습니다.")
    except Exception as e:
        print(f" {site_key} 처리 중 오류 발생: {e}")


def main():
    sites_config = load_site(PEOPLE_CONFIG_PATH)

    for site_key in sites_config:
        process_site(site_key, limit=10)


if __name__ == "__main__":
    main()