import os
from crawler.url_collector import collect_urls
from crawler.utils import load_articles_from_urls, save_json
from analyzer.nlp_processor import NLPProcessor
from config.constants import PROCESSED_DATA_DIR


def build_dataset(site_key="cnn", keywords=["trump"], limit=10):
    print("🔍 Step 1. 기사 URL 수집")
    urls = collect_urls(site_key=site_key, keywords=keywords, limit=limit)
    print(f"🔗 수집된 URL 수: {len(urls)}")

    print("📰 Step 2. 기사 본문 수집")
    articles = load_articles_from_urls(urls)
    print(f"📄 수집된 기사 수: {len(articles)}")

    print("🧠 Step 3. NLP 처리 시작")
    nlp = NLPProcessor()
    processed_articles = nlp.process_articles(articles)

    dataset = []
    for article in processed_articles:
        dataset.append({
            "url": article["url"],
            "title": article["title"],
            "summary": article["summary"],
            "cause_effects": article["cause_effects"]
        })

    print(f"📦 Step 4. 데이터셋 저장 중 ({len(dataset)}개 기사)")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    save_path = os.path.join(PROCESSED_DATA_DIR, f"{site_key}_dataset.json")
    save_json(dataset, save_path)

    print(f"✅ 완료: {save_path}")


if __name__ == "__main__":
    build_dataset()
