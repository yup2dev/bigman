import os
from crawler.url_collector import collect_urls
from crawler.utils import load_articles_from_urls, save_json
from analyzer.nlp_processor import NLPProcessor
from config.constants import PROCESSED_DATA_DIR


def build_dataset(site_key="cnn", keywords=["trump"], limit=10):

    print("🔍 Step 1. 기사 URL 수집")
    urls = collect_urls(site_key=site_key, keywords=keywords, limit=limit)

    print("📰 Step 2. 기사 본문 수집")
    articles = load_articles_from_urls(urls)

    print("🧠 Step 3. NLP 처리 시작")
    nlp = NLPProcessor()  # NLPProcessor 객체 생성
    dataset = []

    for article in articles:
        cleaned_text = nlp.clean_text(article["text"])

        # 1. 요약
        summary = nlp.summarize(cleaned_text)

        # 2. 원인/결과 추출
        cause_effect_pairs = nlp.extract_cause_effect(cleaned_text)

        for cause_effect in cause_effect_pairs:
            dataset.append({
                "url": article["url"],
                "title": article["title"],
                "summary": summary,
                "cause": cause_effect["cause"],
                "effect": cause_effect["effect"]
            })

    print(f"📦 Step 4. 데이터셋 저장 중 ({len(dataset)}개)")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    save_path = os.path.join(PROCESSED_DATA_DIR, f"{site_key}_dataset.json")
    save_json(dataset, save_path)

    print(f"✅ 완료: {save_path}")


if __name__ == "__main__":
    build_dataset()
