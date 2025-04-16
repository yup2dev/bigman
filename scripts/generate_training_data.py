import os
import json
from datetime import datetime
from analyzer.nlp_processor import NLPProcessor


def load_articles_from_directory(date_folder: str):
    """data/processed/날짜 폴더 내 JSON 파일을 읽어 기사 리스트로 반환"""
    all_articles = []
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    directory = os.path.join(base_dir, "data", "processed", date_folder)

    if not os.path.exists(directory):
        print(f"❌ 지정된 폴더가 존재하지 않습니다: {directory}")
        return []

    print(f"📂 날짜별 폴더 스캔 시작: {directory}")

    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    print(f"🔍 JSON 파일 {len(files)}개 발견")

    for filename in files:
        filepath = os.path.join(directory, filename)
        print(f"📄 파일 처리 중: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, list):
                    all_articles.extend(content)
                    print(f"✅ 기사 {len(content)}건 로드됨 (누적: {len(all_articles)}건)")
                elif isinstance(content, dict):
                    all_articles.append(content)
                    print(f"✅ 단일 기사 로드됨 (누적: {len(all_articles)}건)")
                else:
                    print(f"⚠️ 무시됨 (지원되지 않는 형식): {filename}")
        except Exception as e:
            print(f"❌ 파일 읽기 실패: {filename} | {e}")

    print(f"\n📊 처리 완료: 총 {len(all_articles)}개 기사 수집됨\n")
    return all_articles


def main():
    date_input = "2025-04-17"

    if not date_input:
        date_input = datetime.today().strftime('%Y-%m-%d')
        print(f"📅 오늘 날짜 기준: {date_input}")
    else:
        print(f"📅 지정된 날짜: {date_input}")

    articles = load_articles_from_directory(date_input)
    print(f"✅ 총 {len(articles)}개 기사 로드됨")

    nlp = NLPProcessor(model="gpt-3.5-turbo")

    for i, article in enumerate(articles):
        print(f"\n📝 [{i+1}/{len(articles)}] {article.get('title', '제목 없음')}")
        result = nlp.process_article(article)

        if result:
            events = result.get("events", [])
            print(f"✅ 분석 완료: {len(events)}개 이벤트 추출됨")
        else:
            print("⚠️ 분석 실패")


if __name__ == "__main__":
    main()
