import os
import json
import logging
from datetime import datetime
from analyzer.nlp_processor import NLPProcessor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 콘솔 출력
        logging.FileHandler("generate_training_data.log", encoding="utf-8")  # 로그 파일 저장
    ]
)
logger = logging.getLogger(__name__)


def load_articles_from_directory(date_folder: str):
    """data/processed/날짜 폴더 내 JSON 파일을 읽어 기사 리스트로 반환"""
    all_articles = []
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    directory = os.path.join(base_dir, "data", "processed", date_folder)

    if not os.path.exists(directory):
        logger.error(f"지정된 폴더가 존재하지 않습니다: {directory}")
        return []

    logger.info(f"폴더 스캔 시작: {directory}")
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    logger.info(f"JSON 파일 {len(files)}개 발견")

    for filename in files:
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, list):
                    all_articles.extend(content)
                elif isinstance(content, dict):
                    all_articles.append(content)
                else:
                    logger.warning(f"무시됨 (지원되지 않는 형식): {filename}")
        except Exception as e:
            logger.error(f"파일 읽기 실패: {filename} | {e}")

    logger.info(f"처리 완료: 총 {len(all_articles)}개 기사 수집됨")
    return all_articles


def process_articles_by_date(date_input: str = None):
    if not date_input:
        date_input = datetime.today().strftime('%Y-%m-%d')
        logger.info(f"오늘 날짜 기준 실행: {date_input}")
    else:
        try:
            datetime.strptime(date_input, '%Y-%m-%d')
        except ValueError:
            logger.error(f"날짜 형식 오류 (YYYY-MM-DD 형식 필요): {date_input}")
            return

    logger.info(f"기사 분석 시작일: {date_input}")
    articles = load_articles_from_directory(date_input)
    logger.info(f"총 {len(articles)}개 기사 로드됨")

    if not articles:
        logger.warning("분석할 기사가 없습니다.")
        return

    nlp = NLPProcessor(model="gpt-3.5-turbo")

    success_count = 0
    failure_count = 0

    for i, article in enumerate(articles):
        logger.info(f"[{i+1}/{len(articles)}] 제목: {article.get('title', '제목 없음')}")

        try:
            result = nlp.process_article(article)
            if result:
                events = result.get("events", [])
                logger.info(f"분석 성공: {len(events)}개 이벤트 추출됨")
                success_count += 1
            else:
                logger.warning(f"분석 실패 - 파일명 또는 URL: {article.get('url', '알 수 없음')}")
                failure_count += 1
        except Exception as e:
            logger.error(f"기사 분석 중 오류 발생: {e}")
            failure_count += 1

    logger.info(f"\n📊 분석 완료 요약: 성공 {success_count}건 | 실패 {failure_count}건")


if __name__ == "__main__":
    process_articles_by_date("2025-04-19")
