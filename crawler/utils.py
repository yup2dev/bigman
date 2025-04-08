import yaml
import re
import json
import os
from datetime import datetime
from newspaper import Article
from typing import List, Dict
import time

def load_site_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def filter_urls_by_keyword(urls, keywords):
    pattern = re.compile('|'.join(keywords), re.IGNORECASE)
    return [url for url in urls if pattern.search(url)]


def save_json(data, filepath: str, ensure_dir=True, indent=2):
    """
    데이터를 JSON 파일로 저장한다.

    :param data: 저장할 데이터 (dict 또는 list)
    :param filepath: 저장할 경로 (예: 'data/raw/cnn_articles.json')
    :param ensure_dir: True일 경우, 디렉토리가 없으면 생성
    :param indent: JSON 들여쓰기
    """
    # 디렉토리가 없으면 생성
    if ensure_dir:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # datetime 객체를 ISO 8601 형식으로 변환하는 함수
    def datetime_converter(o):
        if isinstance(o, datetime):
            return o.isoformat()  # ISO 포맷으로 변환
        if isinstance(o, type(None)):
            return None  # None을 처리

    # 파일 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=datetime_converter)

    print(f"[+] JSON saved to {filepath}")

def load_json(filepath: str, encoding: str = "utf-8") -> dict:
    """
    지정된 경로의 JSON 파일을 불러와 Python 객체로 반환합니다.
    """
    try:
        with open(filepath, "r", encoding=encoding) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {filepath}")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ JSON 디코딩 실패: {filepath}, 에러: {e}")
        return {}


def load_articles_from_urls(urls: List[str], delay: float = 3.0) -> List[Dict]:
    articles = []
    for url in urls:
        print(f"📥 기사 수집 중: {url}")
        try:
            article = Article(url)
            article.download()
            article.parse()
            articles.append({
                "url": url,
                "title": article.title,
                "text": article.text,
                "published": article.publish_date.isoformat() if article.publish_date else "",
            })
            time.sleep(delay)  # 너무 빠르게 요청하면 사이트 차단 위험
        except Exception as e:
            print(f"❌ 기사 수집 실패: {url} - {e}")
    return articles
