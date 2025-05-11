import os
import json
import torch
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from typing import List, Dict
from utils.util import preprocess_text, similarity_check


def get_embedding(texts: List[str], vectorizer=None) -> torch.Tensor:
    if not vectorizer:
        vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return torch.tensor(tfidf_matrix.toarray(), dtype=torch.float32)


def parse_articles(urls: List[str], existing_articles: List[Dict]) -> List[Dict]:
    seen_urls = set()
    new_articles = []
    all_articles = existing_articles.copy()

    for url in urls:
        if not isinstance(url, str) or not url.startswith("http"):
            print(f" Skipping invalid URL: {url}")
            continue
        if url in seen_urls:
            print(f" Duplicate URL skipped: {url}")
            continue

        try:
            article = Article(url)
            article.download()
            article.parse()

            if not article.text.strip():
                print(f" Skipping empty article: {url}")
                continue

            current_text = preprocess_text(article.text)
            existing_texts = [preprocess_text(a["text"]) for a in all_articles if a.get("text")]

            if similarity_check(current_text, existing_texts):
                print(f" Skipping duplicate article: {article.title}")
                continue

            new_article = {
                "url": url,
                "title": article.title.strip(),
                "text": article.text.strip(),
                "published": (
                    article.publish_date.strftime('%Y-%m-%d %H:%M:%S')
                    if article.publish_date else None
                ),
                "source": article.source_url or url.split("/")[2]
            }

            new_articles.append(new_article)
            all_articles.append(new_article)  # 다음 중복 검사를 위해 누적
            seen_urls.add(url)

        except Exception as e:
            print(f"❌ Failed to parse {url}: {e}")

    return new_articles


def save_articles(articles: List[Dict], site_key: str):  # site_key 추가
    if not articles:
        print("📝 저장할 기사가 없습니다.")
        return

    today = datetime.today().strftime('%Y-%m-%d')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder_path = os.path.join(base_dir, "data", "processed", today)

    os.makedirs(folder_path, exist_ok=True)
    print(f"📁 저장 폴더: {folder_path}")

    # 사이트별 파일명 생성
    filename = f"articles_{site_key}_{today}.json"
    file_path = os.path.join(folder_path, filename)

    # 기존 기사 로드 후 추가
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_articles = json.load(f)
        articles = existing_articles + articles

    # 👉 ① 타입별 그룹
    buckets: Dict[str, List[Dict]] = {}
    for art in articles:
        buckets.setdefault(art.get("doc_type", "unknown"), []).append(art)

    # 👉 ② 타입마다 별도 파일
    for dtype, items in buckets.items():
        fpath = os.path.join(folder_path, f"{site_key}_{dtype}_{today}.json")

        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as fp:
                items = json.load(fp) + items

        with open(fpath, "w", encoding="utf-8") as fp:
            json.dump(items, fp, ensure_ascii=False, indent=2)

        print(f"✅ {dtype:<10} → {fpath} ({len(items)}개)")

def save_articles_with_date(articles: List[Dict], site_key: str, date_str: str):
    if not articles:
        print("📝 No articles to save.")
        return

    try:
        year = datetime.strptime(date_str, '%Y-%m-%d').year
    except ValueError:
        print(f"⚠️ Invalid date format: {date_str}. Expected 'YYYY-MM-DD'.")
        return

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder_path = os.path.join(base_dir, "data", "processed", str(year))

    # Create year folder
    os.makedirs(folder_path, exist_ok=True)
    print(f"📁 Folder created or exists: {folder_path}")

    # Site-specific filename
    filename = f"articles_{site_key}_{date_str}.json"
    file_path = os.path.join(folder_path, filename)

    # Load existing articles and append new ones
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_articles = json.load(f)
            articles = existing_articles + articles
        except Exception as e:
            print(f"⚠️ Failed to load existing articles: {e}")

    # Group articles by doc_type
    buckets: Dict[str, List[Dict]] = {}
    for art in articles:
        buckets.setdefault(art.get("doc_type", "unknown"), []).append(art)

    # Save each doc_type to separate files
    for dtype, items in buckets.items():
        fpath = os.path.join(folder_path, f"{site_key}_{dtype}_{date_str}.json")

        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as fp:
                    existing_items = json.load(fp)
                items = existing_items + items
            except Exception as e:
                print(f"⚠️ Failed to load existing type-specific articles: {e}")

        try:
            with open(fpath, "w", encoding="utf-8") as fp:
                json.dump(items, fp, ensure_ascii=False, indent=2)
            print(f"✅ {dtype:<10} → {fpath} ({len(items)} items)")
        except Exception as e:
            print(f"⚠️ Failed to save file: {e}")
