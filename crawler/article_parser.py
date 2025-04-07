from newspaper import Article
from typing import List, Dict
from datetime import datetime

def parse_articles(urls: List[str]) -> List[Dict]:
    articles = []

    for url in urls:
        if not isinstance(url, str) or not url.startswith("http"):
            print(f"⚠️ Skipping invalid URL: {url}")
            continue

        try:
            article = Article(url)
            article.download()
            article.parse()

            articles.append({
                "url": url,
                "title": article.title.strip(),
                "text": article.text.strip(),
                "published": (
                    article.publish_date.strftime('%Y-%m-%d %H:%M:%S')
                    if article.publish_date else None
                ),
                "source": article.source_url or url.split("/")[2]
            })

        except Exception as e:
            print(f"❌ Failed to parse {url}: {e}")

    return articles
