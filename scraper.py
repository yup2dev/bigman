import requests
from bs4 import BeautifulSoup
from config import site_settings

class NewsScraper:
    def __init__(self, site="bbc", query="trump", max_pages=5):
        self.site = site
        self.settings = site_settings.get(site)
        if not self.settings:
            raise ValueError(f"지원되지 않는 사이트: {site}")

        self.query = query
        self.max_pages = max_pages
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.excluded_keywords = {"sport", "audio"}

    def fetch_news_links(self):
        """뉴스 검색 결과에서 기사 링크 수집 (페이징 포함)"""
        news_links = set()
        for page in range(1, self.max_pages + 1):
            search_url = self.settings["search_url"].format(query=self.query, page=page)
            response = requests.get(search_url, headers=self.headers)
            soup = BeautifulSoup(response.text, "html.parser")

            articles_section = soup.find_all("div", {"data-testid": self.settings["target_class"]})
            for section in articles_section:
                links = section.find_all("a", href=True)
                for a_tag in links:
                    link = a_tag["href"]
                    if link and not any(keyword in link for keyword in self.excluded_keywords):
                        full_url = link if link.startswith("http") else f"{self.settings['base_url']}{link}"
                        news_links.add(full_url)

        return list(news_links)

    def scrape_article(self, url):
        """기사 본문 크롤링"""
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            print(f"❌ 크롤링 실패: {url}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.find(self.settings["title_tag"])
        title = title.text.strip() if title else "No Title"

        date_tag = soup.find(self.settings["date_tag"])
        date = date_tag["datetime"] if date_tag and date_tag.has_attr("datetime") else "No Date"

        article_tag = soup.find("article")
        content = " ".join([p.text.strip() for p in article_tag.find_all("p") if p.text.strip()]) if article_tag else "No content"

        return {"url": url, "title": title, "date": date, "content": content}

    def run(self, num_articles=5):
        """전체 크롤링 실행"""
        news_links = self.fetch_news_links()
        print(f"🔗 총 {len(news_links)}개의 기사 링크를 찾음.")

        for link in news_links[:num_articles]:
            article = self.scrape_article(link)
            if article:
                print(f"🔗 URL: {article['url']}\n📰 제목: {article['title']}\n📅 날짜: {article['date']}\n📜 본문 (일부): {article['content'][:100]}...\n")