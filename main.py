from scraper import NewsScraper

if __name__ == "__main__":
    scraper = NewsScraper(site="bbc", query="trump", max_pages=5)
    scraper.run(num_articles=5)