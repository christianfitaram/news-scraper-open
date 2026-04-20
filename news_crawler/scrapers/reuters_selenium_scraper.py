"""Reuters Selenium scraper."""
from __future__ import annotations

from typing import Any, Dict, Iterator

from news_crawler.scrapers.base_scraper import BaseScraper
from news_crawler.scrapers.streams import scrape_reuters_selenium_stream


class ReutersSeleniumScraper(BaseScraper):
    """Reuters Selenium scraper."""

    def __init__(self) -> None:
        super().__init__("reuters-selenium")

    def scrape(self) -> Iterator[Dict[str, Any]]:
        for article in scrape_reuters_selenium_stream():
            article.setdefault("source", self.source_name)
            if self._validate_article(article):
                yield article
