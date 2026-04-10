"""NPR Selenium scraper."""
from __future__ import annotations

from typing import Iterator, Dict, Any

from news_crawler.scrapers.base_scraper import BaseScraper
from news_crawler.scrapers.streams import scrape_npr_selenium_stream


class NPRScraper(BaseScraper):
    """NPR Selenium scraper."""

    def __init__(self) -> None:
        super().__init__("npr")

    def scrape(self) -> Iterator[Dict[str, Any]]:
        for article in scrape_npr_selenium_stream(check_link_pool=False, track_links=False):
            article.setdefault("source", self.source_name)
            if self._validate_article(article):
                yield article
