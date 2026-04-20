"""France24 Selenium scraper."""
from __future__ import annotations

from typing import Any, Dict, Iterator

from news_crawler.scrapers.base_scraper import BaseScraper
from news_crawler.scrapers.streams import scrape_france24_selenium_stream


class France24Scraper(BaseScraper):
    """France24 Selenium scraper."""

    def __init__(self) -> None:
        super().__init__("france24")

    def scrape(self) -> Iterator[Dict[str, Any]]:
        for article in scrape_france24_selenium_stream(check_link_pool=False, track_links=False):
            article.setdefault("source", self.source_name)
            if self._validate_article(article):
                yield article
