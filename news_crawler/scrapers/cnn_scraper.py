"""CNN scraper."""
from __future__ import annotations

from typing import Iterator, Dict, Any

from news_crawler.scrapers.base_scraper import BaseScraper
from news_crawler.scrapers.streams import scrape_cnn_stream


class CNNScraper(BaseScraper):
    """CNN scraper."""

    def __init__(self) -> None:
        super().__init__("cnn")

    def scrape(self) -> Iterator[Dict[str, Any]]:
        for article in scrape_cnn_stream(check_link_pool=False, track_links=False):
            article.setdefault("source", self.source_name)
            if self._validate_article(article):
                yield article
