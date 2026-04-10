"""Wall Street Journal scraper."""
from __future__ import annotations

from typing import Iterator, Dict, Any

from news_crawler.scrapers.base_scraper import BaseScraper
from news_crawler.scrapers.streams import scrape_wsj_stream


class WSJScraper(BaseScraper):
    """WSJ scraper."""

    def __init__(self) -> None:
        super().__init__("wsj")

    def scrape(self) -> Iterator[Dict[str, Any]]:
        for article in scrape_wsj_stream(check_link_pool=False, track_links=False):
            article.setdefault("source", self.source_name)
            if self._validate_article(article):
                yield article
