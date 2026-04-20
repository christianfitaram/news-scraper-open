"""Deutsche Welle scraper."""
from __future__ import annotations

from typing import Any, Dict, Iterator

from news_crawler.scrapers.base_scraper import BaseScraper
from news_crawler.scrapers.streams import scrape_dw_stream


class DWScraper(BaseScraper):
    """DW scraper."""

    def __init__(self) -> None:
        super().__init__("dw")

    def scrape(self) -> Iterator[Dict[str, Any]]:
        for article in scrape_dw_stream(check_link_pool=False, track_links=False):
            article.setdefault("source", self.source_name)
            if self._validate_article(article):
                yield article
