"""The Guardian Selenium scraper."""
from __future__ import annotations

from typing import Iterator, Dict, Any

from news_crawler.scrapers.base_scraper import BaseScraper
from news_crawler.scrapers.streams import scrape_guardian_selenium_stream


class GuardianSeleniumScraper(BaseScraper):
    """Guardian Selenium scraper."""

    def __init__(self) -> None:
        super().__init__("the-guardian")

    def scrape(self) -> Iterator[Dict[str, Any]]:
        for article in scrape_guardian_selenium_stream(check_link_pool=False, track_links=False):
            article.setdefault("source", self.source_name)
            if self._validate_article(article):
                yield article
