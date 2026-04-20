"""Scraper registry and aggregator."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterator, List, Type

from news_crawler.scrapers.aljazeera_scraper import AlJazeeraScraper
from news_crawler.scrapers.base_scraper import BaseScraper
from news_crawler.scrapers.bbc_scraper import BBCScraper
from news_crawler.scrapers.cnn_scraper import CNNScraper
from news_crawler.scrapers.dw_scraper import DWScraper
from news_crawler.scrapers.france24_scraper import France24Scraper
from news_crawler.scrapers.guardian_scraper import GuardianScraper
from news_crawler.scrapers.guardian_selenium_scraper import GuardianSeleniumScraper
from news_crawler.scrapers.newsapi_scraper import NewsAPIScraper
from news_crawler.scrapers.npr_scraper import NPRScraper
from news_crawler.scrapers.reuters_scraper import ReutersScraper
from news_crawler.scrapers.wsj_scraper import WSJScraper

logger = logging.getLogger(__name__)

_ENABLE_SELENIUM = os.getenv("ENABLE_SELENIUM_SCRAPERS", "1") in {"1", "true", "True"}
_ENABLE_NEWSAPI = os.getenv("ENABLE_NEWSAPI_SCRAPER", "0") in {"1", "true", "True"}

ACTIVE_SCRAPERS: List[Type[BaseScraper]] = [
    BBCScraper,
    CNNScraper,
    WSJScraper,
    AlJazeeraScraper,
    DWScraper,
    GuardianScraper,
    ReutersScraper,
]

if _ENABLE_SELENIUM:
    ACTIVE_SCRAPERS.extend([GuardianSeleniumScraper, France24Scraper, NPRScraper])

if _ENABLE_NEWSAPI:
    ACTIVE_SCRAPERS.append(NewsAPIScraper)


def get_all_articles() -> Iterator[Dict[str, Any]]:
    """Yield articles from all active scrapers."""
    for scraper_class in ACTIVE_SCRAPERS:
        try:
            scraper = scraper_class()
            logger.info("Starting scraper: %s", scraper.source_name)
            for article in scraper.scrape():
                yield article
        except Exception as exc:
            logger.error("Scraper %s failed: %s", scraper_class.__name__, exc, exc_info=True)
