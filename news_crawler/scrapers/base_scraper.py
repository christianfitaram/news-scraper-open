"""Base scraper interface."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Base class for all news scrapers."""

    def __init__(self, source_name: str) -> None:
        self.source_name = source_name
        logger.info("Scraper initialized: %s", source_name)

    @abstractmethod
    def scrape(self) -> Iterator[Dict[str, Any]]:
        """Yield article dicts with title, url, text, source, scraped_at."""
        raise NotImplementedError

    def _validate_article(self, article: Dict[str, Any]) -> bool:
        required_fields = ["title", "url", "text", "source"]
        for field in required_fields:
            if not article.get(field):
                logger.warning("Article missing required field: %s", field)
                return False

        if len(str(article.get("text", "")).strip()) < 100:
            logger.warning("Article text too short: %s chars", len(str(article.get("text", ""))))
            return False
        return True
