"""Text utilities for scraping and normalization."""
from __future__ import annotations

import logging
import re
from typing import Optional

import trafilatura

logger = logging.getLogger(__name__)


def fetch_and_extract(url: str) -> Optional[str]:
    """Fetch a URL and extract main text content using trafilatura."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        return trafilatura.extract(downloaded)
    except Exception as exc:
        logger.warning("Failed to fetch or extract %s: %s", url, exc)
        return None


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace and strip noisy artifacts."""
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized
