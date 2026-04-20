"""Shared pytest fixtures."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from news_crawler.providers._genai_quota import reset_quota_cooldown


@pytest.fixture(autouse=True)
def _reset_genai_quota_cooldown():
    reset_quota_cooldown()
    yield
    reset_quota_cooldown()


@pytest.fixture
def sample_article():
    return {
        "title": "Test Article Title",
        "url": "https://example.com/article",
        "text": "This is a test article with enough text to be valid. " * 10,
        "source": "test-source",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }
