"""Shared pytest fixtures."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest


@pytest.fixture
def sample_article():
    return {
        "title": "Test Article Title",
        "url": "https://example.com/article",
        "text": "This is a test article with enough text to be valid. " * 10,
        "source": "test-source",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }
