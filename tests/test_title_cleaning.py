"""Tests for title normalization helpers used by scrapers."""
from __future__ import annotations

from news_crawler.scrapers.streams import _strip_numeric_prefix


def test_strip_numeric_prefix_removes_list_markers() -> None:
    assert _strip_numeric_prefix("1. France24 headline") == "France24 headline"
    assert _strip_numeric_prefix("2) France24 headline") == "France24 headline"
    assert _strip_numeric_prefix("[3]: France24 headline") == "France24 headline"
    assert _strip_numeric_prefix("#4 - France24 headline") == "France24 headline"


def test_strip_numeric_prefix_preserves_regular_numeric_headline() -> None:
    assert _strip_numeric_prefix("100-day plan announced") == "100-day plan announced"
    assert _strip_numeric_prefix("2026 outlook for Europe") == "2026 outlook for Europe"
