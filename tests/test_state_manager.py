"""Tests for the StateManager."""
from __future__ import annotations

from pathlib import Path

from news_crawler.core.state import StateManager


def test_state_manager_roundtrip(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    manager = StateManager(state_file=str(state_file))

    manager.set_last_batch_id("batch-123")
    manager.set_scraper_state("bbc", last_url="https://example.com", articles_processed=3)

    reloaded = StateManager(state_file=str(state_file))

    assert reloaded.get_last_batch_id() == "batch-123"
    assert reloaded.get_scraper_state("bbc").get("last_url") == "https://example.com"
    assert reloaded.get_scraper_state("bbc").get("articles_processed") == 3
