"""State management for pipeline runs."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from news_crawler.core.config import APP_CONFIG

logger = logging.getLogger(__name__)


class StateManager:
    """Persist and load pipeline state to allow resumable runs."""

    def __init__(self, state_file: Optional[str] = None) -> None:
        self.state_file = Path(state_file or APP_CONFIG.state_file)
        self._state: Dict[str, Any] = {}
        self.load()

    def load(self) -> Dict[str, Any]:
        """Load state from JSON file if present."""
        if self.state_file.exists():
            try:
                self._state = json.loads(self.state_file.read_text(encoding="utf-8"))
                logger.info("State loaded from %s", self.state_file)
            except Exception as exc:
                logger.warning("Failed to load state (%s); starting fresh", exc)
                self._state = {}
        else:
            logger.info("No existing state file; starting fresh")
            self._state = {}
        return self._state

    def save(self) -> None:
        """Persist current state to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(
                json.dumps(self._state, indent=2, default=str),
                encoding="utf-8",
            )
            logger.debug("State saved to %s", self.state_file)
        except Exception as exc:
            logger.error("Failed to save state: %s", exc)

    def get_last_batch_id(self) -> Optional[str]:
        return self._state.get("last_batch_id")

    def set_last_batch_id(self, batch_id: str) -> None:
        self._state["last_batch_id"] = batch_id
        self._state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.save()

    def get_scraper_state(self, scraper_name: str) -> Dict[str, Any]:
        return self._state.get("scrapers", {}).get(scraper_name, {})

    def set_scraper_state(
        self,
        scraper_name: str,
        last_url: Optional[str] = None,
        articles_processed: Optional[int] = None,
        last_run: Optional[str] = None,
    ) -> None:
        scrapers = self._state.setdefault("scrapers", {})
        scraper = scrapers.setdefault(scraper_name, {})

        if last_url is not None:
            scraper["last_url"] = last_url
        if articles_processed is not None:
            scraper["articles_processed"] = articles_processed
        if last_run is not None:
            scraper["last_run"] = last_run

        scraper["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save()

    def get_pipeline_stats(self) -> Dict[str, Any]:
        return self._state.get(
            "stats",
            {"total_articles_processed": 0, "total_batches": 0, "last_run": None},
        )

    def increment_stats(self, articles_count: int) -> None:
        stats = self._state.setdefault(
            "stats",
            {"total_articles_processed": 0, "total_batches": 0, "last_run": None},
        )
        stats["total_articles_processed"] += int(articles_count)
        stats["total_batches"] += 1
        stats["last_run"] = datetime.now(timezone.utc).isoformat()
        self.save()

    def reset(self, scraper_name: Optional[str] = None) -> None:
        if scraper_name:
            scrapers = self._state.get("scrapers", {})
            if scraper_name in scrapers:
                scrapers.pop(scraper_name, None)
                logger.info("State reset for scraper: %s", scraper_name)
        else:
            self._state = {}
            logger.info("All state reset")
        self.save()
