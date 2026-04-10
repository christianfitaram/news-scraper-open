"""Pipeline orchestrator for scraping and processing."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from news_crawler.core.config import APP_CONFIG, WEBHOOK_CONFIG, validate_config
from news_crawler.core.state import StateManager
from news_crawler.processors.classifier import classify_article
from news_crawler.processors.summarizer import smart_summarize
from news_crawler.providers.genai_provider import GenAIProvider
from news_crawler.providers.ollama_provider import OllamaProvider
from news_crawler.providers.webhook_provider import WebhookProvider
from news_crawler.repositories.articles_repository import ArticlesRepository
from news_crawler.repositories.link_pool_repository import LinkPoolRepository
from news_crawler.repositories.metadata_repository import MetadataRepository
from news_crawler.repositories.pipeline_logs_repository import PipelineLogsRepository
from news_crawler.scrapers import get_all_articles

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the scraping, enrichment, and persistence pipeline."""

    def __init__(self, state_manager: Optional[StateManager] = None, dry_run: bool = False) -> None:
        self.state_manager = state_manager or StateManager()
        self.dry_run = dry_run
        self._seen_urls = set()

        require_genai = APP_CONFIG.enable_genai and not APP_CONFIG.enable_ollama and not dry_run
        validate_config(require_db=not dry_run, require_genai=require_genai)

        self.articles_repo = None
        self.link_pool_repo = None
        self.metadata_repo = None
        self.logs_repo = None

        if not dry_run:
            self.articles_repo = ArticlesRepository()
            self.link_pool_repo = LinkPoolRepository()
            self.metadata_repo = MetadataRepository()
            self.logs_repo = PipelineLogsRepository()

        self.genai = None
        self.ollama = None
        if APP_CONFIG.enable_genai:
            try:
                self.genai = GenAIProvider()
            except Exception as exc:
                logger.warning("GenAI disabled: %s", exc)
        if self.genai is None and APP_CONFIG.enable_ollama:
            try:
                self.ollama = OllamaProvider()
            except Exception as exc:
                logger.warning("Ollama disabled: %s", exc)

        self.webhook = None
        if APP_CONFIG.enable_webhooks and (WEBHOOK_CONFIG.embedding_url or WEBHOOK_CONFIG.thread_events_url):
            self.webhook = WebhookProvider()

        logger.info("PipelineOrchestrator initialized (dry_run=%s)", dry_run)

    def run(self, limit: Optional[int] = None) -> Dict[str, Any]:
        batch_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        logger.info("Starting pipeline run (batch_id=%s)", batch_id)

        stats: Dict[str, Any] = {
            "batch_id": batch_id,
            "started_at": start_time.isoformat(),
            "articles_processed": 0,
            "articles_failed": 0,
            "articles_skipped": 0,
            "scrapers_used": set(),
        }

        try:
            for idx, article in enumerate(get_all_articles()):
                if limit and idx >= limit:
                    logger.info("Reached limit of %s articles", limit)
                    break

                try:
                    result = self._process_article(article, batch_id)
                    status = result.get("status")
                    if status == "success":
                        stats["articles_processed"] += 1
                    elif status == "skipped":
                        stats["articles_skipped"] += 1
                    else:
                        stats["articles_failed"] += 1

                    stats["scrapers_used"].add(article.get("source", "unknown"))
                except Exception as exc:
                    logger.error("Error processing article: %s", exc, exc_info=True)
                    stats["articles_failed"] += 1

            end_time = datetime.now(timezone.utc)
            stats["finished_at"] = end_time.isoformat()
            stats["duration_seconds"] = (end_time - start_time).total_seconds()
            stats["scrapers_used"] = list(stats["scrapers_used"])

            if not self.dry_run and self.metadata_repo:
                self._save_batch_metadata(batch_id, stats)
                self.state_manager.set_last_batch_id(batch_id)
                self.state_manager.increment_stats(stats["articles_processed"])

            logger.info(
                "Pipeline run completed: %s processed, %s skipped, %s failed",
                stats["articles_processed"],
                stats["articles_skipped"],
                stats["articles_failed"],
            )
            return stats
        except Exception as exc:
            logger.error("Pipeline run failed: %s", exc, exc_info=True)
            raise

    def _process_article(self, article: Dict[str, Any], batch_id: str) -> Dict[str, str]:
        url = article.get("url", "")
        if not url:
            return {"status": "failed", "reason": "missing_url"}

        if self.dry_run:
            if url in self._seen_urls:
                return {"status": "skipped", "reason": "already_seen"}
            self._seen_urls.add(url)
        else:
            if self.link_pool_repo and self.link_pool_repo.is_processed(url):
                return {"status": "skipped", "reason": "already_processed"}

        article["scraped_at"] = self._normalize_scraped_at(article.get("scraped_at"))

        if self.genai:
            cleaned_data = self.genai.clean_and_extract_entities(article.get("text", ""))
        elif self.ollama:
            cleaned_data = self.ollama.clean_and_extract_entities(article.get("text", ""))
        else:
            cleaned_data = None

        if cleaned_data:
            article["text"] = cleaned_data.get("cleaned_text", article.get("text"))
            article["locations"] = cleaned_data.get("locations", [])
            article["organizations"] = cleaned_data.get("organizations", [])
            article["persons"] = cleaned_data.get("persons", [])

        article["summary"] = smart_summarize(article.get("text", ""))
        article.update(classify_article(article.get("text", "")))

        if not self.dry_run and self.articles_repo and self.link_pool_repo:
            article["sample"] = batch_id
            article_id = self.articles_repo.insert_article(article)
            self.link_pool_repo.mark_as_processed(url, batch_id)

            if self.webhook:
                self.webhook.send_article_webhooks(article_id, article)

            if self.logs_repo:
                self.logs_repo.log_event(
                    action="process_article",
                    actor=article.get("source", "orchestrator"),
                    article_id=str(article_id),
                    batch_id=batch_id,
                    status="success",
                )

        scraper_name = article.get("source", "unknown")
        prev_state = self.state_manager.get_scraper_state(scraper_name)
        prev_count = int(prev_state.get("articles_processed", 0) or 0)
        self.state_manager.set_scraper_state(
            scraper_name=scraper_name,
            last_url=url,
            articles_processed=prev_count + 1,
            last_run=datetime.now(timezone.utc).isoformat(),
        )

        logger.info("Processed: %s", str(article.get("title", "Untitled")))
        return {"status": "success"}

    def _save_batch_metadata(self, batch_id: str, stats: Dict[str, Any]) -> None:
        if not self.metadata_repo:
            return
        try:
            self.metadata_repo.insert_metadata(batch_id, stats)
            logger.debug("Batch metadata saved: %s", batch_id)
        except Exception as exc:
            logger.error("Error saving batch metadata: %s", exc)

    @staticmethod
    def _normalize_scraped_at(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if value is None:
            return datetime.now(timezone.utc)
        if isinstance(value, str):
            text = value.strip()
            if text.endswith("Z"):
                text = f"{text[:-1]}+00:00"
            try:
                parsed = datetime.fromisoformat(text)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
            except ValueError:
                logger.warning("Invalid scraped_at value; using current UTC time: %s", value)
                return datetime.now(timezone.utc)
        logger.warning("Unexpected scraped_at type %s; using current UTC time", type(value).__name__)
        return datetime.now(timezone.utc)
