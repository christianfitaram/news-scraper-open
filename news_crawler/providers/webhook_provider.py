"""Provider for sending webhooks with retries and signing."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict

import requests

from news_crawler.core.config import WEBHOOK_CONFIG

logger = logging.getLogger(__name__)


class WebhookProvider:
    """Send webhooks with HMAC signature and retry backoff."""

    def __init__(self) -> None:
        self.config = WEBHOOK_CONFIG
        logger.info("WebhookProvider initialized")

    def send_article_webhooks(self, article_id: str, article: Dict[str, Any]) -> None:
        scraped_at = self._serialize_scraped_at(article.get("scraped_at"))
        if self.config.embedding_url:
            payload = {
                "article_id": str(article_id),
                "url": article.get("url", ""),
                "title": article.get("title", ""),
                "text": article.get("text", ""),
                "topic": article.get("topic", ""),
                "source": article.get("source", ""),
                "sentiment": article.get("sentiment", {}),
                "scraped_at": scraped_at,
            }
            self._send_webhook(self.config.embedding_url, payload, "embedding")

        if self.config.thread_events_url:
            payload = {
                "article_id": str(article_id),
                "source": article.get("source", ""),
                "scraped_at": scraped_at,
            }
            self._send_webhook(self.config.thread_events_url, payload, "thread-events")

    def _send_webhook(self, url: str, payload: Dict[str, Any], webhook_name: str) -> bool:
        payload_json = json.dumps(payload, default=str)
        signature = ""
        if self.config.signature:
            signature = hmac.new(
                self.config.signature.encode(),
                payload_json.encode(),
                hashlib.sha256,
            ).hexdigest()

        headers = {"Content-Type": "application/json", "X-Signature": f"sha256={signature}"}

        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    url,
                    data=payload_json,
                    headers=headers,
                    timeout=self.config.timeout,
                )

                if response.status_code in {200, 201, 202}:
                    logger.debug("Webhook %s sent successfully", webhook_name)
                    return True
                if response.status_code in {429, 500, 502, 503, 504}:
                    wait_time = 2**attempt
                    logger.warning(
                        "Webhook %s failed with %s; retrying in %ss (attempt %s/%s)",
                        webhook_name,
                        response.status_code,
                        wait_time,
                        attempt + 1,
                        self.config.max_retries,
                    )
                    time.sleep(wait_time)
                    continue

                logger.error(
                    "Webhook %s failed with %s: %s",
                    webhook_name,
                    response.status_code,
                    response.text,
                )
                return False
            except requests.exceptions.Timeout:
                logger.error("Webhook %s timed out (attempt %s)", webhook_name, attempt + 1)
                if attempt < self.config.max_retries - 1:
                    time.sleep(2**attempt)
            except Exception as exc:
                logger.error("Webhook %s error: %s", webhook_name, exc)
                return False

        logger.error("Webhook %s failed after %s attempts", webhook_name, self.config.max_retries)
        return False

    @staticmethod
    def _serialize_scraped_at(value: Any) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        if value is None:
            return ""
        return str(value)
