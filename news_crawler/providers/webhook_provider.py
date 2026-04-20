"""Provider for sending webhooks with retries and signing."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from datetime import datetime
from queue import Queue
from threading import Lock, Thread
from typing import Any, Dict, Optional, Tuple

import requests

from news_crawler.core.config import WEBHOOK_CONFIG

logger = logging.getLogger(__name__)


class WebhookProvider:
    """Queue and dispatch webhooks asynchronously with retry backoff."""

    def __init__(self) -> None:
        self.config = WEBHOOK_CONFIG
        self._queue: Queue[Optional[Tuple[str, Dict[str, Any], str]]] = Queue()
        self._workers: list[Thread] = []
        self._closed = False
        self._closed_lock = Lock()

        for idx in range(max(1, self.config.async_workers)):
            worker = Thread(target=self._worker_loop, name=f"webhook-outbox-{idx}", daemon=True)
            worker.start()
            self._workers.append(worker)

        logger.info("WebhookProvider initialized (async_workers=%s)", len(self._workers))

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
            self._enqueue_webhook(self.config.embedding_url, payload, "embedding")

        if self.config.thread_events_url:
            payload = {
                "article_id": str(article_id),
                "source": article.get("source", ""),
                "scraped_at": scraped_at,
            }
            self._enqueue_webhook(self.config.thread_events_url, payload, "thread-events")

    def _enqueue_webhook(self, url: str, payload: Dict[str, Any], webhook_name: str) -> None:
        with self._closed_lock:
            if self._closed:
                logger.warning("Webhook outbox is closed; dropping %s event", webhook_name)
                return
        self._queue.put((url, payload, webhook_name))

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return

            url, payload, webhook_name = item
            try:
                self._send_webhook(url, payload, webhook_name)
            except Exception as exc:
                logger.error("Webhook outbox worker error for %s: %s", webhook_name, exc)
            finally:
                self._queue.task_done()

    def flush(self, timeout_seconds: float | None = None) -> bool:
        if timeout_seconds is None:
            self._queue.join()
            return True

        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        while time.monotonic() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.05)
        return self._queue.unfinished_tasks == 0

    def close(self) -> None:
        with self._closed_lock:
            if self._closed:
                return
            self._closed = True

        drained = self.flush(timeout_seconds=float(self.config.drain_timeout_seconds))
        if not drained:
            logger.warning(
                "Webhook outbox did not fully drain within %ss",
                self.config.drain_timeout_seconds,
            )

        for _ in self._workers:
            self._queue.put(None)
        for worker in self._workers:
            worker.join(timeout=1.0)

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

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
