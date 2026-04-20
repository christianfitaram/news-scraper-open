"""Tests for webhook retry and signing behavior."""
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from queue import Queue
from threading import Lock
from types import SimpleNamespace

import requests

from news_crawler.providers import webhook_provider as webhook_module
from news_crawler.providers.webhook_provider import WebhookProvider


def _provider(**config_overrides):
    config = {
        "embedding_url": None,
        "thread_events_url": None,
        "signature": "",
        "timeout": 5,
        "max_retries": 3,
        "async_workers": 1,
        "drain_timeout_seconds": 0,
    }
    config.update(config_overrides)
    provider = object.__new__(WebhookProvider)
    provider.config = SimpleNamespace(**config)
    provider._queue = Queue()
    provider._workers = []
    provider._closed = False
    provider._closed_lock = Lock()
    return provider


class _Response:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


def test_send_webhook_success_adds_hmac_signature(monkeypatch):
    captured: dict[str, object] = {}

    def fake_post(url, data, headers, timeout):
        captured.update({"url": url, "data": data, "headers": headers, "timeout": timeout})
        return _Response(202)

    monkeypatch.setattr(webhook_module.requests, "post", fake_post)
    provider = _provider(signature="secret", timeout=9)
    payload = {"article_id": "a1", "title": "Hello"}

    assert provider._send_webhook("https://hooks.example", payload, "embedding") is True

    expected_signature = hmac.new(
        b"secret",
        json.dumps(payload, default=str).encode(),
        hashlib.sha256,
    ).hexdigest()
    assert captured["url"] == "https://hooks.example"
    assert captured["timeout"] == 9
    assert captured["headers"]["X-Signature"] == f"sha256={expected_signature}"


def test_send_webhook_retries_transient_status_then_succeeds(monkeypatch):
    statuses = [_Response(500), _Response(429), _Response(200)]
    sleeps: list[int] = []

    monkeypatch.setattr(webhook_module.requests, "post", lambda *args, **kwargs: statuses.pop(0))
    monkeypatch.setattr(webhook_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    provider = _provider(max_retries=3)

    assert provider._send_webhook("https://hooks.example", {"ok": True}, "embedding") is True
    assert sleeps == [1, 2]


def test_send_webhook_does_not_retry_client_error(monkeypatch):
    calls = 0

    def fake_post(*args, **kwargs):
        del args, kwargs
        nonlocal calls
        calls += 1
        return _Response(400, "bad request")

    monkeypatch.setattr(webhook_module.requests, "post", fake_post)
    provider = _provider(max_retries=3)

    assert provider._send_webhook("https://hooks.example", {"ok": True}, "embedding") is False
    assert calls == 1


def test_send_webhook_retries_timeout(monkeypatch):
    calls = 0
    sleeps: list[int] = []

    def fake_post(*args, **kwargs):
        del args, kwargs
        nonlocal calls
        calls += 1
        if calls == 1:
            raise requests.exceptions.Timeout
        return _Response(200)

    monkeypatch.setattr(webhook_module.requests, "post", fake_post)
    monkeypatch.setattr(webhook_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    provider = _provider(max_retries=2)

    assert provider._send_webhook("https://hooks.example", {"ok": True}, "embedding") is True
    assert sleeps == [1]


def test_send_article_webhooks_enqueues_enabled_destinations():
    provider = _provider(
        embedding_url="https://embedding.example",
        thread_events_url="https://events.example",
    )
    article = {
        "url": "https://example.com/a",
        "title": "Title",
        "text": "Body",
        "topic": "business",
        "source": "example",
        "sentiment": {"label": "NEUTRAL"},
        "scraped_at": datetime(2026, 4, 20, tzinfo=timezone.utc),
    }

    provider.send_article_webhooks("article-1", article)

    first = provider._queue.get_nowait()
    second = provider._queue.get_nowait()
    assert first[0] == "https://embedding.example"
    assert first[2] == "embedding"
    assert first[1]["scraped_at"] == "2026-04-20T00:00:00+00:00"
    assert second[0] == "https://events.example"
    assert second[2] == "thread-events"
