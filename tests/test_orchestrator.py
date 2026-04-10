"""Tests for PipelineOrchestrator in dry-run mode."""
from __future__ import annotations

import pytest
from typing import Iterator, Dict, Any

from news_crawler.core import config as config_module
from news_crawler.core import orchestrator as orchestrator_module
from news_crawler.core.orchestrator import PipelineOrchestrator


def test_orchestrator_dry_run(monkeypatch, sample_article):
    def fake_get_all_articles() -> Iterator[Dict[str, Any]]:
        yield sample_article

    monkeypatch.setattr("news_crawler.core.orchestrator.get_all_articles", fake_get_all_articles)
    monkeypatch.setattr("news_crawler.core.orchestrator.smart_summarize", lambda text: "summary")
    monkeypatch.setattr("news_crawler.core.orchestrator.classify_article", lambda text: {"sentiment": {"label": "NEUTRAL", "score": 0.0}, "topic": "other"})

    orchestrator = PipelineOrchestrator(dry_run=True)
    stats = orchestrator.run(limit=1)

    assert stats["articles_processed"] == 1
    assert stats["articles_failed"] == 0


def test_orchestrator_falls_back_to_ollama_when_genai_is_unconfigured(monkeypatch):
    monkeypatch.setattr(
        orchestrator_module,
        "APP_CONFIG",
        config_module.AppConfig(enable_genai=True, enable_ollama=True, enable_webhooks=False),
    )
    monkeypatch.setattr(
        config_module,
        "MONGO_CONFIG",
        config_module.MongoConfig(uri="mongodb://localhost:27017", db_name="test-db"),
    )
    monkeypatch.setattr(orchestrator_module, "ArticlesRepository", lambda: object())
    monkeypatch.setattr(orchestrator_module, "LinkPoolRepository", lambda: object())
    monkeypatch.setattr(orchestrator_module, "MetadataRepository", lambda: object())
    monkeypatch.setattr(orchestrator_module, "PipelineLogsRepository", lambda: object())
    monkeypatch.setattr(
        orchestrator_module,
        "GenAIProvider",
        lambda: (_ for _ in ()).throw(ValueError("GenAI configuration not available")),
    )

    orchestrator = PipelineOrchestrator(dry_run=False)

    assert orchestrator.genai is None
    assert orchestrator.ollama is not None


def test_orchestrator_requires_genai_when_ollama_is_disabled(monkeypatch):
    monkeypatch.setattr(
        orchestrator_module,
        "APP_CONFIG",
        config_module.AppConfig(enable_genai=True, enable_ollama=False, enable_webhooks=False),
    )
    monkeypatch.setattr(
        config_module,
        "MONGO_CONFIG",
        config_module.MongoConfig(uri="mongodb://localhost:27017", db_name="test-db"),
    )
    monkeypatch.setattr(config_module, "GENAI_CONFIG", None)

    with pytest.raises(ValueError, match="GEMINI_API_KEY or GOOGLE_API_KEY is required"):
        PipelineOrchestrator(dry_run=False)
