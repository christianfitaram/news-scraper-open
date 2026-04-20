"""Tests for PipelineOrchestrator in dry-run mode."""
from __future__ import annotations

from typing import Any, Dict, Iterator

import pytest

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


def test_orchestrator_stores_llm_model_metadata(monkeypatch, sample_article):
    inserted: Dict[str, Any] = {}

    class _ArticlesRepo:
        def insert_article(self, article):
            inserted["article"] = dict(article)
            return "article-id"

    class _LinkPoolRepo:
        def is_processed(self, url):
            del url
            return False

        def mark_as_processed(self, url, batch_id):
            del url, batch_id

    class _MetadataRepo:
        def insert_metadata(self, batch_id, stats):
            del batch_id, stats

    class _LogsRepo:
        def log_event(self, **kwargs):
            del kwargs

    class _GenAIProvider:
        def clean_and_extract_entities(self, text):
            return {
                "cleaned_text": text,
                "locations": [],
                "organizations": [],
                "persons": [],
                "llm_models": {
                    "cleaning": [{"provider": "genai", "model": "gemini-test"}],
                    "entity_extraction": [{"provider": "ollama", "model": "gpt-oss:120b-cloud"}],
                },
            }

    def fake_get_all_articles() -> Iterator[Dict[str, Any]]:
        yield dict(sample_article)

    def fake_summarize(text, return_metadata=False):
        del text
        if return_metadata:
            return "summary", {"summary": [{"provider": "ollama", "model": "gemma4:31b-cloud"}]}
        return "summary"

    def fake_classify(text, return_metadata=False):
        del text
        classification = {"sentiment": {"label": "NEUTRAL", "score": 0.0}, "topic": "other"}
        if return_metadata:
            return classification, {
                "classification": [{"provider": "ollama", "model": "gpt-oss:20b-cloud"}]
            }
        return classification

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
    monkeypatch.setattr(orchestrator_module, "ArticlesRepository", _ArticlesRepo)
    monkeypatch.setattr(orchestrator_module, "LinkPoolRepository", _LinkPoolRepo)
    monkeypatch.setattr(orchestrator_module, "MetadataRepository", _MetadataRepo)
    monkeypatch.setattr(orchestrator_module, "PipelineLogsRepository", _LogsRepo)
    monkeypatch.setattr(orchestrator_module, "GenAIProvider", _GenAIProvider)
    monkeypatch.setattr(orchestrator_module, "get_all_articles", fake_get_all_articles)
    monkeypatch.setattr(orchestrator_module, "smart_summarize", fake_summarize)
    monkeypatch.setattr(orchestrator_module, "classify_article", fake_classify)

    orchestrator = PipelineOrchestrator(dry_run=False)
    stats = orchestrator.run(limit=1)

    assert stats["articles_processed"] == 1
    assert inserted["article"]["llm_models"] == {
        "cleaning": [{"provider": "genai", "model": "gemini-test"}],
        "entity_extraction": [{"provider": "ollama", "model": "gpt-oss:120b-cloud"}],
        "summary": [{"provider": "ollama", "model": "gemma4:31b-cloud"}],
        "classification": [{"provider": "ollama", "model": "gpt-oss:20b-cloud"}],
    }


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
