"""Additional tests for classifier fallback behavior."""
from __future__ import annotations

from news_crawler.core.config import AppConfig
from news_crawler.processors import classifier as classifier_module


def test_classify_article_short_text_returns_metadata_when_requested(monkeypatch):
    monkeypatch.setattr(classifier_module, "APP_CONFIG", AppConfig())

    classification, metadata = classifier_module.classify_article("too short", return_metadata=True)

    assert classification == {"sentiment": {"label": "NEUTRAL", "score": 0.0}, "topic": "other"}
    assert metadata == {}


def test_local_classifier_falls_back_when_both_models_fail(monkeypatch):
    monkeypatch.setattr(classifier_module, "APP_CONFIG", AppConfig())
    monkeypatch.setattr(
        classifier_module,
        "analyze_sentiment",
        lambda text: (_ for _ in ()).throw(RuntimeError("sentiment unavailable")),
    )
    monkeypatch.setattr(
        classifier_module,
        "classify_topic",
        lambda text: (_ for _ in ()).throw(RuntimeError("topic unavailable")),
    )

    result = classifier_module.classify_article("This article is long enough for classification." * 5)

    assert result == {"sentiment": {"label": "NEUTRAL", "score": 0.0}, "topic": "other"}


def test_gemini_sentiment_backend_stores_step_metadata(monkeypatch):
    class _Runtime:
        def sentiment_with_metadata(self, text):
            del text
            return {"label": "POSITIVE", "score": 0.8}, [{"provider": "genai", "model": "sentiment"}]

    monkeypatch.setattr(
        classifier_module,
        "APP_CONFIG",
        AppConfig(sentiment_backend="gemini", topic_backend="local"),
    )
    monkeypatch.setattr("news_crawler.processors.gemini_nlp.get_runtime", lambda: _Runtime())
    monkeypatch.setattr(classifier_module, "classify_topic", lambda text: "technology and innovation")

    classification, metadata = classifier_module.classify_article(
        "This article is long enough for classification." * 5,
        return_metadata=True,
    )

    assert classification["sentiment"]["label"] == "POSITIVE"
    assert classification["topic"] == "technology and innovation"
    assert metadata == {"sentiment": [{"provider": "genai", "model": "sentiment"}]}
