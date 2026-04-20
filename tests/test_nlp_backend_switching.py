"""Tests for local/Gemini NLP backend switching."""
from __future__ import annotations

from news_crawler.core.config import AppConfig
from news_crawler.processors import classifier as classifier_module
from news_crawler.processors import summarizer as summarizer_module


def test_smart_summarize_uses_gemini_backend(monkeypatch):
    class _FakeGeminiRuntime:
        def summarize(self, text: str) -> str:
            return f"gemini:{text[:10]}"

    monkeypatch.setattr(
        summarizer_module,
        "APP_CONFIG",
        AppConfig(summarizer_backend="gemini"),
    )
    monkeypatch.setattr("news_crawler.processors.gemini_nlp.get_runtime", lambda: _FakeGeminiRuntime())

    result = summarizer_module.smart_summarize("A" * 300)

    assert result == "gemini:AAAAAAAAAA"


def test_classify_article_uses_gemini_backend(monkeypatch):
    class _FakeGeminiRuntime:
        def sentiment(self, text: str):
            return {"label": "POSITIVE", "score": 0.91}

        def topic(self, text: str, candidate_labels=None):
            return "technology and innovation"

    monkeypatch.setattr(
        classifier_module,
        "APP_CONFIG",
        AppConfig(sentiment_backend="gemini", topic_backend="gemini"),
    )
    monkeypatch.setattr("news_crawler.processors.gemini_nlp.get_runtime", lambda: _FakeGeminiRuntime())

    result = classifier_module.classify_article("This is a long enough text " * 20)

    assert result["sentiment"]["label"] == "POSITIVE"
    assert result["topic"] == "technology and innovation"


def test_classify_article_prefers_combined_gemini_call(monkeypatch):
    class _FakeGeminiRuntime:
        def __init__(self) -> None:
            self.calls = 0

        def classify(self, text: str, candidate_labels=None):
            self.calls += 1
            return {
                "sentiment": {"label": "NEGATIVE", "score": 0.72},
                "topic": "business and finance",
            }

        def sentiment(self, text: str):
            raise AssertionError("sentiment should not be called when classify() exists")

        def topic(self, text: str, candidate_labels=None):
            raise AssertionError("topic should not be called when classify() exists")

    runtime = _FakeGeminiRuntime()
    monkeypatch.setattr(
        classifier_module,
        "APP_CONFIG",
        AppConfig(sentiment_backend="gemini", topic_backend="gemini"),
    )
    monkeypatch.setattr("news_crawler.processors.gemini_nlp.get_runtime", lambda: runtime)

    result = classifier_module.classify_article("This is a long enough text " * 20)

    assert runtime.calls == 1
    assert result["sentiment"]["label"] == "NEGATIVE"
    assert result["topic"] == "business and finance"


def test_classify_article_falls_back_to_local_on_gemini_error(monkeypatch):
    monkeypatch.setattr(
        classifier_module,
        "APP_CONFIG",
        AppConfig(sentiment_backend="gemini", topic_backend="gemini"),
    )
    monkeypatch.setattr(
        "news_crawler.processors.gemini_nlp.get_runtime",
        lambda: (_ for _ in ()).throw(RuntimeError("gemini unavailable")),
    )
    monkeypatch.setattr(
        classifier_module,
        "analyze_sentiment",
        lambda text: {"label": "NEGATIVE", "score": 0.6},
    )
    monkeypatch.setattr(classifier_module, "classify_topic", lambda text: "business and finance")

    result = classifier_module.classify_article("This is a long enough text " * 20)

    assert result["sentiment"]["label"] == "NEGATIVE"
    assert result["topic"] == "business and finance"
