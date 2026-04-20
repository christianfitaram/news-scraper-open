"""Tests for Gemini NLP payload parsing and normalization."""
from __future__ import annotations

from types import SimpleNamespace

from news_crawler.processors import gemini_nlp

MODEL = {"provider": "genai", "model": "gemini-test"}


def _runtime(max_chunk_chars: int = 12000):
    runtime = object.__new__(gemini_nlp.GeminiNLP)
    runtime.config = SimpleNamespace(model="gemini-test", max_chunk_chars=max_chunk_chars)
    return runtime


def test_payload_reads_parsed_dict_and_model_dump():
    assert gemini_nlp.GeminiNLP._payload(SimpleNamespace(parsed={"summary": "from parsed"})) == {
        "summary": "from parsed"
    }

    class _ParsedModel:
        def model_dump(self):
            return {"summary": "from model_dump"}

    assert gemini_nlp.GeminiNLP._payload(SimpleNamespace(parsed=_ParsedModel())) == {
        "summary": "from model_dump"
    }
    assert gemini_nlp.GeminiNLP._payload(SimpleNamespace(text='{"summary": "from text"}')) == {
        "summary": "from text"
    }
    assert gemini_nlp.GeminiNLP._payload(SimpleNamespace(text="not json")) == {}


def test_sentiment_with_metadata_normalizes_label_and_clamps_score(monkeypatch):
    runtime = _runtime()

    def fake_generate(**kwargs):
        del kwargs
        return SimpleNamespace(text='{"label": "excited", "score": 2.5}'), MODEL

    monkeypatch.setattr(runtime, "_generate_content_with_model_info", fake_generate)

    sentiment, models = runtime.sentiment_with_metadata("A useful article")

    assert sentiment == {"label": "NEUTRAL", "score": 1.0}
    assert models == [MODEL]


def test_topic_with_metadata_rejects_unknown_topic(monkeypatch):
    runtime = _runtime()

    def fake_generate(**kwargs):
        del kwargs
        return SimpleNamespace(text='{"topic": "not a label", "score": 0.9}'), MODEL

    monkeypatch.setattr(runtime, "_generate_content_with_model_info", fake_generate)

    topic, models = runtime.topic_with_metadata(
        "A useful article",
        candidate_labels=["business and finance"],
    )

    assert topic == "other"
    assert models == [MODEL]


def test_classify_with_metadata_normalizes_invalid_payload(monkeypatch):
    runtime = _runtime()

    def fake_generate(**kwargs):
        del kwargs
        return (
            SimpleNamespace(
                text='{"sentiment": {"label": "bad", "score": -4}, "topic": "not valid"}'
            ),
            MODEL,
        )

    monkeypatch.setattr(runtime, "_generate_content_with_model_info", fake_generate)

    classification, models = runtime.classify_with_metadata(
        "A useful article",
        candidate_labels=["business and finance"],
    )

    assert classification == {"sentiment": {"label": "NEUTRAL", "score": 0.0}, "topic": "other"}
    assert models == [MODEL]


def test_summarize_with_metadata_merges_chunk_summaries(monkeypatch):
    runtime = _runtime(max_chunk_chars=10)
    responses = [
        SimpleNamespace(text='{"summary": "first"}'),
        SimpleNamespace(text='{"summary": "second"}'),
    ]

    monkeypatch.setattr(gemini_nlp, "chunk_text", lambda text, max_chars: ["chunk one", "chunk two"])

    def fake_generate(**kwargs):
        del kwargs
        return responses.pop(0), MODEL

    monkeypatch.setattr(runtime, "_generate_content_with_model_info", fake_generate)

    summary, models = runtime.summarize_with_metadata("A" * 250)

    assert summary == "first\nsecond"
    assert models == [MODEL]

