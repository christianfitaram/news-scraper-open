"""Tests for Gemini client recovery on closed-client errors."""
from __future__ import annotations

from types import SimpleNamespace

from news_crawler.processors import gemini_nlp as gemini_nlp_module
from news_crawler.providers import genai_provider as genai_provider_module


def test_gemini_nlp_retries_when_client_is_closed(monkeypatch):
    monkeypatch.setattr(
        gemini_nlp_module,
        "GENAI_CONFIG",
        SimpleNamespace(api_key="test-key", model="gemini-test", max_chunk_chars=12000),
    )

    calls = {"count": 0, "reset_count": 0}

    class _Client:
        class _Models:
            def __init__(self, should_fail: bool):
                self._should_fail = should_fail

            def generate_content(self, model, contents, config):
                del model, contents, config
                if self._should_fail:
                    raise RuntimeError("Cannot send a request, as the client has been closed.")
                return {"ok": True}

        def __init__(self, should_fail: bool):
            self.models = self._Models(should_fail)

    def _fake_get_client(cls):
        calls["count"] += 1
        return _Client(should_fail=calls["count"] == 1)

    def _fake_reset_client(cls):
        calls["reset_count"] += 1

    monkeypatch.setattr(gemini_nlp_module.GeminiNLP, "_get_client", classmethod(_fake_get_client))
    monkeypatch.setattr(gemini_nlp_module.GeminiNLP, "_reset_client", classmethod(_fake_reset_client))

    runtime = gemini_nlp_module.GeminiNLP()
    result = runtime._generate_content(model="gemini-test", contents="hello", config={})

    assert result == {"ok": True}
    assert calls["count"] == 2
    assert calls["reset_count"] == 1


def test_genai_provider_retries_when_client_is_closed(monkeypatch):
    monkeypatch.setattr(
        genai_provider_module,
        "GENAI_CONFIG",
        SimpleNamespace(api_key="test-key", model="gemini-test", temperature=0.0, max_chunk_chars=12000),
    )

    calls = {"count": 0, "reset_count": 0}

    class _Client:
        class _Models:
            def __init__(self, should_fail: bool):
                self._should_fail = should_fail

            def generate_content(self, model, contents, config):
                del model, contents, config
                if self._should_fail:
                    raise RuntimeError("Cannot send a request, as the client has been closed.")
                return {"ok": True}

        def __init__(self, should_fail: bool):
            self.models = self._Models(should_fail)

    def _fake_get_client(cls):
        calls["count"] += 1
        return _Client(should_fail=calls["count"] == 1)

    def _fake_reset_client(cls):
        calls["reset_count"] += 1

    monkeypatch.setattr(genai_provider_module.GenAIProvider, "_get_client", classmethod(_fake_get_client))
    monkeypatch.setattr(genai_provider_module.GenAIProvider, "_reset_client", classmethod(_fake_reset_client))

    provider = genai_provider_module.GenAIProvider()
    result = provider._generate_content(model="gemini-test", contents="hello", config={})

    assert result == {"ok": True}
    assert calls["count"] == 2
    assert calls["reset_count"] == 1
