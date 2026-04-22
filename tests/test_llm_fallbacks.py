"""Tests for Gemini-to-Ollama LLM fallbacks."""
from __future__ import annotations

from types import SimpleNamespace

from news_crawler.providers import genai_provider as genai_provider_module
from news_crawler.providers import ollama_provider as ollama_provider_module
from news_crawler.providers._genai_quota import is_quota_cooldown_active


def test_ollama_provider_tries_fallback_models_in_order(monkeypatch):
    calls: list[str] = []

    def _fake_generate_with_model(self, *, model, prompt, timeout, options=None):
        del self, prompt, timeout, options
        calls.append(model)
        if model != "gemma4:31b-cloud":
            raise RuntimeError(f"quota exhausted for {model}")
        return '{"summary": "fallback summary"}'

    monkeypatch.setattr(ollama_provider_module.OllamaProvider, "_generate_with_model", _fake_generate_with_model)

    provider = ollama_provider_module.OllamaProvider(
        model_sequence=[
            "gpt-oss:120b-cloud",
            "gemma4:31b-cloud",
            "gpt-oss:20b-cloud",
        ]
    )
    response, model = provider.generate_json_with_model(
        system_instruction="Return JSON.",
        contents="Summarize this.",
    )

    assert response == '{"summary": "fallback summary"}'
    assert model == {"provider": "ollama", "model": "gemma4:31b-cloud"}
    assert calls == ["gpt-oss:120b-cloud", "gemma4:31b-cloud"]


def test_default_gemini_fallback_models_include_final_gemma2(monkeypatch):
    monkeypatch.delenv("OLLAMA_FALLBACK_MODELS", raising=False)

    assert ollama_provider_module._gemini_fallback_models() == [
        "gpt-oss:120b-cloud",
        "gemma4:31b-cloud",
        "gpt-oss:20b-cloud",
        "gemma2:9b",
    ]


def test_genai_provider_falls_back_to_ollama_after_quota_error(monkeypatch):
    monkeypatch.setattr(
        genai_provider_module,
        "GENAI_CONFIG",
        SimpleNamespace(api_key="test-key", model="gemini-test", temperature=0.0, max_chunk_chars=12000),
    )

    class _Client:
        class _Models:
            @staticmethod
            def generate_content(model, contents, config):
                del model, contents, config
                raise RuntimeError("429 RESOURCE_EXHAUSTED")

        models = _Models()

    class _FallbackProvider:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def generate_json_with_model(self, *, system_instruction, contents, timeout=None, options=None):
            del timeout, options
            self.calls.append({"system_instruction": system_instruction, "contents": contents})
            return '{"cleaned_text": "ollama cleaned"}', {
                "provider": "ollama",
                "model": "gpt-oss:120b-cloud",
            }

    fallback_provider = _FallbackProvider()

    monkeypatch.setattr(genai_provider_module.GenAIProvider, "_get_client", classmethod(lambda cls: _Client()))
    monkeypatch.setattr(
        ollama_provider_module,
        "get_gemini_fallback_provider",
        lambda: fallback_provider,
    )

    provider = genai_provider_module.GenAIProvider()
    response = provider._generate_content(
        model="gemini-test",
        contents="TEXT TO CLEAN:\nraw",
        config=SimpleNamespace(system_instruction="clean exactly"),
    )

    assert response.text == '{"cleaned_text": "ollama cleaned"}'
    assert fallback_provider.calls == [
        {"system_instruction": "clean exactly", "contents": "TEXT TO CLEAN:\nraw"}
    ]

    response, model = provider._generate_content_with_model_info(
        model="gemini-test",
        contents="TEXT TO CLEAN:\nraw",
        config=SimpleNamespace(system_instruction="clean exactly"),
    )
    assert response.text == '{"cleaned_text": "ollama cleaned"}'
    assert model == {"provider": "ollama", "model": "gpt-oss:120b-cloud"}


def test_genai_provider_skips_gemini_while_quota_cooldown_is_active(monkeypatch):
    monkeypatch.setattr(
        genai_provider_module,
        "GENAI_CONFIG",
        SimpleNamespace(api_key="test-key", model="gemini-test", temperature=0.0, max_chunk_chars=12000),
    )

    calls = {"gemini": 0, "fallback": 0}

    class _Client:
        class _Models:
            @staticmethod
            def generate_content(model, contents, config):
                del model, contents, config
                calls["gemini"] += 1
                raise RuntimeError("429 RESOURCE_EXHAUSTED {'details': [{'retryDelay': '120s'}]}")

        models = _Models()

    class _FallbackProvider:
        def generate_json_with_model(self, *, system_instruction, contents, timeout=None, options=None):
            del system_instruction, contents, timeout, options
            calls["fallback"] += 1
            return '{"cleaned_text": "ollama cleaned"}', {
                "provider": "ollama",
                "model": "gpt-oss:120b-cloud",
            }

    monkeypatch.setattr(genai_provider_module.GenAIProvider, "_get_client", classmethod(lambda cls: _Client()))
    monkeypatch.setattr(
        ollama_provider_module,
        "get_gemini_fallback_provider",
        lambda: _FallbackProvider(),
    )

    provider = genai_provider_module.GenAIProvider()
    provider._generate_content_with_model_info(
        model="gemini-test",
        contents="TEXT TO CLEAN:\nraw",
        config=SimpleNamespace(system_instruction="clean exactly"),
    )
    provider._generate_content_with_model_info(
        model="gemini-test",
        contents="TEXT TO CLEAN:\nraw",
        config=SimpleNamespace(system_instruction="clean exactly"),
    )

    assert is_quota_cooldown_active()
    assert calls == {"gemini": 1, "fallback": 2}
