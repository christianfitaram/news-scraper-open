"""Tests for Ollama entity extraction robustness."""
from __future__ import annotations

from news_crawler.providers.ollama_provider import OllamaProvider


def test_ollama_entity_extraction_accepts_common_key_aliases(monkeypatch):
    responses = [
        '{"cleaned_text": "Rachel Reeves met Deloitte executives in Southwark, London."}',
        '{"places": ["Southwark", "London"], "companies": ["Deloitte"], "people": ["Rachel Reeves"]}',
    ]

    def _fake_generate_result(self, prompt, timeout, options=None):
        del self, prompt, timeout, options
        return responses.pop(0), {"provider": "ollama", "model": "test-model"}

    monkeypatch.setattr(OllamaProvider, "_generate_result", _fake_generate_result)

    result = OllamaProvider(model_sequence=["test-model"]).clean_and_extract_entities(
        "Rachel Reeves met Deloitte executives in Southwark, London."
    )

    assert result["locations"] == ["Southwark", "London"]
    assert result["organizations"] == ["Deloitte"]
    assert result["persons"] == ["Rachel Reeves"]


def test_ollama_entity_extraction_retries_when_first_pass_is_empty(monkeypatch):
    prompts: list[str] = []
    responses = [
        '{"cleaned_text": "Ed Miliband discussed Iran energy policy with Rachel Reeves."}',
        '{"locations": [], "organizations": [], "persons": []}',
        '{"locations": ["Iran"], "organizations": [], "persons": ["Ed Miliband", "Rachel Reeves"]}',
    ]

    def _fake_generate_result(self, prompt, timeout, options=None):
        del self, timeout, options
        prompts.append(prompt)
        return responses.pop(0), {"provider": "ollama", "model": "test-model"}

    monkeypatch.setattr(OllamaProvider, "_generate_result", _fake_generate_result)

    result = OllamaProvider(model_sequence=["test-model"]).clean_and_extract_entities(
        "Ed Miliband discussed Iran energy policy with Rachel Reeves."
    )

    assert "previous extraction returned no entities" in prompts[-1]
    assert result["locations"] == ["Iran"]
    assert result["persons"] == ["Ed Miliband", "Rachel Reeves"]


def test_ollama_cleaner_missing_cleaned_text_uses_local_fallback(monkeypatch):
    prompts: list[str] = []
    responses = [
        '{"text": ""}',
        '{"locations": ["Southwark", "London"], "organizations": ["APCOA"], "persons": []}',
    ]

    def _fake_generate_result(self, prompt, timeout, options=None):
        del self, timeout, options
        prompts.append(prompt)
        return responses.pop(0), {"provider": "ollama", "model": "test-model"}

    monkeypatch.setattr(OllamaProvider, "_generate_result", _fake_generate_result)

    text = "My son was fined in Southwark, London by APCOA after dropping a cigarette butt."
    result = OllamaProvider(model_sequence=["test-model"]).clean_and_extract_entities(
        text
    )

    assert f"EXTRACT FROM THIS TEXT:\n{text}" in prompts[-1]
    assert result["locations"] == ["Southwark", "London"]
    assert result["organizations"] == ["APCOA"]
