"""Provider for GenAI (Gemini) operations."""
from __future__ import annotations

import logging
from threading import Lock, local
from types import SimpleNamespace
from typing import Any, Dict, List

from news_crawler.core.config import GENAI_CONFIG
from news_crawler.providers._entity_common import (
    chunk_text,
    dedupe_preserve_first,
    normalize_entity_lists,
    safe_json_loads,
    simple_clean_fallback,
)
from news_crawler.providers._genai_quota import (
    activate_quota_cooldown,
    is_quota_cooldown_active,
    is_quota_exhausted_error,
    maybe_log_quota_skip,
)

logger = logging.getLogger(__name__)


def _model_info(provider: str, model: str) -> Dict[str, str]:
    return {"provider": provider, "model": model}


def _dedupe_model_infos(models: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    result: List[Dict[str, str]] = []
    for model in models:
        provider = str(model.get("provider", "")).strip()
        name = str(model.get("model", "")).strip()
        if not provider or not name:
            continue
        key = (provider, name)
        if key in seen:
            continue
        seen.add(key)
        result.append({"provider": provider, "model": name})
    return result


def _genai_cleaner_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {"cleaned_text": {"type": "string"}},
        "required": ["cleaned_text"],
    }


def _genai_entity_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "locations": {"type": "array", "items": {"type": "string"}},
            "organizations": {"type": "array", "items": {"type": "string"}},
            "persons": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["locations", "organizations", "persons"],
    }


class GenAIProvider:
    """Provider for Gemini text cleaning and entity extraction."""

    _thread_local = local()
    _client_init_lock = Lock()

    def __init__(self) -> None:
        if not GENAI_CONFIG:
            raise ValueError("GenAI configuration not available")
        self.config = GENAI_CONFIG
        logger.info("GenAIProvider initialized with model %s", self.config.model)

    @classmethod
    def _get_client(cls) -> Any:
        client = getattr(cls._thread_local, "client", None)
        if client is not None:
            return client

        with cls._client_init_lock:
            client = getattr(cls._thread_local, "client", None)
            if client is not None:
                return client
            from google import genai

            if not GENAI_CONFIG:
                raise RuntimeError("Missing GenAI configuration")
            client = genai.Client(api_key=GENAI_CONFIG.api_key)
            cls._thread_local.client = client
            return client

    @classmethod
    def _reset_client(cls) -> None:
        client = getattr(cls._thread_local, "client", None)
        if client is None:
            return
        close_method = getattr(client, "close", None)
        if callable(close_method):
            try:
                close_method()
            except Exception:
                pass
        cls._thread_local.client = None

    @staticmethod
    def _is_closed_client_error(exc: Exception) -> bool:
        message = str(exc).strip().lower()
        return "client has been closed" in message or "cannot send a request" in message

    def _generate_content(self, *, model: str, contents: str, config: Any) -> Any:
        response, _ = self._generate_content_with_model_info(
            model=model,
            contents=contents,
            config=config,
        )
        return response

    def _generate_content_with_model_info(
        self,
        *,
        model: str,
        contents: str,
        config: Any,
    ) -> tuple[Any, Dict[str, str]]:
        if is_quota_cooldown_active():
            maybe_log_quota_skip(logger)
            return self._generate_content_with_ollama_fallback(contents=contents, config=config)

        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                response = self._get_client().models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                return response, _model_info("genai", model)
            except Exception as exc:
                last_exc = exc
                if attempt == 0 and self._is_closed_client_error(exc):
                    logger.warning("GenAI client was closed; recreating and retrying request once.")
                    self._reset_client()
                    continue
                if is_quota_exhausted_error(exc):
                    activate_quota_cooldown(exc)
                logger.warning("GenAI request failed; falling back to Ollama sequence: %s", exc)
                return self._generate_content_with_ollama_fallback(contents=contents, config=config)
        if last_exc is not None:
            if is_quota_exhausted_error(last_exc):
                activate_quota_cooldown(last_exc)
            logger.warning("GenAI request failed; falling back to Ollama sequence: %s", last_exc)
            return self._generate_content_with_ollama_fallback(contents=contents, config=config)
        raise RuntimeError("GenAI generate_content failed without an exception")

    @staticmethod
    def _generate_content_with_ollama_fallback(
        *,
        contents: str,
        config: Any,
    ) -> tuple[Any, Dict[str, str]]:
        from news_crawler.providers.ollama_provider import get_gemini_fallback_provider

        if isinstance(config, dict):
            system_instruction = str(config.get("system_instruction", "") or "")
        else:
            system_instruction = str(getattr(config, "system_instruction", "") or "")
        fallback_provider = get_gemini_fallback_provider()
        if hasattr(fallback_provider, "generate_json_with_model"):
            text, model = fallback_provider.generate_json_with_model(
                system_instruction=system_instruction,
                contents=contents,
            )
            return SimpleNamespace(text=text), model
        text = fallback_provider.generate_json(
            system_instruction=system_instruction,
            contents=contents,
        )
        return SimpleNamespace(text=text), _model_info("ollama", "unknown")

    def _clean_text(self, raw_text: str) -> tuple[str, List[Dict[str, str]]]:
        from google.genai import types

        system_instruction = (
            "Act as a VERBATIM text cleaner. Output ONLY JSON. "
            "Rules: DELETE-ONLY. No paraphrasing, summarizing, or reordering. "
            "Keep text exactly as provided. "
            "REMOVE: URLs, nav/cookie text, ads, bylines, author/publisher info, "
            "image captions, and layout fragments. "
            "FIX: Normalize whitespace and join hyphenated line-breaks."
        )
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.config.temperature,
            response_mime_type="application/json",
            response_schema=_genai_cleaner_schema(),
        )
        try:
            response, model = self._generate_content_with_model_info(
                model=self.config.model,
                contents=f"TEXT TO CLEAN:\n{raw_text}",
                config=config,
            )
            payload = safe_json_loads(getattr(response, "text", "") or "")
            if isinstance(payload, dict):
                return str(payload.get("cleaned_text", "")).strip(), [model]
        except Exception as exc:
            logger.warning("GenAI cleaner failed: %s", exc)
        return simple_clean_fallback(raw_text), []

    def _extract_entities(self, cleaned_text: str) -> tuple[Dict[str, List[str]], List[Dict[str, str]]]:
        from google.genai import types

        chunks = chunk_text(cleaned_text, self.config.max_chunk_chars)
        system_instruction = (
            "Extract named entities ONLY from the provided text. "
            "STRICT RULE: Every entity MUST be a verbatim substring from the input. "
            "Do NOT infer, translate, or correct names. "
            "Deduplicate case-insensitively, keeping the first-seen casing. "
            "Return empty arrays if no entities are found."
        )
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=_genai_entity_schema(),
        )

        locations: List[str] = []
        organizations: List[str] = []
        persons: List[str] = []
        models_used: List[Dict[str, str]] = []
        for chunk in chunks:
            try:
                response, model = self._generate_content_with_model_info(
                    model=self.config.model,
                    contents=f"EXTRACT FROM THIS TEXT:\n{chunk}",
                    config=config,
                )
                payload = safe_json_loads(getattr(response, "text", "") or "")
                if isinstance(payload, dict):
                    normalized = normalize_entity_lists(payload)
                    locations.extend(normalized["locations"])
                    organizations.extend(normalized["organizations"])
                    persons.extend(normalized["persons"])
                    models_used.append(model)
            except Exception as exc:
                logger.warning("GenAI extraction failed for one chunk: %s", exc)

        return (
            {
                "locations": dedupe_preserve_first(locations),
                "organizations": dedupe_preserve_first(organizations),
                "persons": dedupe_preserve_first(persons),
            },
            _dedupe_model_infos(models_used),
        )

    def clean_and_extract_entities(self, text: str, timeout: int | None = None) -> Dict[str, Any]:
        del timeout
        if not text or len(text.strip()) < 50:
            return {
                "cleaned_text": text,
                "locations": [],
                "organizations": [],
                "persons": [],
            }

        cleaned_text, cleaning_models = self._clean_text(text)
        entities, extraction_models = self._extract_entities(cleaned_text)
        return {
            "cleaned_text": cleaned_text,
            "locations": entities["locations"],
            "organizations": entities["organizations"],
            "persons": entities["persons"],
            "llm_models": {
                "cleaning": cleaning_models,
                "entity_extraction": extraction_models,
            },
        }


def call_to_genai_sdk(text: str, timeout: int = 60) -> Dict[str, Any]:
    """Compatibility entry point for the legacy GenAI helper."""
    return GenAIProvider().clean_and_extract_entities(text, timeout=timeout)
