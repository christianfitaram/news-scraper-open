"""Provider for GenAI (Gemini) operations."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from news_crawler.core.config import GENAI_CONFIG
from news_crawler.providers._entity_common import (
    chunk_text,
    dedupe_preserve_first,
    normalize_entity_lists,
    safe_json_loads,
    simple_clean_fallback,
)

logger = logging.getLogger(__name__)


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

    _client: Optional[Any] = None

    def __init__(self) -> None:
        if not GENAI_CONFIG:
            raise ValueError("GenAI configuration not available")
        self.config = GENAI_CONFIG
        logger.info("GenAIProvider initialized with model %s", self.config.model)

    @classmethod
    def _get_client(cls) -> Any:
        if cls._client is None:
            from google import genai

            if not GENAI_CONFIG:
                raise RuntimeError("Missing GenAI configuration")
            cls._client = genai.Client(api_key=GENAI_CONFIG.api_key)
        return cls._client

    def _clean_text(self, raw_text: str) -> str:
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
            response = self._get_client().models.generate_content(
                model=self.config.model,
                contents=f"TEXT TO CLEAN:\n{raw_text}",
                config=config,
            )
            payload = safe_json_loads(getattr(response, "text", "") or "")
            if isinstance(payload, dict):
                return str(payload.get("cleaned_text", "")).strip()
        except Exception as exc:
            logger.warning("GenAI cleaner failed: %s", exc)
        return simple_clean_fallback(raw_text)

    def _extract_entities(self, cleaned_text: str) -> Dict[str, List[str]]:
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
        for chunk in chunks:
            try:
                response = self._get_client().models.generate_content(
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
            except Exception as exc:
                logger.warning("GenAI extraction failed for one chunk: %s", exc)

        return {
            "locations": dedupe_preserve_first(locations),
            "organizations": dedupe_preserve_first(organizations),
            "persons": dedupe_preserve_first(persons),
        }

    def clean_and_extract_entities(self, text: str, timeout: int | None = None) -> Dict[str, Any]:
        del timeout
        if not text or len(text.strip()) < 50:
            return {
                "cleaned_text": text,
                "locations": [],
                "organizations": [],
                "persons": [],
            }

        cleaned_text = self._clean_text(text)
        entities = self._extract_entities(cleaned_text)
        return {
            "cleaned_text": cleaned_text,
            "locations": entities["locations"],
            "organizations": entities["organizations"],
            "persons": entities["persons"],
        }


def call_to_genai_sdk(text: str, timeout: int = 60) -> Dict[str, Any]:
    """Compatibility entry point for the legacy GenAI helper."""
    return GenAIProvider().clean_and_extract_entities(text, timeout=timeout)
