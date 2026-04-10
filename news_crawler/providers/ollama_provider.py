"""Provider for Ollama operations."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from news_crawler.core.config import OLLAMA_CONFIG
from news_crawler.providers._entity_common import (
    chunk_text,
    dedupe_preserve_first,
    normalize_entity_lists,
    safe_json_loads,
    simple_clean_fallback,
)

logger = logging.getLogger(__name__)


class OllamaProvider:
    """Provider for Ollama text cleaning and entity extraction."""

    def __init__(self) -> None:
        self.config = OLLAMA_CONFIG
        logger.info("OllamaProvider initialized with model %s", self.config.model)

    def _build_api_url(self) -> str:
        host = (self.config.host or "localhost").strip()
        if host.startswith(("http://", "https://")):
            base = host.rstrip("/")
        else:
            base = f"http://{host}".rstrip("/")
        parsed = urlparse(base)
        if parsed.port is None:
            base = f"{parsed.scheme}://{parsed.hostname}:{self.config.port}"
        return f"{base}/api/generate"

    def _generate(self, prompt: str, timeout: int, options: Optional[dict] = None) -> str:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        if options:
            payload["options"] = options

        response = requests.post(self._build_api_url(), json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "") if isinstance(data, dict) else ""

    def _clean_text(self, raw_text: str, timeout: int, options: Optional[dict]) -> str:
        system_instruction = (
            "Act as a VERBATIM text cleaner. Output ONLY JSON. "
            "Rules: DELETE-ONLY. No paraphrasing, summarizing, or reordering. "
            "Keep text exactly as provided. "
            "REMOVE: URLs, nav/cookie text, ads, bylines, author/publisher info, "
            "image captions, and layout fragments. "
            "FIX: Normalize whitespace and join hyphenated line-breaks. "
            'Return JSON as {"cleaned_text": "..."}'
        )
        prompt = f"{system_instruction}\n\nTEXT TO CLEAN:\n{raw_text}"
        try:
            payload = safe_json_loads(self._generate(prompt, timeout=timeout, options=options) or "")
            if isinstance(payload, dict):
                return str(payload.get("cleaned_text", "")).strip()
        except Exception as exc:
            logger.warning("Ollama cleaner failed: %s", exc)
        return simple_clean_fallback(raw_text)

    def _extract_entities(
        self,
        cleaned_text: str,
        timeout: int,
        max_chunk_chars: int,
        options: Optional[dict],
    ) -> Dict[str, List[str]]:
        system_instruction = (
            "Extract named entities ONLY from the provided text. "
            "STRICT RULE: Every entity MUST be a verbatim substring from the input. "
            "Do NOT infer, translate, or correct names. "
            "Deduplicate case-insensitively, keeping the first-seen casing. "
            "Return empty arrays if no entities are found. "
            'Return JSON as {"locations": [...], "organizations": [...], "persons": [...]}'
        )

        locations: List[str] = []
        organizations: List[str] = []
        persons: List[str] = []
        for chunk in chunk_text(cleaned_text, max_chunk_chars):
            prompt = f"{system_instruction}\n\nEXTRACT FROM THIS TEXT:\n{chunk}"
            try:
                payload = safe_json_loads(self._generate(prompt, timeout=timeout, options=options) or "")
                if isinstance(payload, dict):
                    normalized = normalize_entity_lists(payload)
                    locations.extend(normalized["locations"])
                    organizations.extend(normalized["organizations"])
                    persons.extend(normalized["persons"])
            except Exception as exc:
                logger.warning("Ollama extraction failed for one chunk: %s", exc)

        return {
            "locations": dedupe_preserve_first(locations),
            "organizations": dedupe_preserve_first(organizations),
            "persons": dedupe_preserve_first(persons),
        }

    def clean_and_extract_entities(self, text: str, timeout: int | None = None) -> Dict[str, Any]:
        if not text or len(text.strip()) < 50:
            return {
                "cleaned_text": text,
                "locations": [],
                "organizations": [],
                "persons": [],
            }

        try:
            max_chunk_chars = int(os.getenv("OLLAMA_MAX_CHUNK_CHARS", "12000"))
        except Exception:
            max_chunk_chars = 12000

        try:
            temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
            options: Optional[dict] = {"temperature": temperature}
        except Exception:
            options = None

        effective_timeout = timeout if timeout is not None else int(os.getenv("OLLAMA_TIMEOUT", "60"))
        cleaned_text = self._clean_text(text, timeout=effective_timeout, options=options)
        entities = self._extract_entities(
            cleaned_text,
            timeout=effective_timeout,
            max_chunk_chars=max_chunk_chars,
            options=options,
        )
        return {
            "cleaned_text": cleaned_text,
            "locations": entities["locations"],
            "organizations": entities["organizations"],
            "persons": entities["persons"],
        }


def call_to_ollama_sdk(text: str, timeout: int = 60) -> Dict[str, Any]:
    """Compatibility entry point for the legacy Ollama helper."""
    return OllamaProvider().clean_and_extract_entities(text, timeout=timeout)
