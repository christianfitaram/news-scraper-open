"""Provider for Ollama operations."""
from __future__ import annotations

import logging
import os
from threading import Lock
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

GEMINI_FALLBACK_OLLAMA_MODELS = (
    "gpt-oss:120b-cloud",
    "gemma4:31b-cloud",
    "gpt-oss:20b-cloud",
)


def _unique_models(models: List[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for model in models:
        clean_model = model.strip()
        if clean_model and clean_model not in seen:
            seen.add(clean_model)
            result.append(clean_model)
    return result


def _gemini_fallback_models() -> List[str]:
    configured = os.getenv("OLLAMA_FALLBACK_MODELS", "")
    if configured.strip():
        return _unique_models(configured.split(","))
    return list(GEMINI_FALLBACK_OLLAMA_MODELS)


def _model_info(model: str) -> Dict[str, str]:
    return {"provider": "ollama", "model": model}


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


def _has_entities(entities: Dict[str, List[str]]) -> bool:
    return any(entities.get(key) for key in ("locations", "organizations", "persons"))


class OllamaProvider:
    """Provider for Ollama text cleaning and entity extraction."""

    def __init__(self, model_sequence: Optional[List[str]] = None) -> None:
        self.config = OLLAMA_CONFIG
        models = model_sequence if model_sequence is not None else [self.config.model]
        self.model_sequence = _unique_models(models)
        if not self.model_sequence:
            self.model_sequence = [self.config.model]
        logger.info("OllamaProvider initialized with models %s", ", ".join(self.model_sequence))

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

    @staticmethod
    def _default_options() -> Optional[dict]:
        try:
            temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
            return {"temperature": temperature}
        except Exception:
            return None

    @staticmethod
    def _default_timeout(timeout: int | None = None) -> int:
        return timeout if timeout is not None else int(os.getenv("OLLAMA_TIMEOUT", "60"))

    def _generate_with_model(
        self,
        *,
        model: str,
        prompt: str,
        timeout: int,
        options: Optional[dict] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
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

    def _generate_result(
        self,
        prompt: str,
        timeout: int,
        options: Optional[dict] = None,
    ) -> tuple[str, Dict[str, str]]:
        last_exc: Exception | None = None
        for idx, model in enumerate(self.model_sequence):
            try:
                if idx > 0:
                    logger.info("Trying Ollama fallback model %s", model)
                text = self._generate_with_model(
                    model=model,
                    prompt=prompt,
                    timeout=timeout,
                    options=options,
                )
                return text, _model_info(model)
            except Exception as exc:
                last_exc = exc
                if idx < len(self.model_sequence) - 1:
                    logger.warning("Ollama model %s failed; trying next fallback: %s", model, exc)
                    continue
                logger.warning("Ollama model %s failed with no fallback remaining: %s", model, exc)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Ollama generation failed without an exception")

    def _generate(self, prompt: str, timeout: int, options: Optional[dict] = None) -> str:
        text, _ = self._generate_result(prompt, timeout=timeout, options=options)
        return text

    def generate_json_with_model(
        self,
        *,
        system_instruction: str,
        contents: str,
        timeout: int | None = None,
        options: Optional[dict] = None,
    ) -> tuple[str, Dict[str, str]]:
        prompt = (
            f"{system_instruction.strip()}\n\n"
            f"{contents.strip()}\n\n"
            "Return only valid JSON. Do not include markdown fences or commentary."
        ).strip()
        return self._generate_result(
            prompt,
            timeout=self._default_timeout(timeout),
            options=options if options is not None else self._default_options(),
        )

    def generate_json(
        self,
        *,
        system_instruction: str,
        contents: str,
        timeout: int | None = None,
        options: Optional[dict] = None,
    ) -> str:
        text, _ = self.generate_json_with_model(
            system_instruction=system_instruction,
            contents=contents,
            timeout=timeout,
            options=options,
        )
        return text

    def _clean_text(
        self,
        raw_text: str,
        timeout: int,
        options: Optional[dict],
    ) -> tuple[str, List[Dict[str, str]]]:
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
            response_text, model = self._generate_result(prompt, timeout=timeout, options=options)
            payload = safe_json_loads(response_text or "")
            if isinstance(payload, dict):
                cleaned_text = str(payload.get("cleaned_text", "") or "").strip()
                if cleaned_text:
                    return cleaned_text, [model]
                logger.warning("Ollama cleaner returned JSON without cleaned_text; using local cleaner fallback")
        except Exception as exc:
            logger.warning("Ollama cleaner failed: %s", exc)
        return simple_clean_fallback(raw_text), []

    def _extract_entities(
        self,
        cleaned_text: str,
        timeout: int,
        max_chunk_chars: int,
        options: Optional[dict],
    ) -> tuple[Dict[str, List[str]], List[Dict[str, str]]]:
        system_instruction = (
            "Extract named entities ONLY from the provided text. "
            "STRICT RULE: Every entity MUST be a verbatim substring from the input. "
            "Do NOT infer, translate, or correct names. "
            "Deduplicate case-insensitively, keeping the first-seen casing. "
            "Locations include cities, countries, regions, boroughs, roads, seas, and named geographic places. "
            "Organizations include companies, brands, government bodies, agencies, political parties, "
            "thinktanks, publishers, apps, services, and institutions. "
            "Persons include named human beings only. "
            "If the text contains names such as Sonos, Southwark, Rachel Reeves, Iran, or Deloitte, "
            "they must be returned in the correct arrays. "
            "Return empty arrays only when the text truly contains no named entities. "
            'Return JSON as {"locations": [...], "organizations": [...], "persons": [...]}'
        )

        locations: List[str] = []
        organizations: List[str] = []
        persons: List[str] = []
        models_used: List[Dict[str, str]] = []
        for chunk in chunk_text(cleaned_text, max_chunk_chars):
            prompt = f"{system_instruction}\n\nEXTRACT FROM THIS TEXT:\n{chunk}"
            try:
                response_text, model = self._generate_result(prompt, timeout=timeout, options=options)
                payload = safe_json_loads(response_text or "")
                if isinstance(payload, dict):
                    normalized = normalize_entity_lists(payload)
                    if not _has_entities(normalized):
                        retry_response_text, retry_model = self._generate_result(
                            self._entity_retry_prompt(system_instruction, chunk),
                            timeout=timeout,
                            options=options,
                        )
                        retry_payload = safe_json_loads(retry_response_text or "")
                        if isinstance(retry_payload, dict):
                            normalized = normalize_entity_lists(retry_payload)
                            model = retry_model
                    if _has_entities(normalized):
                        locations.extend(normalized["locations"])
                        organizations.extend(normalized["organizations"])
                        persons.extend(normalized["persons"])
                    else:
                        logger.warning("Ollama entity extraction returned no entities for a non-empty chunk")
                    models_used.append(model)
            except Exception as exc:
                logger.warning("Ollama extraction failed for one chunk: %s", exc)

        return (
            {
                "locations": dedupe_preserve_first(locations),
                "organizations": dedupe_preserve_first(organizations),
                "persons": dedupe_preserve_first(persons),
            },
            _dedupe_model_infos(models_used),
        )

    @staticmethod
    def _entity_retry_prompt(system_instruction: str, chunk: str) -> str:
        return (
            f"{system_instruction}\n\n"
            "The previous extraction returned no entities. Re-read the text carefully. "
            "Extract visible proper nouns into the exact JSON keys. "
            "Do not use alternative keys such as people, companies, places, or entities. "
            "Do not explain.\n\n"
            f"TEXT:\n{chunk}\n\n"
            'Return exactly {"locations": [...], "organizations": [...], "persons": [...]}.'
        )

    def summarize(self, text: str, timeout: int | None = None) -> str:
        if not text or len(text.strip()) < 200:
            return text.strip()

        system_instruction = (
            "You summarize news text. Return JSON only. Keep core facts, numbers, "
            "dates, places, people, and organizations. Do not invent details. "
            'Return JSON as {"summary": "..."}'
        )

        summaries: List[str] = []
        effective_timeout = self._default_timeout(timeout)
        options = self._default_options()
        for part in chunk_text(text.strip(), 12000):
            payload = safe_json_loads(
                self.generate_json(
                    system_instruction=system_instruction,
                    contents=f"Summarize this article chunk:\n{part}",
                    timeout=effective_timeout,
                    options=options,
                )
                or ""
            )
            if isinstance(payload, dict):
                summary = str(payload.get("summary", "")).strip()
                if summary:
                    summaries.append(summary)

        if not summaries:
            return text.strip()
        if len(summaries) == 1:
            return summaries[0]

        merged = "\n".join(summaries)
        if len(merged) <= 1200:
            return merged

        payload = safe_json_loads(
            self.generate_json(
                system_instruction=system_instruction,
                contents=f"Summarize this merged summary into one concise paragraph:\n{merged}",
                timeout=effective_timeout,
                options=options,
            )
            or ""
        )
        if isinstance(payload, dict):
            final_summary = str(payload.get("summary", "")).strip()
            if final_summary:
                return final_summary
        return merged

    def classify(self, text: str, candidate_labels: Optional[List[str]] = None, timeout: int | None = None) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"sentiment": {"label": "NEUTRAL", "score": 0.0}, "topic": "other"}

        labels = list(candidate_labels or [])
        if not labels:
            labels = [
                "politics and government",
                "sports and athletics",
                "science and research",
                "technology and innovation",
                "health and medicine",
                "business and finance",
                "entertainment and celebrity",
                "crime and justice",
                "climate and environment",
                "education and schools",
                "war and conflict",
                "travel and tourism",
            ]

        labels_literal = ", ".join(labels)
        system_instruction = (
            "Classify sentiment and topic in one pass. "
            "Sentiment label must be exactly one of POSITIVE, NEGATIVE, NEUTRAL. "
            "Topic must be exactly one of the provided labels. "
            'Return JSON as {"sentiment": {"label": "...", "score": 0.0}, "topic": "...", "topic_score": 0.0}'
        )
        payload = safe_json_loads(
            self.generate_json(
                system_instruction=system_instruction,
                contents=(
                    f"Allowed topic labels: {labels_literal}\n\n"
                    f"Text:\n{text.strip()[:4000]}\n\n"
                    "Return fields: sentiment.label, sentiment.score, topic, topic_score."
                ),
                timeout=self._default_timeout(timeout),
            )
            or ""
        )
        if not isinstance(payload, dict):
            return {"sentiment": {"label": "NEUTRAL", "score": 0.0}, "topic": "other"}

        sentiment_payload = payload.get("sentiment", {})
        label = str((sentiment_payload or {}).get("label", "NEUTRAL")).strip().upper()
        if label not in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
            label = "NEUTRAL"
        try:
            score = float((sentiment_payload or {}).get("score", 0.0))
        except Exception:
            score = 0.0
        score = max(0.0, min(1.0, score))

        topic = str(payload.get("topic", "")).strip()
        if topic not in labels:
            topic = "other"

        return {"sentiment": {"label": label, "score": score}, "topic": topic}

    def sentiment(self, text: str, timeout: int | None = None) -> Dict[str, float | str]:
        return self.classify(text, timeout=timeout)["sentiment"]

    def topic(self, text: str, candidate_labels: Optional[List[str]] = None, timeout: int | None = None) -> str:
        return str(self.classify(text, candidate_labels=candidate_labels, timeout=timeout)["topic"])

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

        effective_timeout = self._default_timeout(timeout)
        cleaned_text, cleaning_models = self._clean_text(text, timeout=effective_timeout, options=options)
        entities, extraction_models = self._extract_entities(
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
            "llm_models": {
                "cleaning": cleaning_models,
                "entity_extraction": extraction_models,
            },
        }


def call_to_ollama_sdk(text: str, timeout: int = 60) -> Dict[str, Any]:
    """Compatibility entry point for the legacy Ollama helper."""
    return OllamaProvider().clean_and_extract_entities(text, timeout=timeout)


_gemini_fallback_provider: OllamaProvider | None = None
_gemini_fallback_provider_lock = Lock()


def get_gemini_fallback_provider() -> OllamaProvider:
    """Return the shared Ollama provider used after Gemini request failures."""
    global _gemini_fallback_provider
    if _gemini_fallback_provider is not None:
        return _gemini_fallback_provider
    with _gemini_fallback_provider_lock:
        if _gemini_fallback_provider is None:
            _gemini_fallback_provider = OllamaProvider(model_sequence=_gemini_fallback_models())
    return _gemini_fallback_provider
