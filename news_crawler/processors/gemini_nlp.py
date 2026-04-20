"""Gemini-backed NLP utilities for summarization and classification."""
from __future__ import annotations

import logging
from threading import Lock, local
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

from news_crawler.core.config import GENAI_CONFIG
from news_crawler.providers._entity_common import chunk_text, safe_json_loads
from news_crawler.providers._genai_quota import (
    activate_quota_cooldown,
    is_quota_cooldown_active,
    is_quota_exhausted_error,
    maybe_log_quota_skip,
)

logger = logging.getLogger(__name__)

DEFAULT_TOPICS = [
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
SENTIMENT_LABELS = {"POSITIVE", "NEGATIVE", "NEUTRAL"}


def _model_info(provider: str, model: str) -> Dict[str, str]:
    return {"provider": provider, "model": model}


def _dedupe_model_infos(models: list[Dict[str, str]]) -> list[Dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    result: list[Dict[str, str]] = []
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


def _summary_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }


def _sentiment_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "score": {"type": "number"},
        },
        "required": ["label", "score"],
    }


def _topic_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "score": {"type": "number"},
        },
        "required": ["topic", "score"],
    }


def _classification_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "sentiment": _sentiment_schema(),
            "topic": {"type": "string"},
            "topic_score": {"type": "number"},
        },
        "required": ["sentiment", "topic"],
    }


class GeminiNLP:
    """Thin Gemini runtime used by NLP processors."""

    _thread_local = local()
    _client_init_lock = Lock()

    def __init__(self) -> None:
        if not GENAI_CONFIG:
            raise ValueError("GenAI configuration not available")
        self.config = GENAI_CONFIG

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
                    logger.warning("Gemini client was closed; recreating and retrying request once.")
                    self._reset_client()
                    continue
                if is_quota_exhausted_error(exc):
                    activate_quota_cooldown(exc)
                logger.warning("Gemini request failed; falling back to Ollama sequence: %s", exc)
                return self._generate_content_with_ollama_fallback(contents=contents, config=config)
        if last_exc is not None:
            if is_quota_exhausted_error(last_exc):
                activate_quota_cooldown(last_exc)
            logger.warning("Gemini request failed; falling back to Ollama sequence: %s", last_exc)
            return self._generate_content_with_ollama_fallback(contents=contents, config=config)
        raise RuntimeError("Gemini generate_content failed without an exception")

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

    @staticmethod
    def _payload(response: Any) -> Dict[str, Any]:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, dict):
            return parsed
        if parsed is not None and hasattr(parsed, "model_dump"):
            try:
                dumped = parsed.model_dump()
                if isinstance(dumped, dict):
                    return dumped
            except Exception:
                pass
        text = getattr(response, "text", "") or ""
        payload = safe_json_loads(text)
        return payload if isinstance(payload, dict) else {}

    def summarize(self, text: str) -> str:
        summary, _ = self.summarize_with_metadata(text)
        return summary

    def summarize_with_metadata(self, text: str) -> tuple[str, list[Dict[str, str]]]:
        if not text or len(text.strip()) < 200:
            return text.strip(), []

        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=(
                "You summarize news text. Return JSON only. Keep core facts, numbers, "
                "dates, places, people, and organizations. Do not invent details."
            ),
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=_summary_schema(),
        )

        summaries: list[str] = []
        models_used: list[Dict[str, str]] = []
        for part in chunk_text(text.strip(), self.config.max_chunk_chars):
            response, model = self._generate_content_with_model_info(
                model=self.config.model,
                contents=f"Summarize this article chunk:\n{part}",
                config=config,
            )
            payload = self._payload(response)
            summary = str(payload.get("summary", "")).strip()
            if summary:
                summaries.append(summary)
                models_used.append(model)

        if not summaries:
            return text.strip(), _dedupe_model_infos(models_used)
        if len(summaries) == 1:
            return summaries[0], _dedupe_model_infos(models_used)

        merged = "\n".join(summaries)
        if len(merged) <= 1200:
            return merged, _dedupe_model_infos(models_used)

        # One extra compression pass keeps multi-chunk output concise.
        response, model = self._generate_content_with_model_info(
            model=self.config.model,
            contents=f"Summarize this merged summary into one concise paragraph:\n{merged}",
            config=config,
        )
        models_used.append(model)
        payload = self._payload(response)
        final_summary = str(payload.get("summary", "")).strip()
        if final_summary:
            return final_summary, _dedupe_model_infos(models_used)
        return merged, _dedupe_model_infos(models_used)

    def sentiment(self, text: str) -> Dict[str, float | str]:
        sentiment, _ = self.sentiment_with_metadata(text)
        return sentiment

    def sentiment_with_metadata(self, text: str) -> tuple[Dict[str, float | str], list[Dict[str, str]]]:
        if not text or not text.strip():
            return {"label": "NEUTRAL", "score": 0.0}, []

        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=(
                "Classify sentiment as exactly one of: POSITIVE, NEGATIVE, NEUTRAL. "
                "Return JSON only."
            ),
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=_sentiment_schema(),
        )

        response, model = self._generate_content_with_model_info(
            model=self.config.model,
            contents=f"Analyze sentiment for this text:\n{text.strip()[:4000]}",
            config=config,
        )
        payload = self._payload(response)
        if not payload:
            return {"label": "NEUTRAL", "score": 0.0}, [model]

        label = str(payload.get("label", "NEUTRAL")).strip().upper()
        if label not in SENTIMENT_LABELS:
            label = "NEUTRAL"
        score = float(payload.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        return {"label": label, "score": score}, [model]

    def topic(self, text: str, candidate_labels: Optional[Sequence[str]] = None) -> str:
        topic, _ = self.topic_with_metadata(text, candidate_labels=candidate_labels)
        return topic

    def topic_with_metadata(
        self,
        text: str,
        candidate_labels: Optional[Sequence[str]] = None,
    ) -> tuple[str, list[Dict[str, str]]]:
        if not text or not text.strip():
            return "other", []

        labels = list(candidate_labels) if candidate_labels else list(DEFAULT_TOPICS)
        if not labels:
            return "other", []

        from google.genai import types

        labels_literal = ", ".join(labels)
        config = types.GenerateContentConfig(
            system_instruction=(
                "Classify topic by selecting exactly one label from the provided list. "
                "Return JSON only."
            ),
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=_topic_schema(),
        )
        response, model = self._generate_content_with_model_info(
            model=self.config.model,
            contents=(
                f"Allowed labels: {labels_literal}\n\n"
                f"Text:\n{text.strip()[:4000]}\n\n"
                "Return the best matching label."
            ),
            config=config,
        )
        payload = self._payload(response)
        if not payload:
            return "other", [model]

        topic = str(payload.get("topic", "")).strip()
        return (topic if topic in labels else "other"), [model]

    def classify(self, text: str, candidate_labels: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        classification, _ = self.classify_with_metadata(text, candidate_labels=candidate_labels)
        return classification

    def classify_with_metadata(
        self,
        text: str,
        candidate_labels: Optional[Sequence[str]] = None,
    ) -> tuple[Dict[str, Any], list[Dict[str, str]]]:
        if not text or not text.strip():
            return {"sentiment": {"label": "NEUTRAL", "score": 0.0}, "topic": "other"}, []

        labels = list(candidate_labels) if candidate_labels else list(DEFAULT_TOPICS)
        if not labels:
            return {"sentiment": {"label": "NEUTRAL", "score": 0.0}, "topic": "other"}, []

        from google.genai import types

        labels_literal = ", ".join(labels)
        config = types.GenerateContentConfig(
            system_instruction=(
                "Classify sentiment and topic in one pass. "
                "Sentiment label must be exactly one of POSITIVE, NEGATIVE, NEUTRAL. "
                "Topic must be exactly one of the provided labels. "
                "Return JSON only."
            ),
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=_classification_schema(),
        )
        response, model = self._generate_content_with_model_info(
            model=self.config.model,
            contents=(
                f"Allowed topic labels: {labels_literal}\n\n"
                f"Text:\n{text.strip()[:4000]}\n\n"
                "Return fields: sentiment.label, sentiment.score, topic, topic_score."
            ),
            config=config,
        )
        payload = self._payload(response)
        if not payload:
            return {"sentiment": {"label": "NEUTRAL", "score": 0.0}, "topic": "other"}, [model]

        sentiment_payload = payload.get("sentiment", {}) if isinstance(payload, dict) else {}
        label = str((sentiment_payload or {}).get("label", "NEUTRAL")).strip().upper()
        if label not in SENTIMENT_LABELS:
            label = "NEUTRAL"
        score = float((sentiment_payload or {}).get("score", 0.0))
        score = max(0.0, min(1.0, score))

        topic = str(payload.get("topic", "")).strip()
        if topic not in labels:
            topic = "other"

        return {"sentiment": {"label": label, "score": score}, "topic": topic}, [model]


_runtime: Optional[GeminiNLP] = None
_runtime_lock = Lock()


def get_runtime() -> GeminiNLP:
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = GeminiNLP()
    return _runtime
