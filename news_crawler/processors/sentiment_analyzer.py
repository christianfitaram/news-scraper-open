"""Sentiment analysis utilities with lazy model loading."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

from news_crawler.processors._hf_common import resolve_cache_dir, select_torch_device

logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


@dataclass(frozen=True)
class SentimentResult:
    label: str
    score: float

    def to_json(self) -> str:
        return json.dumps({"label": self.label, "score": self.score}, ensure_ascii=False)


class SentimentDetector:
    """Encapsulates a sentiment pipeline and loads it on first use."""

    def __init__(self, model_name: str = MODEL_NAME, cache_dir: Optional[str] = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir or resolve_cache_dir()
        self._pipeline = None

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        torch, torch_device, pipeline_device = select_torch_device()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        try:
            model.to(torch_device)
        except Exception as exc:
            logger.debug("Unable to move sentiment model to %s: %s", torch_device, exc)

        try:
            self._pipeline = pipeline(
                task="sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=pipeline_device,
                max_length=512,
                truncation=True,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize sentiment pipeline on device %s: %s. Falling back to CPU.",
                pipeline_device,
                exc,
            )
            self._pipeline = pipeline(
                task="sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=-1,
                max_length=512,
                truncation=True,
            )

    def analyze(self, text: str) -> Optional[SentimentResult]:
        """Analyze a single text, returning None for empty inputs."""
        if not text or not text.strip():
            return None
        self._ensure_pipeline()
        result = self._pipeline(text.strip())[0]
        return SentimentResult(label=result["label"], score=float(result["score"]))

    def analyze_batch(self, texts: Iterable[str]) -> List[Optional[SentimentResult]]:
        batch = [text.strip() if text is not None else "" for text in texts]
        if not batch:
            return []
        self._ensure_pipeline()

        indices = [idx for idx, value in enumerate(batch) if value]
        outputs: List[Optional[SentimentResult]] = [None] * len(batch)
        if indices:
            predictions = self._pipeline([batch[idx] for idx in indices])
            for idx, prediction in zip(indices, predictions):
                outputs[idx] = SentimentResult(
                    label=prediction["label"],
                    score=float(prediction["score"]),
                )
        return outputs


_detector_singleton: Optional[SentimentDetector] = None


def _get_detector() -> SentimentDetector:
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = SentimentDetector()
    return _detector_singleton


def analyze_sentiment(text: str) -> dict[str, float | str]:
    """Analyze sentiment for a single text."""
    if not text or not text.strip():
        return {"label": "NEUTRAL", "score": 0.0}
    try:
        result = _get_detector().analyze(text)
    except Exception as exc:
        logger.error("Sentiment analysis failed: %s", exc)
        return {"label": "NEUTRAL", "score": 0.0}
    if result is None:
        return {"label": "NEUTRAL", "score": 0.0}
    return {"label": result.label, "score": float(result.score)}


def get_sentiment(text: str) -> Optional[SentimentResult]:
    return _get_detector().analyze(text)


def get_sentiments(texts: Iterable[str]) -> List[Optional[SentimentResult]]:
    return _get_detector().analyze_batch(texts)
