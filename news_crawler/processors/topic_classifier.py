"""Topic classification utilities with lazy model loading."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from news_crawler.processors._hf_common import resolve_cache_dir, select_torch_device

logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/bart-large-mnli"
DEFAULT_TOPICS: List[str] = [
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


@dataclass(frozen=True)
class TopicResult:
    top_label: str
    top_score: float
    labels: List[str]
    scores: List[float]

    def to_json(self) -> str:
        return json.dumps(
            {
                "top_label": self.top_label,
                "top_score": self.top_score,
                "labels": self.labels,
                "scores": self.scores,
            },
            ensure_ascii=False,
        )


class TopicClassifier:
    """Zero-shot topic classifier wrapper."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        candidate_labels: Optional[Sequence[str]] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir or resolve_cache_dir()
        self.candidate_labels = list(candidate_labels) if candidate_labels else list(DEFAULT_TOPICS)
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
            logger.debug("Unable to move topic model to %s: %s", torch_device, exc)

        try:
            self._pipeline = pipeline(
                task="zero-shot-classification",
                model=model,
                tokenizer=tokenizer,
                device=pipeline_device,
                max_length=512,
                truncation=True,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize topic pipeline on device %s: %s. Falling back to CPU.",
                pipeline_device,
                exc,
            )
            self._pipeline = pipeline(
                task="zero-shot-classification",
                model=model,
                tokenizer=tokenizer,
                device=-1,
                max_length=512,
                truncation=True,
            )

    def classify(
        self,
        text: str,
        candidate_labels: Optional[Sequence[str]] = None,
        multi_label: bool = False,
        top_k: Optional[int] = None,
    ) -> Optional[TopicResult]:
        if not text or not text.strip():
            return None

        labels = list(candidate_labels) if candidate_labels else self.candidate_labels
        if not labels:
            raise ValueError("candidate_labels must be non-empty")

        self._ensure_pipeline()
        prediction = self._pipeline(
            text.strip(),
            candidate_labels=labels,
            multi_label=multi_label,
        )
        labels_out = list(prediction["labels"])
        scores_out = [float(score) for score in prediction["scores"]]
        if top_k is not None:
            labels_out = labels_out[:top_k]
            scores_out = scores_out[:top_k]
        return TopicResult(
            top_label=labels_out[0],
            top_score=scores_out[0],
            labels=labels_out,
            scores=scores_out,
        )

    def classify_batch(
        self,
        texts: Iterable[str],
        candidate_labels: Optional[Sequence[str]] = None,
        multi_label: bool = False,
        top_k: Optional[int] = None,
    ) -> List[Optional[TopicResult]]:
        batch = [text.strip() if text is not None else "" for text in texts]
        if not batch:
            return []

        labels = list(candidate_labels) if candidate_labels else self.candidate_labels
        if not labels:
            raise ValueError("candidate_labels must be non-empty")

        self._ensure_pipeline()
        indices = [idx for idx, value in enumerate(batch) if value]
        outputs: List[Optional[TopicResult]] = [None] * len(batch)
        if indices:
            predictions = self._pipeline(
                [batch[idx] for idx in indices],
                candidate_labels=labels,
                multi_label=multi_label,
            )
            prediction_list = [predictions] if isinstance(predictions, dict) else list(predictions)
            for idx, prediction in zip(indices, prediction_list):
                labels_out = list(prediction["labels"])
                scores_out = [float(score) for score in prediction["scores"]]
                if top_k is not None:
                    labels_out = labels_out[:top_k]
                    scores_out = scores_out[:top_k]
                outputs[idx] = TopicResult(
                    top_label=labels_out[0],
                    top_score=scores_out[0],
                    labels=labels_out,
                    scores=scores_out,
                )
        return outputs


_topic_singleton: Optional[TopicClassifier] = None


def _get_classifier() -> TopicClassifier:
    global _topic_singleton
    if _topic_singleton is None:
        _topic_singleton = TopicClassifier()
    return _topic_singleton


def classify_topic(text: str) -> str:
    """Return the top topic label for a given text."""
    if not text or not text.strip():
        return "other"
    try:
        result = _get_classifier().classify(text)
    except Exception as exc:
        logger.error("Topic classification failed: %s", exc)
        return "other"
    if result is None:
        return "other"
    return result.top_label


def get_topic(
    text: str,
    candidate_labels: Optional[Sequence[str]] = None,
    multi_label: bool = False,
    top_k: Optional[int] = None,
) -> Optional[TopicResult]:
    return _get_classifier().classify(
        text,
        candidate_labels=candidate_labels,
        multi_label=multi_label,
        top_k=top_k,
    )


def get_topics(
    texts: Iterable[str],
    candidate_labels: Optional[Sequence[str]] = None,
    multi_label: bool = False,
    top_k: Optional[int] = None,
) -> List[Optional[TopicResult]]:
    return _get_classifier().classify_batch(
        texts,
        candidate_labels=candidate_labels,
        multi_label=multi_label,
        top_k=top_k,
    )
