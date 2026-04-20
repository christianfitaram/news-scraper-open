"""Article classifier combining sentiment and topic."""
from __future__ import annotations

import logging
from typing import Any, Dict

from news_crawler.core.config import APP_CONFIG
from news_crawler.processors.sentiment_analyzer import analyze_sentiment
from news_crawler.processors.topic_classifier import DEFAULT_TOPICS, classify_topic

logger = logging.getLogger(__name__)


def classify_article(text: str, return_metadata: bool = False) -> Dict[str, Any] | tuple[Dict[str, Any], dict[str, Any]]:
    """Classify text into sentiment and topic."""
    if not text or len(text.strip()) < 20:
        classification = {
            "sentiment": {"label": "NEUTRAL", "score": 0.0},
            "topic": "other",
        }
        return (classification, {}) if return_metadata else classification

    text_sample = text.strip()[:512]

    if APP_CONFIG.sentiment_backend == "gemini" and APP_CONFIG.topic_backend == "gemini":
        try:
            from news_crawler.processors.gemini_nlp import get_runtime

            runtime = get_runtime()
            if return_metadata and hasattr(runtime, "classify_with_metadata"):
                classification, models = runtime.classify_with_metadata(
                    text_sample,
                    candidate_labels=DEFAULT_TOPICS,
                )
                return classification, {"classification": models}
            if hasattr(runtime, "classify"):
                classification = runtime.classify(text_sample, candidate_labels=DEFAULT_TOPICS)
                return (classification, {}) if return_metadata else classification
            sentiment = runtime.sentiment(text_sample)
            topic = runtime.topic(text_sample, candidate_labels=DEFAULT_TOPICS)
            classification = {"sentiment": sentiment, "topic": topic}
            return (classification, {}) if return_metadata else classification
        except Exception as exc:
            logger.warning("Gemini classification failed, falling back to local models: %s", exc)
            try:
                sentiment = analyze_sentiment(text_sample)
            except Exception as local_exc:
                logger.error("Sentiment fallback failed: %s", local_exc)
                sentiment = {"label": "NEUTRAL", "score": 0.0}
            try:
                topic = classify_topic(text_sample)
            except Exception as local_exc:
                logger.error("Topic fallback failed: %s", local_exc)
                topic = "other"
            classification = {"sentiment": sentiment, "topic": topic}
            return (classification, {}) if return_metadata else classification

    metadata: dict[str, Any] = {}
    if APP_CONFIG.sentiment_backend == "gemini":
        try:
            from news_crawler.processors.gemini_nlp import get_runtime

            runtime = get_runtime()
            if return_metadata and hasattr(runtime, "sentiment_with_metadata"):
                sentiment, models = runtime.sentiment_with_metadata(text_sample)
                metadata["sentiment"] = models
            else:
                sentiment = runtime.sentiment(text_sample)
        except Exception as exc:
            logger.warning("Gemini sentiment failed, falling back to local model: %s", exc)
            try:
                sentiment = analyze_sentiment(text_sample)
            except Exception as local_exc:
                logger.error("Sentiment fallback failed: %s", local_exc)
                sentiment = {"label": "NEUTRAL", "score": 0.0}
    else:
        try:
            sentiment = analyze_sentiment(text_sample)
        except Exception as exc:
            logger.error("Sentiment classification failed: %s", exc)
            sentiment = {"label": "NEUTRAL", "score": 0.0}

    if APP_CONFIG.topic_backend == "gemini":
        try:
            from news_crawler.processors.gemini_nlp import get_runtime

            runtime = get_runtime()
            if return_metadata and hasattr(runtime, "topic_with_metadata"):
                topic, models = runtime.topic_with_metadata(text_sample, candidate_labels=DEFAULT_TOPICS)
                metadata["topic"] = models
            else:
                topic = runtime.topic(text_sample, candidate_labels=DEFAULT_TOPICS)
        except Exception as exc:
            logger.warning("Gemini topic failed, falling back to local model: %s", exc)
            try:
                topic = classify_topic(text_sample)
            except Exception as local_exc:
                logger.error("Topic fallback failed: %s", local_exc)
                topic = "other"
    else:
        try:
            topic = classify_topic(text_sample)
        except Exception as exc:
            logger.error("Topic classification failed: %s", exc)
            topic = "other"

    classification = {"sentiment": sentiment, "topic": topic}
    return (classification, metadata) if return_metadata else classification
