"""Article classifier combining sentiment and topic."""
from __future__ import annotations

import logging
from typing import Any, Dict

from news_crawler.processors.sentiment_analyzer import analyze_sentiment
from news_crawler.processors.topic_classifier import classify_topic

logger = logging.getLogger(__name__)


def classify_article(text: str) -> Dict[str, Any]:
    """Classify text into sentiment and topic."""
    if not text or len(text.strip()) < 20:
        return {
            "sentiment": {"label": "NEUTRAL", "score": 0.0},
            "topic": "other",
        }

    text_sample = text.strip()[:512]

    try:
        sentiment = analyze_sentiment(text_sample)
    except Exception as exc:
        logger.error("Sentiment classification failed: %s", exc)
        sentiment = {"label": "NEUTRAL", "score": 0.0}

    try:
        topic = classify_topic(text_sample)
    except Exception as exc:
        logger.error("Topic classification failed: %s", exc)
        topic = "other"

    return {"sentiment": sentiment, "topic": topic}
