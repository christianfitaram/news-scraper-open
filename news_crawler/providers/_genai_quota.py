"""Shared Gemini quota cooldown state."""
from __future__ import annotations

import logging
import os
import re
import time
from threading import Lock

logger = logging.getLogger(__name__)

_DEFAULT_COOLDOWN_SECONDS = int(os.getenv("GENAI_QUOTA_COOLDOWN_SECONDS", "3600"))
_cooldown_until = 0.0
_cooldown_lock = Lock()
_last_skip_log_at = 0.0


def _parse_retry_delay_seconds(message: str) -> int | None:
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", message)
    if match:
        return max(1, int(float(match.group(1))))
    return None


def is_quota_exhausted_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "resource_exhausted" in message
        or "too many requests" in message
        or "quota exceeded" in message
    ) and ("429" in message or "quota" in message)


def activate_quota_cooldown(exc: Exception) -> None:
    delay_seconds = _parse_retry_delay_seconds(str(exc)) or _DEFAULT_COOLDOWN_SECONDS
    until = time.time() + delay_seconds
    global _cooldown_until
    with _cooldown_lock:
        if until > _cooldown_until:
            _cooldown_until = until
            logger.warning(
                "Gemini quota cooldown activated for %ss; routing Gemini requests to Ollama fallback.",
                delay_seconds,
            )


def quota_cooldown_remaining_seconds() -> int:
    with _cooldown_lock:
        remaining = int(_cooldown_until - time.time())
    return max(0, remaining)


def is_quota_cooldown_active() -> bool:
    return quota_cooldown_remaining_seconds() > 0


def maybe_log_quota_skip(logger_: logging.Logger) -> None:
    global _last_skip_log_at
    now = time.time()
    remaining = quota_cooldown_remaining_seconds()
    if remaining <= 0:
        return
    with _cooldown_lock:
        if now - _last_skip_log_at < 300:
            return
        _last_skip_log_at = now
    logger_.warning(
        "Gemini quota cooldown active for another %ss; skipping Gemini and using Ollama fallback.",
        remaining,
    )


def reset_quota_cooldown() -> None:
    global _cooldown_until, _last_skip_log_at
    with _cooldown_lock:
        _cooldown_until = 0.0
        _last_skip_log_at = 0.0
