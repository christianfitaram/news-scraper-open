"""Shared helpers for entity-cleaning providers."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional


def normalize_entity_key(name: str) -> str:
    key = name.strip().lower()
    key = key.replace("'s", "")
    key = key.replace('"', "").replace("'", "")
    key = re.sub(r"\s+", " ", key)
    return key


def dedupe_preserve_first(values: Iterable[str]) -> List[str]:
    seen = set()
    items: List[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if not normalized:
            continue
        key = normalize_entity_key(normalized)
        if key in seen:
            continue
        seen.add(key)
        items.append(normalized)
    return items


def normalize_entity_lists(payload: Dict[str, Any]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    for key in ("locations", "organizations", "persons"):
        value = payload.get(key, [])
        if not isinstance(value, list):
            value = []
        normalized[key] = dedupe_preserve_first(item for item in value if isinstance(item, str))
    return normalized


def chunk_text(text: str, max_chars: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for paragraph in paragraphs:
        extra_len = len(paragraph) + (2 if current else 0)
        if current_len + extra_len <= max_chars:
            current.append(paragraph)
            current_len += extra_len
            continue

        flush()
        if len(paragraph) > max_chars:
            for idx in range(0, len(paragraph), max_chars):
                part = paragraph[idx : idx + max_chars].strip()
                if part:
                    chunks.append(part)
            continue

        current.append(paragraph)
        current_len = len(paragraph)

    flush()
    return [chunk for chunk in chunks if chunk]


def simple_clean_fallback(raw_text: str) -> str:
    text = raw_text or ""
    text = re.sub(r"https?://\S+", "", text)
    junk_patterns = [
        r"Follow .*? news on .*?(?:\.|$)",
        r"Share this.*?(?:\.|$)",
        r"Sign up.*?(?:\.|$)",
        r"Subscribe.*?(?:\.|$)",
        r"Cookie (?:policy|preferences).*?(?:\.|$)",
    ]
    for pattern in junk_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_json_loads(raw: str) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
