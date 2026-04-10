"""Text summarization utilities with lazy model loading."""
from __future__ import annotations

import logging
import re
from typing import Any

from news_crawler.processors._hf_common import resolve_cache_dir, select_torch_device

logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/bart-large-cnn"


def _split_oversized_text(text: str, tokenizer: Any, max_tokens: int) -> list[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return [text]

    parts: list[str] = []
    for idx in range(0, len(token_ids), max_tokens):
        piece = tokenizer.decode(
            token_ids[idx : idx + max_tokens],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if piece:
            parts.append(piece)
    return parts


def _chunk_text(text: str, tokenizer: Any, max_tokens: int = 512) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        for part in _split_oversized_text(sentence, tokenizer, max_tokens):
            token_len = len(tokenizer.encode(part, add_special_tokens=False))
            if current and current_len + token_len > max_tokens:
                chunks.append(current.strip())
                current = part
                current_len = token_len
                continue
            current = f"{current} {part}".strip()
            current_len += token_len

    if current:
        chunks.append(current.strip())
    return chunks


class _SummarizerRuntime:
    def __init__(self, model_name: str = MODEL_NAME, cache_dir: str | None = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir or resolve_cache_dir()
        self._pipeline = None
        self._tokenizer = None

    def _ensure_pipeline(self, preferred_device: str | int = "auto") -> None:
        if self._pipeline is not None and self._tokenizer is not None:
            return

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

        torch, torch_device, pipeline_device = select_torch_device(preferred_device)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=False,
            use_fast=True,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=False,
        )
        try:
            model.to(torch_device)
        except Exception as exc:
            logger.debug("Unable to move summarizer model to %s: %s", torch_device, exc)

        try:
            summarizer = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                device=pipeline_device,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize summarizer on device %s: %s. Falling back to CPU.",
                pipeline_device,
                exc,
            )
            summarizer = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                device=-1,
            )

        self._pipeline = summarizer
        self._tokenizer = tokenizer

    def summarize(self, text: str, preferred_device: str | int = "auto") -> str:
        if not text or len(text.strip()) < 200:
            return text.strip()

        self._ensure_pipeline(preferred_device)
        chunks = _chunk_text(text.strip(), self._tokenizer)
        summaries: list[str] = []

        for chunk in chunks:
            try:
                input_length = len(self._tokenizer.encode(chunk, add_special_tokens=False))
                if input_length < 200:
                    max_len = max(int(input_length * 0.8), 20)
                    min_len = min(10, max_len // 2)
                else:
                    max_len, min_len = 200, 80
                summary = self._pipeline(
                    chunk,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False,
                    truncation=True,
                )[0]["summary_text"]
                summaries.append(summary)
            except Exception as exc:
                logger.warning("Failed to summarize chunk: %s", exc)

        if not summaries:
            return text.strip()

        result = "\n".join(summaries)
        if len(self._tokenizer.encode(result, add_special_tokens=False)) > 512 and result != text.strip():
            return self.summarize(result, preferred_device)
        return result


_runtime: _SummarizerRuntime | None = None


def _get_runtime() -> _SummarizerRuntime:
    global _runtime
    if _runtime is None:
        _runtime = _SummarizerRuntime()
    return _runtime


def smart_summarize(text: str, device: str | int = "auto") -> str:
    """Summarize text with robust fallbacks."""
    if not text:
        return ""
    try:
        return _get_runtime().summarize(text, preferred_device=device)
    except Exception as exc:
        logger.error("Summarization failed: %s", exc)
        return text
