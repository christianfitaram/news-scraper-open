#!/usr/bin/env python3
"""
Download all models your pipeline needs into the local cache (offline-ready).
"""
import os
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent
_env_cache = os.getenv("HF_HOME")
print(_env_cache)
cache_path = Path(_env_cache).expanduser() if _env_cache else (_BASE_DIR.parent / "models" / "transformers")
cache_path = cache_path if cache_path.is_absolute() else (_BASE_DIR / cache_path).resolve()
cache_path.mkdir(parents=True, exist_ok=True)
CACHE_DIR = str(cache_path)
print(f"Using cache dir: {CACHE_DIR}")
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)


def dl_sentiment():
    name = "distilbert-base-uncased-finetuned-sst-2-english"
    print(f"⬇️ {name}")
    AutoTokenizer.from_pretrained(name, cache_dir=CACHE_DIR)
    AutoModelForSequenceClassification.from_pretrained(name, cache_dir=CACHE_DIR)


def dl_topic():
    name = "facebook/bart-large-mnli"
    print(f"⬇️ {name}")
    AutoTokenizer.from_pretrained(name, cache_dir=CACHE_DIR)
    AutoModelForSequenceClassification.from_pretrained(name, cache_dir=CACHE_DIR)


def dl_summarizer():
    name = "facebook/bart-large-cnn"
    print(f"⬇️ {name}")
    AutoTokenizer.from_pretrained(name, cache_dir=CACHE_DIR)
    AutoModelForSeq2SeqLM.from_pretrained(name, cache_dir=CACHE_DIR)


def main():
    dl_sentiment()
    dl_topic()
    dl_summarizer()
    print("✅ All models cached.")


if __name__ == "__main__":
    main()
