"""Shared Hugging Face runtime helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple

from news_crawler.core.config import APP_CONFIG

_ROOT_DIR = Path(__file__).resolve().parents[2]


def resolve_cache_dir() -> str:
    """Return the absolute transformers cache directory, creating it if needed."""
    raw_path = os.getenv("HF_HOME", APP_CONFIG.hf_home)
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (_ROOT_DIR / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def select_torch_device(preferred: str | int = "auto") -> Tuple[Any, Any, int]:
    """Resolve the torch device and the pipeline device index lazily."""
    import torch

    force_cpu = os.getenv("FORCE_CPU", "").strip().lower() in {"1", "true", "yes"}
    if force_cpu:
        return torch, torch.device("cpu"), -1

    if isinstance(preferred, int):
        if preferred >= 0:
            return torch, torch.device("cuda:0"), preferred
        return torch, torch.device("cpu"), -1

    normalized = str(preferred).strip().lower()
    if normalized in {"cpu"}:
        return torch, torch.device("cpu"), -1
    if normalized in {"cuda", "gpu", "cuda:0"}:
        return torch, torch.device("cuda:0"), 0
    if normalized == "mps":
        return torch, torch.device("mps"), -1

    if torch.cuda.is_available():
        return torch, torch.device("cuda:0"), 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch, torch.device("mps"), -1
    return torch, torch.device("cpu"), -1
