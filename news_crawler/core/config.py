"""Centralized configuration for the news crawler."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip() in {"1", "true", "True", "yes", "YES"}


def _env_backend(name: str, default: str = "local") -> str:
    value = os.getenv(name, default).strip().lower()
    if value in {"gemini", "genai", "google"}:
        return "gemini"
    return "local"


@dataclass(frozen=True)
class MongoConfig:
    """MongoDB configuration."""

    uri: str
    db_name: str
    articles_collection: str = "articles"
    link_pool_collection: str = "link_pool"
    metadata_collection: str = "metadata"
    logs_collection: str = "pipeline_logs"

    @classmethod
    def from_env(cls) -> "MongoConfig":
        return cls(
            uri=os.getenv("MONGO_URI", ""),
            db_name=os.getenv("MONGODB_DB", "agents"),
            articles_collection=os.getenv("MONGO_ARTICLES_COLLECTION", "articles"),
            link_pool_collection=os.getenv("MONGO_LINK_POOL_COLLECTION", "link_pool"),
            metadata_collection=os.getenv("MONGO_METADATA_COLLECTION", "metadata"),
            logs_collection=os.getenv("MONGO_LOGS_COLLECTION", "pipeline_logs"),
        )


@dataclass(frozen=True)
class GenAIConfig:
    """Configuration for Gemini / GenAI calls."""

    api_key: str
    model: str = "gemini-2.0-flash"
    temperature: float = 0.0
    top_p: float = 1.0
    max_chunk_chars: int = 12000

    @classmethod
    def from_env(cls) -> Optional["GenAIConfig"]:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            model=os.getenv("GENAI_MODEL", "gemini-2.0-flash"),
            temperature=float(os.getenv("GENAI_TEMPERATURE", "0.0")),
            top_p=float(os.getenv("GENAI_TOP_P", "1.0")),
            max_chunk_chars=int(os.getenv("GENAI_MAX_CHUNK_CHARS", "12000")),
        )


@dataclass(frozen=True)
class WebhookConfig:
    """Webhook configuration."""

    embedding_url: Optional[str] = None
    thread_events_url: Optional[str] = None
    signature: str = ""
    timeout: int = 60
    max_retries: int = 3
    async_workers: int = 2
    drain_timeout_seconds: int = 20

    @classmethod
    def from_env(cls) -> "WebhookConfig":
        return cls(
            embedding_url=os.getenv("WEBHOOK_URL"),
            thread_events_url=os.getenv("WEBHOOK_URL_THREAD_EVENTS"),
            signature=os.getenv("WEBHOOK_SIGNATURE", ""),
            timeout=int(os.getenv("WEBHOOK_TIMEOUT", "60")),
            max_retries=int(os.getenv("WEBHOOK_MAX_RETRIES", "3")),
            async_workers=max(1, int(os.getenv("WEBHOOK_ASYNC_WORKERS", "2"))),
            drain_timeout_seconds=max(0, int(os.getenv("WEBHOOK_DRAIN_TIMEOUT_SECONDS", "20"))),
        )


@dataclass(frozen=True)
class NewsAPIConfig:
    """NewsAPI.org configuration."""

    api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "NewsAPIConfig":
        return cls(api_key=os.getenv("NEWSAPI_KEY"))


@dataclass(frozen=True)
class SeleniumConfig:
    """Selenium / Chrome configuration."""

    chrome_bin: Optional[str] = None
    chromedriver_path: Optional[str] = None
    chromedriver_version: Optional[str] = None
    chrome_user_data_dir: str = "/tmp/chrome-user-data"

    @classmethod
    def from_env(cls) -> "SeleniumConfig":
        return cls(
            chrome_bin=os.getenv("CHROME_BIN"),
            chromedriver_path=os.getenv("CHROMEDRIVER_PATH"),
            chromedriver_version=os.getenv("CHROMEDRIVER_VERSION"),
            chrome_user_data_dir=os.getenv("CHROME_USER_DATA_DIR", "/tmp/chrome-user-data"),
        )


@dataclass(frozen=True)
class AppConfig:
    """General application configuration."""

    app_name: str = "news-scraper-open"
    log_level: str = "INFO"
    hf_home: str = "./models/transformers"
    enable_webhooks: bool = True
    enable_genai: bool = False
    news_fetch_timeout: int = 20
    state_file: str = "crawler_state.json"
    enable_ollama: bool = True
    nlp_backend: str = "local"
    summarizer_backend: str = "local"
    sentiment_backend: str = "local"
    topic_backend: str = "local"

    @classmethod
    def from_env(cls) -> "AppConfig":
        nlp_backend = _env_backend("NLP_BACKEND", "local")
        return cls(
            app_name=os.getenv("APP_NAME", "news-scraper-open"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            hf_home=os.getenv("HF_HOME", "./models/transformers"),
            enable_webhooks=_env_truthy("ENABLE_WEBHOOKS", "1"),
            enable_genai=_env_truthy("ENABLE_GENAI", "0"),
            news_fetch_timeout=int(os.getenv("NEWS_FETCH_TIMEOUT", "20")),
            state_file=os.getenv("STATE_FILE", "crawler_state.json"),
            enable_ollama=_env_truthy("ENABLE_OLLAMA", "1"),
            nlp_backend=nlp_backend,
            summarizer_backend=_env_backend("SUMMARIZER_BACKEND", nlp_backend),
            sentiment_backend=_env_backend("SENTIMENT_BACKEND", nlp_backend),
            topic_backend=_env_backend("TOPIC_BACKEND", nlp_backend),
        )

@dataclass(frozen=True)
class OllamaConfig:
    """Ollama configuration."""

    host: str = "localhost"
    port: int = 11434
    model: str = "gpt-oss:20b-cloud"

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        return cls(
            host=os.getenv("OLLAMA_HOST", "localhost"),
            port=int(os.getenv("OLLAMA_PORT", "11434")),
            model=os.getenv("OLLAMA_MODEL", "gpt-oss:20b-cloud"),
        )


MONGO_CONFIG = MongoConfig.from_env()
GENAI_CONFIG = GenAIConfig.from_env()
WEBHOOK_CONFIG = WebhookConfig.from_env()
NEWSAPI_CONFIG = NewsAPIConfig.from_env()
SELENIUM_CONFIG = SeleniumConfig.from_env()
APP_CONFIG = AppConfig.from_env()
OLLAMA_CONFIG = OllamaConfig.from_env()


def validate_config(*, require_db: bool = True, require_genai: bool = False) -> None:
    """Validate configuration and raise a ValueError with details if invalid."""

    errors = []

    if require_db:
        if not MONGO_CONFIG.uri:
            errors.append("MONGO_URI is required")
        if not MONGO_CONFIG.db_name:
            errors.append("MONGODB_DB is required")

    if require_genai and not GENAI_CONFIG:
        errors.append("GEMINI_API_KEY or GOOGLE_API_KEY is required when GenAI is enabled")

    if errors:
        raise ValueError("Configuration errors: " + ", ".join(errors))
