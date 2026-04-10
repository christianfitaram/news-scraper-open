# NewsFeeder-IA

NewsFeeder-IA is a modular news scraping and enrichment pipeline for collecting articles from multiple sources, deduplicating URLs, cleaning text, extracting entities, generating summaries, classifying content, and persisting structured output to MongoDB.

It is designed as a practical application codebase rather than a framework. The current architecture centers on a single package, `news_crawler/`, with clear separation between scraping, processing, providers, repositories, and orchestration.

## What It Does

- Scrapes articles from RSS and dynamic sources
- Deduplicates already-seen URLs
- Cleans article text and extracts entities with Gemini or Ollama
- Summarizes text with Hugging Face BART
- Classifies sentiment and topic
- Persists enriched results and pipeline metadata to MongoDB
- Supports resumable state and downstream webhooks

## Current LLM Support

The project currently supports two text-cleaning and entity-extraction backends:

- Google GenAI / Gemini
- Ollama

Selection behavior:

- If `ENABLE_GENAI=1` and a valid Google API key is present, Gemini is preferred.
- If Gemini is unavailable and `ENABLE_OLLAMA=1`, startup falls back to Ollama.
- If both are disabled, the pipeline still runs, but LLM-based cleaning and entity extraction are skipped.

This selection happens at startup. Changing `.env` values requires restarting the process.

## Tech Stack

- Python 3.10+
- MongoDB with PyMongo
- Trafilatura, BeautifulSoup4, Feedparser, Selenium
- Hugging Face Transformers and PyTorch
- Google GenAI SDK
- Typer and Rich

## Project Layout

```text
news-crawler-ai/
├── news_crawler/
│   ├── core/           # Config, orchestrator, state
│   ├── db/             # Mongo client
│   ├── processors/     # Summarization and classification
│   ├── providers/      # Gemini, Ollama, webhooks
│   ├── repositories/   # Mongo repositories
│   ├── scrapers/       # Source integrations
│   └── utils/          # Logging and text helpers
├── scripts/            # Bootstrap and maintenance commands
├── tests/              # Pytest suite
└── main.py             # CLI entry point
```

## Installation

Recommended development setup:

```bash
poetry install
poetry shell
```

Alternative setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create a local environment file:

```bash
cp .env.example .env
```

Important variables:

- `MONGO_URI`
- `MONGODB_DB`
- `HF_HOME`
- `ENABLE_GENAI`
- `ENABLE_OLLAMA`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `OLLAMA_HOST`
- `OLLAMA_PORT`
- `OLLAMA_MODEL`
- `ENABLE_SELENIUM_SCRAPERS`
- `ENABLE_NEWSAPI_SCRAPER`
- `NEWSAPI_KEY`
- `ENABLE_WEBHOOKS`
- `WEBHOOK_URL`
- `WEBHOOK_URL_THREAD_EVENTS`

### Minimal Local Modes

Dry-run without MongoDB writes:

```bash
news-crawler run --dry-run --limit 10
```

Gemini-enabled run:

```bash
ENABLE_GENAI=1 ENABLE_OLLAMA=0 news-crawler run --limit 10
```

Ollama-only run:

```bash
ENABLE_GENAI=0 ENABLE_OLLAMA=1 news-crawler run --limit 10
```

## Usage

Installed CLI:

```bash
news-crawler run
news-crawler run --dry-run
news-crawler run --limit 50
news-crawler run --resume
news-crawler run --reset-state
news-crawler run --verbose
news-crawler status
news-crawler reset
```

Direct entrypoint:

```bash
python main.py run
python main.py run --limit 50
python main.py status
```

Maintenance commands:

```bash
news-crawler-bootstrap
news-crawler-bootstrap-indexes
news-crawler-bootstrap-indexes --dedupe
news-crawler-dedupe-articles --limit 100
news-crawler-dedupe-articles --apply
```

## MongoDB Collections

- `articles`: enriched article documents
- `link_pool`: deduplication and processing state
- `metadata`: batch-level statistics
- `pipeline_logs`: structured pipeline events

## Testing

Run the test suite:

```bash
pytest
```

## Operational Notes

- Selenium-based scrapers are inherently more brittle than RSS-based scrapers.
- NewsAPI scraping is optional and only runs when `ENABLE_NEWSAPI_SCRAPER=1` and `NEWSAPI_KEY` is configured.
- Webhooks are optional and only run when enabled and configured.
- Hugging Face models are cached under `HF_HOME`.

## Legal and Usage Notes

- You are responsible for complying with the terms of service, licenses, and access policies of the upstream sites and APIs you use with this project.
- This repository includes scrapers for public news sources, but source markup and access restrictions may change at any time.
- Model-based cleaning, entity extraction, summarization, and classification are best-effort features and may produce inaccurate output.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [SECURITY.md](SECURITY.md).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
